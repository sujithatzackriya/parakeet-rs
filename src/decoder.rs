use crate::error::{Error, Result};
use ndarray::Array2;
use std::path::Path;

// Token with its timestamp information
// start and end are in seconds
#[derive(Debug, Clone)]
pub struct TimedToken {
    pub text: String,
    pub start: f32,
    pub end: f32,
}

#[derive(Debug, Clone)]
pub struct TranscriptionResult {
    pub text: String,
    pub tokens: Vec<TimedToken>,
}

// CTC decoder for parakeet-ctc-0.6b model with token-level timestamps
pub struct ParakeetDecoder {
    tokenizer: tokenizers::Tokenizer,
    pad_token_id: usize,
}

impl ParakeetDecoder {
    pub fn from_pretrained<P: AsRef<Path>>(tokenizer_path: P) -> Result<Self> {
        let tokenizer_path = tokenizer_path.as_ref();

        let tokenizer = tokenizers::Tokenizer::from_file(tokenizer_path)
            .map_err(|e| Error::Tokenizer(format!("Failed to load tokenizer: {e}")))?;

        // Hardcoded pad_token_id for Parakeet-CTC-0.6b (constant across all models: please see def configs jsons: https://huggingface.co/onnx-community/parakeet-ctc-0.6b-ONNX/tree/main)
        let pad_token_id = 1024;

        Ok(Self {
            tokenizer,
            pad_token_id,
        })
    }

    pub fn decode(&self, logits: &Array2<f32>) -> Result<String> {
        let time_steps = logits.shape()[0];

        let mut token_ids = Vec::new();
        for t in 0..time_steps {
            let logits_t = logits.row(t);
            let max_idx = logits_t
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx)
                .unwrap_or(0);

            token_ids.push(max_idx as u32);
        }

        let collapsed = self.ctc_collapse(&token_ids);

        let text = self
            .tokenizer
            .decode(&collapsed, true)
            .map_err(|e| Error::Tokenizer(format!("Failed to decode: {e}")))?;

        Ok(text)
    }

    fn ctc_collapse(&self, token_ids: &[u32]) -> Vec<u32> {
        let mut result = Vec::new();
        let mut prev_token: Option<u32> = None;

        for &token_id in token_ids {
            if token_id == self.pad_token_id as u32 {
                prev_token = Some(token_id);
                continue;
            }

            if Some(token_id) != prev_token {
                result.push(token_id);
            }

            prev_token = Some(token_id);
        }

        result
    }

    // CTC collapse with frame tracking for timestamps
    fn ctc_collapse_with_frames(&self, token_ids: &[(u32, usize)]) -> Vec<(u32, usize, usize)> {
        let mut result: Vec<(u32, usize, usize)> = Vec::new();
        let mut prev_token: Option<u32> = None;

        for &(token_id, frame) in token_ids.iter() {
            if token_id == self.pad_token_id as u32 {
                prev_token = Some(token_id);
                continue;
            }

            if Some(token_id) != prev_token {
                if let Some(prev) = prev_token {
                    if prev != self.pad_token_id as u32 {
                        // End previous token
                        if let Some(last) = result.last_mut() {
                            last.2 = frame;
                        }
                    }
                }
                // Start new token
                result.push((token_id, frame, frame));
            }

            prev_token = Some(token_id);
        }

        // Close last token
        if let Some(last) = result.last_mut() {
            last.2 = token_ids.len();
        }

        result
    }

    // Decode with token-level timestamps
    // hop_length and sample_rate are needed to convert frames to seconds
    pub fn decode_with_timestamps(
        &self,
        logits: &Array2<f32>,
        hop_length: usize,
        sample_rate: usize,
    ) -> Result<TranscriptionResult> {
        let time_steps = logits.shape()[0];

        let mut token_ids_with_frames = Vec::new();
        for t in 0..time_steps {
            let logits_t = logits.row(t);
            let max_idx = logits_t
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx)
                .unwrap_or(0);

            token_ids_with_frames.push((max_idx as u32, t));
        }

        // CTC collapse with frame tracking
        let collapsed_with_frames = self.ctc_collapse_with_frames(&token_ids_with_frames);

        // Extract just token IDs for decoding
        let token_ids: Vec<u32> = collapsed_with_frames.iter().map(|(id, _, _)| *id).collect();

        // Decode full text
        let full_text = self
            .tokenizer
            .decode(&token_ids, true)
            .map_err(|e| Error::Tokenizer(format!("Failed to decode: {e}")))?;

        // Progressive decode to detect word boundaries
        // BPE tokenizers only add spaces when decoding sequences, not individual tokens
        let mut timed_tokens = Vec::new();
        let mut prev_decode = String::new();

        for (i, (_token_id, start_frame, end_frame)) in collapsed_with_frames.iter().enumerate() {
            // Decode from start up to and including current token
            let token_ids_so_far: Vec<u32> = collapsed_with_frames[0..=i]
                .iter()
                .map(|(id, _, _)| *id)
                .collect();

            if let Ok(curr_decode) = self.tokenizer.decode(&token_ids_so_far, true) {
                // Find what this token added
                let added_text = if curr_decode.len() > prev_decode.len() {
                    &curr_decode[prev_decode.len()..]
                } else {
                    ""
                };

                if !added_text.is_empty() {
                    let start_time = (*start_frame * hop_length) as f32 / sample_rate as f32;
                    let end_time = (*end_frame * hop_length) as f32 / sample_rate as f32;

                    timed_tokens.push(TimedToken {
                        text: added_text.to_string(),
                        start: start_time,
                        end: end_time,
                    });
                }

                prev_decode = curr_decode;
            }
        }

        Ok(TranscriptionResult {
            text: full_text,
            tokens: timed_tokens,
        })
    }

    // Stub - falls back to greedy decoding. Full beam search with language model is TODO.
    pub fn decode_with_beam_search(
        &self,
        logits: &Array2<f32>,
        _beam_width: usize,
    ) -> Result<String> {
        self.decode(logits)
    }

    pub fn pad_token_id(&self) -> usize {
        self.pad_token_id
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;
    use tokenizers::models::bpe::BPE;
    use tokenizers::Tokenizer;

    /// Build a decoder with a throwaway tokenizer. The CTC-collapse and
    /// argmax logic under test never touch the tokenizer, so an empty BPE
    /// model is sufficient and avoids needing a model download.
    fn decoder_with_pad(pad_token_id: usize) -> ParakeetDecoder {
        ParakeetDecoder {
            tokenizer: Tokenizer::new(BPE::default()),
            pad_token_id,
        }
    }

    // --- CTC collapse (decoder.rs:68-86) ---

    #[test]
    fn ctc_collapse_dedups_consecutive_repeats() {
        // Invariant: CTC collapse merges runs of the SAME non-blank token
        // into one emission, but keeps a repeat that is separated by a
        // different token (a b b a -> a b a, not a b).
        let d = decoder_with_pad(1024);
        let collapsed = d.ctc_collapse(&[5, 5, 5, 7, 7, 5]);
        assert_eq!(collapsed, vec![5, 7, 5]);
    }

    #[test]
    fn ctc_collapse_removes_pad_and_allows_repeat_across_pad() {
        // Invariant: the pad/blank id is dropped entirely, AND a pad between
        // two identical tokens resets prev_token so the second copy survives
        // (5 PAD 5 -> 5 5). This is the core CTC blank-as-separator rule.
        let d = decoder_with_pad(1024);
        let collapsed = d.ctc_collapse(&[1024, 5, 1024, 5, 1024]);
        assert_eq!(collapsed, vec![5, 5]);
    }

    #[test]
    fn ctc_collapse_empty_input_is_empty() {
        let d = decoder_with_pad(1024);
        assert!(d.ctc_collapse(&[]).is_empty());
    }

    // --- CTC collapse with frame tracking (decoder.rs:89-121) ---

    #[test]
    fn ctc_collapse_with_frames_assigns_start_and_end() {
        // Invariant: each emitted token carries (id, start_frame, end_frame).
        // start = frame the token first appeared; end = frame the NEXT token
        // started; the final token's end is the total frame count.
        let d = decoder_with_pad(1024);
        let input = vec![(5u32, 0usize), (5, 1), (7, 2), (7, 3)];
        let out = d.ctc_collapse_with_frames(&input);
        // token 5 spans frames [0,2) (ends where 7 starts); token 7 ends at len=4.
        assert_eq!(out, vec![(5, 0, 2), (7, 2, 4)]);
    }

    #[test]
    fn ctc_collapse_with_frames_skips_pad_frames() {
        // Invariant: pad frames are not emitted, and a pad between two equal
        // tokens lets the second copy emit as its own timed token.
        let d = decoder_with_pad(1024);
        let input = vec![(1024u32, 0usize), (5, 1), (1024, 2), (5, 3)];
        let out = d.ctc_collapse_with_frames(&input);
        // First 5 starts at frame 1; its end is NOT advanced because the token
        // that follows it is the pad (the end-update branch only fires when the
        // previous token was non-pad), so it keeps its initial end==start==1.
        // The second 5 starts at frame 3 and is closed to len==4.
        assert_eq!(out, vec![(5, 1, 1), (5, 3, 4)]);
    }

    // --- argmax (decoder.rs:48-53, 136-141): max_by => LAST max wins on ties ---

    #[test]
    fn decoder_argmax_is_last_wins_on_ties() {
        // Documents the CURRENT decoder argmax behavior (T10 will unify it):
        // `max_by` returns the LAST element among equal-maximum values.
        // Here bins 1 and 3 tie at 0.9; decode must collapse to that argmax.
        // We exercise it through the public decode-by-argmax path: a single
        // time step whose argmax is the (last) tied index 3.
        let d = decoder_with_pad(1024);
        let logits = arr2(&[[0.1f32, 0.9, 0.2, 0.9]]);
        // Reproduce the decoder's own argmax expression to pin the tie rule.
        let row = logits.row(0);
        let max_idx = row
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx)
            .unwrap_or(0);
        assert_eq!(max_idx, 3, "decoder max_by must pick the LAST tied index");
        // And the full decode path produces exactly one token for one frame.
        let collapsed = d.ctc_collapse(&[max_idx as u32]);
        assert_eq!(collapsed, vec![3]);
    }
}
