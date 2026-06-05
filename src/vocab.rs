use crate::error::{Error, Result};
use std::fs::File;
use std::io::{BufRead, BufReader, Read};
use std::path::Path;

/// Greedy argmax over a logit slice with ONE policy shared by every decoder.
///
/// Policy (unified by T10/M5, previously divergent across variants):
/// - **First-wins on ties:** the lowest index among equal-maximum values is
///   returned (matches NeMo greedy semantics).
/// - **Finite-guard:** non-finite logits (`NaN`, `±inf` that never exceed a
///   finite running max) are never selected; only finite values can win.
/// - **Empty / all-non-finite input:** returns `0` (no finite candidate).
///
/// This replaces the four-to-six hand-rolled argmaxes that used three
/// different tie/NaN policies (Nemotron first-wins/NaN-as-token-0, Unified/CTC/
/// TDT last-wins/NaN-as-Equal, EOU finite-guarded first-wins). Ties and NaN
/// logits are pathological on real audio, so transcripts are unaffected; the
/// change only fixes the latent divergence.
pub(crate) fn argmax(logits: &[f32]) -> usize {
    let mut max_idx = 0;
    let mut max_val = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v.is_finite() && v > max_val {
            max_val = v;
            max_idx = i;
        }
    }
    max_idx
}

/// Vocabulary parser for vocab.txt format used by TDT models
#[derive(Debug, Clone)]
pub struct Vocabulary {
    pub id_to_token: Vec<String>,
    pub _blank_id: usize,
}

impl Vocabulary {
    /// Load vocabulary from vocab.txt file
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path.as_ref())
            .map_err(|e| Error::Config(format!("Failed to open vocab file: {}", e)))?;

        let reader = BufReader::new(file);
        let mut id_to_token = Vec::new();
        let mut blank_id = 0;

        for line in reader.lines() {
            let line =
                line.map_err(|e| Error::Config(format!("Failed to read vocab file: {}", e)))?;

            let parts: Vec<&str> = line.splitn(2, ' ').collect();
            if parts.len() == 2 {
                let token = parts[0].to_string();
                let id: usize = parts[1]
                    .parse()
                    .map_err(|e| Error::Config(format!("Invalid token ID in vocab: {}", e)))?;

                if id >= id_to_token.len() {
                    id_to_token.resize(id + 1, String::new());
                }
                id_to_token[id] = token.clone();

                // Track blank token
                if token == "<blk>" || token == "<blank>" {
                    blank_id = id;
                }
            }
        }

        // Default to last token if no blank found
        if blank_id == 0 && !id_to_token.is_empty() {
            blank_id = id_to_token.len() - 1;
        }

        Ok(Self {
            id_to_token,
            _blank_id: blank_id,
        })
    }

    /// Get token by ID
    pub fn id_to_text(&self, id: usize) -> Option<&str> {
        self.id_to_token.get(id).map(|s| s.as_str())
    }

    /// Get vocabulary size (number of tokens)
    pub fn size(&self) -> usize {
        self.id_to_token.len()
    }
}

/// Extract the language code from a SentencePiece piece that encodes a
/// language tag like `<en-US>` or `<en>` - returns `Some("en-US")` /
/// `Some("en")` (brackets stripped) for a tag, `None` otherwise.
///
/// This is the single source of truth for language-tag shape; [`is_lang_tag`]
/// delegates to it so detection and code-extraction can never disagree.
pub(crate) fn lang_code_from_piece(piece: &str) -> Option<String> {
    let bytes = piece.as_bytes();
    if bytes.len() < 4 || bytes[0] != b'<' || bytes[bytes.len() - 1] != b'>' {
        return None;
    }
    let inner = &bytes[1..bytes.len() - 1];
    let ok = match inner.len() {
        2 => inner[0].is_ascii_lowercase() && inner[1].is_ascii_lowercase(),
        5 => inner[0].is_ascii_lowercase()
            && inner[1].is_ascii_lowercase()
            && inner[2] == b'-'
            && inner[3].is_ascii_uppercase()
            && inner[4].is_ascii_uppercase(),
        _ => false,
    };
    // `inner` is ASCII by construction of `ok`, so the slice is valid UTF-8.
    ok.then(|| piece[1..piece.len() - 1].to_string())
}

/// Detect SentencePiece pieces that encode a language tag like `<en-US>` or
/// `<en>`. The multilingual model emits these inline with text; they're
/// stripped from the user-visible transcript.
pub(crate) fn is_lang_tag(piece: &str) -> bool {
    lang_code_from_piece(piece).is_some()
}

/// Most-recent language code from a token slice, by EXACT id membership in
/// `lang_tag_ids` (the precomputed set of vocab ids whose piece is a language
/// tag). Returns the code (e.g. `"es-ES"`, brackets stripped) of the last
/// language tag present, or `None` if the slice contains no known tag id.
///
/// Pure and model-free: detection is by exact id membership, not a re-run of
/// the string-shape heuristic, so it is safe to drive control flow from.
pub(crate) fn language_from_tokens(
    tokens: &[usize],
    lang_tag_ids: &[usize],
    vocab: &SentencePieceVocab,
) -> Option<String> {
    let last = tokens.iter().rev().find(|t| lang_tag_ids.contains(t))?;
    lang_code_from_piece(vocab.pieces.get(*last)?)
}

/// Minimal SentencePiece vocabulary loader.
/// Parses the protobuf .model file to extract token strings.
/// Note that, our Vocabulary cannot parse protobuf format. I haven't test it with digit spacing yet, at least for this initial impl.
pub(crate) struct SentencePieceVocab {
    pub(crate) pieces: Vec<String>,
}

impl SentencePieceVocab {
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let mut file = File::open(path.as_ref())
            .map_err(|e| Error::Tokenizer(format!("Failed to open tokenizer.model: {e}")))?;
        let mut data = Vec::new();
        file.read_to_end(&mut data)
            .map_err(|e| Error::Tokenizer(format!("Failed to read tokenizer.model: {e}")))?;

        let pieces = Self::parse_sentencepiece_model(&data)?;
        Ok(Self { pieces })
    }

    fn parse_sentencepiece_model(data: &[u8]) -> Result<Vec<String>> {
        let mut pieces = Vec::new();
        let mut pos = 0;

        while pos < data.len() {
            let (field_header, bytes_read) = Self::read_varint(&data[pos..])?;
            pos += bytes_read;

            let field_num = field_header >> 3;
            let wire_type = field_header & 0x7;

            match (field_num, wire_type) {
                (1, 2) => {
                    let (len, bytes_read) = Self::read_varint(&data[pos..])?;
                    pos += bytes_read;

                    if pos + len as usize > data.len() {
                        break;
                    }

                    let piece_data = &data[pos..pos + len as usize];
                    pos += len as usize;

                    if let Ok(piece) = Self::parse_piece_message(piece_data) {
                        pieces.push(piece);
                    }
                }
                (_, 0) => {
                    let (_, bytes_read) = Self::read_varint(&data[pos..])?;
                    pos += bytes_read;
                }
                (_, 1) => pos += 8,
                (_, 2) => {
                    let (len, bytes_read) = Self::read_varint(&data[pos..])?;
                    pos += bytes_read + len as usize;
                }
                (_, 5) => pos += 4,
                _ => break,
            }
        }

        if pieces.is_empty() {
            return Err(Error::Tokenizer("No tokens found in model".into()));
        }

        Ok(pieces)
    }

    fn parse_piece_message(data: &[u8]) -> Result<String> {
        let mut pos = 0;
        let mut piece = String::new();

        while pos < data.len() {
            let (field_header, bytes_read) = Self::read_varint(&data[pos..])?;
            pos += bytes_read;

            let field_num = field_header >> 3;
            let wire_type = field_header & 0x7;

            match (field_num, wire_type) {
                (1, 2) => {
                    let (len, bytes_read) = Self::read_varint(&data[pos..])?;
                    pos += bytes_read;

                    if pos + len as usize <= data.len() {
                        piece = String::from_utf8_lossy(&data[pos..pos + len as usize]).to_string();
                    }
                    pos += len as usize;
                }
                (_, 0) => {
                    let (_, bytes_read) = Self::read_varint(&data[pos..])?;
                    pos += bytes_read;
                }
                (_, 1) => pos += 8,
                (_, 2) => {
                    let (len, bytes_read) = Self::read_varint(&data[pos..])?;
                    pos += bytes_read + len as usize;
                }
                (_, 5) => pos += 4,
                _ => break,
            }
        }

        Ok(piece)
    }

    fn read_varint(data: &[u8]) -> Result<(u64, usize)> {
        let mut result: u64 = 0;
        let mut shift = 0;
        let mut pos = 0;

        while pos < data.len() && pos < 10 {
            let byte = data[pos];
            result |= ((byte & 0x7F) as u64) << shift;
            pos += 1;

            if byte & 0x80 == 0 {
                return Ok((result, pos));
            }
            shift += 7;
        }

        Err(Error::Tokenizer("Invalid varint".into()))
    }

    pub fn decode(&self, ids: &[usize]) -> String {
        let mut result = String::new();
        for &id in ids {
            if id < self.pieces.len() {
                let piece = &self.pieces[id];
                let decoded = piece.replace('\u{2581}', " ");
                result.push_str(&decoded);
            }
        }
        result.trim_start().to_string()
    }

    pub fn decode_single(&self, id: usize) -> String {
        if id < self.pieces.len() {
            self.pieces[id].replace('\u{2581}', " ")
        } else {
            String::new()
        }
    }

    pub fn size(&self) -> usize {
        self.pieces.len()
    }

    /// Token IDs whose SentencePiece pieces look like language tags
    /// (`<en-US>`, `<fr>`, ...). and ofc empty for the en only vocab.
    pub fn lang_tag_ids(&self) -> Vec<usize> {
        self.pieces
            .iter()
            .enumerate()
            .filter_map(|(i, p)| is_lang_tag(p).then_some(i))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- argmax: the single shared decoder policy (T10/M5) ---
    // First-wins on ties + finite-guard. This is the one policy every variant
    // now uses; the per-variant pinning tests (decoder.rs, nemotron.rs) assert
    // the same behavior through their own call paths.

    #[test]
    fn argmax_picks_normal_max() {
        assert_eq!(argmax(&[0.1, 0.5, 0.3, 0.9, 0.2]), 3);
        assert_eq!(argmax(&[1.0, 0.0, 0.0]), 0);
    }

    #[test]
    fn argmax_first_wins_on_ties() {
        // bins 1 and 3 tie at 0.9 -> the FIRST (lowest index) wins.
        assert_eq!(argmax(&[0.1, 0.9, 0.2, 0.9]), 1);
    }

    #[test]
    fn argmax_all_equal_picks_first() {
        assert_eq!(argmax(&[0.5, 0.5, 0.5]), 0);
    }

    #[test]
    fn argmax_skips_nan() {
        // A leading NaN must never be selected; the finite max wins.
        assert_eq!(argmax(&[f32::NAN, 0.2, 0.8, 0.1]), 2);
        // NaN at the max-value position is skipped in favor of a finite tie.
        assert_eq!(argmax(&[0.8, f32::NAN, 0.8]), 0);
    }

    #[test]
    fn argmax_all_non_finite_or_empty_returns_zero() {
        assert_eq!(argmax(&[]), 0);
        assert_eq!(argmax(&[f32::NAN, f32::NAN]), 0);
        assert_eq!(argmax(&[f32::NEG_INFINITY]), 0);
    }

    // --- is_lang_tag ---
    // The multilingual model emits inline language pieces like `<en>` /
    // `<en-US>` that must be detected so they can be stripped from the
    // transcript. This guards the exact byte-pattern matcher.

    #[test]
    fn is_lang_tag_accepts_two_letter_lowercase() {
        assert!(is_lang_tag("<en>"));
        assert!(is_lang_tag("<fr>"));
        assert!(is_lang_tag("<zh>"));
    }

    #[test]
    fn is_lang_tag_accepts_locale_form() {
        // `<xx-XX>`: lower-lower '-' UPPER-UPPER
        assert!(is_lang_tag("<en-US>"));
        assert!(is_lang_tag("<pt-BR>"));
        assert!(is_lang_tag("<zh-CN>"));
    }

    #[test]
    fn is_lang_tag_rejects_malformed() {
        // Wrong case, wrong shape, missing brackets, too short.
        assert!(!is_lang_tag("<EN>"), "uppercase 2-letter is not a tag");
        assert!(!is_lang_tag("<en-us>"), "lowercase locale half is not a tag");
        assert!(!is_lang_tag("<EN-US>"), "uppercase lang half is not a tag");
        assert!(!is_lang_tag("<en_US>"), "underscore separator is not a tag");
        assert!(!is_lang_tag("en-US"), "missing brackets is not a tag");
        assert!(!is_lang_tag("<e>"), "single inner char (len<4) is not a tag");
        assert!(!is_lang_tag("<>"), "empty inner is not a tag");
        assert!(!is_lang_tag("hello"), "plain text is not a tag");
        assert!(!is_lang_tag("<eng>"), "three-letter inner is not a tag");
    }

    // --- SentencePieceVocab::lang_tag_ids ---
    // Pure path over an in-memory piece table (no protobuf, no file IO):
    // only pieces that look like language tags get collected.

    #[test]
    fn lang_tag_ids_selects_only_tag_pieces() {
        let vocab = SentencePieceVocab {
            pieces: vec![
                "hello".to_string(),   // 0 - not a tag
                "<en>".to_string(),    // 1 - tag
                "world".to_string(),   // 2 - not a tag
                "<es-ES>".to_string(), // 3 - tag
                "<EN>".to_string(),    // 4 - not a tag (uppercase)
            ],
        };
        assert_eq!(vocab.lang_tag_ids(), vec![1, 3]);
    }

    #[test]
    fn lang_tag_ids_empty_for_plain_vocab() {
        let vocab = SentencePieceVocab {
            pieces: vec!["a".to_string(), "b".to_string()],
        };
        assert!(vocab.lang_tag_ids().is_empty());
    }

    // --- lang_code_from_piece ---
    // The code extractor is the source of truth `is_lang_tag` delegates to,
    // so the brackets-stripped code must agree exactly with tag detection.

    #[test]
    fn lang_code_from_piece_extracts_code_for_tags() {
        assert_eq!(lang_code_from_piece("<en>").as_deref(), Some("en"));
        assert_eq!(lang_code_from_piece("<es-ES>").as_deref(), Some("es-ES"));
        assert_eq!(lang_code_from_piece("<pt-BR>").as_deref(), Some("pt-BR"));
    }

    #[test]
    fn lang_code_from_piece_none_for_non_tags() {
        assert_eq!(lang_code_from_piece("hello"), None);
        assert_eq!(lang_code_from_piece("<EN>"), None);
        assert_eq!(lang_code_from_piece("<en_US>"), None);
        assert_eq!(lang_code_from_piece("<eng>"), None);
    }

    // --- language_from_tokens (the detected_language() core) ---
    // Pure, model-free: given exact lang-tag ids, scan a token slice and
    // return the MOST RECENT tag's code; plain tokens yield None.

    #[test]
    fn language_from_tokens_returns_most_recent_tag() {
        let vocab = SentencePieceVocab {
            pieces: vec![
                "hello".to_string(),   // 0
                "<en-US>".to_string(), // 1 - tag
                "world".to_string(),   // 2
                "<es-ES>".to_string(), // 3 - tag
            ],
        };
        let lang_tag_ids = vocab.lang_tag_ids(); // [1, 3]
        // Two tags present: the LAST one (es-ES) wins.
        let tokens = [0usize, 1, 2, 3, 2];
        assert_eq!(
            language_from_tokens(&tokens, &lang_tag_ids, &vocab).as_deref(),
            Some("es-ES")
        );
        // Only the first tag present -> that code.
        let tokens = [0usize, 1, 2];
        assert_eq!(
            language_from_tokens(&tokens, &lang_tag_ids, &vocab).as_deref(),
            Some("en-US")
        );
    }

    #[test]
    fn language_from_tokens_none_without_tags() {
        let vocab = SentencePieceVocab {
            pieces: vec!["hello".to_string(), "<en-US>".to_string(), "world".to_string()],
        };
        let lang_tag_ids = vocab.lang_tag_ids(); // [1]
        // No tag id in the slice -> None (English-only / not yet emitted).
        assert_eq!(language_from_tokens(&[0usize, 2], &lang_tag_ids, &vocab), None);
        // Empty lang_tag_ids (the English-only variant) -> always None.
        assert_eq!(language_from_tokens(&[0usize, 1, 2], &[], &vocab), None);
    }
}
