//! A common streaming contract across the bespoke streaming variants.
//!
//! Before this trait, every streaming wrapper (`Nemotron`, `ParakeetEOU`,
//! `ParakeetUnified`, `MultitalkerASR`) invented its own verb, reset spelling,
//! and output shape, so the crate could not be driven generically. The
//! [`StreamingTranscriber`] trait captures the shape they actually share:
//!
//! - feed a chunk of `f32` samples and get an incremental result back,
//! - [`reset`](StreamingTranscriber::reset) the per-stream state for a new
//!   utterance,
//! - optionally [`flush`](StreamingTranscriber::flush) any buffered tail at
//!   stream end.
//!
//! # Associated output type
//!
//! The variants differ in *what* a chunk yields: text variants return a
//! `String`, while the multi-talker pipeline returns one transcript per
//! speaker. That is modelled with an associated [`Output`](StreamingTranscriber::Output)
//! type rather than a single concrete return, so each impl keeps its natural
//! shape:
//!
//! - [`Nemotron`](crate::Nemotron): `Output = String`
//! - [`ParakeetEOU`](crate::ParakeetEOU): `Output = String`
//! - [`ParakeetUnified`](crate::ParakeetUnified): `Output = String`
//! - [`MultitalkerASR`](crate::MultitalkerASR): `Output = Vec<SpeakerTranscript>`
//!
//! # Object safety
//!
//! Because the trait has an associated type, a trait object must name it:
//! `dyn StreamingTranscriber<Output = String>` is a valid object type (the
//! methods are all object-safe), but a bare `dyn StreamingTranscriber` is not,
//! since the associated type is unbound. In practice, prefer a generic bound
//! (`fn run<T: StreamingTranscriber>(t: &mut T)`); reach for a trait object
//! only when you have pinned the output, e.g.
//! `Box<dyn StreamingTranscriber<Output = String>>` to hold any of the
//! text-yielding variants behind one handle.
//!
//! # Not included: Sortformer
//!
//! [`Sortformer`](crate::sortformer) is intentionally left out. It streams
//! *speaker diarization* (per-frame speaker segments), not transcription: its
//! output is `Vec<SpeakerSegment>`, its reset is `reset_state`, and it exposes
//! two distinct streaming entry points (`diarize_chunk` for stateless
//! per-chunk segmentation and `feed` for buffered absolute-timestamp
//! streaming). Forcing it under a transcription trait would misname its domain
//! without giving callers anything they cannot already get from its inherent
//! methods, so it keeps its own surface.

use crate::error::Result;

/// The shared streaming contract: feed audio chunks, reset between utterances,
/// and flush any buffered tail at stream end.
///
/// Implemented by the streaming ASR variants ([`Nemotron`](crate::Nemotron),
/// [`ParakeetEOU`](crate::ParakeetEOU), [`ParakeetUnified`](crate::ParakeetUnified),
/// and [`MultitalkerASR`](crate::MultitalkerASR)). Each trait method delegates
/// to the variant's existing inherent method, so adopting the trait changes no
/// transcription behavior. See the [module docs](self) for the object-safety
/// note and why Sortformer is excluded.
pub trait StreamingTranscriber {
    /// The incremental result a single chunk yields. `String` for the
    /// text variants; a richer type (e.g. `Vec<SpeakerTranscript>`) for
    /// per-speaker output.
    type Output: Default;

    /// Feed one chunk of `f32` audio samples (16 kHz mono) and return the
    /// incremental output decoded from it. Call repeatedly for real-time
    /// streaming; the per-stream decode state carries across calls in order.
    fn transcribe_chunk(&mut self, audio: &[f32]) -> Result<Self::Output>;

    /// Reset all per-stream state so the next chunk starts a fresh utterance.
    ///
    /// Every implementor exposes this with identical meaning, replacing the
    /// previously ad-hoc per-variant reset surface.
    fn reset(&mut self);

    /// Drain any buffered audio that has not yet formed a full chunk and emit
    /// the remaining output. The default returns an empty [`Output`](Self::Output)
    /// for variants that buffer nothing across chunk boundaries; variants with a
    /// trailing buffer (e.g. [`Nemotron`](crate::Nemotron),
    /// [`ParakeetUnified`](crate::ParakeetUnified)) override it to drain the tail.
    fn flush(&mut self) -> Result<Self::Output> {
        Ok(Self::Output::default())
    }
}

#[cfg(test)]
mod tests {
    use super::StreamingTranscriber;
    use crate::error::Result;

    /// Minimal in-memory implementor: proves the trait is usable end to end
    /// (chunk -> reset -> flush) with no model download, and that the `flush`
    /// default fires for a variant that does not override it.
    #[derive(Default)]
    struct Counter {
        seen: usize,
        reset_calls: usize,
    }

    impl StreamingTranscriber for Counter {
        type Output = String;

        fn transcribe_chunk(&mut self, audio: &[f32]) -> Result<String> {
            self.seen += audio.len();
            Ok(format!("[{}]", audio.len()))
        }

        fn reset(&mut self) {
            self.reset_calls += 1;
            self.seen = 0;
        }
        // `flush` intentionally not overridden: exercises the default.
    }

    /// Drive any implementor generically through the full streaming contract.
    /// This is the proof that callers can write model-agnostic streaming code
    /// against the trait bound rather than a concrete variant.
    fn drive<T: StreamingTranscriber>(t: &mut T, chunks: &[&[f32]]) -> Result<Vec<T::Output>> {
        let mut out = Vec::new();
        for c in chunks {
            out.push(t.transcribe_chunk(c)?);
        }
        out.push(t.flush()?);
        Ok(out)
    }

    #[test]
    fn trait_is_usable_through_a_generic_bound() {
        let mut c = Counter::default();
        let outs = drive(&mut c, &[&[0.0; 3], &[0.0; 5]]).unwrap();
        // two chunk outputs + one (default, empty) flush output
        assert_eq!(outs, vec!["[3]".to_string(), "[5]".to_string(), String::new()]);
        assert_eq!(c.seen, 8, "generic driver fed both chunks through the trait");

        c.reset();
        assert_eq!(c.reset_calls, 1);
        assert_eq!(c.seen, 0, "reset() cleared per-stream state via the trait");
    }

    /// Compile-time assertion that the real streaming variants implement the
    /// trait with the expected `Output` types. If any impl is dropped or its
    /// `Output` changes, this fails to compile.
    #[test]
    fn real_variants_implement_the_trait() {
        fn assert_impl<T: StreamingTranscriber<Output = O>, O>() {}
        assert_impl::<crate::Nemotron, String>();
        assert_impl::<crate::ParakeetEOU, String>();
        assert_impl::<crate::ParakeetUnified, String>();
        #[cfg(feature = "multitalker")]
        assert_impl::<crate::MultitalkerASR, Vec<crate::SpeakerTranscript>>();
    }
}
