//! Typed language selection shared across ASR variants.
//!
//! [`Language`] is the single canonical spelling for a target language. It maps
//! to/from the code strings the engines already use (Nemotron's prompt
//! dictionary, Cohere's supported-language list) WITHOUT changing which code a
//! given language resolves to: every [`Language`] has an exact [`Language::as_str`]
//! that is fed unchanged into each engine's existing lookup table, so behavior is
//! preserved. `auto` is a first-class [`Language::Auto`] variant rather than a
//! magic string, and [`Language::Other`] is an escape hatch for codes not
//! enumerated here (e.g. experimental locales) so no code is rejected that worked
//! before.
//!
//! String call sites keep working: [`From<&str>`](Language::from) /
//! [`FromStr`](std::str::FromStr) parse a code into a [`Language`], and engines
//! that take `impl Into<Language>` accept `"ja-JP"` / `"auto"` exactly as before.

use std::borrow::Cow;
use std::fmt;
use std::str::FromStr;

use crate::error::Error;

/// A target language for transcription.
///
/// The enumerated variants cover the most common locales; any other code is
/// carried verbatim in [`Language::Other`]. The [`Language::as_str`] of every
/// value is the exact code string the engines look up, so a [`Language`] always
/// resolves to the same engine behavior as the equivalent `&str` did.
///
/// `#[non_exhaustive]` so additional named locales can be added later without a
/// breaking change.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum Language {
    /// Language-agnostic decoding: the model picks the language itself
    /// (Nemotron prompt index 101). Not a fixed language; a first-class state.
    Auto,
    /// English (US): code `"en-US"` (also the bare `"en"` Nemotron alias and the
    /// Cohere `"en"` code resolve here via [`Language::from`]).
    English,
    /// Spanish (Spain): code `"es-ES"`.
    Spanish,
    /// French (France): code `"fr-FR"`.
    French,
    /// German: code `"de-DE"`.
    German,
    /// Italian: code `"it-IT"`.
    Italian,
    /// Portuguese (Portugal): code `"pt-PT"`.
    Portuguese,
    /// Dutch: code `"nl-NL"`.
    Dutch,
    /// Russian: code `"ru-RU"`.
    Russian,
    /// Arabic: code `"ar"`.
    Arabic,
    /// Hindi: code `"hi-IN"`.
    Hindi,
    /// Japanese: code `"ja-JP"`.
    Japanese,
    /// Korean: code `"ko-KR"`.
    Korean,
    /// Chinese (Simplified): code `"zh-CN"`.
    Chinese,
    /// Turkish: code `"tr-TR"`.
    Turkish,
    /// Vietnamese: code `"vi-VN"`.
    Vietnamese,
    /// Ukrainian: code `"uk-UA"`.
    Ukrainian,
    /// Any code not enumerated above, carried verbatim (e.g. `"qu-PE"`,
    /// `"es-US"`, `"en-GB"`). Resolves through the engine's table exactly as the
    /// raw `&str` would have.
    Other(Cow<'static, str>),
}

impl Language {
    /// The exact language code string for this value, as accepted by the engine
    /// lookup tables (Nemotron's prompt dictionary / Cohere's supported list).
    ///
    /// This is the round-trip target for [`Language::from`]: a code string parsed
    /// into a [`Language`] and rendered back with `as_str` yields the same code
    /// the engine would have looked up directly.
    pub fn as_str(&self) -> &str {
        match self {
            Language::Auto => "auto",
            Language::English => "en-US",
            Language::Spanish => "es-ES",
            Language::French => "fr-FR",
            Language::German => "de-DE",
            Language::Italian => "it-IT",
            Language::Portuguese => "pt-PT",
            Language::Dutch => "nl-NL",
            Language::Russian => "ru-RU",
            Language::Arabic => "ar",
            Language::Hindi => "hi-IN",
            Language::Japanese => "ja-JP",
            Language::Korean => "ko-KR",
            Language::Chinese => "zh-CN",
            Language::Turkish => "tr-TR",
            Language::Vietnamese => "vi-VN",
            Language::Ukrainian => "uk-UA",
            Language::Other(code) => code,
        }
    }

    /// Parse a language code into a [`Language`].
    ///
    /// Known canonical codes map to their named variant; everything else
    /// (aliases like `"en"`, regional codes like `"en-GB"`, and experimental
    /// locales) is carried verbatim in [`Language::Other`] so it resolves through
    /// the engine table unchanged. Infallible by design - validation happens at
    /// the engine (an unknown code errors there, matching the previous `&str`
    /// behavior).
    fn parse(code: &str) -> Language {
        match code {
            "auto" => Language::Auto,
            "en-US" => Language::English,
            "es-ES" => Language::Spanish,
            "fr-FR" => Language::French,
            "de-DE" => Language::German,
            "it-IT" => Language::Italian,
            "pt-PT" => Language::Portuguese,
            "nl-NL" => Language::Dutch,
            "ru-RU" => Language::Russian,
            "ar" => Language::Arabic,
            "hi-IN" => Language::Hindi,
            "ja-JP" => Language::Japanese,
            "ko-KR" => Language::Korean,
            "zh-CN" => Language::Chinese,
            "tr-TR" => Language::Turkish,
            "vi-VN" => Language::Vietnamese,
            "uk-UA" => Language::Ukrainian,
            other => Language::Other(Cow::Owned(other.to_string())),
        }
    }
}

impl From<&str> for Language {
    fn from(code: &str) -> Self {
        Language::parse(code)
    }
}

impl From<String> for Language {
    fn from(code: String) -> Self {
        // Reuse the canonical mapping; only genuinely unknown codes allocate.
        match Language::parse(&code) {
            Language::Other(_) => Language::Other(Cow::Owned(code)),
            named => named,
        }
    }
}

impl From<&Language> for Language {
    fn from(lang: &Language) -> Self {
        lang.clone()
    }
}

impl FromStr for Language {
    type Err = Error;

    /// Infallible in practice (any code becomes [`Language::Other`]); the
    /// `Result` exists only to satisfy [`FromStr`]. Engine-level validation
    /// rejects unknown codes later, matching the original `&str` API.
    fn from_str(code: &str) -> Result<Self, Self::Err> {
        Ok(Language::parse(code))
    }
}

impl fmt::Display for Language {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn auto_is_first_class_and_round_trips() {
        assert_eq!(Language::from("auto"), Language::Auto);
        assert_eq!(Language::Auto.as_str(), "auto");
        assert_eq!(Language::Auto.to_string(), "auto");
    }

    #[test]
    fn named_locales_round_trip_their_code() {
        for code in [
            "en-US", "es-ES", "fr-FR", "de-DE", "it-IT", "pt-PT", "nl-NL", "ru-RU", "ar", "hi-IN",
            "ja-JP", "ko-KR", "zh-CN", "tr-TR", "vi-VN", "uk-UA",
        ] {
            let lang = Language::from(code);
            assert_eq!(lang.as_str(), code, "{code} must round-trip via as_str()");
            assert!(
                !matches!(lang, Language::Other(_)),
                "{code} must map to a named variant, not Other"
            );
        }
    }

    #[test]
    fn unknown_and_alias_codes_pass_through_verbatim() {
        // Aliases (en) and regional/experimental codes are carried verbatim so
        // they resolve through the engine table exactly as the raw &str did.
        for code in ["en", "en-GB", "es-US", "qu-PE", "mi-NZ", "zz-ZZ"] {
            let lang = Language::from(code);
            assert!(matches!(lang, Language::Other(_)), "{code} -> Other");
            assert_eq!(lang.as_str(), code, "{code} passes through unchanged");
        }
    }

    #[test]
    fn from_string_does_not_drop_the_code() {
        assert_eq!(Language::from("ja-JP".to_string()), Language::Japanese);
        assert_eq!(
            Language::from("custom-XX".to_string()).as_str(),
            "custom-XX"
        );
    }

    #[test]
    fn from_str_trait_is_infallible_passthrough() {
        assert_eq!("auto".parse::<Language>().unwrap(), Language::Auto);
        assert_eq!("zz-ZZ".parse::<Language>().unwrap().as_str(), "zz-ZZ");
    }
}
