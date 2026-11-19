//! The one setting that says how much of a turn may start before the turn is
//! committed, and the one rule for keeping what did.
//!
//! The turn machine decides when a turn ends and is not touched by any of
//! this. Early work only moves the slow steps (the transcript, the knowledge
//! search) into the silence the caller leaves anyway; what the turn then does
//! with them is the reuse rule's call:
//!
//! * [`Reuse::SameWordsLoose`]: the same words, any case, any punctuation.
//!   On the paper's workspace a question that differs only by its end mark
//!   got a different passage list in 56% of cases, so this rule can hand the
//!   reply passages its own words would not have found.
//! * [`Reuse::SameWords`]: the same text, character for character.
//! * [`Reuse::SameRequest`]: the same text AND the same query as the
//!   knowledge service is sent it. The request is then identical, so what it
//!   found is what the turn's own search would find: early work can change
//!   when the knowledge arrives, never which knowledge.

use core::fmt;
use core::time::Duration;

/// How long one voice-detector chunk of the caller's audio lasts (512
/// samples at 16 kHz). Silence is counted in whole chunks.
pub const CHUNK: Duration = Duration::from_millis(32);

/// The shortest silence that may start early work: two chunks. One chunk
/// under the detector's threshold is a dip inside a word, not a pause.
pub const MIN_AFTER: Duration = Duration::from_millis(64);

/// The silence early work starts after when the setting names none: four
/// chunks, 128 ms, under half the 320 ms floor the turn's end waits for.
pub const DEFAULT_AFTER: Duration = Duration::from_millis(128);

/// When a turn may take a knowledge search that started before the turn was
/// committed. See the module docs for what each rule risks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Reuse {
    /// The same words, any case, any punctuation or spacing between them.
    SameWordsLoose,
    /// The same text, character for character.
    SameWords,
    /// The same text and the same query: an identical request.
    SameRequest,
}

/// What one knowledge search was (or will be) asked: the caller's words it
/// was made from, and the query exactly as the knowledge service is sent it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Asked<'a> {
    pub words: &'a str,
    pub query: &'a str,
}

impl Reuse {
    /// May the turn that asks `turn` take the answer of the search that
    /// asked `early`?
    pub fn allows(self, early: Asked<'_>, turn: Asked<'_>) -> bool {
        match self {
            Reuse::SameWordsLoose => same_words(early.words, turn.words),
            Reuse::SameWords => early.words == turn.words,
            Reuse::SameRequest => early.words == turn.words && early.query == turn.query,
        }
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            Reuse::SameWordsLoose => "loose",
            Reuse::SameWords => "words",
            Reuse::SameRequest => "request",
        }
    }
}

/// The same words, as a reader hears them: letters and digits, any case,
/// any punctuation or spacing between them. NOT how a knowledge search reads
/// them (see the module docs), which is why it is only one of the rules.
pub fn same_words(a: &str, b: &str) -> bool {
    let words = |s: &str| {
        s.split(|c: char| !c.is_alphanumeric())
            .filter(|w| !w.is_empty())
            .map(str::to_lowercase)
            .collect::<Vec<_>>()
    };
    words(a) == words(b)
}

/// Does the caller's line end on a number ("...order 4 5 1 2?")? An
/// end-of-turn model reads trailing digits as dictation still going, so a
/// finished question ending on one tends to wait out the silence ceiling.
/// The prior behaviour ([`EarlyStart::NumbersOnly`]) searches early only then.
pub fn ends_on_a_number(text: &str) -> bool {
    const SAID: [&str; 11] = [
        "zero", "oh", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    ];
    let last = text
        .trim_end_matches(|c: char| c.is_whitespace() || c.is_ascii_punctuation())
        .rsplit(|c: char| c.is_whitespace() || c == ',')
        .next()
        .unwrap_or("")
        .to_lowercase();
    last.chars().last().is_some_and(|c| c.is_ascii_digit()) || SAID.contains(&last.as_str())
}

/// How much of a turn's work may start before the turn is committed. One
/// setting, so two ways of starting early can never be on together.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum EarlyStart {
    /// Nothing about the caller's question starts before the commit.
    #[default]
    Off,
    /// The prior behaviour: while the end-of-turn model's "unfinished" waits
    /// out the ceiling, a question ending on a number has its knowledge
    /// search started, and the turn takes it under [`Reuse::SameWordsLoose`].
    NumbersOnly,
    /// At the caller's first silence of `after`, before the turn's end is
    /// decided, the words so far are transcribed and searched. If the caller
    /// says nothing more, those words are the turn's transcript and the
    /// search is taken under `reuse`; if they go on, both are dropped.
    FirstSilence { after: Duration, reuse: Reuse },
}

/// Why a setting's text is not an [`EarlyStart`].
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum EarlyStartError {
    #[error("'{0}' unknown; valid: off | numbers | first-silence[:<ms>[:loose|words|request]]")]
    Unknown(String),
    #[error("silence '{0}' is not a whole number of 32 ms chunks, at least 64 ms")]
    Silence(String),
    #[error("reuse rule '{0}' unknown; valid: loose | words | request")]
    Reuse(String),
}

impl EarlyStart {
    /// The configuration the paper measured: start after 128 ms of silence,
    /// keep only an identical request.
    pub const MEASURED: EarlyStart = EarlyStart::FirstSilence {
        after: DEFAULT_AFTER,
        reuse: Reuse::SameRequest,
    };

    /// Read the setting's text: `off`, `numbers`, or `first-silence` with an
    /// optional silence in milliseconds and an optional reuse rule
    /// (`first-silence:128:request` is `first-silence` spelled out).
    pub fn parse(text: &str) -> Result<Self, EarlyStartError> {
        let text = text.trim();
        let mut parts = text.split(':');
        match parts.next().unwrap_or("") {
            "off" if parts.next().is_none() => Ok(EarlyStart::Off),
            "numbers" if parts.next().is_none() => Ok(EarlyStart::NumbersOnly),
            "first-silence" => {
                let after = match parts.next() {
                    None => DEFAULT_AFTER,
                    Some(ms) => parse_after(ms)?,
                };
                let reuse = match parts.next() {
                    None | Some("request") => Reuse::SameRequest,
                    Some("words") => Reuse::SameWords,
                    Some("loose") => Reuse::SameWordsLoose,
                    Some(other) => return Err(EarlyStartError::Reuse(other.to_string())),
                };
                match parts.next() {
                    None => Ok(EarlyStart::FirstSilence { after, reuse }),
                    Some(_) => Err(EarlyStartError::Unknown(text.to_string())),
                }
            }
            _ => Err(EarlyStartError::Unknown(text.to_string())),
        }
    }

    /// The rule a search that started early is taken under.
    pub const fn reuse(self) -> Reuse {
        match self {
            EarlyStart::FirstSilence { reuse, .. } => reuse,
            // Nothing starts early when off; the loose rule is the prior one.
            EarlyStart::Off | EarlyStart::NumbersOnly => Reuse::SameWordsLoose,
        }
    }

    /// The silence the early start waits for (None: it does not start at the
    /// caller's first silence at all).
    pub const fn first_silence(self) -> Option<Duration> {
        match self {
            EarlyStart::FirstSilence { after, .. } => Some(after),
            EarlyStart::Off | EarlyStart::NumbersOnly => None,
        }
    }

    /// Does a look at the words so far start the knowledge search on them?
    /// `ends_on_a_number`: whether those words end on a number.
    pub const fn searches(self, ends_on_a_number: bool) -> bool {
        match self {
            EarlyStart::Off => false,
            EarlyStart::NumbersOnly => ends_on_a_number,
            EarlyStart::FirstSilence { .. } => true,
        }
    }
}

fn parse_after(ms: &str) -> Result<Duration, EarlyStartError> {
    let bad = || EarlyStartError::Silence(ms.to_string());
    let n: u64 = ms.parse().map_err(|_| bad())?;
    let after = Duration::from_millis(n);
    if after < MIN_AFTER || !n.is_multiple_of(CHUNK.as_millis() as u64) {
        return Err(bad());
    }
    Ok(after)
}

impl fmt::Display for EarlyStart {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EarlyStart::Off => f.write_str("off"),
            EarlyStart::NumbersOnly => f.write_str("numbers"),
            EarlyStart::FirstSilence { after, reuse } => {
                write!(f, "first-silence:{}:{}", after.as_millis(), reuse.as_str())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ASKED: Asked<'static> = Asked {
        words: "Where is my order 4 5 1 2?",
        query: "Where is my order 4 5 1 2?",
    };

    #[test]
    fn the_default_starts_nothing_early() {
        assert_eq!(EarlyStart::default(), EarlyStart::Off);
        assert_eq!(EarlyStart::default().first_silence(), None);
        assert!(!EarlyStart::default().searches(true));
        assert_eq!(
            EarlyStart::MEASURED.to_string(),
            "first-silence:128:request"
        );
    }

    #[test]
    fn the_prior_behaviour_searches_early_only_for_a_number() {
        assert_eq!(EarlyStart::NumbersOnly.reuse(), Reuse::SameWordsLoose);
        assert_eq!(EarlyStart::NumbersOnly.first_silence(), None);
        assert!(EarlyStart::NumbersOnly.searches(true));
        assert!(!EarlyStart::NumbersOnly.searches(false));
    }

    #[test]
    fn the_setting_reads_every_choice_and_writes_it_back() {
        for (text, want) in [
            ("off", EarlyStart::Off),
            ("numbers", EarlyStart::NumbersOnly),
            ("first-silence", EarlyStart::MEASURED),
            (
                "first-silence:192",
                EarlyStart::FirstSilence {
                    after: Duration::from_millis(192),
                    reuse: Reuse::SameRequest,
                },
            ),
            (
                "first-silence:64:loose",
                EarlyStart::FirstSilence {
                    after: Duration::from_millis(64),
                    reuse: Reuse::SameWordsLoose,
                },
            ),
            (
                " first-silence:128:words ",
                EarlyStart::FirstSilence {
                    after: Duration::from_millis(128),
                    reuse: Reuse::SameWords,
                },
            ),
        ] {
            let got = EarlyStart::parse(text).unwrap_or_else(|e| panic!("{text:?}: {e}"));
            assert_eq!(got, want, "{text:?}");
            assert_eq!(EarlyStart::parse(&got.to_string()), Ok(got), "{got}");
        }
    }

    #[test]
    fn a_setting_that_names_no_choice_is_refused_with_the_choices() {
        for text in [
            "",
            "on",
            "numbers:128",
            "off:1",
            "first-silence:",
            "first-silence:100",
            "first-silence:32",
            "first-silence:-64",
            "first-silence:128:same",
            "first-silence:128:request:more",
        ] {
            let err = EarlyStart::parse(text).expect_err(text);
            assert!(!err.to_string().is_empty());
        }
        assert!(
            EarlyStart::parse("sometimes")
                .expect_err("unknown")
                .to_string()
                .contains("off | numbers | first-silence")
        );
    }

    #[test]
    fn only_first_silence_starts_at_the_first_silence_and_searches_every_question() {
        let first = EarlyStart::MEASURED;
        assert_eq!(first.first_silence(), Some(DEFAULT_AFTER));
        assert!(first.searches(false) && first.searches(true));
        assert_eq!(first.reuse(), Reuse::SameRequest);
        assert!(!EarlyStart::Off.searches(true));
        assert_eq!(EarlyStart::Off.first_silence(), None);
    }

    /// The three rules on the cases that tell them apart. The punctuation
    /// case is the one the paper's measurement is about: the loose rule keeps
    /// the early search, the two exact rules send the turn's own.
    #[test]
    fn the_rules_differ_exactly_where_the_words_or_the_query_do() {
        let punctuation = Asked {
            words: "Where is my order, 4 5 1 2.",
            query: "Where is my order, 4 5 1 2.",
        };
        let other_number = Asked {
            words: "Where is my order 4 5 1 3?",
            query: "Where is my order 4 5 1 3?",
        };
        // The same words, searched with something added the second time
        // (the call's topic changed between the early look and the turn).
        let other_query = Asked {
            words: ASKED.words,
            query: "Where is my order 4 5 1 2? A refund",
        };
        for (rule, same, punct, number, query) in [
            (Reuse::SameWordsLoose, true, true, false, true),
            (Reuse::SameWords, true, false, false, true),
            (Reuse::SameRequest, true, false, false, false),
        ] {
            assert_eq!(rule.allows(ASKED, ASKED), same, "{rule:?} same");
            assert_eq!(
                rule.allows(ASKED, punctuation),
                punct,
                "{rule:?} punctuation"
            );
            assert_eq!(rule.allows(ASKED, other_number), number, "{rule:?} number");
            assert_eq!(rule.allows(ASKED, other_query), query, "{rule:?} query");
        }
    }

    #[test]
    fn the_same_words_are_told_apart_by_letters_and_digits_only() {
        assert!(same_words(
            "Where is my order, 4 5 1 2?",
            "where is my order 4 5 1 2."
        ));
        assert!(!same_words("order 4512", "order 4513"));
        assert!(!same_words("order 4512", "order 4 5 1 2"));
        assert!(same_words("", " ?! "));
    }

    #[test]
    fn a_line_ending_on_a_number_is_told_apart() {
        for (line, ends) in [
            ("Where is my order, 4 5 1 2?", true),
            ("It's 4512.", true),
            ("four five one two", true),
            ("order 4512, and", false),
            ("Where is my order?", false),
            ("I'd like to", false),
            ("", false),
        ] {
            assert_eq!(ends_on_a_number(line), ends, "{line:?}");
        }
    }
}
