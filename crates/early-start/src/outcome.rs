//! What one turn took from its early start, and the record a turn leaves:
//! facts for measurement, never decisions.

use serde::{Deserialize, Serialize};

/// Where a turn's words came from: decoded after its end was decided, or
/// before it.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EarlyWords {
    /// Decoded once the turn's end was decided: nothing started early, or
    /// the caller went on after it did.
    #[default]
    #[serde(rename = "at-commit")]
    AtCommit,
    /// Decoded before the turn's end was decided, and ready when it was.
    #[serde(rename = "kept")]
    Kept,
    /// The early decode was still running at the turn's end: the turn waited
    /// for it instead of starting its own on the same audio.
    #[serde(rename = "waited")]
    Waited,
}

impl EarlyWords {
    pub const fn as_str(self) -> &'static str {
        match self {
            EarlyWords::AtCommit => "at-commit",
            EarlyWords::Kept => "kept",
            EarlyWords::Waited => "waited",
        }
    }
}

/// What became of the knowledge search a turn was offered from before its
/// end.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EarlySearch {
    /// None was offered: nothing searched early, or the turn searched
    /// nothing at all.
    #[default]
    #[serde(rename = "none")]
    NotOffered,
    /// The reuse rule let the turn take it.
    #[serde(rename = "kept")]
    Kept,
    /// The reuse rule refused it (the turn's words or its query differ): the
    /// turn searched again.
    #[serde(rename = "refused")]
    Refused,
    /// The turn took it and it ended without an answer: the turn searched
    /// again.
    #[serde(rename = "lost")]
    Lost,
}

impl EarlySearch {
    /// The reuse rule's answer for a turn: `offered`, whether a search had
    /// started early; `allowed`, whether the rule let the turn take it.
    pub const fn judge(offered: bool, allowed: bool) -> EarlySearch {
        match (offered, allowed) {
            (true, true) => EarlySearch::Kept,
            (true, false) => EarlySearch::Refused,
            (false, _) => EarlySearch::NotOffered,
        }
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            EarlySearch::NotOffered => "none",
            EarlySearch::Kept => "kept",
            EarlySearch::Refused => "refused",
            EarlySearch::Lost => "lost",
        }
    }
}

/// What one turn took from its early start.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct EarlyUse {
    pub words: EarlyWords,
    pub search: EarlySearch,
}

/// What a turn's knowledge lookup came to.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Knowledge {
    /// The turn asked no knowledge.
    #[default]
    #[serde(rename = "none")]
    None,
    /// The lookup returned passages.
    #[serde(rename = "hit")]
    Hit,
    /// The lookup ran and nothing cleared the gate.
    #[serde(rename = "no-match")]
    NoMatch,
    /// The lookup answered too late for the turn.
    #[serde(rename = "stale")]
    Stale,
    /// The lookup failed and the turn went on without it.
    #[serde(rename = "fail-open")]
    FailOpen,
}

/// One answered turn, as the server logs it: the hops the turn waited for,
/// what it took from its early start, and the passages its reply was
/// grounded on, by id. Never the caller's words. Every hop is in
/// milliseconds; `None` is a hop the turn skipped.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct TurnRecord {
    pub call: String,
    /// The turn's number in its call (1 is the greeting).
    pub turn: u32,
    /// The end-of-turn model.
    pub smart_turn_ms: Option<u32>,
    /// The transcript, from the turn's commit (0 when it was ready).
    pub stt_ms: Option<u32>,
    /// The knowledge lookup, from the turn's commit (0 when it was ready).
    pub kb_ms: Option<u32>,
    pub lookup_ms: Option<u32>,
    /// The language model's first token.
    pub brain_ttft_ms: Option<u32>,
    /// The speech engine's first audio of the reply.
    pub tts_ttfa_ms: Option<u32>,
    /// The silence floor reached, to the first agent audio. The caller's own
    /// wait is longer by the floor itself and by their client's framing.
    pub total_ms: Option<u32>,
    pub kb: Knowledge,
    pub early_words: EarlyWords,
    pub early_search: EarlySearch,
    /// How long before the turn's commit its early start began.
    pub early_lead_ms: Option<u32>,
    /// The end-of-turn model's last verdict before the commit: true, the
    /// turn ended at the floor; false, it waited for more first.
    pub judged_complete: Option<bool>,
    /// The passages the turn's knowledge lookup returned, in order.
    pub passages: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_turns_early_use_starts_as_nothing_and_has_stable_names() {
        assert_eq!(
            EarlyUse::default(),
            EarlyUse {
                words: EarlyWords::AtCommit,
                search: EarlySearch::NotOffered
            }
        );
        assert_eq!(
            [EarlyWords::AtCommit, EarlyWords::Kept, EarlyWords::Waited].map(EarlyWords::as_str),
            ["at-commit", "kept", "waited"]
        );
        assert_eq!(
            [
                EarlySearch::NotOffered,
                EarlySearch::Kept,
                EarlySearch::Refused,
                EarlySearch::Lost
            ]
            .map(EarlySearch::as_str),
            ["none", "kept", "refused", "lost"]
        );
    }

    #[test]
    fn the_rules_answer_is_recorded_as_kept_refused_or_not_offered() {
        assert_eq!(EarlySearch::judge(true, true), EarlySearch::Kept);
        assert_eq!(EarlySearch::judge(true, false), EarlySearch::Refused);
        assert_eq!(EarlySearch::judge(false, true), EarlySearch::NotOffered);
        assert_eq!(EarlySearch::judge(false, false), EarlySearch::NotOffered);
    }

    /// The record's names are the data files' names, so a recorded line
    /// reads back as the same record.
    #[test]
    fn a_turn_record_reads_and_writes_the_data_files_names() {
        let line = r#"{"call":"c1","turn":2,"smart_turn_ms":68,"stt_ms":0,"kb_ms":0,"lookup_ms":null,"brain_ttft_ms":301,"tts_ttfa_ms":128,"total_ms":598,"kb":"hit","early_words":"kept","early_search":"kept","early_lead_ms":259,"judged_complete":true,"passages":["a","b"]}"#;
        let r: TurnRecord = serde_json::from_str(line).expect("a recorded line");
        assert_eq!(r.kb, Knowledge::Hit);
        assert_eq!(r.early_words, EarlyWords::Kept);
        assert_eq!(r.early_search, EarlySearch::Kept);
        assert_eq!(r.early_lead_ms, Some(259));
        assert_eq!(r.passages, ["a", "b"]);
        let back: serde_json::Value =
            serde_json::from_str(&serde_json::to_string(&r).expect("json")).expect("json");
        let want: serde_json::Value = serde_json::from_str(line).expect("json");
        assert_eq!(back, want);
        for (name, kb) in [
            ("\"none\"", Knowledge::None),
            ("\"no-match\"", Knowledge::NoMatch),
            ("\"stale\"", Knowledge::Stale),
            ("\"fail-open\"", Knowledge::FailOpen),
        ] {
            assert_eq!(serde_json::from_str::<Knowledge>(name).expect(name), kb);
        }
    }
}
