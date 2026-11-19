//! The replayed calls: each arm's two halves joined (what the caller said and
//! waited for, and what the server recorded for each turn), each arm
//! summarised, and two arms that heard the same recordings compared question
//! by question.
//!
//! An arm lives in a directory as three files:
//! * `<arm>.json`: its settings ([`ArmSettings`]);
//! * `<arm>.replay.jsonl`: one line per question the caller said ([`Said`]);
//! * `<arm>.turns.jsonl`: one line per turn the server answered
//!   ([`early_start::TurnRecord`], with the time the server logged it).
//!
//! A question's line is joined to the LAST answered turn the server recorded
//! on the same call between the question's start and 300 ms after its turn
//! ended. A long question the server answered in parts leaves earlier
//! records in that window: its other parts.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result, bail};
use early_start::{Knowledge, TurnRecord};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};

use crate::questions::QuestionSet;
use crate::stats::{dist_json, median_95, round_half_up, share_json};
use crate::wer::{word_errors, words};

/// One arm's settings.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ArmSettings {
    pub name: String,
    /// The early start the voice server ran (`numbers` is the prior behaviour).
    pub early_start: String,
    #[serde(default)]
    pub silence_floor_ms: Option<u32>,
    /// The caller's audio frame.
    pub mic_frame_ms: u32,
    /// The distance to the knowledge service, each way.
    pub kb_delay_ms: u32,
    /// Which lookup the knowledge service ran: `existing` or `split`.
    #[serde(default)]
    pub lookup: Option<String>,
    /// Every search the knowledge service ran while the arm ran.
    #[serde(default)]
    pub knowledge_searches: Option<u64>,
    #[serde(default)]
    pub stub_ttft_ms: Option<u32>,
    #[serde(default)]
    pub started: Option<String>,
    #[serde(default)]
    pub ended: Option<String>,
}

/// One question as the caller said it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Said {
    pub arm: String,
    pub qid: String,
    pub call: String,
    /// The question's place in its call.
    pub n: u32,
    pub frame_ms: u32,
    #[serde(default)]
    pub speech_ms: Option<u32>,
    #[serde(default)]
    pub tail_ms: Option<u32>,
    /// From the last voiced sample to the first audio of the reply; None:
    /// no reply came.
    pub wait_ms: Option<i64>,
    /// The agent's audio arrived while the caller was still speaking.
    pub talked_over: bool,
    #[serde(default)]
    pub talked_over_ms: Option<u32>,
    /// What the server heard, one entry per turn it made of the question.
    pub heard: Vec<String>,
    #[serde(default)]
    pub reply: Option<String>,
    #[serde(default)]
    pub reply_audio_ms: Option<u32>,
    #[serde(default)]
    pub error: Option<String>,
    pub began_unix_ms: i64,
    pub at_unix_ms: i64,
}

/// A server record with the time it was logged.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Logged {
    pub at: String,
    #[serde(flatten)]
    pub record: TurnRecord,
}

/// `YYYY-MM-DDTHH:MM:SS[.fraction]Z` as Unix milliseconds, the fraction cut
/// (not rounded) to the millisecond.
pub fn unix_ms(at: &str) -> Option<i64> {
    let b = at.as_bytes();
    if b.len() < 20
        || b[4] != b'-'
        || b[7] != b'-'
        || b[10] != b'T'
        || b[13] != b':'
        || b[16] != b':'
    {
        return None;
    }
    let num = |r: std::ops::Range<usize>| at.get(r)?.parse::<i64>().ok();
    let (y, m, d) = (num(0..4)?, num(5..7)?, num(8..10)?);
    let (hh, mm, ss) = (num(11..13)?, num(14..16)?, num(17..19)?);
    let rest = at.get(19..)?;
    let frac = match rest.strip_prefix('.') {
        Some(f) => {
            let digits = f.strip_suffix('Z')?;
            if digits.is_empty() || !digits.bytes().all(|c| c.is_ascii_digit()) {
                return None;
            }
            let ms: String = digits.chars().chain("000".chars()).take(3).collect();
            ms.parse::<i64>().ok()?
        }
        None if rest == "Z" => 0,
        None => return None,
    };
    if !(1..=12).contains(&m) || !(1..=31).contains(&d) || hh > 23 || mm > 59 || ss > 60 {
        return None;
    }
    // Days from the civil date (proleptic Gregorian).
    let y = if m <= 2 { y - 1 } else { y };
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let mp = (m + 9) % 12;
    let doy = (153 * mp + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    let days = era * 146_097 + doe - 719_468;
    Some(((days * 24 + hh) * 60 + mm) * 60_000 + ss * 1000 + frac)
}

fn read_lines<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<Vec<T>> {
    let text = std::fs::read_to_string(path).with_context(|| format!("{}", path.display()))?;
    text.lines()
        .enumerate()
        .filter(|(_, l)| !l.trim().is_empty())
        .map(|(n, l)| {
            serde_json::from_str(l).with_context(|| format!("{} line {}", path.display(), n + 1))
        })
        .collect()
}

/// One question joined with the server's record of its turn.
#[derive(Debug, Clone)]
pub struct Joined {
    pub said: Said,
    pub record: Option<TurnRecord>,
    /// The turns the server made of the question.
    pub parts: usize,
}

impl Joined {
    fn answered(&self) -> bool {
        self.said.wait_ms.is_some() && !self.said.talked_over
    }

    /// Answered, not talked over, heard as one turn: the timed set.
    fn timed(&self) -> bool {
        self.answered() && self.parts == 1
    }

    fn wait(&self) -> i64 {
        self.said.wait_ms.unwrap_or_default()
    }

    fn heard_text(&self) -> String {
        self.said.heard.join(" ")
    }
}

/// One arm, read and joined.
#[derive(Debug, Clone)]
pub struct Arm {
    pub settings: ArmSettings,
    pub said: usize,
    pub turns: Vec<Joined>,
    pub records: Vec<TurnRecord>,
    pub unjoined: usize,
}

impl Arm {
    /// Read the arm `name` from `dir`.
    pub fn read(dir: &Path, name: &str) -> Result<Arm> {
        let settings: ArmSettings = serde_json::from_str(
            &std::fs::read_to_string(dir.join(format!("{name}.json")))
                .with_context(|| format!("arm {name}"))?,
        )
        .with_context(|| format!("arm {name}"))?;
        let said: Vec<Said> = read_lines::<Said>(&dir.join(format!("{name}.replay.jsonl")))?
            .into_iter()
            .filter(|s| s.arm == name)
            .collect();
        let logged: Vec<Logged> = read_lines(&dir.join(format!("{name}.turns.jsonl")))?;
        let mut at = Vec::with_capacity(logged.len());
        for l in &logged {
            at.push(
                unix_ms(&l.at)
                    .with_context(|| format!("arm {name}: a record logged at {:?}", l.at))?,
            );
        }
        let mut by_call: HashMap<&str, Vec<usize>> = HashMap::new();
        for (i, l) in logged.iter().enumerate() {
            by_call.entry(l.record.call.as_str()).or_default().push(i);
        }
        let mut unjoined = 0;
        let mut turns = Vec::with_capacity(said.len());
        for s in &said {
            let window: Vec<usize> = by_call
                .get(s.call.as_str())
                .map(|v| {
                    v.iter()
                        .copied()
                        .filter(|i| {
                            logged[*i].record.stt_ms.is_some()
                                && at[*i] > s.began_unix_ms
                                && at[*i] <= s.at_unix_ms + 300
                        })
                        .collect()
                })
                .unwrap_or_default();
            let record = match (s.wait_ms, window.last()) {
                (Some(_), Some(i)) => Some(logged[*i].record.clone()),
                _ => None,
            };
            if s.wait_ms.is_some() && record.is_none() {
                unjoined += 1;
            }
            turns.push(Joined {
                said: s.clone(),
                record,
                parts: s.heard.len(),
            });
        }
        Ok(Arm {
            settings,
            said: said.len(),
            turns,
            records: logged.into_iter().map(|l| l.record).collect(),
            unjoined,
        })
    }

    fn timed_with_record(&self) -> Vec<(&Joined, &TurnRecord)> {
        self.turns
            .iter()
            .filter(|t| t.timed())
            .filter_map(|t| t.record.as_ref().map(|r| (t, r)))
            .collect()
    }

    /// The arm's numbers. `pages`: each passage's page, for the right page
    /// among the passages.
    pub fn summary(
        &self,
        questions: &QuestionSet,
        pages: Option<&HashMap<String, String>>,
    ) -> Value {
        let timed: Vec<&Joined> = self.turns.iter().filter(|t| t.timed()).collect();
        let in_parts: Vec<&Joined> = self
            .turns
            .iter()
            .filter(|t| t.answered() && t.parts > 1)
            .collect();
        let with_rec = self.timed_with_record();
        let waits = |v: &[&Joined]| -> Vec<i64> { v.iter().map(|t| t.wait()).collect() };
        let complete: Vec<&Joined> = with_rec
            .iter()
            .filter(|(_, r)| r.judged_complete == Some(true))
            .map(|(t, _)| *t)
            .collect();
        let waited: Vec<&Joined> = with_rec
            .iter()
            .filter(|(_, r)| r.judged_complete == Some(false))
            .map(|(t, _)| *t)
            .collect();
        let hop = |f: fn(&TurnRecord) -> Option<u32>| -> Value {
            let v: Vec<i64> = with_rec
                .iter()
                .filter_map(|(_, r)| f(r))
                .map(i64::from)
                .collect();
            dist_json(&v)
        };
        let count = |f: fn(&TurnRecord) -> &'static str| -> Value {
            let mut c = Map::new();
            for (_, r) in &with_rec {
                let k = f(r);
                let n = c.get(k).and_then(Value::as_u64).unwrap_or(0);
                c.insert(k.to_string(), json!(n + 1));
            }
            Value::Object(c)
        };
        let (mut errors, mut said_words, mut exact) = (0usize, 0usize, 0usize);
        for t in &timed {
            let said = words(
                questions
                    .by_qid(&t.said.qid)
                    .map(|q| q.question.as_str())
                    .unwrap_or(""),
            );
            let heard = words(&t.heard_text());
            errors += word_errors(&said, &heard);
            said_words += said.len();
            exact += usize::from(said == heard);
        }
        let right_page = pages.map(|pages| {
            let searched: Vec<&(&Joined, &TurnRecord)> = with_rec
                .iter()
                .filter(|(_, r)| r.kb != Knowledge::None)
                .collect();
            let found = with_rec
                .iter()
                .filter(|(t, r)| {
                    let gold = questions.by_qid(&t.said.qid).map(|q| q.gold_page.as_str());
                    r.passages
                        .iter()
                        .any(|id| pages.get(id).map(String::as_str) == gold && gold.is_some())
                })
                .count();
            share_json(found, searched.len())
        });
        let s = &self.settings;
        json!({
            "early_start": s.early_start,
            "silence_floor_ms": s.silence_floor_ms.unwrap_or(320),
            "mic_frame_ms": s.mic_frame_ms,
            "kb_delay_ms": s.kb_delay_ms,
            "lookup": s.lookup,
            "questions_said": self.said,
            "heard_as_one_turn": timed.len(),
            "heard_in_parts": in_parts.len(),
            "agent_talked_over": self.turns.iter().filter(|t| t.said.talked_over).count(),
            "no_reply": self.turns.iter().filter(|t| t.said.wait_ms.is_none()).count(),
            "not_joined_to_a_server_record": self.unjoined,
            "caller_wait_ms": dist_json(&waits(&timed)),
            "caller_wait_ms_questions_heard_in_parts": dist_json(&waits(&in_parts)),
            "caller_wait_ms_turn_judged_complete": dist_json(&waits(&complete)),
            "caller_wait_ms_turn_waited_for_more": dist_json(&waits(&waited)),
            "judged_complete": share_json(complete.len(), with_rec.len()),
            "knowledge_searches": s.knowledge_searches,
            "server_turns_that_searched": self.records.iter().filter(|r| r.kb != Knowledge::None).count(),
            "server_ms": {
                "smart_turn": hop(|r| r.smart_turn_ms),
                "stt": hop(|r| r.stt_ms),
                "kb": hop(|r| r.kb_ms),
                "brain_ttft": hop(|r| r.brain_ttft_ms),
                "tts_ttfa": hop(|r| r.tts_ttfa_ms),
                "floor_to_first_audio": hop(|r| r.total_ms),
                "early_lead": hop(|r| r.early_lead_ms),
            },
            "early_words": count(|r| r.early_words.as_str()),
            "early_search": count(|r| r.early_search.as_str()),
            "knowledge": count(|r| knowledge_name(r.kb)),
            "heard_word_error_rate_pct": (said_words > 0).then(|| round_half_up(1000.0 * errors as f64 / said_words as f64) / 10.0),
            "heard_exactly_the_question": share_json(exact, timed.len()),
            "right_page_among_the_passages": right_page,
            "timeline_turns_judged_complete": self.timeline(),
        })
    }

    /// The medians a turn's time line is drawn from, over the timed turns the
    /// end-of-turn model judged complete: the caller's wait, and each hop of
    /// the server's record. `rest` is what the hops' medians leave of the
    /// median from the floor to the first audio (the brain's first sentence,
    /// and the joins between hops); `before_the_floor` is what the median
    /// floor-to-audio leaves of the median wait (the silence floor, and the
    /// caller's audio framing).
    pub fn timeline(&self) -> Value {
        let complete: Vec<(&Joined, &TurnRecord)> = self
            .timed_with_record()
            .into_iter()
            .filter(|(_, r)| r.judged_complete == Some(true))
            .collect();
        let median = |v: Vec<i64>| -> Option<i64> {
            let mut v = v;
            v.sort_unstable();
            crate::stats::quantile(&v, 0.5)
        };
        let of = |f: fn(&TurnRecord) -> Option<u32>| {
            median(
                complete
                    .iter()
                    .filter_map(|(_, r)| f(r))
                    .map(i64::from)
                    .collect(),
            )
        };
        let wait = median(complete.iter().map(|(t, _)| t.wait()).collect());
        let check = of(|r| r.smart_turn_ms);
        let stt = of(|r| r.stt_ms);
        let kb = of(|r| r.kb_ms);
        let brain = of(|r| r.brain_ttft_ms);
        let voice = of(|r| r.tts_ttfa_ms);
        let floor_to_audio = of(|r| r.total_ms);
        let rest = match (floor_to_audio, check, stt, kb, brain, voice) {
            (Some(t), Some(c), Some(s), Some(k), Some(b), Some(v)) => Some(t - c - s - k - b - v),
            _ => None,
        };
        json!({
            "turns": complete.len(),
            "caller_wait": wait,
            "before_the_floor": wait.zip(floor_to_audio).map(|(w, f)| w - f),
            "end_of_turn_model": check,
            "transcript": stt,
            "knowledge": kb,
            "brain_first_token": brain,
            "rest": rest,
            "voice_first_audio": voice,
            "floor_to_first_audio": floor_to_audio,
        })
    }
}

fn knowledge_name(k: Knowledge) -> &'static str {
    match k {
        Knowledge::None => "none",
        Knowledge::Hit => "hit",
        Knowledge::NoMatch => "no-match",
        Knowledge::Stale => "stale",
        Knowledge::FailOpen => "fail-open",
    }
}

/// Two arms that heard the same recordings, compared question by question:
/// `a`'s wait minus `b`'s.
pub fn pair(a: &Arm, b: &Arm) -> Value {
    let by_qid = |arm: &Arm, keep: fn(&Joined) -> bool| -> HashMap<String, Joined> {
        // The last line of a question wins, as a resumed run may say one again.
        arm.turns
            .iter()
            .filter(|t| keep(t))
            .map(|t| (t.said.qid.clone(), t.clone()))
            .collect()
    };
    let b_timed = by_qid(b, Joined::timed);
    let b_answered = by_qid(b, Joined::answered);
    let both: Vec<(&Joined, &Joined)> = a
        .turns
        .iter()
        .filter(|t| t.timed())
        .filter_map(|t| b_timed.get(&t.said.qid).map(|u| (t, u)))
        .collect();
    let every: Vec<i64> = a
        .turns
        .iter()
        .filter(|t| t.answered())
        .filter_map(|t| b_answered.get(&t.said.qid).map(|u| t.wait() - u.wait()))
        .collect();
    let saved: Vec<i64> = both.iter().map(|(x, y)| x.wait() - y.wait()).collect();
    let rec: Vec<(&Joined, &Joined, &TurnRecord, &TurnRecord)> = both
        .iter()
        .filter_map(|(x, y)| Some((*x, *y, x.record.as_ref()?, y.record.as_ref()?)))
        .collect();
    let same_verdict: Vec<&(&Joined, &Joined, &TurnRecord, &TurnRecord)> = rec
        .iter()
        .filter(|(_, _, rx, ry)| rx.judged_complete == ry.judged_complete)
        .collect();
    let bucket = |want: bool| -> Value {
        let v: Vec<i64> = same_verdict
            .iter()
            .filter(|(_, _, rx, _)| rx.judged_complete == Some(want))
            .map(|(x, y, _, _)| x.wait() - y.wait())
            .collect();
        with_interval(dist_json(&v), &v)
    };
    let heard_same = both
        .iter()
        .filter(|(x, y)| x.heard_text() == y.heard_text())
        .count();
    let heard_same_words = both
        .iter()
        .filter(|(x, y)| words(&x.heard_text()) == words(&y.heard_text()))
        .count();
    let searched: Vec<&(&Joined, &Joined, &TurnRecord, &TurnRecord)> = rec
        .iter()
        .filter(|(_, _, rx, ry)| rx.kb != Knowledge::None && ry.kb != Knowledge::None)
        .collect();
    let same_list = searched
        .iter()
        .filter(|(_, _, rx, ry)| rx.passages == ry.passages)
        .count();
    let same_lead = searched
        .iter()
        .filter(|(_, _, rx, ry)| rx.passages.first() == ry.passages.first())
        .count();
    let heard_same_searched: Vec<&&(&Joined, &Joined, &TurnRecord, &TurnRecord)> = searched
        .iter()
        .filter(|(x, y, _, _)| x.heard_text() == y.heard_text())
        .collect();
    let same_list_same_words = heard_same_searched
        .iter()
        .filter(|(_, _, rx, ry)| rx.passages == ry.passages)
        .count();
    let faster = saved.iter().filter(|v| **v > 0).count();
    let mut saved_json = with_interval(dist_json(&saved), &saved);
    if let Some(m) = saved_json.as_object_mut() {
        m.insert("faster".into(), share_json(faster, saved.len()));
    }
    json!({
        "questions_heard_as_one_turn_in_both": both.len(),
        "wait_saved_ms": saved_json,
        "wait_saved_ms_every_answered_question": with_interval(dist_json(&every), &every),
        "wait_saved_ms_where_both_judged_complete": bucket(true),
        "wait_saved_ms_where_both_waited_for_more": bucket(false),
        "same_end_of_turn_verdict": share_json(same_verdict.len(), rec.len()),
        "heard_the_same_text": share_json(heard_same, both.len()),
        "heard_the_same_words": share_json(heard_same_words, both.len()),
        "same_passage_list": share_json(same_list, searched.len()),
        "same_first_passage": share_json(same_lead, searched.len()),
        "same_passage_list_when_the_same_text_was_heard": share_json(same_list_same_words, heard_same_searched.len()),
    })
}

fn with_interval(mut d: Value, values: &[i64]) -> Value {
    if let Some(m) = d.as_object_mut() {
        m.insert("median_95".into(), median_95(values));
    }
    d
}

/// Every arm in `dir` (a `<arm>.json` each), by name.
pub fn arm_names(dir: &Path) -> Result<Vec<String>> {
    let mut names: Vec<String> = std::fs::read_dir(dir)
        .with_context(|| format!("{}", dir.display()))?
        .filter_map(|e| e.ok())
        .filter_map(|e| e.file_name().to_str().map(str::to_string))
        .filter_map(|f| f.strip_suffix(".json").map(str::to_string))
        .filter(|n| !n.contains('.'))
        .collect();
    names.sort();
    Ok(names)
}

/// Each passage's page, from `{"id": "...", "page": "..."}` lines.
pub fn read_pages(path: &Path) -> Result<HashMap<String, String>> {
    #[derive(Deserialize)]
    struct Line {
        id: String,
        page: String,
    }
    Ok(read_lines::<Line>(path)?
        .into_iter()
        .map(|l| (l.id, l.page))
        .collect())
}

/// The whole report: every arm in `dir`, and each of `pairs` (`a=b`).
pub fn report(
    dir: &Path,
    questions: &QuestionSet,
    pages: Option<&HashMap<String, String>>,
    pairs: &[(String, String)],
) -> Result<Value> {
    let mut arms: HashMap<String, Arm> = HashMap::new();
    let mut summaries = Map::new();
    for name in arm_names(dir)? {
        let arm = Arm::read(dir, &name)?;
        summaries.insert(name.clone(), arm.summary(questions, pages));
        arms.insert(name, arm);
    }
    if arms.is_empty() {
        bail!("{} holds no arms", dir.display());
    }
    let mut compared = Map::new();
    for (a, b) in pairs {
        let key = format!("{a}={b}");
        match (arms.get(a), arms.get(b)) {
            (Some(x), Some(y)) => compared.insert(key, pair(x, y)),
            _ => compared.insert(key, json!({"error": "no such arm"})),
        };
    }
    Ok(json!({"arms": summaries, "pairs": compared}))
}

/// `a=b,c=d` (or one pair a line) as pairs; a line starting with `#` is a
/// note.
pub fn parse_pairs(text: &str) -> Result<Vec<(String, String)>> {
    text.lines()
        .map(str::trim)
        .filter(|l| !l.starts_with('#'))
        .flat_map(|l| l.split(','))
        .map(str::trim)
        .filter(|p| !p.is_empty())
        .map(|p| match p.split_once('=') {
            Some((a, b)) if !a.is_empty() && !b.is_empty() => Ok((a.to_string(), b.to_string())),
            _ => bail!("a pair is two arms joined by '=': {p:?}"),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_log_time_reads_as_unix_milliseconds_cut_to_the_millisecond() {
        assert_eq!(
            unix_ms("2026-11-19T11:47:00.533349Z"),
            Some(1_795_088_820_533)
        );
        assert_eq!(
            unix_ms("2026-11-19T11:47:00.533999Z"),
            Some(1_795_088_820_533)
        );
        assert_eq!(unix_ms("2026-11-19T11:47:00.5Z"), Some(1_795_088_820_500));
        assert_eq!(unix_ms("2026-11-19T11:47:00Z"), Some(1_795_088_820_000));
        assert_eq!(unix_ms("1970-01-01T00:00:00Z"), Some(0));
        assert_eq!(unix_ms("2000-03-01T00:00:00Z"), Some(951_868_800_000));
        for bad in [
            "",
            "2026-11-19 11:47:00Z",
            "2026-11-19T11:47:00",
            "2026-13-01T00:00:00Z",
            "2026-11-19T11:47:00.Z",
        ] {
            assert_eq!(unix_ms(bad), None, "{bad}");
        }
    }

    #[test]
    fn pairs_read_from_a_list() {
        assert_eq!(
            parse_pairs("a=b, c=d\n# a note\n").expect("pairs"),
            vec![("a".into(), "b".into()), ("c".into(), "d".into())]
        );
        assert!(parse_pairs("a").is_err());
        assert!(parse_pairs("=b").is_err());
    }
}
