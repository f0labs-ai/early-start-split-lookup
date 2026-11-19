//! The context-stability audit over recorded lookups: how good the lookup is
//! in each written form of the questions, the flip rate of each change, the
//! reuse audit, the settle curve, and the repeat floor (the same lookups run
//! twice). Offline: it reads what `ctxaudit forms|prefixes` recorded.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context as _, Result};
use ctxstab::{
    Context, FlipReport, Form, ReusePair, RuleAudit, Settling, Share, audit_rule, flip_report,
    settle_curve,
};
use early_start::{Asked, Reuse, turn_query};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::questions::{Question, QuestionSet};

/// The settle curve's last row gathers this many words still to come, and more.
pub const SETTLE_CAP: u32 = 8;

/// One recorded lookup.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Looked {
    pub qid: String,
    /// A [`Form`] name, or `prefix_<k>_of_<n>` for a question's first `k` of
    /// `n` words.
    pub form: String,
    pub n_words: usize,
    pub ids: Vec<String>,
    pub pages: Vec<String>,
    #[serde(default)]
    pub scores: Vec<f64>,
    #[serde(default)]
    pub ms: u64,
}

impl Looked {
    pub fn context(&self) -> Context {
        Context {
            ids: self.ids.clone(),
            pages: self.pages.clone(),
        }
    }
}

pub fn read_looked(path: &Path) -> Result<Vec<Looked>> {
    let text =
        std::fs::read_to_string(path).with_context(|| format!("lookups {}", path.display()))?;
    text.lines()
        .enumerate()
        .filter(|(_, l)| !l.trim().is_empty())
        .map(|(n, l)| {
            serde_json::from_str(l).with_context(|| format!("{} line {}", path.display(), n + 1))
        })
        .collect()
}

fn round1(x: f64) -> f64 {
    (x * 10.0).round() / 10.0
}

pub fn share_json(s: Share) -> Value {
    json!({
        "k": s.k,
        "n": s.n,
        "pct": s.pct().map(round1),
        "range95": s.range95().map(|(lo, hi)| [round1(lo), round1(hi)]),
    })
}

fn flip_json(r: &FlipReport) -> Value {
    json!({
        "flip_rate_pct": r.flip_rate_pct().map(round1),
        "same_list": share_json(r.same_list),
        "same_set": share_json(r.same_set),
        "same_lead": share_json(r.same_lead),
        "kept_mean_pct": (r.kept_mean * 1000.0).round() / 10.0,
        "right_page_in_both": r.right_in_both,
        "right_page_lost": r.right_lost,
        "right_page_gained": r.right_gained,
    })
}

fn audit_json(a: &RuleAudit) -> Value {
    json!({
        "rule": a.rule,
        "reused": share_json(a.reused),
        "stale_of_reused": share_json(a.stale),
        "harmful_of_reused": share_json(a.harmful),
    })
}

/// The measures over one group of questions (all, or one style).
fn report_group(
    questions: &[&Question],
    by_form: &HashMap<(String, Form), Looked>,
    settle: Option<&HashMap<String, Vec<(u32, Context)>>>,
) -> Value {
    // Only questions looked up in every form count: one table, one n.
    let whole: Vec<&&Question> = questions
        .iter()
        .filter(|q| {
            Form::ALL
                .iter()
                .all(|f| by_form.contains_key(&(q.qid.clone(), *f)))
        })
        .collect();
    let looked = |q: &Question, f: Form| &by_form[&(q.qid.clone(), f)];
    let contexts: HashMap<(&str, Form), Context> = whole
        .iter()
        .flat_map(|q| Form::ALL.map(|f| ((q.qid.as_str(), f), looked(q, f).context())))
        .collect();

    let mut forms = serde_json::Map::new();
    for f in Form::ALL {
        let (mut page, mut first, mut source, mut nothing) = (
            Share::default(),
            Share::default(),
            Share::default(),
            Share::default(),
        );
        let mut tops: Vec<f64> = Vec::new();
        for q in &whole {
            let l = looked(q, f);
            page.add(l.pages.iter().any(|p| p == &q.gold_page));
            first.add(l.pages.first() == Some(&q.gold_page));
            source.add(l.ids.contains(&q.gold_id));
            nothing.add(l.ids.is_empty());
            tops.extend(l.scores.first());
        }
        tops.sort_by(f64::total_cmp);
        forms.insert(
            f.as_str().to_string(),
            json!({
                "right_page_among_the_passages": share_json(page),
                "right_page_first": share_json(first),
                "source_passage_among_the_passages": share_json(source),
                "nothing_found": share_json(nothing),
                "median_top_score": tops.get(tops.len() / 2),
            }),
        );
    }

    let mut flips = serde_json::Map::new();
    for f in [Form::NoEndMark, Form::Plain, Form::LastWordMissing] {
        let report = flip_report(whole.iter().map(|q| {
            (
                &contexts[&(q.qid.as_str(), Form::AsWritten)],
                &contexts[&(q.qid.as_str(), f)],
                q.gold_page.as_str(),
            )
        }));
        flips.insert(f.as_str().to_string(), flip_json(&report));
    }

    // The reuse audit: the early search was made on one form, the turn's own
    // words are the question as written. The first scenario is the control.
    let texts: HashMap<(&str, Form), (String, String)> = whole
        .iter()
        .flat_map(|q| {
            Form::ALL.map(|f| {
                let words = f.of(&q.question);
                let query = turn_query(&words);
                ((q.qid.as_str(), f), (words, query))
            })
        })
        .collect();
    let mut audit = serde_json::Map::new();
    for (scenario, early_form) in [
        ("the_same_words", Form::AsWritten),
        ("an_end_mark_missing", Form::NoEndMark),
        ("no_capitals_no_punctuation", Form::Plain),
        ("the_last_word_still_to_come", Form::LastWordMissing),
    ] {
        let rules: Vec<Value> = [Reuse::SameWordsLoose, Reuse::SameWords, Reuse::SameRequest]
            .into_iter()
            .map(|rule| {
                let pairs = whole.iter().map(|q| {
                    let id = q.qid.as_str();
                    let (early_words, early_query) = &texts[&(id, early_form)];
                    let (turn_words, turn_query) = &texts[&(id, Form::AsWritten)];
                    ReusePair {
                        early: Asked {
                            words: early_words,
                            query: early_query,
                        },
                        turn: Asked {
                            words: turn_words,
                            query: turn_query,
                        },
                        early_found: &contexts[&(id, early_form)],
                        turn_found: &contexts[&(id, Form::AsWritten)],
                        right_page: q.gold_page.as_str(),
                    }
                });
                audit_json(&audit_rule(rule, pairs))
            })
            .collect();
        audit.insert(scenario.to_string(), Value::Array(rules));
    }

    let settle = settle.map(|by_q| {
        let curve = settle_curve(
            whole.iter().filter_map(|q| {
                by_q.get(&q.qid).map(|rows| Settling {
                    whole: &contexts[&(q.qid.as_str(), Form::AsWritten)],
                    by_words_left: rows,
                    right_page: q.gold_page.as_str(),
                })
            }),
            SETTLE_CAP,
        );
        json!({
            "questions": curve.questions,
            "whole_question_right_page": share_json(curve.whole_right_page),
            "by_words_still_to_come": curve.rows.iter().map(|r| json!({
                "words_left": r.words_left,
                "same_list": share_json(r.same_list),
                "same_lead": share_json(r.same_lead),
                "right_page_among_the_passages": share_json(r.right_page),
                "nothing_found": share_json(r.nothing_found),
            })).collect::<Vec<_>>(),
            "settled_at_least_n_words_before_the_end": curve.settled.iter().map(|s| share_json(*s)).collect::<Vec<_>>(),
        })
    });

    json!({
        "questions": whole.len(),
        "by_form": forms,
        "against_as_written": flips,
        "reuse_audit": audit,
        "settle": settle,
    })
}

/// The audit of recorded `forms` lookups (and `prefixes`, for the settle
/// curve): over all questions, the short ones and the long ones.
pub fn report(set: &QuestionSet, forms: &[Looked], prefixes: Option<&[Looked]>) -> Result<Value> {
    let mut by_form = HashMap::new();
    let mut lookups = 0usize;
    for l in forms {
        if let Some(f) = Form::parse(&l.form) {
            lookups += 1;
            by_form.insert((l.qid.clone(), f), l.clone());
        }
    }
    let settle = match prefixes {
        None => None,
        Some(lines) => {
            let mut by_q: HashMap<String, Vec<(u32, Context)>> = HashMap::new();
            for l in lines {
                // "prefix_<k>_of_<n>": n - k words still to come.
                let mut parts = l.form.split('_');
                let (Some("prefix"), Some(k), Some("of"), Some(n)) =
                    (parts.next(), parts.next(), parts.next(), parts.next())
                else {
                    continue;
                };
                let (k, n): (u32, u32) = (k.parse()?, n.parse()?);
                lookups += 1;
                by_q.entry(l.qid.clone())
                    .or_default()
                    .push((n.saturating_sub(k), l.context()));
            }
            Some(by_q)
        }
    };
    let all: Vec<&Question> = set.questions.iter().collect();
    let styled = |style: &str| -> Vec<&Question> {
        set.questions.iter().filter(|q| q.style == style).collect()
    };
    Ok(json!({
        "lookups": lookups,
        "all": report_group(&all, &by_form, settle.as_ref()),
        "short_questions": report_group(&styled("short"), &by_form, None),
        "long_questions": report_group(&styled("long"), &by_form, None),
    }))
}

/// The same lookups run twice: how often the SAME request got a different
/// passage list. A lookup that answers one request one way has a flip rate
/// of zero here; every other flip rate stands on this floor.
pub fn again(set: &QuestionSet, a: &[Looked], b: &[Looked]) -> Value {
    let right: HashMap<&str, &str> = set
        .questions
        .iter()
        .map(|q| (q.qid.as_str(), q.gold_page.as_str()))
        .collect();
    let b: HashMap<(&str, &str), &Looked> = b
        .iter()
        .map(|l| ((l.qid.as_str(), l.form.as_str()), l))
        .collect();
    let pairs: Vec<(&Looked, &Looked)> = a
        .iter()
        .filter_map(|l| {
            b.get(&(l.qid.as_str(), l.form.as_str()))
                .map(|other| (l, *other))
        })
        .collect();
    let contexts: Vec<(Context, Context, &str)> = pairs
        .iter()
        .map(|(x, y)| {
            (
                x.context(),
                y.context(),
                right.get(x.qid.as_str()).copied().unwrap_or(""),
            )
        })
        .collect();
    let flips = flip_report(contexts.iter().map(|(x, y, page)| (x, y, *page)));
    // The scores of the passages both runs returned: equal, or how far apart.
    let (mut equal, mut scored, mut widest) = (Share::default(), 0u32, 0.0_f64);
    let mut moved = Share::default();
    for (x, y) in &pairs {
        let mut any = false;
        for (id, score) in x.ids.iter().zip(&x.scores) {
            let Some(other) = y
                .ids
                .iter()
                .position(|o| o == id)
                .and_then(|i| y.scores.get(i))
            else {
                continue;
            };
            let gap = (score - other).abs();
            equal.add(gap < 1e-9);
            scored += 1;
            widest = widest.max(gap);
            any |= gap >= 1e-9;
        }
        moved.add(any);
    }
    json!({
        "lookups_in_both_runs": pairs.len(),
        "repeat": flip_json(&flips),
        "lookups_whose_scores_moved": share_json(moved),
        "shared_passages_scored": scored,
        "shared_passages_with_the_same_score": share_json(equal),
        "widest_score_gap": (widest * 10000.0).round() / 10000.0,
    })
}
