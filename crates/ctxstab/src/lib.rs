//! Context stability: how the knowledge a turn would get changes when its
//! words change, and what a reuse rule risks when it keeps early work across
//! such a change. Pure: no audio, no clock, no IO.
//!
//! A voice agent that starts a turn's knowledge search before the turn has
//! ended must decide whether the early answer is still the turn's. That
//! decision is only as good as the search is stable, so stability is
//! measured here on the search's OUTPUT, the list of passages a reply is
//! grounded on, not on its ranking quality:
//!
//! * **flip rate**: over a question set, the share of questions whose
//!   passage list is not the same list after a change to the words
//!   ([`FlipReport`]). The changes that matter are the ones with no meaning
//!   (an end mark, capitals) and the caller's next word.
//! * **lead stability**: the share whose FIRST passage stays.
//! * **answer flips**: questions where the page holding the answer is in
//!   one list and not the other: lost, or gained.
//! * **repeat floor**: the flip rate of the identical request asked twice.
//!   Every other flip rate stands on it.
//! * **settle lead**: how many words before a question's end its passage
//!   list stops changing ([`settle_lead`]); the settle curve is its
//!   distribution ([`SettleCurve`]).
//! * **reuse audit**: for one reuse rule on (early words, turn words) pairs,
//!   its yield (how often it keeps the early search), its stale share (how
//!   often what it kept is a different list from the turn's own) and its
//!   harmful share (how often keeping it lost the right page)
//!   ([`audit_rule`]).

use early_start::{Asked, Reuse};
use serde::{Deserialize, Serialize};

/// What one lookup found, as far as stability goes: the passages in order
/// and the page each came from.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct Context {
    pub ids: Vec<String>,
    pub pages: Vec<String>,
}

impl Context {
    /// The same passages in the same order.
    pub fn same_list(&self, other: &Context) -> bool {
        self.ids == other.ids
    }

    /// The same passages, in any order.
    pub fn same_set(&self, other: &Context) -> bool {
        self.ids.len() == other.ids.len() && self.ids.iter().all(|id| other.ids.contains(id))
    }

    /// The same first passage (two empty lists lead with the same nothing).
    pub fn same_lead(&self, other: &Context) -> bool {
        self.ids.first() == other.ids.first()
    }

    /// The share of this list's passages that `other` also holds. An empty
    /// list is wholly kept by an empty one and not at all by any other.
    pub fn kept_in(&self, other: &Context) -> f64 {
        if self.ids.is_empty() {
            return if other.ids.is_empty() { 1.0 } else { 0.0 };
        }
        let kept = self.ids.iter().filter(|id| other.ids.contains(id)).count();
        kept as f64 / self.ids.len() as f64
    }

    /// Does any passage come from `page`?
    pub fn has_page(&self, page: &str) -> bool {
        self.pages.iter().any(|p| p == page)
    }
}

/// The written forms one question is looked up in. The first is the
/// reference; the next two change nothing a listener would hear; the last is
/// the question one word before its end.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Form {
    AsWritten,
    NoEndMark,
    Plain,
    LastWordMissing,
}

impl Form {
    pub const ALL: [Form; 4] = [
        Form::AsWritten,
        Form::NoEndMark,
        Form::Plain,
        Form::LastWordMissing,
    ];

    pub const fn as_str(self) -> &'static str {
        match self {
            Form::AsWritten => "as_written",
            Form::NoEndMark => "no_end_mark",
            Form::Plain => "plain",
            Form::LastWordMissing => "last_word_missing",
        }
    }

    pub fn parse(s: &str) -> Option<Form> {
        Form::ALL.into_iter().find(|f| f.as_str() == s)
    }

    /// The question in this form.
    pub fn of(self, question: &str) -> String {
        match self {
            Form::AsWritten => question.to_string(),
            Form::NoEndMark => strip_end(question).to_string(),
            Form::Plain => plain(question),
            Form::LastWordMissing => {
                let mut words: Vec<&str> = question.split_whitespace().collect();
                words.pop();
                strip_end(&words.join(" ")).to_string()
            }
        }
    }
}

/// The text without the punctuation and spaces it ends on.
fn strip_end(text: &str) -> &str {
    text.trim_end_matches(|c: char| c.is_whitespace() || ",.?!\u{2026}:;".contains(c))
}

/// Lower case, letters digits and apostrophes only, single spaces: how a
/// transcript without punctuation reads.
fn plain(text: &str) -> String {
    let kept: String = text
        .to_lowercase()
        .chars()
        .map(|c| {
            if c.is_alphanumeric() || c.is_whitespace() || c == '\'' {
                c
            } else {
                ' '
            }
        })
        .collect();
    kept.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Every beginning of the question from `from_words` words up, short of the
/// whole: (words in it, words in the question, its text without an end mark).
pub fn prefixes(question: &str, from_words: usize) -> Vec<(usize, usize, String)> {
    let words: Vec<&str> = question.split_whitespace().collect();
    (from_words.max(1)..words.len())
        .map(|k| (k, words.len(), strip_end(&words[..k].join(" ")).to_string()))
        .collect()
}

/// `k` of `n`, with the 95% range a share from a sample deserves (Wilson).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize)]
pub struct Share {
    pub k: u32,
    pub n: u32,
}

impl Share {
    pub fn add(&mut self, hit: bool) {
        self.n += 1;
        self.k += u32::from(hit);
    }

    /// The share in percent (None of nothing).
    pub fn pct(self) -> Option<f64> {
        (self.n > 0).then(|| 100.0 * f64::from(self.k) / f64::from(self.n))
    }

    /// The Wilson 95% range in percent.
    pub fn range95(self) -> Option<(f64, f64)> {
        if self.n == 0 {
            return None;
        }
        let (z, n) = (1.96_f64, f64::from(self.n));
        let p = f64::from(self.k) / n;
        let denom = 1.0 + z * z / n;
        let centre = p + z * z / (2.0 * n);
        let half = z * (p * (1.0 - p) / n + z * z / (4.0 * n * n)).sqrt();
        Some((
            100.0 * ((centre - half) / denom).max(0.0),
            100.0 * ((centre + half) / denom).min(1.0),
        ))
    }
}

/// What one change to the words does to the context, over a question set.
#[derive(Debug, Clone, PartialEq, Default, Serialize)]
pub struct FlipReport {
    /// Same passages, same order. The flip rate is the rest.
    pub same_list: Share,
    pub same_set: Share,
    /// Same first passage (lead stability).
    pub same_lead: Share,
    /// Mean share of the reference's passages the changed list keeps.
    pub kept_mean: f64,
    /// The right page: in both lists, only in the reference's (lost by the
    /// change), only in the changed one (gained).
    pub right_in_both: u32,
    pub right_lost: u32,
    pub right_gained: u32,
}

impl FlipReport {
    /// The share of questions whose passage list changed, in percent.
    pub fn flip_rate_pct(&self) -> Option<f64> {
        self.same_list.pct().map(|same| 100.0 - same)
    }
}

/// Compare each question's reference context with its changed one.
/// The third item of each pair is the page that holds the question's answer.
pub fn flip_report<'a>(
    pairs: impl IntoIterator<Item = (&'a Context, &'a Context, &'a str)>,
) -> FlipReport {
    let mut r = FlipReport::default();
    let mut kept = 0.0;
    for (reference, changed, right_page) in pairs {
        r.same_list.add(reference.same_list(changed));
        r.same_set.add(reference.same_set(changed));
        r.same_lead.add(reference.same_lead(changed));
        kept += reference.kept_in(changed);
        match (reference.has_page(right_page), changed.has_page(right_page)) {
            (true, true) => r.right_in_both += 1,
            (true, false) => r.right_lost += 1,
            (false, true) => r.right_gained += 1,
            (false, false) => {}
        }
    }
    if r.same_list.n > 0 {
        r.kept_mean = kept / f64::from(r.same_list.n);
    }
    r
}

/// How many words before a question's end its passage list stops changing:
/// the largest L such that every beginning with L or fewer words still to
/// come already gets the final list. 0: the last word still changes it.
/// `by_words_left`: (words still to come, the context of that beginning).
pub fn settle_lead(whole: &Context, by_words_left: &[(u32, Context)]) -> u32 {
    let mut sorted: Vec<&(u32, Context)> = by_words_left.iter().collect();
    sorted.sort_by_key(|(left, _)| *left);
    let mut lead = 0;
    for (left, ctx) in sorted {
        // A gap in the beginnings (none looked up at this distance) ends it.
        if *left != lead + 1 || !ctx.same_list(whole) {
            break;
        }
        lead = *left;
    }
    lead
}

/// One row of the settle curve: all beginnings with this many words still to
/// come (the last row gathers `cap` and more).
#[derive(Debug, Clone, PartialEq, Default, Serialize)]
pub struct SettleRow {
    pub words_left: u32,
    pub same_list: Share,
    pub same_lead: Share,
    /// The right page already among the passages.
    pub right_page: Share,
    pub nothing_found: Share,
}

/// How early the context settles, over a question set.
#[derive(Debug, Clone, PartialEq, Default, Serialize)]
pub struct SettleCurve {
    pub questions: u32,
    /// The right page among the passages of the whole question.
    pub whole_right_page: Share,
    pub rows: Vec<SettleRow>,
    /// `settled[i]`: the share of questions whose settle lead is at least
    /// `i + 1` words.
    pub settled: Vec<Share>,
}

/// One question for the settle curve: its whole context, its beginnings'
/// contexts by words still to come, and the page holding its answer.
pub struct Settling<'a> {
    pub whole: &'a Context,
    pub by_words_left: &'a [(u32, Context)],
    pub right_page: &'a str,
}

/// The settle curve up to `cap` words before the end.
pub fn settle_curve<'a>(
    questions: impl IntoIterator<Item = Settling<'a>>,
    cap: u32,
) -> SettleCurve {
    let cap = cap.max(1);
    let mut curve = SettleCurve {
        rows: (1..=cap)
            .map(|words_left| SettleRow {
                words_left,
                ..SettleRow::default()
            })
            .collect(),
        settled: vec![Share::default(); cap as usize],
        ..SettleCurve::default()
    };
    for q in questions {
        curve.questions += 1;
        curve.whole_right_page.add(q.whole.has_page(q.right_page));
        for (left, ctx) in q.by_words_left {
            let row = &mut curve.rows[((*left).clamp(1, cap) - 1) as usize];
            row.same_list.add(ctx.same_list(q.whole));
            row.same_lead.add(ctx.same_lead(q.whole));
            row.right_page.add(ctx.has_page(q.right_page));
            row.nothing_found.add(ctx.ids.is_empty());
        }
        let lead = settle_lead(q.whole, q.by_words_left);
        for (i, share) in curve.settled.iter_mut().enumerate() {
            share.add(lead > i as u32);
        }
    }
    curve
}

/// One (early, turn) pair for the reuse audit: what was searched early, what
/// the turn itself asks, what each found, and the page holding the answer.
pub struct ReusePair<'a> {
    pub early: Asked<'a>,
    pub turn: Asked<'a>,
    pub early_found: &'a Context,
    pub turn_found: &'a Context,
    pub right_page: &'a str,
}

/// What one reuse rule does over a set of pairs.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct RuleAudit {
    pub rule: &'static str,
    /// Yield: the pairs where the rule keeps the early search.
    pub reused: Share,
    /// Of those, the ones where the early list is NOT the turn's own list.
    pub stale: Share,
    /// Of those, the ones where the right page was in the turn's own list
    /// and is not in the kept one.
    pub harmful: Share,
}

/// Audit `rule` on `pairs`.
pub fn audit_rule<'a>(rule: Reuse, pairs: impl IntoIterator<Item = ReusePair<'a>>) -> RuleAudit {
    let mut audit = RuleAudit {
        rule: rule.as_str(),
        reused: Share::default(),
        stale: Share::default(),
        harmful: Share::default(),
    };
    for p in pairs {
        let reused = rule.allows(p.early, p.turn);
        audit.reused.add(reused);
        if reused {
            audit.stale.add(!p.early_found.same_list(p.turn_found));
            audit
                .harmful
                .add(p.turn_found.has_page(p.right_page) && !p.early_found.has_page(p.right_page));
        }
    }
    audit
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ctx(ids: &[&str]) -> Context {
        Context {
            ids: ids.iter().map(|s| s.to_string()).collect(),
            // Passage "a1" lives on page "a", "b2" on "b".
            pages: ids.iter().map(|s| s[..1].to_string()).collect(),
        }
    }

    #[test]
    fn a_question_in_its_four_forms() {
        let q = "How do I, um, reset my Password?";
        assert_eq!(Form::AsWritten.of(q), q);
        assert_eq!(Form::NoEndMark.of(q), "How do I, um, reset my Password");
        assert_eq!(Form::Plain.of(q), "how do i um reset my password");
        assert_eq!(Form::LastWordMissing.of(q), "How do I, um, reset my");
        assert_eq!(Form::Plain.of("It's 4,512\u{2026} OK?!"), "it's 4 512 ok");
        assert_eq!(Form::LastWordMissing.of("Thanks."), "");
        for f in Form::ALL {
            assert_eq!(Form::parse(f.as_str()), Some(f));
        }
        assert_eq!(Form::parse("spoken"), None);
    }

    #[test]
    fn the_beginnings_of_a_question_stop_short_of_the_whole() {
        assert_eq!(
            prefixes("Where is my order, please?", 3),
            vec![
                (3, 5, "Where is my".to_string()),
                (4, 5, "Where is my order".to_string()),
            ]
        );
        assert!(prefixes("Where is it?", 3).is_empty());
    }

    #[test]
    fn two_contexts_are_compared_as_lists_sets_and_leads() {
        let (a, b, c) = (
            ctx(&["a1", "b2", "c3"]),
            ctx(&["a1", "c3", "b2"]),
            ctx(&["b2", "a1"]),
        );
        assert!(a.same_list(&a) && !a.same_list(&b));
        assert!(a.same_set(&b) && !a.same_set(&c));
        assert!(a.same_lead(&b) && !a.same_lead(&c));
        assert!(ctx(&[]).same_lead(&ctx(&[])) && !ctx(&[]).same_lead(&a));
        assert_eq!(a.kept_in(&c), 2.0 / 3.0);
        assert_eq!(ctx(&[]).kept_in(&ctx(&[])), 1.0);
        assert_eq!(ctx(&[]).kept_in(&a), 0.0);
        assert!(a.has_page("b") && !c.has_page("c"));
    }

    #[test]
    fn a_share_knows_its_range() {
        let s = Share { k: 130, n: 406 };
        assert!((s.pct().unwrap() - 32.02).abs() < 0.01);
        let (lo, hi) = s.range95().unwrap();
        assert!(
            (lo - 27.7).abs() < 0.1 && (hi - 36.7).abs() < 0.1,
            "{lo} {hi}"
        );
        assert_eq!(Share::default().pct(), None);
        assert_eq!(Share::default().range95(), None);
        let all = Share { k: 5, n: 5 }.range95().unwrap();
        assert!(all.1 <= 100.0 && all.0 > 50.0);
    }

    #[test]
    fn a_flip_report_counts_lists_leads_and_the_right_page() {
        let whole = [
            ctx(&["a1", "b2"]),
            ctx(&["a1", "b2"]),
            ctx(&["c3"]),
            ctx(&["d4"]),
        ];
        let changed = [
            ctx(&["a1", "b2"]),
            ctx(&["a1", "c3"]),
            ctx(&["a1"]),
            ctx(&[]),
        ];
        // The right pages: a (kept twice), then a (gained), then d (lost).
        let right = ["a", "a", "a", "d"];
        let r = flip_report((0..4).map(|i| (&whole[i], &changed[i], right[i])));
        assert_eq!(r.same_list, Share { k: 1, n: 4 });
        assert_eq!(r.same_lead, Share { k: 2, n: 4 });
        assert_eq!(r.flip_rate_pct(), Some(75.0));
        assert_eq!((r.right_in_both, r.right_lost, r.right_gained), (2, 1, 1));
        assert_eq!(r.kept_mean, (1.0 + 0.5 + 0.0 + 0.0) / 4.0);
        assert_eq!(flip_report(std::iter::empty()).flip_rate_pct(), None);
    }

    #[test]
    fn the_settle_lead_is_the_run_of_final_lists_before_the_end() {
        let whole = ctx(&["a1", "b2"]);
        let other = ctx(&["a1", "c3"]);
        let at = |pairs: &[(u32, &Context)]| -> Vec<(u32, Context)> {
            pairs.iter().map(|(l, c)| (*l, (*c).clone())).collect()
        };
        assert_eq!(settle_lead(&whole, &at(&[(1, &other), (2, &whole)])), 0);
        assert_eq!(
            settle_lead(&whole, &at(&[(2, &whole), (1, &whole), (3, &other)])),
            2
        );
        assert_eq!(
            settle_lead(&whole, &at(&[(1, &whole), (3, &whole)])),
            1,
            "a gap ends it"
        );
        assert_eq!(settle_lead(&whole, &[]), 0);
    }

    #[test]
    fn the_settle_curve_gathers_rows_and_leads() {
        let whole = ctx(&["a1"]);
        let (same, other, none) = (ctx(&["a1"]), ctx(&["b2"]), ctx(&[]));
        let q1 = [
            (1, same.clone()),
            (2, same.clone()),
            (3, other.clone()),
            (9, none.clone()),
        ];
        let q2 = [(1, other.clone()), (2, same.clone())];
        let curve = settle_curve(
            [
                Settling {
                    whole: &whole,
                    by_words_left: &q1,
                    right_page: "a",
                },
                Settling {
                    whole: &whole,
                    by_words_left: &q2,
                    right_page: "a",
                },
            ],
            3,
        );
        assert_eq!(curve.questions, 2);
        assert_eq!(curve.whole_right_page, Share { k: 2, n: 2 });
        assert_eq!(curve.rows[0].same_list, Share { k: 1, n: 2 });
        assert_eq!(curve.rows[1].same_list, Share { k: 2, n: 2 });
        // Three words left and nine (gathered at the cap): neither is final.
        assert_eq!(curve.rows[2].same_list, Share { k: 0, n: 2 });
        assert_eq!(curve.rows[2].nothing_found, Share { k: 1, n: 2 });
        assert_eq!(curve.rows[0].right_page, Share { k: 1, n: 2 });
        // Settle leads: 2 and 0.
        assert_eq!(
            curve.settled,
            vec![
                Share { k: 1, n: 2 },
                Share { k: 1, n: 2 },
                Share { k: 0, n: 2 }
            ]
        );
    }

    fn audit_pairs<'a>(other: &'a Context, own: &'a Context) -> Vec<ReusePair<'a>> {
        let turn = Asked {
            words: "Where is it?",
            query: "Where is it?",
        };
        let pair = |early: &'static str, early_found: &'a Context| ReusePair {
            early: Asked {
                words: early,
                query: early,
            },
            turn,
            early_found,
            turn_found: own,
            right_page: "a",
        };
        // An end mark missing (another list found), the turn's own words
        // (the same list), and the last word still to come.
        vec![
            pair("Where is it", other),
            pair("Where is it?", own),
            pair("Where is", other),
        ]
    }

    /// The audit on the case the paper is about: the early words differ from
    /// the turn's by an end mark, and the search found another list. The
    /// loose rule keeps it (stale, and here the right page is lost); the
    /// exact rules keep only the identical words.
    #[test]
    fn the_reuse_audit_tells_the_rules_apart() {
        let (other, own) = (ctx(&["b2", "c3"]), ctx(&["a1", "b2"]));
        let loose = audit_rule(Reuse::SameWordsLoose, audit_pairs(&other, &own));
        assert_eq!(loose.reused, Share { k: 2, n: 3 });
        assert_eq!(loose.stale, Share { k: 1, n: 2 });
        assert_eq!(loose.harmful, Share { k: 1, n: 2 });
        for rule in [Reuse::SameWords, Reuse::SameRequest] {
            let exact = audit_rule(rule, audit_pairs(&other, &own));
            assert_eq!(exact.reused, Share { k: 1, n: 3 }, "{rule:?}");
            assert_eq!(exact.stale, Share { k: 0, n: 1 }, "{rule:?}");
            assert_eq!(exact.harmful, Share { k: 0, n: 1 }, "{rule:?}");
        }
    }
}
