//! The paper's numbers, regenerated from the released data.
//!
//! The reports the experiment printed are kept in `tests/expected/`; each
//! test here recomputes one from `data/` with this crate's code and requires
//! every value to agree. The last tests read the paper's tables off the
//! regenerated reports.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::path::{Path, PathBuf};

use measure::{audit, calls, questions};
use serde_json::Value;

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn data(rel: &str) -> PathBuf {
    root().join("data").join(rel)
}

fn expected(name: &str) -> Value {
    let p = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/expected")
        .join(name);
    serde_json::from_str(&std::fs::read_to_string(&p).expect("expected report")).expect("json")
}

fn questions() -> questions::QuestionSet {
    questions::read(&data("questions.json")).expect("questions")
}

fn looked(rel: &str) -> Vec<audit::Looked> {
    audit::read_looked(&data(rel)).expect("lookups")
}

/// Every value in `want` is in `got`, equal (numbers as numbers). Keys in
/// `skip` are not compared; keys only `got` has are allowed.
fn agrees(want: &Value, got: &Value, path: &str, skip: &[&str], diffs: &mut Vec<String>) {
    match (want, got) {
        (Value::Object(w), Value::Object(g)) => {
            for (k, wv) in w {
                if skip.contains(&k.as_str()) {
                    continue;
                }
                match g.get(k) {
                    Some(gv) => agrees(wv, gv, &format!("{path}.{k}"), skip, diffs),
                    None => diffs.push(format!("{path}.{k}: missing")),
                }
            }
        }
        (Value::Array(w), Value::Array(g)) if w.len() == g.len() => {
            for (i, (wv, gv)) in w.iter().zip(g).enumerate() {
                agrees(wv, gv, &format!("{path}[{i}]"), skip, diffs);
            }
        }
        (Value::Number(w), Value::Number(g)) => {
            if w.as_f64() != g.as_f64() {
                diffs.push(format!("{path}: want {w}, got {g}"));
            }
        }
        (w, g) => {
            if w != g {
                diffs.push(format!("{path}: want {w}, got {g}"));
            }
        }
    }
}

fn assert_agrees(want: &Value, got: &Value, skip: &[&str]) {
    let mut diffs = Vec::new();
    agrees(want, got, "", skip, &mut diffs);
    assert!(
        diffs.is_empty(),
        "{} values differ:\n{}",
        diffs.len(),
        diffs
            .iter()
            .take(25)
            .cloned()
            .collect::<Vec<_>>()
            .join("\n")
    );
}

fn calls_report() -> Value {
    let pages = calls::read_pages(&data("passage_pages.jsonl")).expect("pages");
    let pairs =
        calls::parse_pairs(&std::fs::read_to_string(data("calls/pairs.txt")).expect("pairs"))
            .expect("pairs");
    calls::report(&data("calls"), &questions(), Some(&pages), &pairs).expect("report")
}

/// The calls: every arm and every pair (round 1, 2 and 3). The searches of
/// the round-1 arms were counted from the knowledge service's log after the
/// report was printed, so they are compared on their own below.
#[test]
fn the_calls_report_regenerates_exactly() {
    assert_agrees(
        &expected("calls_report.json"),
        &calls_report(),
        &["knowledge_searches"],
    );
}

#[test]
fn the_context_stability_audit_regenerates_exactly() {
    let set = questions();
    let box_run =
        audit::report(&set, &looked("lookups/box_forms_run1.jsonl"), None).expect("report");
    assert_agrees(&expected("box_forms_run1_report.json"), &box_run, &[]);
    let shared = audit::report(
        &set,
        &looked("lookups/shared_forms.jsonl"),
        Some(&looked("lookups/shared_prefixes.jsonl")),
    )
    .expect("report");
    assert_agrees(&expected("shared_forms_report.json"), &shared, &[]);
}

#[test]
fn the_split_lookup_comparison_regenerates_exactly() {
    let set = questions();
    let (existing, split) = (
        looked("lookups/box_existing.jsonl"),
        looked("lookups/box_split.jsonl"),
    );
    assert_agrees(
        &expected("existing_vs_split.json"),
        &audit::again(&set, &existing, &split),
        &[],
    );
    assert_agrees(
        &expected("box_existing_report.json"),
        &audit::report(&set, &existing, None).expect("report"),
        &[],
    );
    assert_agrees(
        &expected("box_split_report.json"),
        &audit::report(&set, &split, None).expect("report"),
        &[],
    );
}

#[test]
fn the_oversampling_runs_regenerate_exactly() {
    let set = questions();
    for (run, want) in [
        (
            "lookups/box_oversampling3_run1.jsonl",
            "box_oversampling3_report.json",
        ),
        (
            "lookups/box_oversampling16_run1.jsonl",
            "box_oversampling16_report.json",
        ),
    ] {
        assert_agrees(
            &expected(want),
            &audit::report(&set, &looked(run), None).expect("report"),
            &[],
        );
    }
}

fn pct(v: &Value) -> f64 {
    v["pct"].as_f64().expect("a share")
}

fn range(v: &Value) -> (f64, f64) {
    (
        v["range95"][0].as_f64().expect("low"),
        v["range95"][1].as_f64().expect("high"),
    )
}

/// Table 1 and Table 2: the context stability of the lookup and the reuse
/// audit, 405 questions on the test box.
#[test]
fn tables_1_and_2_read_off_the_regenerated_audit() {
    let r =
        audit::report(&questions(), &looked("lookups/box_forms_run1.jsonl"), None).expect("report");
    let all = &r["all"];
    assert_eq!(all["questions"], 405);
    let form = |f: &str| &all["by_form"][f];
    for (f, page, first) in [
        ("as_written", 91.4, 72.8),
        ("no_end_mark", 90.6, 72.3),
        ("plain", 88.9, 74.6),
        ("last_word_missing", 77.0, 59.5),
    ] {
        assert_eq!(pct(&form(f)["right_page_among_the_passages"]), page, "{f}");
        assert_eq!(pct(&form(f)["right_page_first"]), first, "{f}");
    }
    let flip = |f: &str| &all["against_as_written"][f];
    for (f, rate, lo, hi, lead, lost, gained) in [
        ("no_end_mark", 56.0, 51.2, 60.8, 88.4, 7, 4),
        ("plain", 73.6, 69.1, 77.6, 82.5, 12, 2),
        ("last_word_missing", 87.2, 83.5, 90.1, 63.2, 63, 5),
    ] {
        assert_eq!(flip(f)["flip_rate_pct"].as_f64(), Some(rate), "{f}");
        // The flip rate's interval is the same list's, turned over.
        let (same_lo, same_hi) = range(&flip(f)["same_list"]);
        assert_eq!(((100.0 - same_hi) * 10.0).round() / 10.0, lo, "{f}");
        assert_eq!(((100.0 - same_lo) * 10.0).round() / 10.0, hi, "{f}");
        assert_eq!(pct(&flip(f)["same_lead"]), lead, "{f}");
        assert_eq!(
            (
                flip(f)["right_page_lost"].as_u64(),
                flip(f)["right_page_gained"].as_u64()
            ),
            (Some(lost), Some(gained)),
            "{f}"
        );
    }
    let rule = |scenario: &str, name: &str| -> Value {
        all["reuse_audit"][scenario]
            .as_array()
            .expect("rules")
            .iter()
            .find(|r| r["rule"] == name)
            .cloned()
            .expect("the rule")
    };
    for (scenario, stale, stale_range, harmful, harmful_range) in [
        ("an_end_mark_missing", 56.0, (51.2, 60.8), 1.7, (0.8, 3.5)),
        (
            "no_capitals_no_punctuation",
            73.6,
            (69.1, 77.6),
            3.0,
            (1.7, 5.1),
        ),
    ] {
        let loose = rule(scenario, "loose");
        assert_eq!(pct(&loose["reused"]), 100.0, "{scenario}");
        assert_eq!(pct(&loose["stale_of_reused"]), stale, "{scenario}");
        assert_eq!(range(&loose["stale_of_reused"]), stale_range, "{scenario}");
        assert_eq!(pct(&loose["harmful_of_reused"]), harmful, "{scenario}");
        assert_eq!(
            range(&loose["harmful_of_reused"]),
            harmful_range,
            "{scenario}"
        );
        for exact in ["words", "request"] {
            assert_eq!(
                pct(&rule(scenario, exact)["reused"]),
                0.0,
                "{scenario} {exact}"
            );
        }
    }
    for name in ["loose", "words", "request"] {
        let same = rule("the_same_words", name);
        assert_eq!(pct(&same["reused"]), 100.0);
        assert_eq!(pct(&same["stale_of_reused"]), 0.0);
        assert_eq!(
            pct(&rule("the_last_word_still_to_come", name)["reused"]),
            0.0
        );
    }
}

/// The repeat floor: the identical lookups, run three times, 1,623 each.
#[test]
fn the_repeat_floor_is_one_and_a_half_to_two_percent() {
    let set = questions();
    let runs = [
        looked("lookups/box_forms_run1.jsonl"),
        looked("lookups/box_forms_run2.jsonl"),
        looked("lookups/box_forms_run3.jsonl"),
    ];
    let mut rates: Vec<f64> = [(0, 1), (0, 2), (1, 2)]
        .iter()
        .map(|(a, b)| {
            let r = audit::again(&set, &runs[*a], &runs[*b]);
            assert_eq!(r["lookups_in_both_runs"], 1623);
            r["repeat"]["flip_rate_pct"].as_f64().expect("a rate")
        })
        .collect();
    rates.sort_by(f64::total_cmp);
    assert_eq!(rates, [1.5, 1.7, 2.0]);
}

/// The settle curve, 100 long questions against the shared index.
#[test]
fn the_settle_curve_reads_off_the_shared_index_audit() {
    let r = audit::report(
        &questions(),
        &looked("lookups/shared_forms.jsonl"),
        Some(&looked("lookups/shared_prefixes.jsonl")),
    )
    .expect("report");
    let settle = &r["all"]["settle"];
    assert_eq!(settle["questions"], 100);
    assert_eq!(pct(&settle["whole_question_right_page"]), 89.0);
    let settled: Vec<f64> = settle["settled_at_least_n_words_before_the_end"]
        .as_array()
        .expect("curve")
        .iter()
        .map(pct)
        .collect();
    assert_eq!(&settled[..5], &[19.0, 13.0, 10.0, 5.0, 4.0]);
    assert_eq!(
        range(&settle["settled_at_least_n_words_before_the_end"][0]),
        (12.5, 27.8)
    );
    assert_eq!(
        pct(&settle["by_words_still_to_come"][0]["right_page_among_the_passages"]),
        76.0
    );
}

fn pair<'a>(report: &'a Value, name: &str) -> &'a Value {
    &report["pairs"][name]
}

/// Table 3: early start against the prior behaviour, paired by question.
#[test]
fn table_3_reads_off_the_regenerated_calls_report() {
    let r = calls_report();
    for (name, pairs, saved, lo, hi) in [
        ("base-f32=early-f32", 139, 137, 133, 142),
        ("base-f256=early-f256", 154, 142, 128, 146),
        ("base-s=early-s", 72, 139, 132, 146),
        ("base-s-d60=early-s-d60", 72, 220, 213, 228),
        ("base-s-d120=early-s-d120", 71, 243, 233, 255),
        ("base-s-d120=early64-s-d120", 71, 309, 305, 319),
        ("base-s-d120-split=early-s-d120-split", 72, 254, 244, 260),
    ] {
        let p = pair(&r, name);
        assert_eq!(p["questions_heard_as_one_turn_in_both"], pairs, "{name}");
        assert_eq!(p["wait_saved_ms"]["p50"], saved, "{name}");
        assert_eq!(
            p["wait_saved_ms"]["median_95"],
            serde_json::json!([lo, hi]),
            "{name}"
        );
        assert_eq!(pct(&p["same_end_of_turn_verdict"]), 100.0, "{name}");
        assert_eq!(pct(&p["heard_the_same_words"]), 100.0, "{name}");
    }
    let arm = |a: &str| {
        r["arms"][a]["caller_wait_ms"]["p50"]
            .as_i64()
            .expect("a wait")
    };
    assert_eq!((arm("base-f32"), arm("early-f32")), (1066, 930));
    assert_eq!((arm("base-f256"), arm("early-f256")), (1211, 1065));
    assert_eq!(
        (
            arm("base-s-d120"),
            arm("early-s-d120"),
            arm("early64-s-d120")
        ),
        (1292, 1059, 989)
    );
    // Distance, frames and the floor, each paired the same way.
    let saved = |name: &str| {
        (
            pair(&r, name)["wait_saved_ms"]["p50"].as_i64(),
            pair(&r, name)["wait_saved_ms"]["median_95"].clone(),
        )
    };
    assert_eq!(
        saved("early-s=early-s-d60"),
        (Some(-21), serde_json::json!([-34, -18]))
    );
    assert_eq!(
        saved("base-s=base-s-d60"),
        (Some(-107), serde_json::json!([-111, -98]))
    );
    assert_eq!(saved("base-f32=base-f256").0, Some(-105));
    assert_eq!(saved("early-f32=early-f256").0, Some(-119));
    assert_eq!(saved("base-s=base-s-fl192").0, Some(143));
    assert_eq!(saved("base-s-fl192=early-s-fl192").0, Some(117));
    assert_eq!(saved("base-s=early-s-fl192").0, Some(258));
}

/// The searches each arm sent the knowledge service (Section 5.4).
#[test]
fn the_searches_each_arm_sent() {
    let searches = |arm: &str| -> u64 {
        let s: calls::ArmSettings = serde_json::from_str(
            &std::fs::read_to_string(data(&format!("calls/{arm}.json"))).expect("arm"),
        )
        .expect("arm");
        s.knowledge_searches.expect("counted")
    };
    assert_eq!((searches("base-f32"), searches("early-f32")), (265, 338));
    assert_eq!((searches("base-f256"), searches("early-f256")), (248, 360));
}

/// Table 5: the split lookup against the existing lookup, 1,623 lookups.
#[test]
fn table_5_reads_off_the_regenerated_comparison() {
    let set = questions();
    let (existing, split) = (
        looked("lookups/box_existing.jsonl"),
        looked("lookups/box_split.jsonl"),
    );
    let again = audit::again(&set, &existing, &split);
    assert_eq!(again["repeat"]["flip_rate_pct"].as_f64(), Some(2.5));
    assert_eq!(pct(&again["repeat"]["same_lead"]), 99.5);
    assert_eq!(
        (
            again["repeat"]["right_page_lost"].as_u64(),
            again["repeat"]["right_page_gained"].as_u64()
        ),
        (Some(1), Some(0))
    );
    for run in [&existing, &split] {
        let r = audit::report(&set, run, None).expect("report");
        assert_eq!(
            pct(&r["all"]["by_form"]["as_written"]["right_page_among_the_passages"]),
            91.4
        );
        assert_eq!(
            pct(&r["all"]["by_form"]["as_written"]["right_page_first"]),
            72.8
        );
        assert_eq!(
            pct(&r["all"]["by_form"]["as_written"]["nothing_found"]),
            1.0
        );
    }
}
