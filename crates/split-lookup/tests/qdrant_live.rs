//! The lookup against a real Qdrant: load passages, build the codes from the
//! database, and look up in both modes. Runs only when
//! SPLIT_LOOKUP_QDRANT_URL names a Qdrant to use (the CI starts one); it
//! makes its own collection and drops it after.
//!
//!   SPLIT_LOOKUP_QDRANT_URL=http://127.0.0.1:6333 cargo test -p split-lookup --test qdrant_live

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::HashMap;
use std::time::Duration;

use split_lookup::qdrant::{Point, Qdrant};
use split_lookup::tei::{NoRerank, SavedQueryVectors};
use split_lookup::{Embed, Lookup, LookupConfig, Mode, SearchRequest, ServiceError};

const DIM: usize = 64;

fn unit_vector(seed: u64) -> Vec<f32> {
    let mut s = seed;
    let v: Vec<f32> = (0..DIM)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 33) as f32 / (1u64 << 31) as f32) - 0.5
        })
        .collect();
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    v.into_iter().map(|x| x / norm).collect()
}

/// An embedder that knows a few questions.
struct Known(HashMap<String, Vec<f32>>);

impl Embed for Known {
    async fn embed_query(&self, text: &str) -> Result<Vec<f32>, ServiceError> {
        self.0.get(text).cloned().ok_or(ServiceError::Unreachable {
            service: "embedder",
            detail: "unknown".into(),
        })
    }
}

#[tokio::test]
async fn a_split_lookup_finds_what_the_graph_lookup_finds_in_one_request() {
    let Ok(url) = std::env::var("SPLIT_LOOKUP_QDRANT_URL") else {
        eprintln!("skip: SPLIT_LOOKUP_QDRANT_URL is not set");
        return;
    };
    let collection = format!("split_lookup_test_{}", std::process::id());
    let store = Qdrant::new(&url, &collection, None, Duration::from_secs(30)).expect("client");
    if store.exists().await.expect("qdrant answers") {
        store.delete().await.expect("drop the old one");
    }
    store.create(DIM).await.expect("create");
    // 600 passages: 40 pages of 15, with words that share some terms.
    let points: Vec<Point> = (0..600u64)
        .map(|i| {
            Point::new(
                format!("passage-{i}"),
                format!("how to change setting number {} on page {}", i % 50, i % 40),
                format!("page-{}", i % 40),
                None,
                unit_vector(i + 1),
            )
        })
        .collect();
    for batch in points.chunks(128) {
        store.upsert(batch).await.expect("upsert");
    }
    assert_eq!(store.count().await.expect("count"), 600);

    let mut known = HashMap::new();
    known.insert(
        "how to change setting number 7".to_string(),
        unit_vector(208),
    );
    // Stop words only: no keyword leg, so the dense leg alone answers.
    known.insert("how to".to_string(), unit_vector(208));
    let mut lookup = Lookup::new(
        Known(known),
        NoRerank,
        store.clone(),
        LookupConfig {
            shortlist: 512,
            candidates: 16,
            oversampling: 3.0,
        },
    )
    .expect("lookup");
    assert_eq!(lookup.load_codes().await.expect("codes"), 600);
    assert_eq!(lookup.codes().map(|c| c.bytes()), Some(600 * 8));

    let ask = SearchRequest {
        query: "how to change setting number 7".into(),
        top_k: 16,
        score_threshold: None,
        mode: None,
    };
    let graph = lookup.search(&ask, Mode::Graph).await.expect("graph");
    let split = lookup.search(&ask, Mode::Split).await.expect("split");
    assert_eq!((graph.mode, split.mode), (Mode::Graph, Mode::Split));
    assert_eq!(split.fell_back, None);
    assert!(!split.reranked, "no reranker: the fused order");
    assert_eq!(split.hits.len(), 16);
    // Passage 207 is the question's own vector: first by the dense leg.
    assert!(split.hits.iter().any(|h| h.id == "passage-207"));
    let ids =
        |o: &split_lookup::SearchOutcome| o.hits.iter().map(|h| h.id.clone()).collect::<Vec<_>>();
    // Fused with the keyword leg, the two modes agree on the lead and on
    // almost every passage; their dense legs rank differently (approximate
    // against exact), and the fusion carries that into the order.
    assert_eq!(graph.hits[0].id, split.hits[0].id);
    let shared = ids(&split)
        .iter()
        .filter(|id| ids(&graph).contains(id))
        .count();
    assert!(
        shared >= 14,
        "{shared} of 16 shared: {:?} {:?}",
        ids(&graph),
        ids(&split)
    );
    assert!(
        split
            .hits
            .iter()
            .all(|h| h.page.starts_with("page-") && h.text.contains("setting"))
    );

    // The dense leg alone: the split's IS the exact search over every
    // passage, because the shortlist of 512 holds the exact top 16 here.
    let dense_only = SearchRequest {
        query: "how to".into(),
        ..ask.clone()
    };
    let split_dense = lookup
        .search(&dense_only, Mode::Split)
        .await
        .expect("split");
    let q = unit_vector(208);
    let mut exact: Vec<(u64, f32)> = (0..600u64)
        .map(|i| {
            (
                i,
                unit_vector(i + 1)
                    .iter()
                    .zip(&q)
                    .map(|(a, b)| a * b)
                    .sum::<f32>(),
            )
        })
        .collect();
    exact.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap().then(a.0.cmp(&b.0)));
    let want: Vec<String> = exact
        .iter()
        .take(16)
        .map(|(i, _)| format!("passage-{i}"))
        .collect();
    assert_eq!(ids(&split_dense), want, "the split's dense leg is exact");

    // A question the embedder does not know: the keyword leg alone.
    let keywords = lookup
        .search(
            &SearchRequest {
                query: "setting number 33".into(),
                ..ask.clone()
            },
            Mode::Split,
        )
        .await
        .expect("keywords");
    assert!(!keywords.embedded);
    assert!(!keywords.hits.is_empty());

    // Saved vectors replay the same lookup without a model. (Compared on the
    // dense leg alone: these templated passages tie on keyword scores, and
    // the database may order tied passages differently from one identical
    // request to the next, which is what a lookup's repeat floor measures.)
    let saved = SavedQueryVectors::read(&format!(
        "{}\n",
        serde_json::json!({"query": dense_only.query, "vector": unit_vector(208)})
    ))
    .expect("saved");
    let mut replay = Lookup::new(saved, NoRerank, store.clone(), lookup.config()).expect("lookup");
    replay.load_codes().await.expect("codes");
    let again = replay
        .search(&dense_only, Mode::Split)
        .await
        .expect("replay");
    assert_eq!(ids(&again), ids(&split_dense));

    store.delete().await.expect("drop");
}
