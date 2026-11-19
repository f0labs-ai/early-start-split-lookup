//! The lookup: embed, shortlist (split mode), one hybrid search, rerank and
//! gate. Generic over its three services ([`Embed`], [`Rerank`], [`Store`]).
//!
//! Degradation, in order of what can fail:
//! * no codes, or codes of another dimension: the split runs as a graph
//!   lookup and says so (`fell_back`);
//! * the store refuses the split's request: the same lookup again as a graph
//!   lookup, and says so;
//! * the embedder is away: the keyword leg alone answers (`embedded: false`);
//! * the reranker is away: the fused order stands, without the score gate,
//!   whose scale is the reranker's (`reranked: false`).
//!
//! Only the store failing for both modes fails the lookup.

use std::time::Instant;

use onebit::{Centring, Codes};
use serde::{Deserialize, Serialize};

use crate::bm25;
use crate::contract::{Candidate, DenseLeg, Embed, Hit, HybridQuery, Rerank, ServiceError, Store};

/// How the dense leg is asked.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, clap::ValueEnum)]
#[serde(rename_all = "lowercase")]
pub enum Mode {
    /// The database's own approximate search over its quantized vectors.
    Graph,
    /// The shortlist from the one-bit codes, scored exactly by the database.
    Split,
}

impl Mode {
    pub const fn as_str(self) -> &'static str {
        match self {
            Mode::Graph => "graph",
            Mode::Split => "split",
        }
    }
}

/// A lookup's fixed settings.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LookupConfig {
    /// How many passages the split shortlists (the paper: 512).
    pub shortlist: usize,
    /// How many fused candidates are reranked (the paper: 32).
    pub candidates: usize,
    /// The graph mode's rescoring factor (the database's default here: 3).
    pub oversampling: f64,
}

impl Default for LookupConfig {
    fn default() -> Self {
        LookupConfig {
            shortlist: 512,
            candidates: 32,
            oversampling: 3.0,
        }
    }
}

impl LookupConfig {
    /// The longest shortlist one request carries.
    pub const MAX_SHORTLIST: usize = 4096;
    /// The most candidates one lookup reranks.
    pub const MAX_CANDIDATES: usize = 256;

    pub fn validate(&self) -> Result<(), LookupError> {
        if !(16..=Self::MAX_SHORTLIST).contains(&self.shortlist) {
            return Err(LookupError::Invalid(format!(
                "a shortlist of {} passages: 16 to {}",
                self.shortlist,
                Self::MAX_SHORTLIST
            )));
        }
        if !(1..=Self::MAX_CANDIDATES).contains(&self.candidates) {
            return Err(LookupError::Invalid(format!(
                "{} candidates: 1 to {}",
                self.candidates,
                Self::MAX_CANDIDATES
            )));
        }
        if !self.oversampling.is_finite() || self.oversampling < 1.0 {
            return Err(LookupError::Invalid(format!(
                "an oversampling of {}: a number of at least 1",
                self.oversampling
            )));
        }
        Ok(())
    }
}

fn default_top_k() -> usize {
    4
}

/// One lookup as a caller asks it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SearchRequest {
    pub query: String,
    #[serde(default = "default_top_k")]
    pub top_k: usize,
    /// The least reranker score a passage needs (the paper: 0.12).
    #[serde(default)]
    pub score_threshold: Option<f32>,
    /// The mode to run; the lookup's default when absent.
    #[serde(default)]
    pub mode: Option<Mode>,
}

impl SearchRequest {
    /// The most passages one lookup returns.
    pub const MAX_TOP_K: usize = 64;

    pub fn validate(&self) -> Result<(), LookupError> {
        if self.query.trim().is_empty() {
            return Err(LookupError::Invalid("the query is empty".into()));
        }
        if self.query.chars().count() > early_start::WIRE_QUERY_MAX_CHARS {
            return Err(LookupError::Invalid(format!(
                "the query is over {} characters",
                early_start::WIRE_QUERY_MAX_CHARS
            )));
        }
        if !(1..=Self::MAX_TOP_K).contains(&self.top_k) {
            return Err(LookupError::Invalid(format!(
                "top_k {}: 1 to {}",
                self.top_k,
                Self::MAX_TOP_K
            )));
        }
        if self.score_threshold.is_some_and(|t| !t.is_finite()) {
            return Err(LookupError::Invalid(
                "the score threshold is not a number".into(),
            ));
        }
        Ok(())
    }
}

/// Where one lookup's time went.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct Timings {
    pub embed_ms: f64,
    pub shortlist_us: u64,
    pub store_ms: f64,
    pub rerank_ms: f64,
    pub total_ms: f64,
}

/// What one lookup found, and how.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SearchOutcome {
    pub hits: Vec<Hit>,
    /// The mode the dense leg really ran in.
    pub mode: Mode,
    /// Why a split lookup ran as a graph lookup, when it did.
    pub fell_back: Option<String>,
    /// False when the embedder was away and the keyword leg answered alone.
    pub embedded: bool,
    /// False when the reranker was away and the fused order stands.
    pub reranked: bool,
    /// How many fused candidates the store returned.
    pub candidates: usize,
    pub timings: Timings,
}

#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum LookupError {
    #[error("invalid: {0}")]
    Invalid(String),
    #[error(transparent)]
    Service(#[from] ServiceError),
    #[error("the codes: {0}")]
    Codes(#[from] onebit::CodesError),
}

/// A lookup over one store, with the codes of its passages once loaded.
pub struct Lookup<E, R, S> {
    embed: E,
    rerank: R,
    store: S,
    codes: Option<Codes>,
    cfg: LookupConfig,
}

fn ms_since(t: Instant) -> f64 {
    (t.elapsed().as_secs_f64() * 10_000.0).round() / 10.0
}

impl<E: Embed, R: Rerank, S: Store> Lookup<E, R, S> {
    pub fn new(embed: E, rerank: R, store: S, cfg: LookupConfig) -> Result<Self, LookupError> {
        cfg.validate()?;
        Ok(Lookup {
            embed,
            rerank,
            store,
            codes: None,
            cfg,
        })
    }

    /// Read every passage's vector from the store once and keep its signs
    /// (around zero, as measured). Returns how many passages.
    /// Budget: one scroll of the collection, at start-up only.
    pub async fn load_codes(&mut self) -> Result<usize, LookupError> {
        let (ids, vectors) = self.store.dense_vectors().await?;
        let dim = vectors.first().map(Vec::len).unwrap_or(0);
        let codes = Codes::build(dim.max(1), ids, &vectors, Centring::Zero)?;
        let n = codes.len();
        self.codes = Some(codes);
        Ok(n)
    }

    /// Codes already made (from a cache), in place of [`Lookup::load_codes`].
    pub fn with_codes(mut self, codes: Codes) -> Self {
        self.codes = Some(codes);
        self
    }

    pub fn codes(&self) -> Option<&Codes> {
        self.codes.as_ref()
    }

    pub fn config(&self) -> LookupConfig {
        self.cfg
    }

    /// One lookup. `default_mode` runs when the request names none.
    pub async fn search(
        &self,
        req: &SearchRequest,
        default_mode: Mode,
    ) -> Result<SearchOutcome, LookupError> {
        req.validate()?;
        let started = Instant::now();
        let mut timings = Timings::default();
        let asked = req.mode.unwrap_or(default_mode);
        let sparse = bm25::encode_query(&req.query);

        // 1. The question's vector. Without it the keyword leg answers alone.
        let t = Instant::now();
        let dense = match self.embed.embed_query(&req.query).await {
            Ok(v) if !v.is_empty() => Some(v),
            Ok(_) => None,
            Err(e) => {
                tracing::warn!(error = %e, "embedder_away_keyword_leg_only");
                None
            }
        };
        timings.embed_ms = ms_since(t);

        // 2. The shortlist, from the codes in memory.
        let graph = DenseLeg::Graph {
            oversampling: self.cfg.oversampling,
        };
        let mut fell_back = None;
        let (mut leg, mut mode) = (graph.clone(), Mode::Graph);
        if asked == Mode::Split {
            let t = Instant::now();
            match (&self.codes, &dense) {
                (Some(codes), Some(v)) => match codes.shortlist(v, self.cfg.shortlist) {
                    Ok(ids) => (leg, mode) = (DenseLeg::Shortlist(ids), Mode::Split),
                    Err(e) => fell_back = Some(format!("shortlist: {e}")),
                },
                (None, _) => fell_back = Some("no codes loaded".into()),
                (_, None) => fell_back = Some("no question vector".into()),
            }
            timings.shortlist_us = t.elapsed().as_micros() as u64;
        }

        // 3. One request to the store; a refused split is asked again as a
        // graph lookup.
        let t = Instant::now();
        let limit = self.cfg.candidates;
        let first = self
            .store
            .hybrid(HybridQuery {
                dense: dense.as_deref(),
                sparse: &sparse,
                dense_leg: &leg,
                limit,
            })
            .await;
        let candidates = match first {
            Ok(c) => c,
            Err(e) if mode == Mode::Split => {
                tracing::warn!(error = %e, "split_refused_asked_as_graph");
                fell_back = Some(format!("store: {e}"));
                (leg, mode) = (graph, Mode::Graph);
                self.store
                    .hybrid(HybridQuery {
                        dense: dense.as_deref(),
                        sparse: &sparse,
                        dense_leg: &leg,
                        limit,
                    })
                    .await?
            }
            Err(e) => return Err(e.into()),
        };
        timings.store_ms = ms_since(t);

        // 4. Rerank and gate.
        let t = Instant::now();
        let (hits, reranked) = self
            .order(&req.query, &candidates, req.top_k, req.score_threshold)
            .await;
        timings.rerank_ms = ms_since(t);
        timings.total_ms = ms_since(started);
        Ok(SearchOutcome {
            hits,
            mode,
            fell_back,
            embedded: dense.is_some(),
            reranked,
            candidates: candidates.len(),
            timings,
        })
    }

    /// The candidates in the reranker's order, those under the gate dropped,
    /// the first `top_k` kept; equal scores keep the fused order. Without the
    /// reranker, the fused order and no gate.
    async fn order(
        &self,
        query: &str,
        candidates: &[Candidate],
        top_k: usize,
        threshold: Option<f32>,
    ) -> (Vec<Hit>, bool) {
        if candidates.is_empty() {
            return (Vec::new(), true);
        }
        let texts: Vec<&str> = candidates.iter().map(|c| c.text.as_str()).collect();
        let hit = |c: &Candidate, score: f32| Hit {
            id: c.id.clone(),
            page: c.page.clone(),
            text: c.text.clone(),
            score,
        };
        match self.rerank.rerank(query, &texts).await {
            Ok(scores) if scores.len() == candidates.len() => {
                let mut ranked: Vec<(usize, f32)> = scores
                    .into_iter()
                    .enumerate()
                    .filter(|(_, s)| threshold.is_none_or(|t| *s >= t))
                    .collect();
                ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                ranked.truncate(top_k);
                let hits = ranked
                    .into_iter()
                    .map(|(i, s)| hit(&candidates[i], s))
                    .collect();
                (hits, true)
            }
            other => {
                let why = match other {
                    Ok(scores) => format!("{} scores for {} texts", scores.len(), candidates.len()),
                    Err(e) => e.to_string(),
                };
                tracing::warn!(error = %why, "reranker_away_fused_order");
                let hits = candidates
                    .iter()
                    .take(top_k)
                    .map(|c| hit(c, c.fused_score))
                    .collect();
                (hits, false)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::sync::Mutex;

    /// A store that searches exactly in memory: the graph mode over every
    /// passage, the split over the shortlist only. It remembers what it was
    /// asked, and can refuse shortlists.
    struct MemoryStore {
        ids: Vec<String>,
        vectors: Vec<Vec<f32>>,
        texts: Vec<String>,
        legs: Mutex<Vec<DenseLeg>>,
        refuse_shortlists: bool,
    }

    impl MemoryStore {
        fn new(n: usize, dim: usize) -> Self {
            let vectors: Vec<Vec<f32>> = (0..n).map(|i| unit_vector(i as u64 + 1, dim)).collect();
            MemoryStore {
                ids: (0..n).map(|i| format!("p{i}")).collect(),
                vectors,
                texts: (0..n).map(|i| format!("passage number {i}")).collect(),
                legs: Mutex::new(Vec::new()),
                refuse_shortlists: false,
            }
        }
    }

    fn unit_vector(seed: u64, dim: usize) -> Vec<f32> {
        let mut s = seed;
        let v: Vec<f32> = (0..dim)
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

    fn dot(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).map(|(x, y)| x * y).sum()
    }

    impl Store for MemoryStore {
        async fn hybrid(&self, q: HybridQuery<'_>) -> Result<Vec<Candidate>, ServiceError> {
            self.legs.lock().unwrap().push(q.dense_leg.clone());
            if self.refuse_shortlists && matches!(q.dense_leg, DenseLeg::Shortlist(_)) {
                return Err(ServiceError::Status {
                    service: "store",
                    status: 400,
                    detail: "too many ids".into(),
                });
            }
            let Some(dense) = q.dense else {
                return Ok(Vec::new());
            };
            let allowed: Option<Vec<&String>> = match q.dense_leg {
                DenseLeg::Shortlist(ids) => Some(ids.iter().collect()),
                DenseLeg::Graph { .. } => None,
            };
            let mut scored: Vec<(usize, f32)> = (0..self.ids.len())
                .filter(|i| allowed.as_ref().is_none_or(|a| a.contains(&&self.ids[*i])))
                .map(|i| (i, dot(&self.vectors[i], dense)))
                .collect();
            scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap().then(a.0.cmp(&b.0)));
            scored.truncate(q.limit);
            Ok(scored
                .into_iter()
                .enumerate()
                .map(|(rank, (i, _))| Candidate {
                    id: self.ids[i].clone(),
                    text: self.texts[i].clone(),
                    page: format!("page-{}", i % 7),
                    fused_score: 1.0 / (rank as f32 + 2.0),
                })
                .collect())
        }

        async fn dense_vectors(&self) -> Result<(Vec<String>, Vec<Vec<f32>>), ServiceError> {
            Ok((self.ids.clone(), self.vectors.clone()))
        }
    }

    /// An embedder that answers fixed vectors by text, or is away.
    struct FixedEmbed(HashMap<String, Vec<f32>>);

    impl Embed for FixedEmbed {
        async fn embed_query(&self, text: &str) -> Result<Vec<f32>, ServiceError> {
            self.0.get(text).cloned().ok_or(ServiceError::Unreachable {
                service: "embedder",
                detail: "away".into(),
            })
        }
    }

    /// A reranker that scores by the passage's number (higher number, higher
    /// score), or is away.
    struct NumberRerank {
        away: bool,
    }

    impl Rerank for NumberRerank {
        async fn rerank(&self, _query: &str, texts: &[&str]) -> Result<Vec<f32>, ServiceError> {
            if self.away {
                return Err(ServiceError::Unreachable {
                    service: "reranker",
                    detail: "away".into(),
                });
            }
            Ok(texts
                .iter()
                .map(|t| {
                    let n: f32 = t
                        .rsplit(' ')
                        .next()
                        .and_then(|n| n.parse().ok())
                        .unwrap_or(0.0);
                    n / 1000.0
                })
                .collect())
        }
    }

    const DIM: usize = 64;

    fn lookup(
        store: MemoryStore,
        rerank_away: bool,
    ) -> Lookup<FixedEmbed, NumberRerank, MemoryStore> {
        let mut known = HashMap::new();
        known.insert("where is it".to_string(), store.vectors[17].clone());
        Lookup::new(
            FixedEmbed(known),
            NumberRerank { away: rerank_away },
            store,
            LookupConfig {
                shortlist: 64,
                candidates: 8,
                oversampling: 3.0,
            },
        )
        .expect("valid")
    }

    fn ask(query: &str, mode: Option<Mode>) -> SearchRequest {
        SearchRequest {
            query: query.into(),
            top_k: 4,
            score_threshold: None,
            mode,
        }
    }

    #[tokio::test]
    async fn a_split_sends_the_shortlist_and_finds_what_the_graph_finds() {
        let mut l = lookup(MemoryStore::new(300, DIM), false);
        assert_eq!(l.load_codes().await.expect("codes"), 300);
        let graph = l
            .search(&ask("where is it", None), Mode::Graph)
            .await
            .expect("graph");
        let split = l
            .search(&ask("where is it", None), Mode::Split)
            .await
            .expect("split");
        assert_eq!((graph.mode, split.mode), (Mode::Graph, Mode::Split));
        assert_eq!(split.fell_back, None);
        assert_eq!(
            graph.hits, split.hits,
            "the shortlist holds the exact top 8"
        );
        let legs = l.store.legs.lock().unwrap().clone();
        assert!(matches!(&legs[0], DenseLeg::Graph { oversampling } if *oversampling == 3.0));
        assert!(matches!(&legs[1], DenseLeg::Shortlist(ids) if ids.len() == 64));
        assert!(split.embedded && split.reranked);
        assert_eq!(split.candidates, 8);
    }

    #[tokio::test]
    async fn the_reranker_orders_and_the_gate_drops() {
        let mut l = lookup(MemoryStore::new(300, DIM), false);
        l.load_codes().await.expect("codes");
        let out = l
            .search(&ask("where is it", Some(Mode::Split)), Mode::Graph)
            .await
            .expect("split");
        let scores: Vec<f32> = out.hits.iter().map(|h| h.score).collect();
        assert!(scores.windows(2).all(|w| w[0] >= w[1]), "{scores:?}");
        let gate = scores[1];
        let gated = l
            .search(
                &SearchRequest {
                    score_threshold: Some(gate),
                    ..ask("where is it", Some(Mode::Split))
                },
                Mode::Graph,
            )
            .await
            .expect("gated");
        assert_eq!(gated.hits.len(), 2, "only the two at or over the gate");
        assert!(gated.hits.iter().all(|h| h.score >= gate));
    }

    #[tokio::test]
    async fn a_split_without_codes_runs_as_a_graph_lookup_and_says_so() {
        let l = lookup(MemoryStore::new(100, DIM), false);
        let out = l
            .search(&ask("where is it", None), Mode::Split)
            .await
            .expect("lookup");
        assert_eq!(out.mode, Mode::Graph);
        assert_eq!(out.fell_back.as_deref(), Some("no codes loaded"));
        assert_eq!(out.hits.len(), 4);
    }

    #[tokio::test]
    async fn codes_of_another_dimension_fall_back_too() {
        let l = lookup(MemoryStore::new(100, DIM), false).with_codes(
            Codes::build(8, vec!["x".into()], &[vec![1.0; 8]], Centring::Zero).expect("codes"),
        );
        let out = l
            .search(&ask("where is it", None), Mode::Split)
            .await
            .expect("lookup");
        assert_eq!(out.mode, Mode::Graph);
        assert!(out.fell_back.expect("why").starts_with("shortlist:"));
    }

    #[tokio::test]
    async fn a_refused_split_is_asked_again_as_a_graph_lookup() {
        let mut store = MemoryStore::new(200, DIM);
        store.refuse_shortlists = true;
        let mut l = lookup(store, false);
        l.load_codes().await.expect("codes");
        let out = l
            .search(&ask("where is it", None), Mode::Split)
            .await
            .expect("lookup");
        assert_eq!(out.mode, Mode::Graph);
        assert!(out.fell_back.expect("why").contains("too many ids"));
        assert_eq!(out.hits.len(), 4);
        assert_eq!(l.store.legs.lock().unwrap().len(), 2, "split, then graph");
    }

    #[tokio::test]
    async fn without_the_embedder_the_keyword_leg_answers_alone() {
        let mut l = lookup(MemoryStore::new(50, DIM), false);
        l.load_codes().await.expect("codes");
        let out = l
            .search(&ask("something else", None), Mode::Split)
            .await
            .expect("lookup");
        assert!(!out.embedded);
        assert_eq!(out.fell_back.as_deref(), Some("no question vector"));
    }

    #[tokio::test]
    async fn without_the_reranker_the_fused_order_stands_ungated() {
        let mut l = lookup(MemoryStore::new(300, DIM), true);
        l.load_codes().await.expect("codes");
        let out = l
            .search(
                &SearchRequest {
                    score_threshold: Some(0.99),
                    ..ask("where is it", None)
                },
                Mode::Split,
            )
            .await
            .expect("lookup");
        assert!(!out.reranked);
        assert_eq!(
            out.hits.len(),
            4,
            "the gate is the reranker's scale: not applied"
        );
        assert_eq!(out.hits[0].id, "p17", "the question's own passage first");
    }

    #[tokio::test]
    async fn a_request_out_of_bounds_is_refused() {
        let l = lookup(MemoryStore::new(20, DIM), false);
        for bad in [
            ask("  ", None),
            SearchRequest {
                top_k: 0,
                ..ask("where is it", None)
            },
            SearchRequest {
                top_k: 65,
                ..ask("where is it", None)
            },
            SearchRequest {
                score_threshold: Some(f32::NAN),
                ..ask("where is it", None)
            },
            ask(&"x".repeat(4001), None),
        ] {
            assert!(matches!(
                l.search(&bad, Mode::Split).await,
                Err(LookupError::Invalid(_))
            ));
        }
    }

    #[test]
    fn a_config_out_of_bounds_is_refused() {
        for bad in [
            LookupConfig {
                shortlist: 8,
                ..LookupConfig::default()
            },
            LookupConfig {
                shortlist: 5000,
                ..LookupConfig::default()
            },
            LookupConfig {
                candidates: 0,
                ..LookupConfig::default()
            },
            LookupConfig {
                oversampling: 0.5,
                ..LookupConfig::default()
            },
            LookupConfig {
                oversampling: f64::INFINITY,
                ..LookupConfig::default()
            },
        ] {
            assert!(bad.validate().is_err(), "{bad:?}");
        }
        assert!(LookupConfig::default().validate().is_ok());
    }

    #[test]
    fn a_request_reads_with_the_papers_defaults() {
        let r: SearchRequest = serde_json::from_str(r#"{"query":"hi"}"#).expect("json");
        assert_eq!((r.top_k, r.score_threshold, r.mode), (4, None, None));
        let r: SearchRequest =
            serde_json::from_str(r#"{"query":"hi","mode":"split","score_threshold":0.12}"#)
                .expect("json");
        assert_eq!(r.mode, Some(Mode::Split));
    }
}
