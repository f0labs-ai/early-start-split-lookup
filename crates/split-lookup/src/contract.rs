//! The boundary of a lookup: what it asks of the three services it depends
//! on, and what they answer. The lookup is written against these traits only,
//! so an embedder, a reranker or a store can be replaced without touching it.

use std::future::Future;

use serde::{Deserialize, Serialize};

/// A keyword vector: token ids ascending, one weight each.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct SparseVec {
    pub indices: Vec<u32>,
    pub values: Vec<f32>,
}

impl SparseVec {
    pub fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }
}

/// How the hybrid search's dense leg runs.
#[derive(Debug, Clone, PartialEq)]
pub enum DenseLeg {
    /// The database's own approximate search: its graph over quantized
    /// vectors, rescoring `oversampling` times as many candidates as the leg
    /// returns with the full vectors.
    Graph { oversampling: f64 },
    /// Only these passages, scored exactly with the full vectors: no graph,
    /// no quantized pass. The ids are the store's point ids.
    Shortlist(Vec<String>),
}

/// One hybrid search: a dense leg and a keyword leg, fused by reciprocal
/// rank. `dense` is None when the embedder is away: the keyword leg alone
/// then answers.
#[derive(Debug, Clone, Copy)]
pub struct HybridQuery<'a> {
    pub dense: Option<&'a [f32]>,
    pub sparse: &'a SparseVec,
    pub dense_leg: &'a DenseLeg,
    /// How many fused candidates to return. Each leg is asked for twice as
    /// many.
    pub limit: usize,
}

/// A passage the hybrid search returned, in fused order.
#[derive(Debug, Clone, PartialEq)]
pub struct Candidate {
    pub id: String,
    pub text: String,
    pub page: String,
    /// The fusion's score: rank-based, not on the reranker's scale.
    pub fused_score: f32,
}

/// A passage a lookup returns.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Hit {
    pub id: String,
    pub page: String,
    pub text: String,
    /// The reranker's score, or the fusion's when the reranker was away.
    pub score: f32,
}

/// Why a service did not answer.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ServiceError {
    #[error("{service} unreachable: {detail}")]
    Unreachable {
        service: &'static str,
        detail: String,
    },
    #[error("{service} answered {status}: {detail}")]
    Status {
        service: &'static str,
        status: u16,
        detail: String,
    },
    #[error("{service} answered something unreadable: {detail}")]
    Malformed {
        service: &'static str,
        detail: String,
    },
}

/// Turns a question into its dense vector.
pub trait Embed: Send + Sync {
    fn embed_query(
        &self,
        text: &str,
    ) -> impl Future<Output = Result<Vec<f32>, ServiceError>> + Send;
}

/// Scores how well each text answers the query: one score per text, in the
/// texts' order.
pub trait Rerank: Send + Sync {
    fn rerank(
        &self,
        query: &str,
        texts: &[&str],
    ) -> impl Future<Output = Result<Vec<f32>, ServiceError>> + Send;
}

/// The vector database.
pub trait Store: Send + Sync {
    /// One hybrid search, in one request.
    fn hybrid(
        &self,
        query: HybridQuery<'_>,
    ) -> impl Future<Output = Result<Vec<Candidate>, ServiceError>> + Send;

    /// Every passage's point id and dense vector, for building the one-bit
    /// codes. Read once, when the lookup starts.
    fn dense_vectors(
        &self,
    ) -> impl Future<Output = Result<(Vec<String>, Vec<Vec<f32>>), ServiceError>> + Send;
}
