//! The embedder and the reranker over the HTTP interface of Hugging Face's
//! text-embeddings-inference (`POST /embed`, `POST /rerank`), and an embedder
//! that replays saved query vectors (no model at all).
//!
//! Instruction-tuned embedders want a prefix: multilingual E5 embeds
//! questions as `"query: ..."` and passages as `"passage: ..."`.

use std::collections::HashMap;
use std::time::Duration;

use serde::Deserialize;
use serde_json::json;

use crate::contract::{Embed, Rerank, ServiceError};

fn client(timeout: Duration, service: &'static str) -> Result<reqwest::Client, ServiceError> {
    reqwest::Client::builder()
        .timeout(timeout)
        .tcp_nodelay(true)
        .build()
        .map_err(|e| ServiceError::Unreachable {
            service,
            detail: e.to_string(),
        })
}

async fn post<T: for<'de> Deserialize<'de>>(
    http: &reqwest::Client,
    url: &str,
    token: Option<&str>,
    body: serde_json::Value,
    service: &'static str,
) -> Result<T, ServiceError> {
    let mut req = http.post(url).json(&body);
    if let Some(t) = token {
        req = req.bearer_auth(t);
    }
    let res = req.send().await.map_err(|e| ServiceError::Unreachable {
        service,
        detail: e.to_string(),
    })?;
    let status = res.status();
    if !status.is_success() {
        let detail: String = res
            .text()
            .await
            .unwrap_or_default()
            .chars()
            .take(300)
            .collect();
        return Err(ServiceError::Status {
            service,
            status: status.as_u16(),
            detail,
        });
    }
    res.json().await.map_err(|e| ServiceError::Malformed {
        service,
        detail: e.to_string(),
    })
}

/// An embedding server speaking text-embeddings-inference's `/embed`.
#[derive(Debug, Clone)]
pub struct TeiEmbed {
    http: reqwest::Client,
    url: String,
    prefix: String,
    token: Option<String>,
}

impl TeiEmbed {
    const SERVICE: &'static str = "embedder";

    /// `prefix` is put before every text (`"query: "` for questions to E5).
    pub fn new(
        base: &str,
        prefix: &str,
        token: Option<String>,
        timeout: Duration,
    ) -> Result<Self, ServiceError> {
        Ok(TeiEmbed {
            http: client(timeout, Self::SERVICE)?,
            url: format!("{}/embed", base.trim_end_matches('/')),
            prefix: prefix.to_string(),
            token,
        })
    }

    /// Many texts at once, unit length, in order.
    pub async fn embed_many(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, ServiceError> {
        let inputs: Vec<String> = texts
            .iter()
            .map(|t| format!("{}{t}", self.prefix))
            .collect();
        let vectors: Vec<Vec<f32>> = post(
            &self.http,
            &self.url,
            self.token.as_deref(),
            json!({"inputs": inputs, "normalize": true, "truncate": true}),
            Self::SERVICE,
        )
        .await?;
        if vectors.len() != texts.len() {
            return Err(ServiceError::Malformed {
                service: Self::SERVICE,
                detail: format!("{} vectors for {} texts", vectors.len(), texts.len()),
            });
        }
        Ok(vectors)
    }
}

impl Embed for TeiEmbed {
    async fn embed_query(&self, text: &str) -> Result<Vec<f32>, ServiceError> {
        let mut v = self.embed_many(&[text]).await?;
        v.pop().ok_or(ServiceError::Malformed {
            service: Self::SERVICE,
            detail: "no vector".into(),
        })
    }
}

/// A reranking server speaking text-embeddings-inference's `/rerank`.
#[derive(Debug, Clone)]
pub struct TeiRerank {
    http: reqwest::Client,
    url: String,
    raw_scores: bool,
    token: Option<String>,
}

#[derive(Deserialize)]
struct Ranked {
    index: usize,
    score: f32,
}

impl TeiRerank {
    const SERVICE: &'static str = "reranker";

    /// `raw_scores`: the model's logits instead of probabilities. A score
    /// gate is on whichever scale is chosen here.
    pub fn new(
        base: &str,
        raw_scores: bool,
        token: Option<String>,
        timeout: Duration,
    ) -> Result<Self, ServiceError> {
        Ok(TeiRerank {
            http: client(timeout, Self::SERVICE)?,
            url: format!("{}/rerank", base.trim_end_matches('/')),
            raw_scores,
            token,
        })
    }
}

impl Rerank for TeiRerank {
    async fn rerank(&self, query: &str, texts: &[&str]) -> Result<Vec<f32>, ServiceError> {
        let ranked: Vec<Ranked> = post(
            &self.http,
            &self.url,
            self.token.as_deref(),
            json!({"query": query, "texts": texts, "raw_scores": self.raw_scores, "truncate": true}),
            Self::SERVICE,
        )
        .await?;
        let mut scores = vec![f32::NAN; texts.len()];
        for r in ranked {
            if let Some(s) = scores.get_mut(r.index) {
                *s = r.score;
            }
        }
        if scores.iter().any(|s| s.is_nan()) {
            return Err(ServiceError::Malformed {
                service: Self::SERVICE,
                detail: "a text was not scored".into(),
            });
        }
        Ok(scores)
    }
}

/// Saved query vectors, by the exact query text: a lookup replayed without
/// an embedding model.
#[derive(Debug, Clone, Default)]
pub struct SavedQueryVectors(HashMap<String, Vec<f32>>);

#[derive(Deserialize)]
struct SavedLine {
    query: String,
    vector: Vec<f32>,
}

impl SavedQueryVectors {
    /// Read `{"query": "...", "vector": [...]}` lines.
    pub fn read(text: &str) -> Result<Self, serde_json::Error> {
        let mut by_query = HashMap::new();
        for line in text.lines().filter(|l| !l.trim().is_empty()) {
            let saved: SavedLine = serde_json::from_str(line)?;
            by_query.insert(saved.query, saved.vector);
        }
        Ok(SavedQueryVectors(by_query))
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

impl Embed for SavedQueryVectors {
    async fn embed_query(&self, text: &str) -> Result<Vec<f32>, ServiceError> {
        self.0
            .get(text)
            .cloned()
            .ok_or_else(|| ServiceError::Unreachable {
                service: "saved query vectors",
                detail: format!("no saved vector for {text:?}"),
            })
    }
}

/// A reranker that is not there: every lookup keeps the fused order.
#[derive(Debug, Clone, Copy, Default)]
pub struct NoRerank;

impl Rerank for NoRerank {
    async fn rerank(&self, _query: &str, _texts: &[&str]) -> Result<Vec<f32>, ServiceError> {
        Err(ServiceError::Unreachable {
            service: "reranker",
            detail: "none configured".into(),
        })
    }
}

/// Either embedder, chosen when the service starts.
#[derive(Debug, Clone)]
pub enum AnyEmbed {
    Tei(TeiEmbed),
    Saved(SavedQueryVectors),
}

impl Embed for AnyEmbed {
    async fn embed_query(&self, text: &str) -> Result<Vec<f32>, ServiceError> {
        match self {
            AnyEmbed::Tei(e) => e.embed_query(text).await,
            AnyEmbed::Saved(e) => e.embed_query(text).await,
        }
    }
}

/// Either reranker, chosen when the service starts.
#[derive(Debug, Clone)]
pub enum AnyRerank {
    Tei(TeiRerank),
    None(NoRerank),
}

impl Rerank for AnyRerank {
    async fn rerank(&self, query: &str, texts: &[&str]) -> Result<Vec<f32>, ServiceError> {
        match self {
            AnyRerank::Tei(r) => r.rerank(query, texts).await,
            AnyRerank::None(r) => r.rerank(query, texts).await,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn saved_vectors_answer_their_exact_query_only() {
        let saved = SavedQueryVectors::read(
            "{\"query\":\"Where is it?\",\"vector\":[0.5,0.5]}\n\n{\"query\":\"b\",\"vector\":[1.0]}\n",
        )
        .expect("lines");
        assert_eq!(saved.len(), 2);
        assert_eq!(
            saved.embed_query("Where is it?").await.expect("saved"),
            [0.5, 0.5]
        );
        assert!(saved.embed_query("Where is it").await.is_err());
        assert!(SavedQueryVectors::read("not json").is_err());
    }

    #[tokio::test]
    async fn no_reranker_always_answers_away() {
        assert!(NoRerank.rerank("q", &["a"]).await.is_err());
    }
}
