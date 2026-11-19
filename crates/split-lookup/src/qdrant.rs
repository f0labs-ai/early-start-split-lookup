//! The store, over Qdrant's REST interface (tested with 1.16).
//!
//! One collection holds one workspace's passages: a named dense vector
//! `dense` (cosine, full vectors on disk, binary quantization held in
//! memory) and a named keyword vector `sparse` (inverse document frequency
//! applied by the database). Each point carries `pid` (the passage's own id),
//! `text`, `page` and, optionally, `heading`.
//!
//! A hybrid search is ONE request to `/points/query`: two prefetches (dense,
//! keyword) fused by reciprocal rank. In split mode the dense prefetch is
//! confined to the shortlisted point ids with `has_id` and scored exactly.

use std::time::Duration;

use serde_json::{Value, json};

use crate::bm25;
use crate::contract::{Candidate, DenseLeg, HybridQuery, ServiceError, SparseVec, Store};

const SERVICE: &str = "vector database";

/// A passage as the loader writes it.
#[derive(Debug, Clone, PartialEq)]
pub struct Point {
    /// The passage's own id; the point id is derived from it.
    pub pid: String,
    pub text: String,
    pub page: String,
    pub heading: Option<String>,
    pub dense: Vec<f32>,
    pub sparse: SparseVec,
}

impl Point {
    /// A passage with its keyword vector made from its text.
    pub fn new(
        pid: String,
        text: String,
        page: String,
        heading: Option<String>,
        dense: Vec<f32>,
    ) -> Self {
        let sparse = bm25::encode_passage(&text);
        Point {
            pid,
            text,
            page,
            heading,
            dense,
            sparse,
        }
    }
}

/// The point id Qdrant stores for a passage id: the id itself when it is an
/// unsigned integer or a UUID (Qdrant's two kinds of id), else a UUID made
/// from a 128-bit hash of it.
pub fn point_id(pid: &str) -> Value {
    if let Ok(n) = pid.parse::<u64>() {
        return json!(n);
    }
    if is_uuid(pid) {
        return json!(pid.to_ascii_lowercase());
    }
    let parts: Vec<u32> = (0..4u32)
        .map(|seed| bm25::murmur3_32(pid.as_bytes(), seed))
        .collect();
    let hex = format!(
        "{:08x}{:08x}{:08x}{:08x}",
        parts[0], parts[1], parts[2], parts[3]
    );
    json!(format!(
        "{}-{}-{}-{}-{}",
        &hex[0..8],
        &hex[8..12],
        &hex[12..16],
        &hex[16..20],
        &hex[20..32]
    ))
}

fn is_uuid(s: &str) -> bool {
    let b = s.as_bytes();
    b.len() == 36
        && b.iter().enumerate().all(|(i, c)| match i {
            8 | 13 | 18 | 23 => *c == b'-',
            _ => c.is_ascii_hexdigit(),
        })
}

/// A point id as a string, whatever its JSON kind.
fn id_text(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

/// A point id string back into the JSON kind Qdrant expects.
fn id_value(s: &str) -> Value {
    match s.parse::<u64>() {
        Ok(n) => json!(n),
        Err(_) => json!(s),
    }
}

/// One collection of one Qdrant instance.
#[derive(Debug, Clone)]
pub struct Qdrant {
    http: reqwest::Client,
    base: String,
    collection: String,
    api_key: Option<String>,
}

impl Qdrant {
    /// `timeout` bounds every request; a lookup's request is small, a scroll
    /// page or an upsert batch larger.
    pub fn new(
        base: &str,
        collection: &str,
        api_key: Option<String>,
        timeout: Duration,
    ) -> Result<Self, ServiceError> {
        let http = reqwest::Client::builder()
            .timeout(timeout)
            .tcp_nodelay(true)
            .build()
            .map_err(|e| ServiceError::Unreachable {
                service: SERVICE,
                detail: e.to_string(),
            })?;
        Ok(Qdrant {
            http,
            base: base.trim_end_matches('/').to_string(),
            collection: collection.to_string(),
            api_key,
        })
    }

    async fn call(
        &self,
        method: reqwest::Method,
        path: &str,
        body: Option<Value>,
    ) -> Result<Value, ServiceError> {
        let mut req = self.http.request(method, format!("{}{path}", self.base));
        if let Some(key) = &self.api_key {
            req = req.header("api-key", key);
        }
        if let Some(body) = body {
            req = req.json(&body);
        }
        let res = req.send().await.map_err(|e| ServiceError::Unreachable {
            service: SERVICE,
            detail: e.to_string(),
        })?;
        let status = res.status();
        let text = res.text().await.map_err(|e| ServiceError::Malformed {
            service: SERVICE,
            detail: e.to_string(),
        })?;
        if !status.is_success() {
            return Err(ServiceError::Status {
                service: SERVICE,
                status: status.as_u16(),
                detail: text.chars().take(300).collect(),
            });
        }
        serde_json::from_str(&text).map_err(|e| ServiceError::Malformed {
            service: SERVICE,
            detail: e.to_string(),
        })
    }

    /// Does the collection exist?
    pub async fn exists(&self) -> Result<bool, ServiceError> {
        let v = self
            .call(
                reqwest::Method::GET,
                &format!("/collections/{}/exists", self.collection),
                None,
            )
            .await?;
        Ok(v.pointer("/result/exists")
            .and_then(Value::as_bool)
            .unwrap_or(false))
    }

    /// Make the collection for `dim`-dimensional dense vectors.
    pub async fn create(&self, dim: usize) -> Result<(), ServiceError> {
        self.call(
            reqwest::Method::PUT,
            &format!("/collections/{}", self.collection),
            Some(json!({
                "vectors": {"dense": {
                    "size": dim,
                    "distance": "Cosine",
                    "on_disk": true,
                    "quantization_config": {"binary": {"always_ram": true}},
                }},
                "sparse_vectors": {"sparse": {"modifier": "idf"}},
            })),
        )
        .await?;
        Ok(())
    }

    /// Drop the collection.
    pub async fn delete(&self) -> Result<(), ServiceError> {
        self.call(
            reqwest::Method::DELETE,
            &format!("/collections/{}", self.collection),
            None,
        )
        .await?;
        Ok(())
    }

    /// Write `points`, waiting until they are searchable.
    pub async fn upsert(&self, points: &[Point]) -> Result<(), ServiceError> {
        let points: Vec<Value> = points
            .iter()
            .map(|p| {
                let mut payload = json!({"pid": p.pid, "text": p.text, "page": p.page});
                if let Some(h) = &p.heading {
                    payload["heading"] = json!(h);
                }
                json!({
                    "id": point_id(&p.pid),
                    "vector": {
                        "dense": p.dense,
                        "sparse": {"indices": p.sparse.indices, "values": p.sparse.values},
                    },
                    "payload": payload,
                })
            })
            .collect();
        self.call(
            reqwest::Method::PUT,
            &format!("/collections/{}/points?wait=true", self.collection),
            Some(json!({ "points": points })),
        )
        .await?;
        Ok(())
    }

    /// How many points the collection holds.
    pub async fn count(&self) -> Result<u64, ServiceError> {
        let v = self
            .call(
                reqwest::Method::POST,
                &format!("/collections/{}/points/count", self.collection),
                Some(json!({"exact": true})),
            )
            .await?;
        Ok(v.pointer("/result/count")
            .and_then(Value::as_u64)
            .unwrap_or(0))
    }
}

/// The dense prefetch for one leg.
fn dense_prefetch(dense: &[f32], leg: &DenseLeg, limit: usize) -> Value {
    match leg {
        DenseLeg::Graph { oversampling } => json!({
            "query": dense,
            "using": "dense",
            "limit": limit,
            "params": {"quantization": {"rescore": true, "oversampling": oversampling}},
        }),
        DenseLeg::Shortlist(ids) => json!({
            "query": dense,
            "using": "dense",
            "limit": limit,
            "filter": {"must": [{"has_id": ids.iter().map(|s| id_value(s)).collect::<Vec<_>>()}]},
            "params": {"exact": true},
        }),
    }
}

/// The body of one hybrid search: each leg asked for twice the fused limit.
pub fn hybrid_body(q: &HybridQuery<'_>) -> Option<Value> {
    let leg_limit = q.limit * 2;
    let mut prefetch = Vec::new();
    if let Some(dense) = q.dense {
        prefetch.push(dense_prefetch(dense, q.dense_leg, leg_limit));
    }
    if !q.sparse.is_empty() {
        prefetch.push(json!({
            "query": {"indices": q.sparse.indices, "values": q.sparse.values},
            "using": "sparse",
            "limit": leg_limit,
        }));
    }
    if prefetch.is_empty() {
        return None;
    }
    Some(json!({
        "prefetch": prefetch,
        "query": {"fusion": "rrf"},
        "limit": q.limit,
        "with_payload": ["pid", "text", "page"],
    }))
}

impl Store for Qdrant {
    async fn hybrid(&self, q: HybridQuery<'_>) -> Result<Vec<Candidate>, ServiceError> {
        let Some(body) = hybrid_body(&q) else {
            return Ok(Vec::new());
        };
        let v = self
            .call(
                reqwest::Method::POST,
                &format!("/collections/{}/points/query", self.collection),
                Some(body),
            )
            .await?;
        let points = v
            .pointer("/result/points")
            .and_then(Value::as_array)
            .ok_or_else(|| ServiceError::Malformed {
                service: SERVICE,
                detail: "no result.points".into(),
            })?;
        Ok(points
            .iter()
            .map(|p| {
                let field = |k: &str| p.pointer(&format!("/payload/{k}")).and_then(Value::as_str);
                Candidate {
                    id: field("pid")
                        .map(str::to_string)
                        .unwrap_or_else(|| id_text(&p["id"])),
                    text: field("text").unwrap_or_default().to_string(),
                    page: field("page").unwrap_or_default().to_string(),
                    fused_score: p["score"].as_f64().unwrap_or(0.0) as f32,
                }
            })
            .collect())
    }

    async fn dense_vectors(&self) -> Result<(Vec<String>, Vec<Vec<f32>>), ServiceError> {
        let (mut ids, mut vectors) = (Vec::new(), Vec::new());
        let mut offset: Option<Value> = None;
        loop {
            let mut body = json!({"limit": 512, "with_payload": false, "with_vector": ["dense"]});
            if let Some(o) = &offset {
                body["offset"] = o.clone();
            }
            let v = self
                .call(
                    reqwest::Method::POST,
                    &format!("/collections/{}/points/scroll", self.collection),
                    Some(body),
                )
                .await?;
            for p in v
                .pointer("/result/points")
                .and_then(Value::as_array)
                .map(Vec::as_slice)
                .unwrap_or_default()
            {
                let Some(dense) = p.pointer("/vector/dense").and_then(Value::as_array) else {
                    continue;
                };
                ids.push(id_text(&p["id"]));
                vectors.push(
                    dense
                        .iter()
                        .map(|x| x.as_f64().unwrap_or(0.0) as f32)
                        .collect(),
                );
            }
            match v.pointer("/result/next_page_offset") {
                Some(o) if !o.is_null() => offset = Some(o.clone()),
                _ => break,
            }
        }
        Ok((ids, vectors))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_passage_id_becomes_a_point_id_qdrant_accepts() {
        assert_eq!(point_id("42"), json!(42));
        let uuid = "D7528B29-4637-55F9-AD70-29980BBD48B5";
        assert_eq!(point_id(uuid), json!(uuid.to_ascii_lowercase()));
        let made = point_id("help-centre/reset-password#3");
        let made = made.as_str().expect("a uuid");
        assert!(is_uuid(made), "{made}");
        assert_eq!(
            point_id("help-centre/reset-password#3").as_str(),
            Some(made),
            "stable"
        );
        assert_ne!(
            point_id("help-centre/reset-password#4").as_str(),
            Some(made)
        );
    }

    #[test]
    fn a_split_confines_the_dense_leg_and_scores_it_exactly() {
        let sparse = SparseVec {
            indices: vec![7],
            values: vec![1.0],
        };
        let leg = DenseLeg::Shortlist(vec![
            "3".into(),
            "d7528b29-4637-55f9-ad70-29980bbd48b5".into(),
        ]);
        let body = hybrid_body(&HybridQuery {
            dense: Some(&[0.1, 0.2]),
            sparse: &sparse,
            dense_leg: &leg,
            limit: 32,
        })
        .expect("a body");
        assert_eq!(body["limit"], 32);
        assert_eq!(body["query"], json!({"fusion": "rrf"}));
        let dense = &body["prefetch"][0];
        assert_eq!(
            dense["limit"], 64,
            "each leg asked for twice the fused limit"
        );
        assert_eq!(dense["params"], json!({"exact": true}));
        assert_eq!(
            dense["filter"]["must"][0]["has_id"],
            json!([3, "d7528b29-4637-55f9-ad70-29980bbd48b5"])
        );
        assert_eq!(body["prefetch"][1]["using"], "sparse");
    }

    #[test]
    fn a_graph_lookup_rescores_the_quantized_search() {
        let leg = DenseLeg::Graph { oversampling: 3.0 };
        let body = hybrid_body(&HybridQuery {
            dense: Some(&[0.1]),
            sparse: &SparseVec::default(),
            dense_leg: &leg,
            limit: 8,
        })
        .expect("a body");
        let prefetch = body["prefetch"].as_array().expect("prefetch");
        assert_eq!(
            prefetch.len(),
            1,
            "no keyword leg for a query without keywords"
        );
        assert_eq!(
            prefetch[0]["params"],
            json!({"quantization": {"rescore": true, "oversampling": 3.0}})
        );
        assert!(prefetch[0].get("filter").is_none());
    }

    #[test]
    fn nothing_to_ask_asks_nothing() {
        let leg = DenseLeg::Graph { oversampling: 3.0 };
        assert!(
            hybrid_body(&HybridQuery {
                dense: None,
                sparse: &SparseVec::default(),
                dense_leg: &leg,
                limit: 8,
            })
            .is_none()
        );
    }
}
