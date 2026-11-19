//! The embedder and reranker clients against a stand-in server speaking
//! text-embeddings-inference's interface: what they send, and how they read
//! what comes back.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::sync::{Arc, Mutex};
use std::time::Duration;

use axum::routing::post;
use axum::{Json, Router};
use serde_json::{Value, json};
use split_lookup::tei::{TeiEmbed, TeiRerank};
use split_lookup::{Embed, Rerank, ServiceError};

/// A stand-in server: `/embed` answers a 3-dimensional vector a text, and
/// `/rerank` answers the texts' scores in DESCENDING order of score (as the
/// real server does), each with its index. It keeps every body it got.
async fn server(broken: bool) -> (String, Arc<Mutex<Vec<Value>>>) {
    let seen: Arc<Mutex<Vec<Value>>> = Arc::default();
    let (s1, s2) = (seen.clone(), seen.clone());
    let app = Router::new()
        .route(
            "/embed",
            post(move |Json(body): Json<Value>| {
                let seen = s1.clone();
                async move {
                    seen.lock().unwrap().push(body.clone());
                    let n = body["inputs"].as_array().map(Vec::len).unwrap_or(0);
                    let n = if broken { n + 1 } else { n };
                    Json(json!(
                        (0..n).map(|i| vec![i as f32, 1.0, 0.0]).collect::<Vec<_>>()
                    ))
                }
            }),
        )
        .route(
            "/rerank",
            post(move |Json(body): Json<Value>| {
                let seen = s2.clone();
                async move {
                    seen.lock().unwrap().push(body.clone());
                    let n = body["texts"].as_array().map(Vec::len).unwrap_or(0);
                    // Text i scores i / 10: the highest index first.
                    let mut ranked: Vec<Value> = (0..n)
                        .rev()
                        .map(|i| json!({"index": i, "score": i as f32 / 10.0}))
                        .collect();
                    if broken {
                        ranked.pop();
                    }
                    Json(Value::Array(ranked))
                }
            }),
        );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind");
    let url = format!("http://{}", listener.local_addr().expect("addr"));
    tokio::spawn(async move { axum::serve(listener, app).await });
    (url, seen)
}

#[tokio::test]
async fn questions_are_embedded_with_their_prefix_normalised_and_truncated() {
    let (url, seen) = server(false).await;
    let e = TeiEmbed::new(&url, "query: ", None, Duration::from_secs(5)).expect("client");
    let v = e.embed_query("Where is it?").await.expect("a vector");
    assert_eq!(v, [0.0, 1.0, 0.0]);
    let many = e.embed_many(&["a", "b"]).await.expect("vectors");
    assert_eq!(many.len(), 2);
    let bodies = seen.lock().unwrap().clone();
    assert_eq!(
        bodies[0],
        json!({"inputs": ["query: Where is it?"], "normalize": true, "truncate": true})
    );
    assert_eq!(bodies[1]["inputs"], json!(["query: a", "query: b"]));
}

#[tokio::test]
async fn rerank_scores_come_back_in_the_texts_order() {
    let (url, seen) = server(false).await;
    let r = TeiRerank::new(&url, false, None, Duration::from_secs(5)).expect("client");
    let scores = r.rerank("q", &["t0", "t1", "t2"]).await.expect("scores");
    assert_eq!(scores, [0.0, 0.1, 0.2], "by text, not by rank");
    let body = seen.lock().unwrap()[0].clone();
    assert_eq!(
        body,
        json!({"query": "q", "texts": ["t0", "t1", "t2"], "raw_scores": false, "truncate": true})
    );
}

#[tokio::test]
async fn a_server_that_answers_the_wrong_count_is_an_error() {
    let (url, _) = server(true).await;
    let e = TeiEmbed::new(&url, "", None, Duration::from_secs(5)).expect("client");
    assert!(matches!(
        e.embed_many(&["a"]).await,
        Err(ServiceError::Malformed { .. })
    ));
    let r = TeiRerank::new(&url, false, None, Duration::from_secs(5)).expect("client");
    assert!(matches!(
        r.rerank("q", &["a", "b"]).await,
        Err(ServiceError::Malformed { .. })
    ));
}

#[tokio::test]
async fn a_server_that_is_not_there_is_unreachable() {
    let e =
        TeiEmbed::new("http://127.0.0.1:9", "", None, Duration::from_millis(500)).expect("client");
    assert!(matches!(
        e.embed_query("a").await,
        Err(ServiceError::Unreachable { .. })
    ));
}
