//! near: the lookup service on the box with the models.
//!
//!   near --embed-url http://127.0.0.1:8080 --rerank-url http://127.0.0.1:8081
//!        [--qdrant-url http://127.0.0.1:6333] [--collection passages]
//!        [--mode split|graph] [--shortlist 512] [--candidates 32]
//!
//!   POST /v1/search  {"query": "...", "top_k": 4, "score_threshold": 0.12, "mode": "split"}
//!   GET  /healthz    the process answers
//!   GET  /readyz     the codes are loaded
//!
//! Every flag can come from the environment (NEAR_*). With
//! `--saved-query-vectors` in place of `--embed-url` it replays saved query
//! vectors instead of embedding. Without `--rerank-url` the fused order
//! stands. A service listening beyond the loopback interface needs a bearer
//! token (NEAR_TOKEN) from its callers.

use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result, bail};
use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use clap::Parser;
use serde_json::json;
use split_lookup::qdrant::Qdrant;
use split_lookup::tei::{AnyEmbed, AnyRerank, NoRerank, SavedQueryVectors, TeiEmbed, TeiRerank};
use split_lookup::{Lookup, LookupConfig, LookupError, Mode, SearchRequest};

#[derive(Parser, Debug)]
#[command(about = "The lookup service on the box with the models.")]
#[command(group(clap::ArgGroup::new("embedder").required(true).args(["embed_url", "saved_query_vectors"])))]
struct Args {
    #[arg(long, env = "NEAR_QDRANT_URL", default_value = "http://127.0.0.1:6333")]
    qdrant_url: String,
    #[arg(long, env = "NEAR_QDRANT_API_KEY", hide_env_values = true)]
    qdrant_api_key: Option<String>,
    #[arg(long, env = "NEAR_COLLECTION", default_value = "passages")]
    collection: String,
    /// A text-embeddings-inference server for the questions.
    #[arg(long, env = "NEAR_EMBED_URL")]
    embed_url: Option<String>,
    #[arg(long, env = "NEAR_QUERY_PREFIX", default_value = "query: ")]
    query_prefix: String,
    /// Saved query vectors (`{"query", "vector"}` lines) instead of a model.
    #[arg(long, env = "NEAR_SAVED_QUERY_VECTORS")]
    saved_query_vectors: Option<PathBuf>,
    /// A text-embeddings-inference server for reranking.
    #[arg(long, env = "NEAR_RERANK_URL")]
    rerank_url: Option<String>,
    /// Gate on the reranker's logits instead of its probabilities.
    #[arg(long, env = "NEAR_RAW_SCORES")]
    raw_scores: bool,
    /// The mode a request that names none runs in.
    #[arg(long, env = "NEAR_MODE", value_enum, default_value_t = Mode::Split)]
    mode: Mode,
    #[arg(long, env = "NEAR_SHORTLIST", default_value_t = 512)]
    shortlist: usize,
    #[arg(long, env = "NEAR_CANDIDATES", default_value_t = 32)]
    candidates: usize,
    #[arg(long, env = "NEAR_OVERSAMPLING", default_value_t = 3.0)]
    oversampling: f64,
    #[arg(long, env = "NEAR_LISTEN", default_value = "127.0.0.1:8084")]
    listen: SocketAddr,
    /// The bearer token callers must present.
    #[arg(long, env = "NEAR_TOKEN", hide_env_values = true)]
    token: Option<String>,
    /// Every request to a service this lookup depends on is bounded by this.
    #[arg(long, env = "NEAR_TIMEOUT_MS", default_value_t = 10_000)]
    timeout_ms: u64,
}

type NearLookup = Lookup<AnyEmbed, AnyRerank, Qdrant>;

struct App {
    lookup: NearLookup,
    mode: Mode,
    token: Option<String>,
}

/// Equal without telling how far two tokens agree.
fn same_token(a: &[u8], b: &[u8]) -> bool {
    a.len() == b.len() && a.iter().zip(b).fold(0u8, |acc, (x, y)| acc | (x ^ y)) == 0
}

fn refuse(status: StatusCode, detail: &str) -> Response {
    (status, Json(json!({"detail": detail}))).into_response()
}

async fn search(
    State(app): State<Arc<App>>,
    headers: HeaderMap,
    Json(req): Json<SearchRequest>,
) -> Response {
    if let Some(token) = &app.token {
        let given = headers
            .get("authorization")
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.strip_prefix("Bearer "))
            .unwrap_or("");
        if !same_token(given.as_bytes(), token.as_bytes()) {
            return refuse(StatusCode::UNAUTHORIZED, "missing or wrong bearer token");
        }
    }
    let trace = headers
        .get("traceparent")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_string();
    match app.lookup.search(&req, app.mode).await {
        Ok(out) => {
            let t = out.timings;
            tracing::info!(
                traceparent = %trace,
                mode = out.mode.as_str(),
                fell_back = out.fell_back.as_deref().unwrap_or(""),
                embedded = out.embedded,
                reranked = out.reranked,
                candidates = out.candidates,
                returned = out.hits.len(),
                embed_ms = t.embed_ms,
                shortlist_us = t.shortlist_us,
                store_ms = t.store_ms,
                rerank_ms = t.rerank_ms,
                total_ms = t.total_ms,
                "search_done"
            );
            Json(out).into_response()
        }
        Err(LookupError::Invalid(why)) => refuse(StatusCode::UNPROCESSABLE_ENTITY, &why),
        Err(e) => {
            tracing::error!(traceparent = %trace, error = %e, "search_failed");
            refuse(StatusCode::BAD_GATEWAY, &e.to_string())
        }
    }
}

async fn healthz() -> Json<serde_json::Value> {
    Json(json!({"status": "ok"}))
}

async fn readyz(State(app): State<Arc<App>>) -> Json<serde_json::Value> {
    let codes = app.lookup.codes();
    Json(json!({
        "status": "ready",
        "codes": codes.map(|c| c.len()),
        "codes_bytes": codes.map(|c| c.bytes()),
        "default_mode": app.mode.as_str(),
        "shortlist": app.lookup.config().shortlist,
        "candidates": app.lookup.config().candidates,
    }))
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .json()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .init();
    let args = Args::parse();
    if args.token.is_none() && !args.listen.ip().is_loopback() {
        bail!(
            "listening on {} with no bearer token: set NEAR_TOKEN, or listen on the loopback interface",
            args.listen
        );
    }
    let timeout = Duration::from_millis(args.timeout_ms);
    let store = Qdrant::new(
        &args.qdrant_url,
        &args.collection,
        args.qdrant_api_key.clone(),
        timeout,
    )?;
    let embed = match (&args.embed_url, &args.saved_query_vectors) {
        (Some(url), _) => AnyEmbed::Tei(TeiEmbed::new(url, &args.query_prefix, None, timeout)?),
        (None, Some(path)) => {
            let text = std::fs::read_to_string(path)
                .with_context(|| format!("saved query vectors {}", path.display()))?;
            AnyEmbed::Saved(
                SavedQueryVectors::read(&text)
                    .with_context(|| format!("saved query vectors {}", path.display()))?,
            )
        }
        (None, None) => bail!("an embedder: --embed-url or --saved-query-vectors"),
    };
    let rerank = match &args.rerank_url {
        Some(url) => AnyRerank::Tei(TeiRerank::new(url, args.raw_scores, None, timeout)?),
        None => {
            tracing::warn!("no reranker: every lookup keeps the fused order");
            AnyRerank::None(NoRerank)
        }
    };
    let cfg = LookupConfig {
        shortlist: args.shortlist,
        candidates: args.candidates,
        oversampling: args.oversampling,
    };
    let mut lookup = Lookup::new(embed, rerank, store, cfg)?;
    let started = std::time::Instant::now();
    let passages = lookup
        .load_codes()
        .await
        .context("reading the passages' vectors for their codes")?;
    tracing::info!(
        passages,
        bytes = lookup.codes().map(|c| c.bytes()),
        ms = started.elapsed().as_millis() as u64,
        "codes_loaded"
    );
    let app = Arc::new(App {
        lookup,
        mode: args.mode,
        token: args.token.clone(),
    });
    let router = Router::new()
        .route("/v1/search", post(search))
        .route("/healthz", get(healthz))
        .route("/readyz", get(readyz))
        .with_state(app);
    let listener = tokio::net::TcpListener::bind(args.listen)
        .await
        .with_context(|| format!("listen on {}", args.listen))?;
    tracing::info!(listen = %args.listen, mode = args.mode.as_str(), collection = %args.collection, "near_listening");
    axum::serve(listener, router)
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
        })
        .await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokens_compare_whole() {
        assert!(same_token(b"secret", b"secret"));
        assert!(!same_token(b"secret", b"secreT"));
        assert!(!same_token(b"secret", b"secret!"));
        assert!(!same_token(b"", b"x"));
    }

    #[test]
    fn an_embedder_is_required() {
        assert!(Args::try_parse_from(["near"]).is_err());
        let a =
            Args::try_parse_from(["near", "--embed-url", "http://127.0.0.1:8080"]).expect("args");
        assert_eq!((a.mode, a.shortlist, a.candidates), (Mode::Split, 512, 32));
    }
}
