//! ingest: load a workspace's passages into the vector database the lookup
//! searches.
//!
//!   ingest --passages passages.jsonl [--embed-url http://127.0.0.1:8080]
//!          [--qdrant-url http://127.0.0.1:6333] [--collection passages] [--recreate]
//!
//! One passage a line: `{"id": "...", "text": "...", "page": "...",
//! "heading": "...", "vector": [...]}`. `heading` is optional; a passage
//! without `vector` is embedded through `--embed-url` with `--passage-prefix`
//! before its text. Each passage's keyword vector is made from its text.
//! Prints a JSON summary.

use std::path::PathBuf;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use clap::Parser;
use serde::Deserialize;
use serde_json::json;
use split_lookup::qdrant::{Point, Qdrant};
use split_lookup::tei::TeiEmbed;

#[derive(Parser, Debug)]
#[command(about = "Load a workspace's passages into the vector database.")]
struct Args {
    #[arg(long)]
    passages: PathBuf,
    #[arg(
        long,
        env = "INGEST_QDRANT_URL",
        default_value = "http://127.0.0.1:6333"
    )]
    qdrant_url: String,
    #[arg(long, env = "INGEST_QDRANT_API_KEY", hide_env_values = true)]
    qdrant_api_key: Option<String>,
    #[arg(long, env = "INGEST_COLLECTION", default_value = "passages")]
    collection: String,
    /// A text-embeddings-inference server, for passages without a vector.
    #[arg(long, env = "INGEST_EMBED_URL")]
    embed_url: Option<String>,
    #[arg(long, default_value = "passage: ")]
    passage_prefix: String,
    /// Drop the collection first.
    #[arg(long)]
    recreate: bool,
    #[arg(long, default_value_t = 64)]
    batch: usize,
}

#[derive(Deserialize)]
struct Line {
    id: String,
    text: String,
    page: String,
    #[serde(default)]
    heading: Option<String>,
    #[serde(default)]
    vector: Option<Vec<f32>>,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    if !(1..=1024).contains(&args.batch) {
        bail!("--batch {}: 1 to 1024", args.batch);
    }
    let started = Instant::now();
    let text = std::fs::read_to_string(&args.passages)
        .with_context(|| format!("passages {}", args.passages.display()))?;
    let mut lines = Vec::new();
    for (n, raw) in text
        .lines()
        .enumerate()
        .filter(|(_, l)| !l.trim().is_empty())
    {
        let line: Line = serde_json::from_str(raw)
            .with_context(|| format!("{} line {}", args.passages.display(), n + 1))?;
        if line.id.trim().is_empty() || line.text.trim().is_empty() {
            bail!(
                "{} line {}: a passage needs an id and text",
                args.passages.display(),
                n + 1
            );
        }
        lines.push(line);
    }
    if lines.is_empty() {
        bail!("{} holds no passages", args.passages.display());
    }

    // Vectors for the passages that came without one.
    let missing: Vec<usize> = (0..lines.len())
        .filter(|i| lines[*i].vector.is_none())
        .collect();
    if !missing.is_empty() {
        let Some(url) = &args.embed_url else {
            bail!(
                "{} passages have no vector: give --embed-url",
                missing.len()
            );
        };
        let embedder = TeiEmbed::new(url, &args.passage_prefix, None, Duration::from_secs(120))?;
        for chunk in missing.chunks(32) {
            let texts: Vec<&str> = chunk.iter().map(|i| lines[*i].text.as_str()).collect();
            let vectors = embedder.embed_many(&texts).await?;
            for (i, v) in chunk.iter().zip(vectors) {
                lines[*i].vector = Some(v);
            }
        }
    }
    let dim = lines[0].vector.as_ref().map(Vec::len).unwrap_or(0);
    if dim == 0 {
        bail!("the first passage's vector is empty");
    }
    if let Some((n, _)) = lines
        .iter()
        .enumerate()
        .find(|(_, l)| l.vector.as_ref().map(Vec::len) != Some(dim))
    {
        bail!(
            "passage {} has a vector of another size than {dim}",
            lines[n].id
        );
    }

    let store = Qdrant::new(
        &args.qdrant_url,
        &args.collection,
        args.qdrant_api_key.clone(),
        Duration::from_secs(120),
    )?;
    if args.recreate && store.exists().await? {
        store.delete().await?;
    }
    if !store.exists().await? {
        store.create(dim).await?;
    }
    let points: Vec<Point> = lines
        .into_iter()
        .map(|l| {
            Point::new(
                l.id,
                l.text,
                l.page,
                l.heading,
                l.vector.unwrap_or_default(),
            )
        })
        .collect();
    for batch in points.chunks(args.batch) {
        store.upsert(batch).await?;
    }
    let held = store.count().await?;
    println!(
        "{}",
        json!({
            "collection": args.collection,
            "passages": points.len(),
            "embedded_here": missing.len(),
            "dimensions": dim,
            "points_in_collection": held,
            "seconds": (started.elapsed().as_secs_f64() * 10.0).round() / 10.0,
        })
    );
    Ok(())
}
