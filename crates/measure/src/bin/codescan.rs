//! codescan: the first stage of a split lookup, measured on a real
//! workspace. Does a shortlist made from one-bit codes still hold what an
//! exact search finds, and what does making it cost?
//!
//!   codescan --passages passages.jsonl --questions questions.json
//!            (--query-vectors vectors.jsonl | --embed-url <tei> [--query-prefix "query: "])
//!            [--qdrant-url <url> --collection <name> [--oversampling 3] [--qdrant-filter <json>]]
//!
//! `passages.jsonl`: one passage a line, `{"id", "page", "vector"}`.
//! `vectors.jsonl`: one question a line, `{"qid", "vector"}`, kept from an
//! earlier run (or made here and kept, with `--keep-query-vectors`).
//!
//! For every question the exact search (cosine over every vector) is the
//! reference. Each way of shortlisting, at each shortlist size, is asked: are
//! the reference's first 4 (and 10) passages all in the shortlist? Is the
//! right page among the first 4 (and 10) once the shortlist is scored
//! exactly, as the database would score it? And how long does the shortlist
//! take on one core?
//!
//! With a vector database, its own dense leg (the graph over quantized
//! vectors, rescored) and the split's database half (exact scores for the
//! shortlist alone) are both held against the exact top 64.
//!
//! Prints JSON. Reads only; the network calls are the embedder and the
//! database's searches.

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, anyhow, bail};
use clap::Parser;
use ctxstab::Share;
use early_start::turn_query;
use measure::questions::{self, Question};
use onebit::{Centring, Codes, Shortlisted};
use serde::Deserialize;
use serde_json::{Value, json};
use split_lookup::tei::TeiEmbed;

/// The shortlist sizes asked.
const SIZES: [usize; 8] = [64, 128, 192, 256, 384, 512, 768, 1024];

/// How many times each question's shortlist is timed (the fastest counts:
/// the machine's other work only ever adds).
const TIMED_ROUNDS: usize = 5;

/// The passages a lookup's dense leg asks for: twice its 32 candidates.
const DENSE_LEG: usize = 64;

/// The shortlist a split lookup sends the database.
const SPLIT_SHORTLIST: usize = 512;

#[derive(Parser)]
#[command(about = "Shortlists from one-bit codes, held against the exact search.")]
#[command(group(clap::ArgGroup::new("queries").required(true).args(["query_vectors", "embed_url"])))]
struct Args {
    #[arg(long)]
    passages: PathBuf,
    #[arg(long)]
    questions: PathBuf,
    #[arg(long)]
    query_vectors: Option<PathBuf>,
    #[arg(long)]
    embed_url: Option<String>,
    #[arg(long, default_value = "query: ")]
    query_prefix: String,
    /// Keep the vectors made through --embed-url here.
    #[arg(long)]
    keep_query_vectors: Option<PathBuf>,
    #[arg(long)]
    qdrant_url: Option<String>,
    #[arg(long, default_value = "passages")]
    collection: String,
    #[arg(long, default_value_t = 3.0)]
    oversampling: f64,
    /// A filter every database search carries (JSON), e.g. one workspace of
    /// a shared collection.
    #[arg(long)]
    qdrant_filter: Option<String>,
}

#[derive(Deserialize)]
struct PassageLine {
    id: String,
    #[serde(default)]
    page: String,
    vector: Vec<f32>,
}

/// A workspace's passages: ids, unit vectors and pages, in file order.
struct Passages {
    ids: Vec<String>,
    vectors: Vec<Vec<f32>>,
    pages: Vec<String>,
}

fn unit(mut v: Vec<f32>) -> Vec<f32> {
    let norm = v
        .iter()
        .map(|x| f64::from(*x) * f64::from(*x))
        .sum::<f64>()
        .sqrt() as f32;
    if norm > 0.0 {
        for x in &mut v {
            *x /= norm;
        }
    }
    v
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn read_passages(path: &std::path::Path) -> Result<Passages> {
    let text =
        std::fs::read_to_string(path).with_context(|| format!("passages {}", path.display()))?;
    let (mut ids, mut vectors, mut pages) = (Vec::new(), Vec::new(), Vec::new());
    for (n, line) in text
        .lines()
        .enumerate()
        .filter(|(_, l)| !l.trim().is_empty())
    {
        let p: PassageLine = serde_json::from_str(line)
            .with_context(|| format!("{} line {}", path.display(), n + 1))?;
        ids.push(p.id);
        vectors.push(unit(p.vector));
        pages.push(p.page);
    }
    let Some(dim) = vectors.first().map(Vec::len) else {
        bail!("{} holds no passages", path.display());
    };
    if vectors.iter().any(|v| v.len() != dim) {
        bail!("{}: vectors of more than one size", path.display());
    }
    Ok(Passages {
        ids,
        vectors,
        pages,
    })
}

/// The questions' vectors: from the kept file, else from the embedder.
async fn question_vectors(args: &Args, questions: &[Question]) -> Result<Vec<Vec<f32>>> {
    if let Some(path) = &args.query_vectors {
        let text = std::fs::read_to_string(path).with_context(|| format!("{}", path.display()))?;
        let by_qid: HashMap<String, Vec<f32>> = text
            .lines()
            .filter_map(|l| serde_json::from_str::<Value>(l).ok())
            .filter_map(|v| {
                let qid = v["qid"].as_str()?.to_string();
                let vector = serde_json::from_value(v["vector"].clone()).ok()?;
                Some((qid, vector))
            })
            .collect();
        return questions
            .iter()
            .map(|q| {
                by_qid.get(&q.qid).cloned().map(unit).ok_or_else(|| {
                    anyhow!("{} has no vector for question {}", path.display(), q.qid)
                })
            })
            .collect();
    }
    let url = args
        .embed_url
        .as_deref()
        .ok_or_else(|| anyhow!("an embedder or kept vectors"))?;
    let embedder = TeiEmbed::new(url, &args.query_prefix, None, Duration::from_secs(60))?;
    let mut out = Vec::with_capacity(questions.len());
    for chunk in questions.chunks(32) {
        let texts: Vec<String> = chunk.iter().map(|q| turn_query(&q.question)).collect();
        let refs: Vec<&str> = texts.iter().map(String::as_str).collect();
        out.extend(embedder.embed_many(&refs).await?.into_iter().map(unit));
    }
    if let Some(path) = &args.keep_query_vectors {
        let lines: Vec<String> = questions
            .iter()
            .zip(&out)
            .map(|(q, v)| json!({"qid": q.qid, "vector": v}).to_string())
            .collect();
        std::fs::write(path, lines.join("\n") + "\n")
            .with_context(|| format!("keep {}", path.display()))?;
    }
    Ok(out)
}

/// The exact search: every passage by cosine, the nearest `k` first.
fn exact(passages: &Passages, q: &[f32], k: usize) -> Vec<u32> {
    let mut scored: Vec<(u32, f32)> = passages
        .vectors
        .iter()
        .enumerate()
        .map(|(i, v)| (i as u32, dot(v, q)))
        .collect();
    scored.sort_unstable_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.0.cmp(&b.0))
    });
    scored.truncate(k);
    scored.into_iter().map(|(i, _)| i).collect()
}

/// A shortlist scored exactly (what the database does with its ids): the
/// nearest `k` of it.
fn rescored(passages: &Passages, shortlist: &[Shortlisted], q: &[f32], k: usize) -> Vec<u32> {
    let mut scored: Vec<(u32, f32)> = shortlist
        .iter()
        .map(|s| (s.index, dot(&passages.vectors[s.index as usize], q)))
        .collect();
    scored.sort_unstable_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.0.cmp(&b.0))
    });
    scored.truncate(k);
    scored.into_iter().map(|(i, _)| i).collect()
}

#[derive(Clone, Copy)]
enum Method {
    Bits,
    Signs,
}

impl Method {
    fn shortlist(self, codes: &Codes, q: &[f32], k: usize) -> Result<Vec<Shortlisted>> {
        Ok(match self {
            Method::Bits => codes.nearest_by_bits(q, k)?,
            Method::Signs => codes.nearest_by_signs(q, k)?,
        })
    }
}

fn share(s: Share) -> Value {
    json!({ "k": s.k, "n": s.n, "pct": s.pct(), "range95": s.range95() })
}

fn micros(sorted: &[u128], q: f64) -> u128 {
    if sorted.is_empty() {
        return 0;
    }
    sorted[((q * sorted.len() as f64).ceil() as usize).clamp(1, sorted.len()) - 1]
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    let passages = read_passages(&args.passages)?;
    let set = questions::read(&args.questions)?;
    let vectors = question_vectors(&args, &set.questions).await?;
    let dim = passages.vectors[0].len();
    if vectors.iter().any(|v| v.len() != dim) {
        bail!("the questions' vectors are not {dim} long, as the passages' are");
    }

    // The reference: the exact search's first 10, and whether the right page is among them.
    let reference: Vec<Vec<u32>> = vectors.iter().map(|q| exact(&passages, q, 10)).collect();
    let has_page = |ids: &[u32], q: &Question| {
        ids.iter()
            .any(|i| passages.pages[*i as usize] == q.gold_page)
    };
    let mut exact_page = (Share::default(), Share::default());
    for (ids, q) in reference.iter().zip(&set.questions) {
        exact_page.0.add(has_page(&ids[..ids.len().min(4)], q));
        exact_page.1.add(has_page(ids, q));
    }

    let mut ways = Vec::new();
    for (name, method, centring) in [
        ("bits, signs around zero", Method::Bits, Centring::Zero),
        ("bits, signs around the mean", Method::Bits, Centring::Mean),
        (
            "the question's numbers against signs around zero",
            Method::Signs,
            Centring::Zero,
        ),
        (
            "the question's numbers against signs around the mean",
            Method::Signs,
            Centring::Mean,
        ),
    ] {
        let t = Instant::now();
        let codes = Codes::build(dim, passages.ids.clone(), &passages.vectors, centring)?;
        let build_ms = t.elapsed().as_secs_f64() * 1000.0;
        let mut sizes = Vec::new();
        for size in SIZES {
            let (mut top4, mut top10, mut page4, mut page10) = (
                Share::default(),
                Share::default(),
                Share::default(),
                Share::default(),
            );
            let mut recall10 = 0.0;
            for ((q, want), question) in vectors.iter().zip(&reference).zip(&set.questions) {
                let shortlist = method.shortlist(&codes, q, size)?;
                let held = |ids: &[u32]| {
                    ids.iter()
                        .filter(|i| shortlist.iter().any(|s| s.index == **i))
                        .count()
                };
                top4.add(held(&want[..want.len().min(4)]) == want.len().min(4));
                top10.add(held(want) == want.len());
                recall10 += held(want) as f64 / want.len().max(1) as f64;
                let again = rescored(&passages, &shortlist, q, 10);
                page4.add(has_page(&again[..again.len().min(4)], question));
                page10.add(has_page(&again, question));
            }
            sizes.push(json!({
                "shortlist": size,
                "holds_all_of_the_exact_first_4": share(top4),
                "holds_all_of_the_exact_first_10": share(top10),
                "mean_share_of_the_exact_first_10_held_pct": (recall10 / vectors.len().max(1) as f64 * 1000.0).round() / 10.0,
                "right_page_in_the_first_4_after_exact_scoring": share(page4),
                "right_page_in_the_first_10_after_exact_scoring": share(page10),
            }));
        }
        // What one shortlist of 512 costs on one core: each question's fastest of a few tries.
        let mut took: Vec<u128> = vectors
            .iter()
            .map(|q| {
                (0..TIMED_ROUNDS)
                    .map(|_| {
                        let t = Instant::now();
                        let n = method
                            .shortlist(&codes, q, 512)
                            .map(|s| s.len())
                            .unwrap_or(0);
                        std::hint::black_box(n);
                        t.elapsed().as_micros()
                    })
                    .min()
                    .unwrap_or(0)
            })
            .collect();
        took.sort_unstable();
        ways.push(json!({
            "way": name,
            "codes_bytes": codes.bytes(),
            "build_ms": (build_ms * 100.0).round() / 100.0,
            "one_shortlist_of_512_us": { "p50": micros(&took, 0.5), "p95": micros(&took, 0.95), "slowest": took.last() },
            "by_shortlist": sizes,
        }));
    }

    // The exact search over every vector, timed the same way: what the codes stand in for.
    let mut exact_took: Vec<u128> = vectors
        .iter()
        .map(|q| {
            (0..TIMED_ROUNDS)
                .map(|_| {
                    let t = Instant::now();
                    std::hint::black_box(exact(&passages, q, 10).len());
                    t.elapsed().as_micros()
                })
                .min()
                .unwrap_or(0)
        })
        .collect();
    exact_took.sort_unstable();

    let dense_leg = match &args.qdrant_url {
        Some(url) => Some(dense_leg(&args, url, &passages, &vectors).await?),
        None => None,
    };

    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "dense_leg": dense_leg,
            "passages": passages.vectors.len(),
            "dimensions": dim,
            "questions": vectors.len(),
            "exact_vectors_bytes": passages.vectors.len() * dim * 4,
            "exact_search": {
                "right_page_in_the_first_4": share(exact_page.0),
                "right_page_in_the_first_10": share(exact_page.1),
                "one_search_us": { "p50": micros(&exact_took, 0.5), "p95": micros(&exact_took, 0.95) },
            },
            "ways": ways,
        }))?
    );
    Ok(())
}

/// One dense query to the vector database: the point ids it returns, in
/// order, and how long it took.
async fn dense_query(
    http: &reqwest::Client,
    url: &str,
    body: &Value,
) -> Result<(Vec<String>, Duration)> {
    let t = Instant::now();
    let v: Value = http
        .post(url)
        .json(body)
        .send()
        .await
        .context("vector database")?
        .error_for_status()
        .context("vector database")?
        .json()
        .await
        .context("vector database reply")?;
    let took = t.elapsed();
    let points = v
        .pointer("/result/points")
        .and_then(Value::as_array)
        .ok_or_else(|| anyhow!("the vector database answered no points"))?;
    let id = |p: &Value| match &p["id"] {
        Value::String(s) => s.clone(),
        other => other.to_string(),
    };
    Ok((points.iter().map(id).collect(), took))
}

/// The database's own dense leg against the split's, both held against the
/// exact dense top 64 over the workspace's vectors. The passages' ids must
/// be the database's point ids.
async fn dense_leg(
    args: &Args,
    url: &str,
    passages: &Passages,
    questions: &[Vec<f32>],
) -> Result<Value> {
    let endpoint = format!(
        "{}/collections/{}/points/query",
        url.trim_end_matches('/'),
        args.collection
    );
    let http = reqwest::Client::builder()
        .timeout(Duration::from_secs(30))
        .build()?;
    let filter: Value = match &args.qdrant_filter {
        Some(f) => serde_json::from_str(f).context("--qdrant-filter is not JSON")?,
        None => json!({"must": []}),
    };
    let codes = Codes::build(
        passages.vectors[0].len(),
        passages.ids.clone(),
        &passages.vectors,
        Centring::Zero,
    )?;
    let known: HashSet<&str> = passages.ids.iter().map(String::as_str).collect();

    let (mut db_recall, mut split_recall) = (0.0f64, 0.0f64);
    let (mut db_all, mut split_all, mut same_set) =
        (Share::default(), Share::default(), Share::default());
    let (mut db_ms, mut split_ms) = (Vec::new(), Vec::new());
    let mut unknown = 0usize;
    for q in questions {
        let exact_ids: HashSet<&str> = exact(passages, q, DENSE_LEG)
            .iter()
            .map(|i| passages.ids[*i as usize].as_str())
            .collect();
        // The database's own dense leg, as a graph lookup asks it.
        let (db, took) = dense_query(
            &http,
            &endpoint,
            &json!({
                "query": q, "using": "dense", "filter": filter, "limit": DENSE_LEG,
                "params": { "quantization": { "rescore": true, "oversampling": args.oversampling } },
                "with_payload": false,
            }),
        )
        .await?;
        db_ms.push(took.as_micros());
        unknown += db.iter().filter(|id| !known.contains(id.as_str())).count();
        // The split: the shortlist from the codes, scored exactly by the
        // database, nothing else asked of it.
        let shortlist: Vec<Value> = codes
            .shortlist(q, SPLIT_SHORTLIST)?
            .into_iter()
            .map(|id| match id.parse::<u64>() {
                Ok(n) => json!(n),
                Err(_) => json!(id),
            })
            .collect();
        let mut restricted = filter.clone();
        match restricted.get_mut("must").and_then(Value::as_array_mut) {
            Some(must) => must.push(json!({ "has_id": shortlist })),
            None => restricted["must"] = json!([{ "has_id": shortlist }]),
        }
        let (split, took) = dense_query(
            &http,
            &endpoint,
            &json!({
                "query": q, "using": "dense", "filter": restricted, "limit": DENSE_LEG,
                "params": { "exact": true }, "with_payload": false,
            }),
        )
        .await?;
        split_ms.push(took.as_micros());
        let held = |ids: &[String]| {
            ids.iter()
                .filter(|id| exact_ids.contains(id.as_str()))
                .count()
        };
        let (d, sp) = (held(&db), held(&split));
        db_recall += d as f64 / DENSE_LEG as f64;
        split_recall += sp as f64 / DENSE_LEG as f64;
        db_all.add(d == DENSE_LEG);
        split_all.add(sp == DENSE_LEG);
        let a: std::collections::BTreeSet<&String> = db.iter().collect();
        let b: std::collections::BTreeSet<&String> = split.iter().collect();
        same_set.add(a == b);
    }
    db_ms.sort_unstable();
    split_ms.sort_unstable();
    let n = questions.len().max(1) as f64;
    let ms = |v: &[u128], q: f64| (micros(v, q) as f64 / 100.0).round() / 10.0;
    Ok(json!({
        "leg": DENSE_LEG,
        "oversampling": args.oversampling,
        "ids_not_in_the_passages": unknown,
        "database_dense_leg": {
            "mean_share_of_the_exact_top_64_pct": (db_recall / n * 1000.0).round() / 10.0,
            "all_of_the_exact_top_64": share(db_all),
            "one_query_ms": { "p50": ms(&db_ms, 0.5), "p95": ms(&db_ms, 0.95) },
        },
        "split_dense_leg": {
            "shortlist": SPLIT_SHORTLIST,
            "way": "the question's numbers against signs around zero",
            "mean_share_of_the_exact_top_64_pct": (split_recall / n * 1000.0).round() / 10.0,
            "all_of_the_exact_top_64": share(split_all),
            "database_rescoring_ms": { "p50": ms(&split_ms, 0.5), "p95": ms(&split_ms, 0.95) },
        },
        "the_two_legs_return_the_same_64": share(same_set),
    }))
}
