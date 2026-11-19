//! ctxaudit: the context-stability audit. It asks a lookup service each
//! question exactly as a turn would (the query a turn builds from its words,
//! the same top k and score gate), records the passage LISTS that come back,
//! and measures them.
//!
//!   ctxaudit forms    <questions.json> <out.jsonl> --endpoint <url> [--limit N]
//!       every question in four written forms: as written, without its end
//!       mark, lower case without punctuation, and without its last word
//!   ctxaudit prefixes <questions.json> <out.jsonl> --endpoint <url> [--limit N]
//!       every beginning of each long question, from three words up
//!   ctxaudit report   <questions.json> <forms.jsonl> [--prefixes <prefixes.jsonl>]
//!       the measures, as JSON: how good the lookup is in each form, the
//!       flip rate of each change, the reuse audit, the settle curve
//!   ctxaudit again    <questions.json> <run_a.jsonl> <run_b.jsonl>
//!       the same lookups run twice: the repeat floor
//!
//! The endpoint speaks the reference lookup's interface (`POST /v1/search`
//! of `near`). Both lookup modes append one JSON line per lookup and skip
//! what the file already holds, so a run can be stopped and resumed. The
//! bearer token, when the service needs one, comes from CTXAUDIT_TOKEN.
//! Read-only: it searches and writes nothing to the service.

use std::collections::HashSet;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, anyhow, bail};
use clap::{Parser, Subcommand};
use ctxstab::{Form, prefixes};
use early_start::turn_query;
use futures_util::StreamExt;
use measure::audit::{self, Looked};
use measure::questions::{self, QuestionSet};
use serde_json::json;
use split_lookup::{Mode, SearchOutcome, SearchRequest};

#[derive(Parser)]
#[command(about = "The context-stability audit.")]
struct Args {
    #[command(subcommand)]
    run: Run,
}

#[derive(clap::Args, Clone)]
struct Lane {
    /// The lookup service, e.g. http://127.0.0.1:8084/v1/search
    #[arg(long)]
    endpoint: String,
    #[arg(long, default_value_t = 4)]
    top_k: usize,
    #[arg(long, default_value_t = 0.12)]
    threshold: f32,
    /// The mode to ask for; the service's default when absent.
    #[arg(long, value_enum)]
    mode: Option<Mode>,
    #[arg(long, default_value_t = 3)]
    concurrency: usize,
    /// One lookup's budget.
    #[arg(long, default_value_t = 60_000)]
    timeout_ms: u64,
}

#[derive(Subcommand)]
enum Run {
    Forms {
        questions: PathBuf,
        out: PathBuf,
        #[command(flatten)]
        lane: Lane,
        #[arg(long)]
        limit: Option<usize>,
    },
    Prefixes {
        questions: PathBuf,
        out: PathBuf,
        #[command(flatten)]
        lane: Lane,
        #[arg(long)]
        limit: Option<usize>,
    },
    Report {
        questions: PathBuf,
        forms: PathBuf,
        #[arg(long)]
        prefixes: Option<PathBuf>,
    },
    Again {
        questions: PathBuf,
        run_a: PathBuf,
        run_b: PathBuf,
    },
}

struct Client {
    http: reqwest::Client,
    lane: Lane,
    token: Option<String>,
}

impl Client {
    fn new(lane: Lane) -> Result<Self> {
        if lane.top_k == 0 || lane.concurrency == 0 {
            bail!("--top-k and --concurrency must be at least 1");
        }
        Ok(Client {
            http: reqwest::Client::builder()
                .timeout(Duration::from_millis(lane.timeout_ms))
                .build()
                .context("http client")?,
            token: std::env::var("CTXAUDIT_TOKEN")
                .ok()
                .filter(|t| !t.is_empty()),
            lane,
        })
    }

    /// One lookup, the way a turn asks. One retry.
    async fn look(&self, words: &str) -> Result<(SearchOutcome, u64)> {
        let body = SearchRequest {
            query: turn_query(words),
            top_k: self.lane.top_k,
            score_threshold: Some(self.lane.threshold),
            mode: self.lane.mode,
        };
        let started = Instant::now();
        let mut last = anyhow!("no attempt");
        for _ in 0..2 {
            let mut req = self.http.post(&self.lane.endpoint).json(&body);
            if let Some(t) = &self.token {
                req = req.bearer_auth(t);
            }
            match req.send().await {
                Ok(r) if r.status().is_success() => match r.json::<SearchOutcome>().await {
                    Ok(out) => return Ok((out, started.elapsed().as_millis() as u64)),
                    Err(e) => last = anyhow!("search body: {e}"),
                },
                Ok(r) => last = anyhow!("search answered {}", r.status()),
                Err(e) => last = anyhow!("search request: {e}"),
            }
        }
        Err(last)
    }
}

/// Run `jobs` (question id, form name, words), appending each answer to
/// `out`; what `out` already holds is skipped.
async fn run_lookups(jobs: Vec<(String, String, String)>, out: &Path, lane: Lane) -> Result<()> {
    let done: HashSet<(String, String)> = if out.exists() {
        audit::read_looked(out)?
            .into_iter()
            .map(|l| (l.qid, l.form))
            .collect()
    } else {
        HashSet::new()
    };
    let jobs: Vec<_> = jobs
        .into_iter()
        .filter(|(qid, form, words)| {
            !words.is_empty() && !done.contains(&(qid.clone(), form.clone()))
        })
        .collect();
    let client = Client::new(lane)?;
    eprintln!(
        "{} lookups to run ({} already in {}), {} at a time, at {}",
        jobs.len(),
        done.len(),
        out.display(),
        client.lane.concurrency,
        client.lane.endpoint
    );
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(out)
        .with_context(|| format!("output {}", out.display()))?;
    let started = Instant::now();
    let (mut ok, mut failed) = (0u32, 0u32);
    let client = &client;
    let mut answers =
        futures_util::stream::iter(jobs.into_iter().map(|(qid, form, words)| async move {
            let found = client.look(&words).await;
            (qid, form, words, found)
        }))
        .buffer_unordered(client.lane.concurrency);
    while let Some((qid, form, words, found)) = answers.next().await {
        match found {
            Ok((out, ms)) => {
                let line = Looked {
                    qid,
                    form,
                    n_words: words.split_whitespace().count(),
                    ids: out.hits.iter().map(|h| h.id.clone()).collect(),
                    pages: out.hits.iter().map(|h| h.page.clone()).collect(),
                    scores: out.hits.iter().map(|h| f64::from(h.score)).collect(),
                    ms,
                };
                writeln!(file, "{}", serde_json::to_string(&line)?)?;
                ok += 1;
            }
            Err(e) => {
                eprintln!("{qid} {form}: {e}");
                failed += 1;
            }
        }
        if (ok + failed) % 100 == 0 {
            eprintln!(
                "{} done in {}s ({failed} failed)",
                ok + failed,
                started.elapsed().as_secs()
            );
        }
    }
    println!(
        "{}",
        json!({"lookups_ok": ok, "failed": failed, "seconds": started.elapsed().as_secs()})
    );
    if failed > 0 {
        bail!("{failed} lookups failed; run again to retry them");
    }
    Ok(())
}

fn form_jobs(set: &QuestionSet, limit: Option<usize>) -> Vec<(String, String, String)> {
    set.questions
        .iter()
        .take(limit.unwrap_or(usize::MAX))
        .flat_map(|q| Form::ALL.map(|f| (q.qid.clone(), f.as_str().to_string(), f.of(&q.question))))
        .collect()
}

fn prefix_jobs(set: &QuestionSet, limit: Option<usize>) -> Vec<(String, String, String)> {
    set.questions
        .iter()
        .filter(|q| q.style == "long")
        .take(limit.unwrap_or(usize::MAX))
        .flat_map(|q| {
            prefixes(&q.question, 3)
                .into_iter()
                .map(|(k, n, words)| (q.qid.clone(), format!("prefix_{k}_of_{n}"), words))
        })
        .collect()
}

#[tokio::main]
async fn main() -> Result<()> {
    match Args::parse().run {
        Run::Forms {
            questions,
            out,
            lane,
            limit,
        } => {
            let set = questions::read(&questions)?;
            run_lookups(form_jobs(&set, limit), &out, lane).await
        }
        Run::Prefixes {
            questions,
            out,
            lane,
            limit,
        } => {
            let set = questions::read(&questions)?;
            run_lookups(prefix_jobs(&set, limit), &out, lane).await
        }
        Run::Report {
            questions,
            forms,
            prefixes,
        } => {
            let set = questions::read(&questions)?;
            let forms = audit::read_looked(&forms)?;
            let prefixes = prefixes.map(|p| audit::read_looked(&p)).transpose()?;
            println!(
                "{}",
                serde_json::to_string_pretty(&audit::report(&set, &forms, prefixes.as_deref())?)?
            );
            Ok(())
        }
        Run::Again {
            questions,
            run_a,
            run_b,
        } => {
            let set = questions::read(&questions)?;
            let (a, b) = (audit::read_looked(&run_a)?, audit::read_looked(&run_b)?);
            println!(
                "{}",
                serde_json::to_string_pretty(&audit::again(&set, &a, &b))?
            );
            Ok(())
        }
    }
}
