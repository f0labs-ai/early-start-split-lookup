//! report: the replayed calls' numbers. Every arm's waits and server hops,
//! and every pair of arms compared question by question with the median of
//! the paired differences and its seeded bootstrap interval.
//!
//!   report <calls-dir> --questions <questions.json> [--pages <passage_pages.jsonl>]
//!          [--pairs base=early,base2=early2]
//!
//! Without `--pairs`, the pairs listed in `<calls-dir>/pairs.txt` (one
//! `a=b` a line). `--pages` adds the right page among each turn's passages.
//! Reads only.

use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;
use measure::calls;
use measure::questions;

#[derive(Parser)]
#[command(about = "The replayed calls' numbers.")]
struct Args {
    calls: PathBuf,
    #[arg(long)]
    questions: PathBuf,
    #[arg(long)]
    pages: Option<PathBuf>,
    #[arg(long)]
    pairs: Option<String>,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let set = questions::read(&args.questions)?;
    let pages = args.pages.as_deref().map(calls::read_pages).transpose()?;
    let pairs = match &args.pairs {
        Some(p) => calls::parse_pairs(p)?,
        None => {
            let listed = args.calls.join("pairs.txt");
            if listed.exists() {
                calls::parse_pairs(&std::fs::read_to_string(&listed)?)?
            } else {
                Vec::new()
            }
        }
    };
    let out = calls::report(&args.calls, &set, pages.as_ref(), &pairs)?;
    println!("{}", serde_json::to_string_pretty(&out)?);
    Ok(())
}
