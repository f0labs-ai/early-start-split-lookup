//! lastvoiced: where the speech ends in each recording, the anchor of the
//! caller's wait.
//!
//!   lastvoiced <file.wav>...
//!
//! One JSON line per file: its length, the end of its last voiced window
//! (`measure::voiced`), and the near-silent tail after it, in samples and
//! milliseconds. Recordings must be 16 kHz mono 16-bit, the caller's format.

use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use clap::Parser;
use measure::voiced::voiced_end;
use serde_json::json;

#[derive(Parser)]
#[command(about = "Where the speech ends in each recording.")]
struct Args {
    #[arg(required = true)]
    files: Vec<PathBuf>,
}

fn main() -> Result<()> {
    for path in Args::parse().files {
        let mut reader =
            hound::WavReader::open(&path).with_context(|| format!("{}", path.display()))?;
        let spec = reader.spec();
        if spec.sample_rate != 16_000 || spec.channels != 1 || spec.bits_per_sample != 16 {
            bail!(
                "{}: {} Hz, {} channels, {} bits; the caller's format is 16 kHz mono 16-bit",
                path.display(),
                spec.sample_rate,
                spec.channels,
                spec.bits_per_sample
            );
        }
        let samples: Vec<i16> = reader
            .samples::<i16>()
            .collect::<Result<_, _>>()
            .with_context(|| format!("{}", path.display()))?;
        let end = voiced_end(&samples);
        let ms = |n: usize| (n as f64 / 16.0).round() as u64;
        println!(
            "{}",
            json!({
                "file": path.display().to_string(),
                "samples": samples.len(),
                "ms": ms(samples.len()),
                "voiced_end_sample": end,
                "voiced_end_ms": end.map(ms),
                "tail_ms": end.map(|e| ms(samples.len() - e)),
            })
        );
    }
    Ok(())
}
