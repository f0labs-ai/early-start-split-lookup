//! The statistics of the paper: nearest-rank quantiles, shares, and the
//! seeded percentile bootstrap of a median. Every rounding is the one the
//! published numbers were computed with (to the nearest, halves up), so the
//! numbers reproduce exactly.

use serde::Serialize;
use serde_json::{Value, json};

/// To the nearest integer, halves toward positive infinity.
pub fn round_half_up(x: f64) -> f64 {
    let f = x.floor();
    if x - f >= 0.5 { f + 1.0 } else { f }
}

/// The nearest-rank quantile `f` of `sorted` (ascending): the value at rank
/// `ceil(f * n)`, counted from 1 and kept within the list.
pub fn quantile<T: Copy>(sorted: &[T], f: f64) -> Option<T> {
    if sorted.is_empty() {
        return None;
    }
    let n = sorted.len() as i64;
    let rank = ((f * n as f64).ceil() as i64 - 1).clamp(0, n - 1);
    Some(sorted[rank as usize])
}

/// A distribution of whole milliseconds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct Dist {
    pub n: usize,
    pub p50: i64,
    pub p90: i64,
    pub p95: i64,
    pub mean: i64,
    pub min: i64,
    pub max: i64,
}

/// The distribution of `values`; None of nothing.
pub fn dist(values: &[i64]) -> Option<Dist> {
    let mut s = values.to_vec();
    s.sort_unstable();
    let (min, max) = (*s.first()?, *s.last()?);
    let sum: f64 = s.iter().map(|v| *v as f64).sum();
    Some(Dist {
        n: s.len(),
        p50: quantile(&s, 0.5)?,
        p90: quantile(&s, 0.9)?,
        p95: quantile(&s, 0.95)?,
        mean: round_half_up(sum / s.len() as f64) as i64,
        min,
        max,
    })
}

/// A distribution as JSON: `{"n": 0}` of nothing.
pub fn dist_json(values: &[i64]) -> Value {
    match dist(values) {
        Some(d) => json!(d),
        None => json!({"n": 0}),
    }
}

/// `k` of `n`, as a percentage to one decimal.
pub fn share_json(k: usize, n: usize) -> Value {
    let pct = (n > 0).then(|| round_half_up(1000.0 * k as f64 / n as f64) / 10.0);
    json!({"k": k, "n": n, "pct": pct})
}

/// The Mulberry32 generator: the same stream of numbers in [0, 1) on every
/// machine for one seed.
#[derive(Debug, Clone)]
pub struct Mulberry32(u32);

impl Mulberry32 {
    pub const fn new(seed: u32) -> Self {
        Mulberry32(seed)
    }

    pub fn next_f64(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x6d2b_79f5);
        let s = self.0;
        let mut t = (s ^ (s >> 15)).wrapping_mul(1 | s);
        t = t.wrapping_add((t ^ (t >> 7)).wrapping_mul(61 | t)) ^ t;
        f64::from(t ^ (t >> 14)) / 4_294_967_296.0
    }
}

/// The seed every published interval was drawn with.
pub const BOOTSTRAP_SEED: u32 = 20_261_119;
/// Resamples per interval.
pub const BOOTSTRAP_ROUNDS: usize = 4_000;

/// The 95% percentile-bootstrap interval of the median of `values`:
/// `rounds` resamples with replacement, each one's median, and the 2.5th and
/// 97.5th nearest-rank percentiles of those medians. None of nothing.
pub fn boot_median(values: &[i64], rounds: usize, seed: u32) -> Option<[i64; 2]> {
    if values.is_empty() {
        return None;
    }
    let mut r = Mulberry32::new(seed);
    let n = values.len();
    let mut medians = Vec::with_capacity(rounds);
    let mut sample = Vec::with_capacity(n);
    for _ in 0..rounds {
        sample.clear();
        for _ in 0..n {
            sample.push(values[(r.next_f64() * n as f64).floor() as usize]);
        }
        sample.sort_unstable();
        medians.push(quantile(&sample, 0.5)?);
    }
    medians.sort_unstable();
    Some([quantile(&medians, 0.025)?, quantile(&medians, 0.975)?])
}

/// [`boot_median`] with the published seed and rounds, as JSON.
pub fn median_95(values: &[i64]) -> Value {
    json!(boot_median(values, BOOTSTRAP_ROUNDS, BOOTSTRAP_SEED))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn halves_round_up_both_sides_of_zero() {
        assert_eq!(round_half_up(2.5), 3.0);
        assert_eq!(round_half_up(-2.5), -2.0);
        assert_eq!(round_half_up(-2.6), -3.0);
        assert_eq!(round_half_up(0.49999999999999994), 0.0);
        assert_eq!(round_half_up(986.0), 986.0);
    }

    #[test]
    fn the_nearest_rank_quantile_takes_the_lower_middle() {
        let s = [1, 2, 3, 4];
        assert_eq!(quantile(&s, 0.5), Some(2));
        assert_eq!(quantile(&s, 0.9), Some(4));
        assert_eq!(quantile(&s, 0.0), Some(1));
        assert_eq!(quantile::<i64>(&[], 0.5), None);
        assert_eq!(quantile(&[7], 0.95), Some(7));
    }

    #[test]
    fn a_distribution_and_a_share() {
        let d = dist(&[30, 10, 20, -5]).expect("four values");
        assert_eq!((d.n, d.p50, d.min, d.max, d.mean), (4, 10, -5, 30, 14));
        assert_eq!(dist_json(&[]), json!({"n": 0}));
        assert_eq!(share_json(137, 139)["pct"], json!(98.6));
        assert_eq!(share_json(0, 0)["pct"], Value::Null);
    }

    /// The generator's first numbers for seed 1, as the reference
    /// implementation prints them.
    #[test]
    fn the_generator_is_mulberry32() {
        let mut r = Mulberry32::new(1);
        let first: Vec<f64> = (0..3).map(|_| r.next_f64()).collect();
        assert_eq!(first[0], 0.6270739405881613);
        assert_eq!(first[1], 0.002735721180215478);
        assert_eq!(first[2], 0.5274470399599522);
        let mut r = Mulberry32::new(BOOTSTRAP_SEED);
        assert_eq!(r.next_f64(), 0.5820677196606994);
        assert_eq!(r.next_f64(), 0.3669674713164568);
    }

    #[test]
    fn the_interval_holds_the_median_and_repeats_exactly() {
        let v: Vec<i64> = (0..139).map(|i| 100 + (i * 37) % 80).collect();
        let a = boot_median(&v, 4000, BOOTSTRAP_SEED).expect("interval");
        let b = boot_median(&v, 4000, BOOTSTRAP_SEED).expect("interval");
        assert_eq!(a, b);
        let med = dist(&v).expect("dist").p50;
        assert!(a[0] <= med && med <= a[1], "{a:?} {med}");
        assert_eq!(boot_median(&[], 10, 1), None);
    }
}
