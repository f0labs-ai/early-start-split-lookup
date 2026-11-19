//! What one shortlist costs on one core: both methods, two workspace sizes,
//! 1,024 dimensions, a shortlist of 512.
//!
//!   cargo bench -p onebit

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use onebit::{Centring, Codes};

/// Vectors that are the same on every machine.
fn vectors(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut s = seed;
    let mut next = move || {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((s >> 33) as f32 / (1u64 << 31) as f32) - 0.5
    };
    (0..n).map(|_| (0..dim).map(|_| next()).collect()).collect()
}

fn shortlist(c: &mut Criterion) {
    const DIM: usize = 1024;
    let query = vectors(1, DIM, 99).remove(0);
    let mut group = c.benchmark_group("shortlist_of_512");
    for passages in [3_000usize, 100_000] {
        let v = vectors(passages, DIM, 7);
        let ids = (0..passages).map(|i| i.to_string()).collect();
        let Ok(codes) = Codes::build(DIM, ids, &v, Centring::Zero) else {
            return;
        };
        group.bench_with_input(BenchmarkId::new("bits", passages), &codes, |b, codes| {
            b.iter(|| codes.nearest_by_bits(black_box(&query), 512))
        });
        group.bench_with_input(BenchmarkId::new("signs", passages), &codes, |b, codes| {
            b.iter(|| codes.nearest_by_signs(black_box(&query), 512))
        });
    }
    group.finish();
}

criterion_group!(benches, shortlist);
criterion_main!(benches);
