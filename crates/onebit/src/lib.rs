//! A workspace's passages as ONE BIT per dimension of their dense vectors:
//! the shortlist half of a split lookup.
//!
//! The database keeps the exact vectors, the text and who may read what. The
//! box that runs the models keeps only the signs: 1,024 dimensions make 128
//! bytes a passage, so a million passages fit in 128 MB. With the question
//! embedded on that box, it shortlists the few hundred passages worth an
//! exact look itself, in microseconds, and asks the database for those alone,
//! by id, in one request. A code is not a passage.
//!
//! Two ways to shortlist, because they cost and keep differently:
//!
//! * [`Codes::nearest_by_bits`]: the question is cut to its signs too, and
//!   the codes are ranked by how many bits agree (Hamming distance). The
//!   cheapest: an XOR and a popcount per 64 dimensions.
//! * [`Codes::nearest_by_signs`]: the question keeps its numbers and each
//!   code scores the sum of `q_i * s_i` with `s_i = +1` where the passage's
//!   dimension is above the centre and `-1` where not. This is the asymmetric
//!   distance of product quantization applied to one-bit codes. A table of
//!   256 partial sums per byte of the code, built once per question, makes it
//!   one table read per byte per passage, and it keeps more of what an exact
//!   search finds at the same shortlist size.
//!
//! [`Centring::Mean`] subtracts the workspace's mean vector before the signs
//! are taken (from passages and question alike): the vectors of one
//! embedding model share a strong common direction, and signs taken around
//! zero spend bits on it.
//!
//! Pure: numbers in, indices out. No IO, no clock.

use serde::{Deserialize, Serialize};

/// Whether signs are taken around zero or around the workspace's mean vector.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Centring {
    Zero,
    Mean,
}

/// Why a set of codes can't be built, read or asked.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum CodesError {
    #[error("a code needs at least one dimension")]
    NoDimensions,
    #[error("vector {index} has {got} dimensions, the workspace's vectors have {want}")]
    Dimension {
        index: usize,
        got: usize,
        want: usize,
    },
    #[error("the question's vector has {got} dimensions, the codes have {want}")]
    QueryDimension { got: usize, want: usize },
    #[error("{ids} ids for {vectors} vectors")]
    Mismatch { ids: usize, vectors: usize },
    #[error("the stored codes are malformed: {0}")]
    Wire(&'static str),
}

/// One shortlisted passage: its place in the build order, and its score.
/// Higher is nearer, on either method's own scale (bits: dimensions that
/// agree; signs: the signed sum of the question's numbers).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Shortlisted {
    pub index: u32,
    pub score: f32,
}

/// A workspace's codes as they are stored or sent: the passage ids in order,
/// the centre (when the signs were taken around the mean), and the signs as
/// little-endian u64 words, `ceil(dim / 64)` per passage. Bit `d % 64` of
/// word `d / 64` is set when dimension `d` is above the centre.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WireCodes {
    pub dim: usize,
    #[serde(default)]
    pub centre: Option<Vec<f32>>,
    pub ids: Vec<String>,
    pub words: Vec<u64>,
}

/// A workspace's passages as sign codes, with their ids.
#[derive(Debug, Clone)]
pub struct Codes {
    dim: usize,
    /// u64 words per code.
    words: usize,
    /// `len * words`, passage after passage.
    bits: Vec<u64>,
    /// What was subtracted before the signs were taken.
    centre: Option<Vec<f32>>,
    ids: Vec<String>,
}

impl Codes {
    /// The codes of `vectors` (all `dim` long, one per id), signs taken as
    /// `centring` says. Budget: one pass (two with [`Centring::Mean`]); built
    /// when a workspace's codes are loaded, never on a lookup's path.
    pub fn build(
        dim: usize,
        ids: Vec<String>,
        vectors: &[Vec<f32>],
        centring: Centring,
    ) -> Result<Self, CodesError> {
        if dim == 0 {
            return Err(CodesError::NoDimensions);
        }
        if ids.len() != vectors.len() {
            return Err(CodesError::Mismatch {
                ids: ids.len(),
                vectors: vectors.len(),
            });
        }
        if let Some((index, v)) = vectors.iter().enumerate().find(|(_, v)| v.len() != dim) {
            return Err(CodesError::Dimension {
                index,
                got: v.len(),
                want: dim,
            });
        }
        let centre = match centring {
            Centring::Zero => None,
            Centring::Mean => {
                let mut mean = vec![0f64; dim];
                for v in vectors {
                    for (m, x) in mean.iter_mut().zip(v) {
                        *m += f64::from(*x);
                    }
                }
                let n = vectors.len().max(1) as f64;
                Some(mean.into_iter().map(|m| (m / n) as f32).collect())
            }
        };
        let words = dim.div_ceil(64);
        let mut bits = vec![0u64; vectors.len() * words];
        for (code, v) in bits.chunks_mut(words).zip(vectors) {
            write_signs(code, v, centre.as_deref());
        }
        Ok(Codes {
            dim,
            words,
            bits,
            centre,
            ids,
        })
    }

    /// Codes as they were stored or sent.
    pub fn from_wire(w: WireCodes) -> Result<Self, CodesError> {
        if w.dim == 0 {
            return Err(CodesError::NoDimensions);
        }
        let words = w.dim.div_ceil(64);
        if w.words.len() != w.ids.len() * words {
            return Err(CodesError::Wire("the words do not make one code per id"));
        }
        if w.centre.as_ref().is_some_and(|c| c.len() != w.dim) {
            return Err(CodesError::Wire("the centre is not one number a dimension"));
        }
        Ok(Codes {
            dim: w.dim,
            words,
            bits: w.words,
            centre: w.centre,
            ids: w.ids,
        })
    }

    pub fn to_wire(&self) -> WireCodes {
        WireCodes {
            dim: self.dim,
            centre: self.centre.clone(),
            ids: self.ids.clone(),
            words: self.bits.clone(),
        }
    }

    /// How many passages.
    pub fn len(&self) -> usize {
        self.ids.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    /// The memory the signs take, in bytes.
    pub fn bytes(&self) -> usize {
        self.bits.len() * 8
    }

    /// The id of the passage at `index` in build order.
    pub fn id(&self, index: u32) -> Option<&str> {
        self.ids.get(index as usize).map(String::as_str)
    }

    fn check(&self, query: &[f32]) -> Result<(), CodesError> {
        if query.len() == self.dim {
            Ok(())
        } else {
            Err(CodesError::QueryDimension {
                got: query.len(),
                want: self.dim,
            })
        }
    }

    /// The `k` passages whose signs agree most with the question's signs
    /// (smallest Hamming distance), nearest first; ties in build order.
    /// Budget: one XOR and popcount per 64 dimensions per passage, about
    /// 50 us for 3,000 passages of 1,024 dimensions on one core.
    pub fn nearest_by_bits(&self, query: &[f32], k: usize) -> Result<Vec<Shortlisted>, CodesError> {
        self.check(query)?;
        let mut q = vec![0u64; self.words];
        write_signs(&mut q, query, self.centre.as_deref());
        let scored = self.bits.chunks(self.words).map(|code| {
            let differ: u32 = code.iter().zip(&q).map(|(a, b)| (a ^ b).count_ones()).sum();
            (self.dim as u32 - differ) as f32
        });
        Ok(top(scored, k))
    }

    /// The `k` passages with the highest signed sum of the question's own
    /// numbers against each code's signs, nearest first; ties in build order.
    /// Budget: 256 sums per byte of a code once per question, then one table
    /// read per byte per passage, about 0.3 ms for 3,000 passages of 1,024
    /// dimensions on one core.
    pub fn nearest_by_signs(
        &self,
        query: &[f32],
        k: usize,
    ) -> Result<Vec<Shortlisted>, CodesError> {
        self.check(query)?;
        let centred: Vec<f32> = match &self.centre {
            Some(c) => query.iter().zip(c).map(|(q, c)| q - c).collect(),
            None => query.to_vec(),
        };
        // table[b * 256 + byte]: the sum over the byte's 8 dimensions of +q
        // (bit set) or -q (bit clear).
        let bytes = self.dim.div_ceil(8);
        let mut table = vec![0f32; bytes * 256];
        for (b, t) in table.chunks_mut(256).enumerate() {
            let q = |j: usize| centred.get(b * 8 + j).copied().unwrap_or(0.0);
            t[0] = -(0..8).map(q).sum::<f32>();
            for byte in 1..256usize {
                // One more bit than the entry without its lowest set bit.
                let low = byte.trailing_zeros() as usize;
                t[byte] = t[byte & (byte - 1)] + 2.0 * q(low);
            }
        }
        let scored = self.bits.chunks(self.words).map(|code| {
            let mut sum = 0f32;
            for (w, word) in code.iter().enumerate() {
                for (i, byte) in word.to_le_bytes().into_iter().enumerate() {
                    let b = w * 8 + i;
                    if b < bytes {
                        sum += table[b * 256 + usize::from(byte)];
                    }
                }
            }
            sum
        });
        Ok(top(scored, k))
    }

    /// The ids of the `k` nearest passages by [`Codes::nearest_by_signs`],
    /// nearest first: the shortlist a split lookup sends the database.
    pub fn shortlist(&self, query: &[f32], k: usize) -> Result<Vec<String>, CodesError> {
        Ok(self
            .nearest_by_signs(query, k)?
            .into_iter()
            .filter_map(|s| self.ids.get(s.index as usize).cloned())
            .collect())
    }
}

/// `v`'s signs (above `centre`, or above zero) into `code`.
fn write_signs(code: &mut [u64], v: &[f32], centre: Option<&[f32]>) {
    for (d, x) in v.iter().enumerate() {
        let above = match centre {
            Some(c) => *x > c[d],
            None => *x > 0.0,
        };
        if above {
            code[d / 64] |= 1u64 << (d % 64);
        }
    }
}

/// The `k` highest of `scores`, highest first; equal scores in index order.
fn top(scores: impl Iterator<Item = f32>, k: usize) -> Vec<Shortlisted> {
    let mut all: Vec<Shortlisted> = scores
        .enumerate()
        .map(|(i, score)| Shortlisted {
            index: i as u32,
            score,
        })
        .collect();
    let nearer = |a: &Shortlisted, b: &Shortlisted| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.index.cmp(&b.index))
    };
    let k = k.min(all.len());
    if k == 0 {
        return Vec::new();
    }
    if k < all.len() {
        all.select_nth_unstable_by(k - 1, nearer);
        all.truncate(k);
    }
    all.sort_unstable_by(nearer);
    all
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Vectors that are the same on every machine (no random crate).
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

    fn ids(n: usize) -> Vec<String> {
        (0..n).map(|i| format!("p{i}")).collect()
    }

    fn build(dim: usize, v: &[Vec<f32>], centring: Centring) -> Codes {
        Codes::build(dim, ids(v.len()), v, centring).expect("codes")
    }

    #[test]
    fn a_code_is_one_bit_a_dimension() {
        let codes = build(1024, &vectors(10, 1024, 1), Centring::Zero);
        assert_eq!((codes.len(), codes.dim()), (10, 1024));
        assert_eq!(
            codes.bytes(),
            10 * 128,
            "128 bytes a passage at 1,024 dimensions"
        );
        assert_eq!(
            build(70, &vectors(3, 70, 1), Centring::Zero).bytes(),
            3 * 16
        );
        assert!(build(8, &[], Centring::Mean).is_empty());
        assert_eq!(codes.id(3), Some("p3"));
        assert_eq!(codes.id(10), None);
    }

    #[test]
    fn wrong_shapes_are_refused() {
        assert_eq!(
            Codes::build(0, vec![], &[], Centring::Zero).unwrap_err(),
            CodesError::NoDimensions
        );
        assert_eq!(
            Codes::build(4, ids(1), &vectors(2, 4, 1), Centring::Zero).unwrap_err(),
            CodesError::Mismatch { ids: 1, vectors: 2 }
        );
        let mut v = vectors(3, 16, 2);
        v[1].pop();
        assert_eq!(
            Codes::build(16, ids(3), &v, Centring::Zero).unwrap_err(),
            CodesError::Dimension {
                index: 1,
                got: 15,
                want: 16
            }
        );
        let codes = build(16, &vectors(3, 16, 2), Centring::Zero);
        assert_eq!(
            codes.nearest_by_bits(&[0.0; 8], 2).unwrap_err(),
            CodesError::QueryDimension { got: 8, want: 16 }
        );
        assert!(codes.nearest_by_signs(&[0.0; 17], 2).is_err());
        assert!(codes.shortlist(&[0.0; 3], 2).is_err());
    }

    #[test]
    fn codes_survive_the_wire_and_a_malformed_wire_is_refused() {
        for centring in [Centring::Zero, Centring::Mean] {
            let codes = build(1024, &vectors(10, 1024, 1), centring);
            let text = serde_json::to_string(&codes.to_wire()).expect("json");
            let back = Codes::from_wire(serde_json::from_str(&text).expect("json")).expect("wire");
            let q = &vectors(1, 1024, 9)[0];
            assert_eq!(
                back.shortlist(q, 5).unwrap(),
                codes.shortlist(q, 5).unwrap()
            );
            assert_eq!(
                back.nearest_by_bits(q, 5).unwrap(),
                codes.nearest_by_bits(q, 5).unwrap()
            );
        }
        let bad = |w: WireCodes| Codes::from_wire(w).is_err();
        assert!(bad(WireCodes {
            dim: 64,
            centre: None,
            ids: ids(2),
            words: vec![0; 3]
        }));
        assert!(bad(WireCodes {
            dim: 64,
            centre: Some(vec![0.0; 3]),
            ids: ids(1),
            words: vec![0; 1]
        }));
        assert!(bad(WireCodes {
            dim: 0,
            centre: None,
            ids: vec![],
            words: vec![]
        }));
    }

    #[test]
    fn the_nearest_by_bits_are_the_codes_that_differ_least() {
        // Four passages of 4 dimensions, by their signs: ++++, +++-, ++--, ----.
        let v = vec![
            vec![1.0, 1.0, 1.0, 1.0],
            vec![1.0, 1.0, 1.0, -1.0],
            vec![1.0, 1.0, -1.0, -1.0],
            vec![-1.0, -1.0, -1.0, -1.0],
        ];
        let codes = build(4, &v, Centring::Zero);
        let near = codes
            .nearest_by_bits(&[0.3, 0.2, 0.9, 0.1], 3)
            .expect("shortlist");
        assert_eq!(near.iter().map(|s| s.index).collect::<Vec<_>>(), [0, 1, 2]);
        assert_eq!(
            near.iter().map(|s| s.score).collect::<Vec<_>>(),
            [4.0, 3.0, 2.0]
        );
        // More asked for than there are: every passage, nearest first.
        let all = codes.nearest_by_bits(&[-1.0; 4], 9).expect("all");
        assert_eq!((all.len(), all[0].index), (4, 3));
        assert!(
            codes
                .nearest_by_bits(&[1.0; 4], 0)
                .expect("none")
                .is_empty()
        );
    }

    #[test]
    fn equal_scores_come_in_build_order() {
        let v = vec![vec![1.0, -1.0], vec![1.0, -1.0], vec![1.0, -1.0]];
        let codes = build(2, &v, Centring::Zero);
        let order: Vec<u32> = codes
            .nearest_by_bits(&[1.0, -1.0], 2)
            .expect("two")
            .iter()
            .map(|s| s.index)
            .collect();
        assert_eq!(order, [0, 1]);
    }

    #[test]
    fn the_sign_scores_are_the_sum_of_the_questions_numbers_by_sign() {
        for (dim, centring) in [
            (1024, Centring::Zero),
            (1024, Centring::Mean),
            (70, Centring::Zero),
            (9, Centring::Mean),
        ] {
            let v = vectors(200, dim, 7);
            let codes = build(dim, &v, centring);
            let q = &vectors(1, dim, 99)[0];
            // The slow way: the sum of (q - centre)_d * (+1 or -1) for every passage.
            let centre: Vec<f32> = match centring {
                Centring::Zero => vec![0.0; dim],
                Centring::Mean => (0..dim)
                    .map(|d| (v.iter().map(|x| f64::from(x[d])).sum::<f64>() / 200.0) as f32)
                    .collect(),
            };
            let mut slow: Vec<(usize, f32)> = v
                .iter()
                .enumerate()
                .map(|(i, x)| {
                    let s = (0..dim)
                        .map(|d| {
                            if x[d] > centre[d] {
                                q[d] - centre[d]
                            } else {
                                centre[d] - q[d]
                            }
                        })
                        .sum();
                    (i, s)
                })
                .collect();
            slow.sort_by(|a, b| b.1.partial_cmp(&a.1).expect("finite").then(a.0.cmp(&b.0)));
            let fast = codes.nearest_by_signs(q, 20).expect("shortlist");
            for (f, s) in fast.iter().zip(&slow) {
                assert!(
                    (f.score - s.1).abs() < 1e-3,
                    "{dim} {centring:?}: {} vs {}",
                    f.score,
                    s.1
                );
            }
            // The same passages, allowing neighbours a rounding apart to swap.
            let top_fast: std::collections::BTreeSet<u32> =
                fast.iter().take(15).map(|s| s.index).collect();
            let top_slow: std::collections::BTreeSet<u32> =
                slow.iter().take(20).map(|s| s.0 as u32).collect();
            assert!(top_fast.is_subset(&top_slow), "{dim} {centring:?}");
            // The shortlist is the same order, by id.
            let ids = codes.shortlist(q, 5).expect("ids");
            let want: Vec<String> = fast
                .iter()
                .take(5)
                .map(|s| format!("p{}", s.index))
                .collect();
            assert_eq!(ids, want);
        }
    }

    #[test]
    fn signs_around_the_mean_differ_from_signs_around_zero_when_the_vectors_share_a_direction() {
        // Every vector sits on the same side of zero in every dimension:
        // around zero all codes are equal, around the mean they are not.
        let v: Vec<Vec<f32>> = vectors(50, 64, 3)
            .into_iter()
            .map(|x| x.into_iter().map(|a| a + 5.0).collect())
            .collect();
        let zero = build(64, &v, Centring::Zero);
        let mean = build(64, &v, Centring::Mean);
        let q = &v[17];
        let by_zero = zero.nearest_by_bits(q, 50).expect("all");
        assert!(
            by_zero.iter().all(|s| s.score == 64.0),
            "signs around zero cannot tell them apart"
        );
        let by_mean = mean.nearest_by_bits(q, 1).expect("one");
        assert_eq!(
            by_mean[0].index, 17,
            "around the mean a passage finds itself"
        );
    }
}
