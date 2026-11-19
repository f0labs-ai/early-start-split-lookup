//! BM25 keyword vectors, the lexical half of the hybrid search.
//!
//! Values are term weights; the database applies the inverse document
//! frequency itself (the collection's keyword vectors carry `modifier: idf`).
//! The passage side uses the standard BM25 saturation of term frequency
//! (k1 = 1.2, b = 0.75, a fixed average length of 256 tokens); the query side
//! is the raw term frequency.
//!
//! Tokens are lower-cased, split on anything that is not a letter or digit,
//! stripped of one-letter tokens and a compact English stop list, and stemmed
//! with the Snowball English stemmer. A token's id is the 32-bit MurmurHash3
//! of its stem (seed 0). Strong for Latin-script languages, weak for CJK; the
//! dense leg carries the rest.

use std::collections::HashMap;
use std::sync::LazyLock;

use rust_stemmers::{Algorithm, Stemmer};

use crate::contract::SparseVec;

const K1: f32 = 1.2;
const B: f32 = 0.75;
const AVG_LEN: f32 = 256.0;

/// The high-frequency core of English. Other languages pass through.
const STOPWORDS: [&str; 107] = [
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "if", "in", "into", "is", "it",
    "its", "no", "not", "of", "on", "or", "such", "that", "the", "their", "then", "there", "these",
    "they", "this", "to", "was", "will", "with", "we", "you", "your", "i", "me", "my", "our", "us",
    "he", "she", "his", "her", "them", "from", "have", "has", "had", "do", "does", "did", "can",
    "could", "should", "would", "may", "might", "been", "being", "am", "were", "what", "when",
    "where", "which", "who", "whom", "why", "how", "all", "any", "both", "each", "few", "more",
    "most", "other", "some", "than", "too", "very", "s", "t", "just", "don", "now", "up", "down",
    "out", "about", "over", "under", "again", "once", "here", "only", "own", "same", "so", "also",
    "after", "before", "between", "during",
];

static STEMMER: LazyLock<Stemmer> = LazyLock::new(|| Stemmer::create(Algorithm::English));

fn tokenize(text: &str) -> Vec<String> {
    text.to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|t| t.len() > 1 && !STOPWORDS.contains(t))
        .map(|t| STEMMER.stem(t).into_owned())
        .collect()
}

/// MurmurHash3, x86 32-bit variant.
pub fn murmur3_32(data: &[u8], seed: u32) -> u32 {
    const C1: u32 = 0xcc9e_2d51;
    const C2: u32 = 0x1b87_3593;
    let mut h = seed;
    let (blocks, tail) = data.as_chunks::<4>();
    for block in blocks {
        let mut k = u32::from_le_bytes(*block);
        k = k.wrapping_mul(C1).rotate_left(15).wrapping_mul(C2);
        h ^= k;
        h = h.rotate_left(13).wrapping_mul(5).wrapping_add(0xe654_6b64);
    }
    let mut k = 0u32;
    for (i, byte) in tail.iter().enumerate() {
        k |= u32::from(*byte) << (8 * i);
    }
    if !tail.is_empty() {
        k = k.wrapping_mul(C1).rotate_left(15).wrapping_mul(C2);
        h ^= k;
    }
    h ^= data.len() as u32;
    h ^= h >> 16;
    h = h.wrapping_mul(0x85eb_ca6b);
    h ^= h >> 13;
    h = h.wrapping_mul(0xc2b2_ae35);
    h ^ (h >> 16)
}

fn token_id(token: &str) -> u32 {
    murmur3_32(token.as_bytes(), 0)
}

fn term_frequencies(tokens: &[String]) -> HashMap<u32, f32> {
    let mut tf: HashMap<u32, f32> = HashMap::new();
    for t in tokens {
        *tf.entry(token_id(t)).or_insert(0.0) += 1.0;
    }
    tf
}

fn sorted(pairs: impl IntoIterator<Item = (u32, f32)>) -> SparseVec {
    let mut pairs: Vec<(u32, f32)> = pairs.into_iter().collect();
    pairs.sort_unstable_by_key(|(id, _)| *id);
    SparseVec {
        indices: pairs.iter().map(|(id, _)| *id).collect(),
        values: pairs.iter().map(|(_, v)| *v).collect(),
    }
}

/// A passage's keyword vector: BM25 term-frequency saturation with length
/// normalisation.
pub fn encode_passage(text: &str) -> SparseVec {
    let tokens = tokenize(text);
    let norm = K1 * (1.0 - B + B * tokens.len() as f32 / AVG_LEN);
    sorted(
        term_frequencies(&tokens)
            .into_iter()
            .map(|(id, f)| (id, f * (K1 + 1.0) / (f + norm))),
    )
}

/// A question's keyword vector: raw term frequency. The database's inverse
/// document frequency supplies the discrimination.
pub fn encode_query(text: &str) -> SparseVec {
    sorted(term_frequencies(&tokenize(text)))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Published MurmurHash3 x86_32 vectors.
    #[test]
    fn the_hash_is_murmur3_x86_32() {
        assert_eq!(murmur3_32(b"", 0), 0);
        assert_eq!(murmur3_32(b"", 1), 0x514e_28b7);
        assert_eq!(murmur3_32(b"", 0xffff_ffff), 0x81f1_6f39);
        assert_eq!(murmur3_32(&[0, 0, 0, 0], 0), 0x2362_f9de);
        assert_eq!(murmur3_32(b"aaaa", 0x9747_b28c), 0x5a97_808a);
        assert_eq!(murmur3_32(b"aaa", 0x9747_b28c), 0x283e_0130);
        assert_eq!(murmur3_32(b"aa", 0x9747_b28c), 0x5d21_1726);
        assert_eq!(murmur3_32(b"a", 0x9747_b28c), 0x7fa0_9ea6);
        assert_eq!(murmur3_32(b"Hello, world!", 0x9747_b28c), 0x2488_4cba);
        assert_eq!(
            murmur3_32(b"The quick brown fox jumps over the lazy dog", 0x9747_b28c),
            0x2fa8_26cd
        );
    }

    #[test]
    fn a_query_hits_the_ids_of_the_passage_that_holds_its_words() {
        let d = encode_passage("Reset the SAML toggle in settings");
        let q = encode_query("saml toggle resetting");
        assert!(!q.indices.is_empty());
        for qi in &q.indices {
            assert!(
                d.indices.contains(qi),
                "query token {qi} missing from the passage"
            );
        }
        assert!(d.indices.windows(2).all(|w| w[0] < w[1]), "ids ascending");
    }

    #[test]
    fn stop_words_and_one_letter_tokens_carry_nothing() {
        assert!(encode_query("the and of a I").indices.is_empty());
        assert!(encode_query("").indices.is_empty());
    }

    #[test]
    fn a_passages_weights_saturate() {
        let v = encode_passage("billing billing billing");
        assert_eq!(v.indices.len(), 1);
        assert!(v.values[0] > 1.0 && v.values[0] < K1 + 1.0);
        let q = encode_query("billing billing");
        assert_eq!(q.values, [2.0]);
    }
}
