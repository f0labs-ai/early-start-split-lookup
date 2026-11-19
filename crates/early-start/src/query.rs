//! The query a turn sends for its words. The identical-request rule compares
//! these strings, so they must be built the same way for the early search
//! and for the turn.

/// The longest query a knowledge service is sent, in characters.
pub const WIRE_QUERY_MAX_CHARS: usize = 4000;

/// Fillers a spoken turn often opens with, peeled from the front of a query.
const LEADING_FILLERS: [&str; 9] = [
    "so", "um", "uh", "well", "okay", "ok", "like", "yeah", "hmm",
];

/// Turn a spoken transcript into a findable retrieval query: collapse
/// whitespace, drop immediately repeated words (recognizer hesitation), peel
/// a leading run of fillers. Never returns empty: a noise-only turn falls
/// back to its raw text.
pub fn clean_query(text: &str) -> String {
    let mut deduped: Vec<&str> = Vec::new();
    for w in text.split_whitespace() {
        if deduped.last().is_none_or(|p| !p.eq_ignore_ascii_case(w)) {
            deduped.push(w);
        }
    }
    while let Some(first) = deduped.first() {
        let head = first
            .trim_matches(|c| matches!(c, ',' | '.' | '?' | '!'))
            .to_lowercase();
        if LEADING_FILLERS.contains(&head.as_str()) {
            deduped.remove(0);
        } else {
            break;
        }
    }
    let out = deduped.join(" ");
    let out = out.trim();
    if !out.is_empty() {
        return out.to_string();
    }
    let fallback = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if fallback.is_empty() {
        text.to_string()
    } else {
        fallback
    }
}

/// The query a turn sends the knowledge service for `words`: cleaned, then
/// cut to the wire's length.
pub fn turn_query(words: &str) -> String {
    clean_query(words)
        .trim()
        .chars()
        .take(WIRE_QUERY_MAX_CHARS)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_query_drops_repeats_and_peels_fillers() {
        assert_eq!(
            clean_query("what what are the the products"),
            "what are the products"
        );
        assert_eq!(clean_query("so um how do I reset"), "how do I reset");
        assert_eq!(clean_query("   spaced   out  "), "spaced out");
        assert_eq!(clean_query("so, well, okay"), "so, well, okay");
        assert_eq!(clean_query(""), "");
    }

    #[test]
    fn a_turns_query_keeps_its_punctuation_and_is_cut_to_the_wire() {
        assert_eq!(turn_query("Where is my order?"), "Where is my order?");
        assert_ne!(
            turn_query("Where is my order?"),
            turn_query("Where is my order")
        );
        let long: Vec<String> = (0..2000).map(|i| format!("w{i}")).collect();
        assert_eq!(
            turn_query(&long.join(" ")).chars().count(),
            WIRE_QUERY_MAX_CHARS
        );
    }
}
