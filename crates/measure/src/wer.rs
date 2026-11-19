//! Word errors between what a caller said and what the server heard.

/// The words of `text` as compared: lower case, letters `a` to `z`, digits
/// and apostrophes; everything else separates words.
pub fn words(text: &str) -> Vec<String> {
    let kept: String = text
        .to_lowercase()
        .chars()
        .map(|c| {
            if c.is_ascii_lowercase() || c.is_ascii_digit() || c == '\'' || c == ' ' {
                c
            } else {
                ' '
            }
        })
        .collect();
    kept.split_whitespace().map(str::to_string).collect()
}

/// The fewest word insertions, deletions and substitutions that turn `said`
/// into `heard`.
pub fn word_errors(said: &[String], heard: &[String]) -> usize {
    let mut row: Vec<usize> = (0..=heard.len()).collect();
    for (i, s) in said.iter().enumerate() {
        let mut diagonal = row[0];
        row[0] = i + 1;
        for (j, h) in heard.iter().enumerate() {
            let above = row[j + 1];
            row[j + 1] = (above + 1)
                .min(row[j] + 1)
                .min(diagonal + usize::from(s != h));
            diagonal = above;
        }
    }
    row[heard.len()]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn words_are_compared_without_case_or_punctuation() {
        assert_eq!(
            words("How do I, um, reset it?"),
            ["how", "do", "i", "um", "reset", "it"]
        );
        assert_eq!(words("It's 4,512."), ["it's", "4", "512"]);
        assert_eq!(words("café"), ["caf"]);
    }

    #[test]
    fn word_errors_count_each_edit_once() {
        let w = |s: &str| words(s);
        assert_eq!(word_errors(&w("a b c"), &w("a b c")), 0);
        assert_eq!(word_errors(&w("a b c"), &w("a c")), 1);
        assert_eq!(word_errors(&w("a b c"), &w("a x c d")), 2);
        assert_eq!(word_errors(&w(""), &w("a b")), 2);
        assert_eq!(word_errors(&w("a b"), &w("")), 2);
    }
}
