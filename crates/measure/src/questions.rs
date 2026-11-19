//! The question set: held-out questions over one workspace's knowledge, each
//! with the page that answers it. `ctxaudit` asks them as text; a replay
//! caller says them, one recording per question (`<qid>.wav`).

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuestionSet {
    pub questions: Vec<Question>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Question {
    pub qid: String,
    /// How it was written: `short` (about 7 words) or `long` (about 15).
    #[serde(default)]
    pub style: String,
    pub question: String,
    /// The passage it was written from, and that passage's page: the right
    /// answer's page.
    #[serde(default)]
    pub gold_id: String,
    pub gold_page: String,
}

impl QuestionSet {
    pub fn by_qid(&self, qid: &str) -> Option<&Question> {
        self.questions.iter().find(|q| q.qid == qid)
    }
}

/// Read a question file.
pub fn read(path: &std::path::Path) -> Result<QuestionSet> {
    let text =
        std::fs::read_to_string(path).with_context(|| format!("questions {}", path.display()))?;
    serde_json::from_str(&text).with_context(|| format!("questions {}", path.display()))
}
