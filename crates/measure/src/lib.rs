//! The measurement tools and the analysis behind every table of the paper.
//!
//! * [`voiced`]: where the speech in a recording ends: the anchor of the
//!   caller's wait.
//! * [`stats`]: nearest-rank quantiles, shares, and the seeded bootstrap
//!   interval of a median.
//! * [`calls`]: the replayed calls, joined with the server's per-turn
//!   records: each arm's waits and hops, and each pair of arms compared
//!   question by question.
//! * [`audit`]: the context-stability audit over recorded lookups.
//! * [`questions`]: the question set.
//! * [`wer`]: word errors between what was said and what was heard.

pub mod audit;
pub mod calls;
pub mod questions;
pub mod stats;
pub mod voiced;
pub mod wer;
