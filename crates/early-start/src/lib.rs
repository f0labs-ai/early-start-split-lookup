//! Early start: begin a voice agent's slow steps (the transcript and the
//! knowledge search) at the caller's first short silence, before the turn's
//! end is decided, and keep them only when the turn asks exactly what they
//! asked. The early start guesses WHEN the caller may be done, never WHAT
//! they will say.
//!
//! The crate is sans-IO: no audio, no clock, no tasks. A turn loop feeds it
//! the voice detector's verdict on every 32 ms chunk ([`SilenceClock`]) and
//! acts on the signals it returns; at the commit it asks [`Reuse::allows`]
//! whether a search that started early may stand for the turn's own, and it
//! records what the turn took ([`EarlyUse`], [`TurnRecord`]).
//!
//! * [`EarlyStart`]: the one setting. Off, numbers (the prior behaviour), or
//!   first-silence with an onset and a reuse rule.
//! * [`TurnTiming`]: the silence floor and the early start, validated
//!   together, so an early start that would begin no earlier than the
//!   end-of-turn model is refused when the loop is built.
//! * [`SilenceClock`]: the per-chunk rule. When early work starts, when it
//!   is dropped, and when the end-of-turn model is asked.
//! * [`Reuse`]: when a turn may take a search that started early.
//! * [`turn_query`]: the query a turn sends for its words, so the identical
//!   request rule compares what the knowledge service is really asked.
//!
//! # Example
//!
//! ```
//! use early_start::{
//!     Asked, Chunk, DEFAULT_FLOOR, EarlyStart, Phase, SilenceClock, Thresholds, TurnTiming,
//!     turn_query,
//! };
//!
//! let timing = TurnTiming::new(DEFAULT_FLOOR, EarlyStart::MEASURED).expect("128 ms is under the 320 ms floor");
//! let mut clock = SilenceClock::new(timing, Thresholds::default());
//! for _ in 0..40 {
//!     clock.on_chunk(Chunk::heard(0.9), Phase::CallerSpeaking); // the caller speaks
//! }
//! let silent: Vec<_> = (0..10)
//!     .map(|_| clock.on_chunk(Chunk::heard(0.0), Phase::CallerSpeaking))
//!     .collect();
//! assert!(silent[3].start_early, "the fourth silent chunk: transcribe and search the words so far");
//! assert!(silent[9].floor_reached, "the tenth: ask the end-of-turn model, exactly as before");
//! assert!(clock.take_early(), "only silence followed: the early words are the turn's");
//!
//! let words = "Where is my order?";
//! let query = turn_query(words);
//! let early = Asked { words, query: &query };
//! assert!(EarlyStart::MEASURED.reuse().allows(early, early));
//! let without_the_mark = turn_query("Where is my order");
//! let turn = Asked { words: "Where is my order", query: &without_the_mark };
//! assert!(!EarlyStart::MEASURED.reuse().allows(early, turn), "not the identical request");
//! ```

mod clock;
mod outcome;
mod query;
mod setting;
mod timing;

pub use clock::{Chunk, Phase, Signals, SilenceClock, Thresholds};
pub use outcome::{EarlySearch, EarlyUse, EarlyWords, Knowledge, TurnRecord};
pub use query::{WIRE_QUERY_MAX_CHARS, clean_query, turn_query};
pub use setting::{
    Asked, CHUNK, DEFAULT_AFTER, EarlyStart, EarlyStartError, MIN_AFTER, Reuse, ends_on_a_number,
    same_words,
};
pub use timing::{DEFAULT_FLOOR, MAX_FLOOR_CHUNKS, MIN_FLOOR_CHUNKS, TimingError, TurnTiming};
