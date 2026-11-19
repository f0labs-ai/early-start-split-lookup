//! The silence floor (how long the caller must be silent before the
//! end-of-turn model is asked) and the early start, validated together: an
//! early start must begin before the floor, or it would begin with the
//! end-of-turn model and save nothing.

use core::time::Duration;

use crate::setting::{CHUNK, EarlyStart, MIN_AFTER};

/// The silence the end-of-turn model is asked after when nothing else is
/// set: ten chunks.
pub const DEFAULT_FLOOR: Duration = Duration::from_millis(320);

/// The shortest floor, in chunks: 128 ms. Under it a breath between two words
/// would ask the end-of-turn model.
pub const MIN_FLOOR_CHUNKS: u32 = 4;

/// The longest floor, in chunks: 640 ms.
pub const MAX_FLOOR_CHUNKS: u32 = 20;

/// Why a floor or an early start can't be what its setting asks.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum TimingError {
    #[error(
        "a silence floor of {ms} ms is not a whole number of {chunk_ms} ms chunks from {min_ms} to {max_ms} ms"
    )]
    Floor {
        ms: u64,
        min_ms: u64,
        max_ms: u64,
        chunk_ms: u64,
    },
    #[error(
        "an early start after {after_ms} ms of silence leaves no silence before the end of the turn is asked for at {floor_ms} ms; valid: {min_ms} to {max_ms} ms"
    )]
    EarlyTooLate {
        after_ms: u64,
        floor_ms: u64,
        min_ms: u64,
        max_ms: u64,
    },
}

/// A turn loop's silence timing: the floor in whole chunks and the early
/// start that runs ahead of it. Built once, when the loop is configured.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TurnTiming {
    floor_chunks: u32,
    early: EarlyStart,
}

impl Default for TurnTiming {
    fn default() -> Self {
        TurnTiming {
            floor_chunks: (DEFAULT_FLOOR.as_millis() / CHUNK.as_millis()) as u32,
            early: EarlyStart::Off,
        }
    }
}

impl TurnTiming {
    /// A floor of `floor` (whole chunks, 128 to 640 ms) and `early` ahead of
    /// it. Refused when the early start would not begin before the floor.
    pub fn new(floor: Duration, early: EarlyStart) -> Result<Self, TimingError> {
        let ms = floor.as_millis() as u64;
        let chunk = CHUNK.as_millis() as u64;
        let (min, max) = (
            u64::from(MIN_FLOOR_CHUNKS) * chunk,
            u64::from(MAX_FLOOR_CHUNKS) * chunk,
        );
        if !ms.is_multiple_of(chunk) || ms < min || ms > max {
            return Err(TimingError::Floor {
                ms,
                min_ms: min,
                max_ms: max,
                chunk_ms: chunk,
            });
        }
        if let Some(after) = early.first_silence() {
            let after_ms = after.as_millis() as u64;
            if after_ms >= ms {
                return Err(TimingError::EarlyTooLate {
                    after_ms,
                    floor_ms: ms,
                    min_ms: MIN_AFTER.as_millis() as u64,
                    max_ms: ms - chunk,
                });
            }
        }
        Ok(TurnTiming {
            floor_chunks: (ms / chunk) as u32,
            early,
        })
    }

    /// The silence the end-of-turn model is asked after.
    pub fn floor(&self) -> Duration {
        CHUNK * self.floor_chunks
    }

    /// The same, in chunks.
    pub const fn floor_chunks(&self) -> u32 {
        self.floor_chunks
    }

    pub const fn early(&self) -> EarlyStart {
        self.early
    }

    /// The silent chunks the early start waits for (None: it does not start
    /// at the caller's first silence). An onset that is not a whole number of
    /// chunks waits for the next whole chunk.
    pub fn early_after_chunks(&self) -> Option<u32> {
        self.early.first_silence().map(|after| {
            (after.as_millis() as u32)
                .div_ceil(CHUNK.as_millis() as u32)
                .max(1)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::setting::Reuse;

    fn at(ms: u64) -> EarlyStart {
        EarlyStart::FirstSilence {
            after: Duration::from_millis(ms),
            reuse: Reuse::SameRequest,
        }
    }

    #[test]
    fn the_default_floor_is_ten_chunks_and_starts_nothing_early() {
        let t = TurnTiming::default();
        assert_eq!(t.floor(), Duration::from_millis(320));
        assert_eq!(t.floor_chunks(), 10);
        assert_eq!(t.early_after_chunks(), None);
    }

    /// The floor is a setting: whole chunks, 128 to 640 ms.
    #[test]
    fn the_floor_is_set_in_whole_chunks_within_bounds() {
        let floor = |ms| TurnTiming::new(Duration::from_millis(ms), EarlyStart::Off);
        assert_eq!(floor(192).expect("six chunks").floor_chunks(), 6);
        assert_eq!(floor(128).expect("the shortest").floor_chunks(), 4);
        assert_eq!(floor(640).expect("the longest").floor_chunks(), 20);
        for bad in [0, 96, 200, 672, 1500] {
            let err = floor(bad).expect_err("refused");
            assert_eq!(
                err,
                TimingError::Floor {
                    ms: bad,
                    min_ms: 128,
                    max_ms: 640,
                    chunk_ms: 32
                }
            );
        }
    }

    /// An early start must end before the floor: at or after it, the look
    /// would start with the end-of-turn model, not ahead of it.
    #[test]
    fn an_early_start_at_the_floor_is_refused() {
        assert!(TurnTiming::new(DEFAULT_FLOOR, at(288)).is_ok());
        let err = TurnTiming::new(DEFAULT_FLOOR, at(320)).expect_err("the floor is 320 ms");
        assert_eq!(
            err,
            TimingError::EarlyTooLate {
                after_ms: 320,
                floor_ms: 320,
                min_ms: 64,
                max_ms: 288
            }
        );
        // 128 ms fits a 192 ms floor, not a 128 ms one.
        assert!(TurnTiming::new(Duration::from_millis(192), at(128)).is_ok());
        assert!(matches!(
            TurnTiming::new(Duration::from_millis(128), at(128)),
            Err(TimingError::EarlyTooLate { max_ms: 96, .. })
        ));
        assert!(TurnTiming::new(DEFAULT_FLOOR, EarlyStart::NumbersOnly).is_ok());
    }

    #[test]
    fn the_onset_is_counted_in_whole_chunks() {
        let t = TurnTiming::new(DEFAULT_FLOOR, EarlyStart::MEASURED).expect("valid");
        assert_eq!(t.early_after_chunks(), Some(4));
        // An onset set in code, off the chunk grid, waits for the next chunk.
        let t = TurnTiming::new(DEFAULT_FLOOR, at(100)).expect("valid");
        assert_eq!(t.early_after_chunks(), Some(4));
        let t = TurnTiming::new(DEFAULT_FLOOR, at(64)).expect("valid");
        assert_eq!(t.early_after_chunks(), Some(2));
    }
}
