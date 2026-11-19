//! The per-chunk silence rule of a turn loop: when the caller's silence
//! reaches the early-start onset (start early work), when it reaches the
//! floor (ask the end-of-turn model), and when early work must be dropped
//! because the caller went on.
//!
//! The rule for early work is deliberately simple. It stands only while
//! nothing but silence has followed it: then the words heard at that silence
//! are the turn's whole words. Any chunk the detector does not call silent
//! drops it, including a chunk between the detector's two thresholds and a
//! loud chunk the room gate discounts. The safe reading of such a chunk is
//! that the caller may have said more.
//!
//! The ceiling (how long a turn the end-of-turn model called unfinished may
//! wait for more) belongs to the host's turn machine, not to this clock.

use crate::timing::TurnTiming;

/// The voice detector's two thresholds. A chunk at or above `speech_on` is
/// speech; below `speech_off` it is silence; between them it is neither, and
/// the silence run neither grows nor restarts.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Thresholds {
    pub speech_on: f32,
    pub speech_off: f32,
}

impl Default for Thresholds {
    fn default() -> Self {
        Thresholds {
            speech_on: 0.5,
            speech_off: 0.35,
        }
    }
}

/// One 32 ms chunk as the detector judged it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Chunk {
    /// The detector's speech probability for the chunk.
    pub prob: f32,
    /// Whether the chunk stood above the room's noise margin. A chunk that
    /// did not counts as silence for the turn's timing (music in a cafe never
    /// opens a turn), but still drops early work when it sounded like speech.
    /// A loop without a room gate passes `true`.
    pub clears_room: bool,
}

impl Chunk {
    /// A chunk with no room gate in the way.
    pub const fn heard(prob: f32) -> Self {
        Chunk {
            prob,
            clears_room: true,
        }
    }
}

/// Where the host's turn machine stands while the caller may hold the floor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Phase {
    /// The caller is speaking, or has paused short of the floor.
    CallerSpeaking,
    /// The floor was reached; the end-of-turn model is deciding.
    Endpointing,
    /// The end-of-turn model called the turn unfinished; the loop waits for
    /// more, up to its ceiling.
    AwaitingMore,
}

/// What one chunk asks the host to do. Several can be set by one chunk only
/// in the combinations the rule allows: a chunk that drops early work is
/// never silent, so it never also starts it.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Signals {
    /// The caller began speaking.
    pub speech_began: bool,
    /// The caller's first short silence: transcribe the words so far and
    /// start the knowledge search on them. Nothing is said, and the turn's
    /// end is still decided at the floor, exactly as without it.
    pub start_early: bool,
    /// The caller went on, or may have: the commit may no longer take what
    /// started early. In [`Phase::CallerSpeaking`] nothing else waits on that
    /// work, so the host can abort it.
    pub drop_early: bool,
    /// The floor: ask the end-of-turn model whether the turn is complete.
    pub floor_reached: bool,
}

/// The silence rule, fed one chunk at a time.
#[derive(Debug, Clone)]
pub struct SilenceClock {
    timing: TurnTiming,
    thresholds: Thresholds,
    speaking: bool,
    silent_run: u32,
    early_stands: bool,
}

impl SilenceClock {
    pub fn new(timing: TurnTiming, thresholds: Thresholds) -> Self {
        SilenceClock {
            timing,
            thresholds,
            speaking: false,
            silent_run: 0,
            early_stands: false,
        }
    }

    /// Judge one chunk. `phase`: where the host's turn machine stands.
    /// Budget: a few comparisons; called 31.25 times a second per call.
    pub fn on_chunk(&mut self, chunk: Chunk, phase: Phase) -> Signals {
        let mut signals = Signals::default();
        let heard = chunk.prob >= self.thresholds.speech_on;
        // Who holds the floor is the room gate's call as well: speech the
        // room itself could have made counts as silence.
        let prob = if chunk.clears_room { chunk.prob } else { 0.0 };

        if self.early_stands && (heard || prob >= self.thresholds.speech_off) {
            self.early_stands = false;
            signals.drop_early = true;
        }

        if prob >= self.thresholds.speech_on {
            self.silent_run = 0;
            if !self.speaking {
                self.speaking = true;
                signals.speech_began = true;
            }
        } else if prob < self.thresholds.speech_off && self.speaking {
            self.silent_run += 1;
            if self.timing.early_after_chunks() == Some(self.silent_run)
                && phase == Phase::CallerSpeaking
                && !self.early_stands
            {
                self.early_stands = true;
                signals.start_early = true;
            }
            if self.silent_run >= self.timing.floor_chunks() {
                self.speaking = false;
                self.silent_run = 0;
                signals.floor_reached = true;
            }
        }
        signals
    }

    /// The commit's claim: does what started early stand for this turn (only
    /// silence followed it)? Taking it clears it.
    pub fn take_early(&mut self) -> bool {
        core::mem::take(&mut self.early_stands)
    }

    /// Everything was cancelled (the caller hung up, the reply was
    /// interrupted): nothing that started early stands.
    pub fn cancel(&mut self) {
        self.early_stands = false;
    }

    /// Whether early work stands right now.
    pub const fn early_stands(&self) -> bool {
        self.early_stands
    }

    pub const fn timing(&self) -> TurnTiming {
        self.timing
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::setting::{EarlyStart, Reuse};
    use crate::timing::DEFAULT_FLOOR;
    use core::time::Duration;

    const SPEECH: Chunk = Chunk::heard(0.9);
    const SILENCE: Chunk = Chunk::heard(0.0);

    fn clock(floor_ms: u64, early: EarlyStart) -> SilenceClock {
        let timing = TurnTiming::new(Duration::from_millis(floor_ms), early).expect("valid");
        SilenceClock::new(timing, Thresholds::default())
    }

    /// Feed `n` copies of `chunk` and gather every signal raised, by chunk.
    fn feed(c: &mut SilenceClock, chunk: Chunk, n: usize, phase: Phase) -> Vec<Signals> {
        (0..n).map(|_| c.on_chunk(chunk, phase)).collect()
    }

    fn first<F: Fn(&Signals) -> bool>(signals: &[Signals], f: F) -> Option<usize> {
        signals.iter().position(f).map(|i| i + 1)
    }

    /// THE early start: the fourth silent chunk starts the early work, the
    /// tenth reaches the floor, and the early work still stands for the
    /// commit because only silence followed it.
    #[test]
    fn the_fourth_silent_chunk_starts_early_work_and_the_tenth_reaches_the_floor() {
        let mut c = clock(320, EarlyStart::MEASURED);
        let spoken = feed(&mut c, SPEECH, 40, Phase::CallerSpeaking);
        assert!(spoken[0].speech_began);
        assert!(spoken[1..].iter().all(|s| *s == Signals::default()));
        let silent = feed(&mut c, SILENCE, 10, Phase::CallerSpeaking);
        assert_eq!(first(&silent, |s| s.start_early), Some(4));
        assert_eq!(first(&silent, |s| s.floor_reached), Some(10));
        assert_eq!(silent.iter().filter(|s| s.start_early).count(), 1);
        assert!(silent.iter().all(|s| !s.drop_early));
        assert!(c.take_early(), "the commit takes it");
        assert!(!c.take_early(), "taken once");
    }

    #[test]
    fn three_silent_chunks_are_not_yet_the_first_short_silence() {
        let mut c = clock(320, EarlyStart::MEASURED);
        feed(&mut c, SPEECH, 10, Phase::CallerSpeaking);
        let silent = feed(&mut c, SILENCE, 3, Phase::CallerSpeaking);
        assert!(silent.iter().all(|s| !s.start_early));
        assert!(!c.early_stands());
    }

    /// The caller goes on after their first short silence: what started there
    /// is dropped, and their NEXT first silence, on the whole question,
    /// starts it again.
    #[test]
    fn a_caller_who_goes_on_drops_it_and_their_next_first_silence_starts_again() {
        let mut c = clock(320, EarlyStart::MEASURED);
        feed(&mut c, SPEECH, 20, Phase::CallerSpeaking);
        feed(&mut c, SILENCE, 4, Phase::CallerSpeaking);
        assert!(c.early_stands());
        let went_on = c.on_chunk(SPEECH, Phase::CallerSpeaking);
        assert!(went_on.drop_early && !went_on.speech_began);
        assert!(!c.early_stands());
        feed(&mut c, SPEECH, 20, Phase::CallerSpeaking);
        let silent = feed(&mut c, SILENCE, 10, Phase::CallerSpeaking);
        assert_eq!(first(&silent, |s| s.start_early), Some(4));
        assert_eq!(first(&silent, |s| s.floor_reached), Some(10));
        assert!(c.take_early());
    }

    /// A sound the detector calls neither speech nor silence, after the early
    /// start: the safe reading is that the caller may have said more. Early
    /// work goes; the silence run neither restarts nor grows, so the floor
    /// comes one chunk later and nothing starts early again in this run.
    #[test]
    fn a_chunk_between_the_thresholds_drops_early_work_without_restarting_the_run() {
        let mut c = clock(320, EarlyStart::MEASURED);
        feed(&mut c, SPEECH, 20, Phase::CallerSpeaking);
        feed(&mut c, SILENCE, 4, Phase::CallerSpeaking);
        let between = c.on_chunk(Chunk::heard(0.4), Phase::CallerSpeaking);
        assert_eq!(
            between,
            Signals {
                drop_early: true,
                ..Signals::default()
            }
        );
        let rest = feed(&mut c, SILENCE, 6, Phase::CallerSpeaking);
        assert_eq!(first(&rest, |s| s.floor_reached), Some(6));
        assert!(rest.iter().all(|s| !s.start_early));
        assert!(
            !c.take_early(),
            "the turn decodes and searches its own words"
        );
    }

    /// A loud chunk the room gate discounts: silence for the turn's timing,
    /// but it sounded like speech, so early work goes.
    #[test]
    fn a_loud_room_counts_as_silence_but_still_drops_early_work() {
        let mut c = clock(320, EarlyStart::MEASURED);
        feed(&mut c, SPEECH, 20, Phase::CallerSpeaking);
        feed(&mut c, SILENCE, 4, Phase::CallerSpeaking);
        let room = c.on_chunk(
            Chunk {
                prob: 0.9,
                clears_room: false,
            },
            Phase::CallerSpeaking,
        );
        assert!(room.drop_early && !room.floor_reached);
        let rest = feed(&mut c, SILENCE, 5, Phase::CallerSpeaking);
        assert_eq!(
            first(&rest, |s| s.floor_reached),
            Some(5),
            "the room's chunk was the run's fifth"
        );
        assert!(!c.take_early());
    }

    #[test]
    fn off_and_numbers_never_start_at_the_first_silence() {
        for early in [EarlyStart::Off, EarlyStart::NumbersOnly] {
            let mut c = clock(320, early);
            feed(&mut c, SPEECH, 20, Phase::CallerSpeaking);
            let silent = feed(&mut c, SILENCE, 10, Phase::CallerSpeaking);
            assert!(silent.iter().all(|s| !s.start_early), "{early}");
            assert_eq!(first(&silent, |s| s.floor_reached), Some(10), "{early}");
            assert!(!c.take_early());
        }
    }

    /// Early work starts only while the caller holds the floor: once the
    /// end-of-turn model is deciding, or the loop awaits more, the host's own
    /// pause handling owns the words.
    #[test]
    fn nothing_starts_early_outside_the_callers_turn() {
        for phase in [Phase::Endpointing, Phase::AwaitingMore] {
            let mut c = clock(320, EarlyStart::MEASURED);
            feed(&mut c, SPEECH, 20, phase);
            let silent = feed(&mut c, SILENCE, 10, phase);
            assert!(silent.iter().all(|s| !s.start_early), "{phase:?}");
        }
    }

    /// A shorter floor asks for the turn's end sooner: at the sixth silent
    /// chunk instead of the tenth. The early start keeps its onset.
    #[test]
    fn a_shorter_floor_reaches_the_floor_sooner() {
        let mut c = clock(192, EarlyStart::MEASURED);
        feed(&mut c, SPEECH, 20, Phase::CallerSpeaking);
        let silent = feed(&mut c, SILENCE, 6, Phase::CallerSpeaking);
        assert_eq!(first(&silent, |s| s.start_early), Some(4));
        assert_eq!(first(&silent, |s| s.floor_reached), Some(6));
        let earlier = EarlyStart::FirstSilence {
            after: Duration::from_millis(64),
            reuse: Reuse::SameRequest,
        };
        let mut c = clock(DEFAULT_FLOOR.as_millis() as u64, earlier);
        feed(&mut c, SPEECH, 20, Phase::CallerSpeaking);
        let silent = feed(&mut c, SILENCE, 10, Phase::CallerSpeaking);
        assert_eq!(first(&silent, |s| s.start_early), Some(2));
    }

    #[test]
    fn silence_before_anyone_speaks_starts_nothing() {
        let mut c = clock(320, EarlyStart::MEASURED);
        let silent = feed(&mut c, SILENCE, 30, Phase::CallerSpeaking);
        assert!(silent.iter().all(|s| *s == Signals::default()));
    }

    #[test]
    fn a_cancel_leaves_nothing_standing() {
        let mut c = clock(320, EarlyStart::MEASURED);
        feed(&mut c, SPEECH, 20, Phase::CallerSpeaking);
        feed(&mut c, SILENCE, 4, Phase::CallerSpeaking);
        assert!(c.early_stands());
        c.cancel();
        assert!(!c.take_early());
    }
}
