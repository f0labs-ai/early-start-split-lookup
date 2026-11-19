//! Where the speech in a recorded line ends.
//!
//! A caller's wait for a reply starts at the last sound they made: not at
//! the end of the recording, which trails a tail of near-silence that
//! differs from clip to clip, and not at the frame their client happened to
//! send it in. This names that sample, the same way for every clip: the
//! recording is cut into 20 ms windows, and the speech ends with the last
//! window whose level is within 26 dB of the loudest one (and above an
//! absolute floor, so a silent clip has no speech at all).
//!
//! Pure: samples in, an index out.

/// One window: 20 ms at 16 kHz.
pub const WINDOW: usize = 320;

/// A window counts as speech at this share of the loudest window's level
/// (0.05 is 26 dB down)...
const OF_PEAK: f64 = 0.05;

/// ...and never below this level (16-bit RMS; about -50 dB full scale).
const FLOOR_RMS: f64 = 100.0;

fn rms(window: &[i16]) -> f64 {
    if window.is_empty() {
        return 0.0;
    }
    let sum: f64 = window.iter().map(|&s| f64::from(s) * f64::from(s)).sum();
    (sum / window.len() as f64).sqrt()
}

/// One past the index of the last voiced sample of `samples` (16 kHz mono):
/// the end of the last window that still holds speech. None: no speech.
pub fn voiced_end(samples: &[i16]) -> Option<usize> {
    let levels: Vec<f64> = samples.chunks(WINDOW).map(rms).collect();
    let peak = levels.iter().copied().fold(0.0_f64, f64::max);
    let bar = (peak * OF_PEAK).max(FLOOR_RMS);
    let last = levels.iter().rposition(|&l| l >= bar)?;
    Some(((last + 1) * WINDOW).min(samples.len()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tone(samples: usize, level: i16) -> Vec<i16> {
        (0..samples)
            .map(|i| if i % 2 == 0 { level } else { -level })
            .collect()
    }

    #[test]
    fn a_silent_clip_has_no_speech() {
        assert_eq!(voiced_end(&[]), None);
        assert_eq!(voiced_end(&vec![0; 16_000]), None);
        assert_eq!(voiced_end(&tone(16_000, 20)), None, "under the floor");
    }

    #[test]
    fn speech_ends_with_its_last_loud_window_whatever_trails_it() {
        // 1 s of speech, then 0.5 s of a tail 40 dB under it.
        let mut clip = tone(16_000, 8_000);
        clip.extend(tone(8_000, 80));
        assert_eq!(voiced_end(&clip), Some(16_000));
        assert_eq!(voiced_end(&tone(16_000, 8_000)), Some(16_000));
        clip.extend(vec![0; 32_000]);
        assert_eq!(voiced_end(&clip), Some(16_000));
    }

    #[test]
    fn a_quiet_last_word_still_counts() {
        // A last word 20 dB under the loudest one is speech (26 dB is the bar).
        let mut clip = tone(8_000, 10_000);
        clip.extend(tone(3_200, 1_000));
        clip.extend(vec![0; 4_800]);
        assert_eq!(voiced_end(&clip), Some(11_200));
    }

    #[test]
    fn a_clip_that_ends_inside_a_window_ends_at_its_last_sample() {
        assert_eq!(voiced_end(&tone(1_000, 8_000)), Some(1_000));
    }
}
