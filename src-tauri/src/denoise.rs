//! Optional spectral noise reduction — telling steady background noise apart from voice.
//!
//! The rest of the audio pipeline (`chunking.rs`) is energy-based: it measures how *loud* a frame
//! is. Loudness alone cannot separate a voice from road hum, a fan, an air-conditioner, or steady
//! outdoor noise — they can be just as loud as speech. That separation needs the *frequency*
//! content, which is what this module adds.
//!
//! Approach: short-time Fourier transform (STFT). We estimate the noise's spectrum from the
//! quietest frames of the clip — in a noisy room the pauses between words are noise-only, so they
//! reveal exactly what the background sounds like — then subtract that spectral profile from every
//! frame. Bins dominated by the noise floor (where there is no voice) are pulled down; bins where
//! the voice sits well above the noise are kept. A per-bin floor keeps the subtraction gentle so it
//! does not carve holes ("musical noise") that would hurt transcription.
//!
//! Safety by design: when a clip has NO steady noise, its quietest frames are near-silent, so the
//! estimated noise spectrum is ~zero and the transform is ~identity — clean speech passes through
//! essentially untouched. That is why it is safe to run, and why it is exposed as an opt-in setting
//! rather than forced on: on real hardware it should be tuned against real noise (a mic this code
//! has never had). A trained voice-activity model (Silero VAD) is the heavier follow-up for the
//! hardest, non-stationary noise; this is the pure-DSP layer that needs no model and no download.

use rustfft::{num_complex::Complex, FftPlanner};
use std::f32::consts::PI;

/// FFT frame size (32 ms at 16 kHz) — long enough to resolve speech formants, short enough to
/// track the fast changes in speech.
const N: usize = 512;
/// Hop between frames — 50% overlap, at which the square-root-Hann analysis+synthesis windows sum
/// to a constant, giving exact reconstruction when the spectral gain is 1 (the clean-audio case).
const HOP: usize = 256;
/// Fraction of the quietest frames used to estimate the noise spectrum. Kept small so it latches
/// onto genuine pauses (noise only) and does not pull quiet *speech* into the noise profile, which
/// would make it subtract the voice from itself.
const NOISE_FRAMES_FRACTION: f32 = 0.1;
/// Never attenuate a bin below this fraction of its own magnitude. Bounds suppression to about
/// -20 dB per bin, which removes steady noise while avoiding the warbly "musical noise" that
/// aggressive subtraction produces and that would degrade, not help, transcription.
const SPECTRAL_FLOOR: f32 = 0.1;

fn sqrt_hann(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| (0.5 * (1.0 - (2.0 * PI * i as f32 / n as f32).cos())).sqrt())
        .collect()
}

/// Reduce steady background noise in `samples` (16 kHz mono, f32 in [-1, 1]).
///
/// `strength` scales how much of the estimated noise spectrum is subtracted; ~1.0 is gentle,
/// ~2.0 is aggressive. Returns a buffer the same length as the input. Clips shorter than one FFT
/// frame, or with no estimable noise, are returned effectively unchanged.
pub fn reduce_noise(samples: &[f32], strength: f32) -> Vec<f32> {
    if samples.len() < N || strength <= 0.0 {
        return samples.to_vec();
    }

    let win = sqrt_hann(N);
    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(N);
    let ifft = planner.plan_fft_inverse(N);

    let orig_len = samples.len();
    let n_frames = (orig_len - 1) / HOP + 1;
    let padded_len = (n_frames - 1) * HOP + N;
    let mut buf = vec![0.0f32; padded_len];
    buf[..orig_len].copy_from_slice(samples);

    // --- Analysis: one complex spectrum per frame, plus its energy for noise picking. ---
    let mut spectra: Vec<Vec<Complex<f32>>> = Vec::with_capacity(n_frames);
    let mut energies: Vec<(usize, f32)> = Vec::with_capacity(n_frames);
    for f in 0..n_frames {
        let start = f * HOP;
        let mut frame: Vec<Complex<f32>> =
            (0..N).map(|i| Complex::new(buf[start + i] * win[i], 0.0)).collect();
        fft.process(&mut frame);
        let energy: f32 = frame.iter().take(N / 2 + 1).map(|c| c.norm_sqr()).sum();
        energies.push((f, energy));
        spectra.push(frame);
    }

    // --- Noise estimate: mean magnitude spectrum of the quietest frames (the pauses). ---
    energies.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    let n_noise = ((n_frames as f32 * NOISE_FRAMES_FRACTION).ceil() as usize).clamp(1, n_frames);
    let mut noise_mag = vec![0.0f32; N];
    for &(f, _) in energies.iter().take(n_noise) {
        for k in 0..N {
            noise_mag[k] += spectra[f][k].norm();
        }
    }
    for m in noise_mag.iter_mut() {
        *m /= n_noise as f32;
    }

    // --- Spectral subtraction + overlap-add synthesis. ---
    let mut out = vec![0.0f32; padded_len];
    for f in 0..n_frames {
        let mut frame = std::mem::take(&mut spectra[f]);
        for k in 0..N {
            let mag = frame[k].norm();
            if mag > 1e-9 {
                let cleaned = (mag - strength * noise_mag[k]).max(SPECTRAL_FLOOR * mag);
                let gain = cleaned / mag;
                frame[k] *= gain;
            }
        }
        ifft.process(&mut frame);
        let start = f * HOP;
        for i in 0..N {
            // rustfft's inverse is unnormalised (scales by N); the synthesis window completes the
            // sqrt-Hann pair so 50%-overlap frames add back to unity gain on clean audio.
            out[start + i] += frame[i].re / N as f32 * win[i];
        }
    }
    out.truncate(orig_len);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    const SR: usize = 16000;

    fn tone(secs: f32, amp: f32) -> Vec<f32> {
        let n = (secs * SR as f32) as usize;
        (0..n).map(|i| (i as f32 * 0.35).sin() * amp).collect()
    }

    fn silence(secs: f32) -> Vec<f32> {
        vec![0.0; (secs * SR as f32) as usize]
    }

    /// Deterministic pseudo-white noise so tests are reproducible.
    fn noise(n: usize, amp: f32) -> Vec<f32> {
        let mut state: u32 = 0x1234_5678;
        (0..n)
            .map(|_| {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                ((state >> 8) as f32 / (1u32 << 24) as f32 - 0.5) * 2.0 * amp
            })
            .collect()
    }

    fn rms(s: &[f32]) -> f32 {
        if s.is_empty() {
            return 0.0;
        }
        (s.iter().map(|x| x * x).sum::<f32>() / s.len() as f32).sqrt()
    }

    #[test]
    fn short_input_is_returned_unchanged() {
        let s = tone(0.01, 0.4); // shorter than one FFT frame
        assert_eq!(reduce_noise(&s, 1.5), s);
    }

    #[test]
    fn output_length_matches_input() {
        let s = tone(1.3, 0.4);
        assert_eq!(reduce_noise(&s, 1.5).len(), s.len());
    }

    #[test]
    fn near_identity_on_clean_speech() {
        // Speech-like: tone segments separated by a real pause, with NO added noise. The quiet
        // frames are true silence, so the noise estimate is ~zero and the voice must survive.
        let mut sig = tone(1.0, 0.4);
        sig.extend(silence(0.4));
        sig.extend(tone(1.0, 0.4));
        let out = reduce_noise(&sig, 1.5);

        // Compare interior RMS (avoid the STFT edge frames, which are under-normalised).
        let a = SR / 2;
        let b = SR; // first tone, interior
        let before = rms(&sig[a..b]);
        let after = rms(&out[a..b]);
        assert!(
            (after - before).abs() / before < 0.15,
            "clean voice was altered too much: {before} -> {after}"
        );
    }

    #[test]
    fn reduces_steady_noise_while_keeping_voice() {
        // Steady noise across the whole clip; a tone (the "voice") only in the middle second. The
        // start/end are noise-only pauses — exactly what the estimator should latch onto.
        let total = 3.0;
        let n = (total * SR as f32) as usize;
        let bg = noise(n, 0.15);
        let mut sig = bg.clone();
        let voice_start = SR; // 1.0s
        let voice_end = 2 * SR; // 2.0s
        for i in voice_start..voice_end {
            sig[i] += (i as f32 * 0.35).sin() * 0.4;
        }
        let out = reduce_noise(&sig, 2.0);

        // Noise-only region (0.3s..0.8s): energy should drop clearly.
        let noise_before = rms(&sig[(0.3 * SR as f32) as usize..(0.8 * SR as f32) as usize]);
        let noise_after = rms(&out[(0.3 * SR as f32) as usize..(0.8 * SR as f32) as usize]);
        assert!(
            noise_after < 0.6 * noise_before,
            "steady noise not reduced enough: {noise_before} -> {noise_after}"
        );

        // Voice region (1.2s..1.8s): most of the energy must remain (voice preserved).
        let voice_before = rms(&sig[(1.2 * SR as f32) as usize..(1.8 * SR as f32) as usize]);
        let voice_after = rms(&out[(1.2 * SR as f32) as usize..(1.8 * SR as f32) as usize]);
        assert!(
            voice_after > 0.5 * voice_before,
            "voice was over-suppressed: {voice_before} -> {voice_after}"
        );
    }

    #[test]
    fn zero_or_negative_strength_is_a_no_op() {
        let s = tone(1.0, 0.4);
        assert_eq!(reduce_noise(&s, 0.0), s);
    }
}
