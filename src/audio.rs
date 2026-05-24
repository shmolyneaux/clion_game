use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use crate::script_bridge::SoundCmd;

pub const SAMPLE_RATE: u32 = 44100;
const TAU: f32 = std::f32::consts::TAU;

/// Mono PCM buffer of normalized `[-1.0, 1.0]` samples, owned by the mixer's
/// sample bank. `Arc`'d so a long-running voice can keep the data alive even
/// if the script later replaces the registration.
struct SampleBuffer {
    sample_rate: u32,
    samples: Vec<f32>,
}

enum VoiceSource {
    Sine {
        /// Cycles per output sample.
        phase_inc: f32,
        amp: f32,
    },
    Square {
        phase_inc: f32,
        duty: f32,
        amp: f32,
    },
    Sample {
        buffer: Arc<SampleBuffer>,
        amp: f32,
        /// Source samples per output sample. Combines user `speed` with the
        /// source's native sample rate vs. the output sample rate.
        source_step: f32,
        fade_in_samples: u64,
        fade_out_samples: u64,
    },
}

struct Voice {
    source: VoiceSource,
    /// Output frames still to wait before this voice becomes audible.
    delay_remaining: u64,
    /// Output frames produced so far (excluding delay).
    position: u64,
    /// Total output frames this voice will produce.
    total_samples: u64,
    /// Equal-power gains for L/R channels, precomputed from pan.
    gain_l: f32,
    gain_r: f32,
}

/// Equal-power pan law. `pan` in `[-1, 1]` (-1 = full left, 0 = center,
/// +1 = full right). Returns `(gain_l, gain_r)`; center yields ~0.707 on
/// each side so total power stays constant as the source moves across.
fn pan_gains(pan: f32) -> (f32, f32) {
    let p = pan.clamp(-1.0, 1.0);
    let angle = (p + 1.0) * std::f32::consts::FRAC_PI_4;
    (angle.cos(), angle.sin())
}

#[derive(Default)]
struct Mixer {
    voices: Vec<Voice>,
    bank: HashMap<u32, Arc<SampleBuffer>>,
}

fn mixer() -> &'static Mutex<Mixer> {
    static MIXER: OnceLock<Mutex<Mixer>> = OnceLock::new();
    MIXER.get_or_init(|| Mutex::new(Mixer::default()))
}

/// Take commands from the script and translate them into either bank entries
/// or active voices. Called from the main/script thread; brief lock.
pub fn submit<I: IntoIterator<Item = SoundCmd>>(cmds: I) {
    let mut m = mixer().lock().unwrap();
    for cmd in cmds {
        match cmd {
            SoundCmd::CreateSample { sample_id, sample_rate, samples } => {
                m.bank.insert(
                    sample_id,
                    Arc::new(SampleBuffer { sample_rate, samples }),
                );
            }
            SoundCmd::Sine { freq, amp, duration, delay, pan } => {
                let total = (duration.max(0.0) * SAMPLE_RATE as f32) as u64;
                if total == 0 {
                    continue;
                }
                let (gain_l, gain_r) = pan_gains(pan);
                m.voices.push(Voice {
                    source: VoiceSource::Sine {
                        phase_inc: freq / SAMPLE_RATE as f32,
                        amp,
                    },
                    delay_remaining: (delay.max(0.0) * SAMPLE_RATE as f32) as u64,
                    position: 0,
                    total_samples: total,
                    gain_l,
                    gain_r,
                });
            }
            SoundCmd::Square { freq, duty, amp, duration, delay, pan } => {
                let total = (duration.max(0.0) * SAMPLE_RATE as f32) as u64;
                if total == 0 {
                    continue;
                }
                let (gain_l, gain_r) = pan_gains(pan);
                m.voices.push(Voice {
                    source: VoiceSource::Square {
                        phase_inc: freq / SAMPLE_RATE as f32,
                        duty: duty.clamp(0.0, 1.0),
                        amp,
                    },
                    delay_remaining: (delay.max(0.0) * SAMPLE_RATE as f32) as u64,
                    position: 0,
                    total_samples: total,
                    gain_l,
                    gain_r,
                });
            }
            SoundCmd::PlaySample { sample_id, amp, speed, fade_in, fade_out, delay, pan } => {
                let Some(buffer) = m.bank.get(&sample_id).cloned() else {
                    continue;
                };
                let speed = speed.max(0.0001);
                let source_step = speed * buffer.sample_rate as f32 / SAMPLE_RATE as f32;
                let total =
                    (buffer.samples.len() as f32 / source_step).floor().max(0.0) as u64;
                if total == 0 {
                    continue;
                }
                let fade_in_samples = (fade_in.max(0.0) * SAMPLE_RATE as f32) as u64;
                let fade_out_samples = (fade_out.max(0.0) * SAMPLE_RATE as f32) as u64;
                let (gain_l, gain_r) = pan_gains(pan);
                m.voices.push(Voice {
                    source: VoiceSource::Sample {
                        buffer,
                        amp,
                        source_step,
                        fade_in_samples: fade_in_samples.min(total),
                        fade_out_samples: fade_out_samples.min(total),
                    },
                    delay_remaining: (delay.max(0.0) * SAMPLE_RATE as f32) as u64,
                    position: 0,
                    total_samples: total,
                    gain_l,
                    gain_r,
                });
            }
        }
    }
}

/// Mix all active voices into an interleaved L/R `i16` output buffer. Called
/// on the audio thread. The lock is held for the buffer's duration (typically
/// <1ms of work), which is acceptable here given submit's brief critical
/// section.
pub fn mix(stream: &mut [i16]) {
    let mut m = mixer().lock().unwrap();
    let voices = std::mem::take(&mut m.voices);
    let mut next_voices = Vec::with_capacity(voices.len());
    let frames = stream.len() / 2;
    let mut accum = vec![0.0f32; frames * 2];

    for mut voice in voices {
        let mut f = 0;
        while f < frames && voice.delay_remaining > 0 {
            voice.delay_remaining -= 1;
            f += 1;
        }
        while f < frames && voice.position < voice.total_samples {
            let s = sample_voice(&voice);
            accum[f * 2] += s * voice.gain_l;
            accum[f * 2 + 1] += s * voice.gain_r;
            voice.position += 1;
            f += 1;
        }
        if voice.position < voice.total_samples {
            next_voices.push(voice);
        }
    }

    m.voices = next_voices;
    drop(m);

    for (dst, &s) in stream.iter_mut().zip(accum.iter()) {
        let clipped = s / (1.0 + s.abs());
        *dst = (clipped * i16::MAX as f32) as i16;
    }
}

fn sample_voice(voice: &Voice) -> f32 {
    let raw = match &voice.source {
        VoiceSource::Sine { phase_inc, amp } => {
            (TAU * phase_inc * voice.position as f32).sin() * *amp
        }
        VoiceSource::Square { phase_inc, duty, amp } => {
            let phase = (phase_inc * voice.position as f32).fract();
            if phase < *duty { *amp } else { -*amp }
        }
        VoiceSource::Sample { buffer, amp, source_step, .. } => {
            let pos = voice.position as f32 * source_step;
            let i0 = pos.floor() as usize;
            let frac = pos - pos.floor();
            let s0 = buffer.samples.get(i0).copied().unwrap_or(0.0);
            let s1 = buffer.samples.get(i0 + 1).copied().unwrap_or(s0);
            (s0 + (s1 - s0) * frac) * *amp
        }
    };

    let env = match &voice.source {
        VoiceSource::Sample { fade_in_samples, fade_out_samples, .. } => {
            let pos = voice.position;
            let mut e = 1.0f32;
            if pos < *fade_in_samples {
                e = e.min(pos as f32 / *fade_in_samples as f32);
            }
            let tail = voice.total_samples - pos;
            if tail < *fade_out_samples {
                e = e.min(tail as f32 / *fade_out_samples as f32);
            }
            e
        }
        _ => {
            const PROC_FADE: u64 = SAMPLE_RATE as u64 / 200;
            let pos = voice.position;
            let mut e = 1.0f32;
            if pos < PROC_FADE {
                e = e.min(pos as f32 / PROC_FADE as f32);
            }
            let tail = voice.total_samples - pos;
            if tail < PROC_FADE {
                e = e.min(tail as f32 / PROC_FADE as f32);
            }
            e
        }
    };

    raw * env
}
