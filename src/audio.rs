use std::cell::UnsafeCell;
use std::collections::HashMap;
use std::mem::MaybeUninit;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::script_bridge::SoundCmd;

pub const SAMPLE_RATE: u32 = 44100;
const TAU: f32 = std::f32::consts::TAU;
const NUM_BUSES: usize = crate::script_bridge::NUM_BUSES as usize;

/// Mono PCM buffer of normalized `[-1.0, 1.0]` samples, owned by the mixer's
/// sample bank. Entries are never replaced or removed, so voices reference
/// their buffer by `sample_id` and look it up at mix time.
struct SampleBuffer {
    sample_rate: u32,
    samples: Vec<f32>,
}

#[derive(Clone, Copy)]
enum VoiceSource {
    Sine {
        /// Cycles per output sample.
        phase_inc: f32,
    },
    Square {
        phase_inc: f32,
        duty: f32,
    },
    Sample {
        sample_id: u32,
        /// Source samples per output sample. Combines user `speed` with the
        /// source's native sample rate vs. the output sample rate.
        source_step: f32,
    },
}

/// A value that linearly slews toward a target over a number of samples, so
/// gain/pan changes don't click. `step`/`remaining` are precomputed when a new
/// target is set.
#[derive(Clone, Copy)]
struct Ramp {
    current: f32,
    target: f32,
    step: f32,
    remaining: u64,
}

impl Ramp {
    fn new(value: f32) -> Self {
        Self { current: value, target: value, step: 0.0, remaining: 0 }
    }

    fn is_settled(&self) -> bool {
        self.remaining == 0
    }

    fn set_target(&mut self, target: f32, ramp_samples: u64) {
        self.target = target;
        if ramp_samples == 0 {
            self.current = target;
            self.step = 0.0;
            self.remaining = 0;
        } else {
            self.step = (target - self.current) / ramp_samples as f32;
            self.remaining = ramp_samples;
        }
    }

    #[inline]
    fn next(&mut self) -> f32 {
        let v = self.current;
        if self.remaining > 0 {
            self.remaining -= 1;
            if self.remaining == 0 {
                self.current = self.target;
            } else {
                self.current += self.step;
            }
        }
        v
    }

    /// Advance `n` samples in O(1), returning the resulting value. Used to skip
    /// a whole buffer's worth of a bus ramp without a per-sample loop.
    fn advance(&mut self, n: u64) -> f32 {
        if self.remaining > 0 {
            if n >= self.remaining {
                self.current = self.target;
                self.remaining = 0;
            } else {
                self.current += self.step * n as f32;
                self.remaining -= n;
            }
        }
        self.current
    }
}

enum Envelope {
    FadeInOut { fade_in: u64, fade_out: u64 },
    Adsr { attack: u64, decay: u64, sustain: f32, release: u64 },
}

struct Voice {
    id: u32,
    bus: u32,
    source: VoiceSource,
    /// Output frames still to wait before this voice becomes audible.
    delay_remaining: u64,
    /// Output frames produced so far (excluding delay).
    position: u64,
    /// Total output frames this voice will produce.
    total_samples: u64,
    envelope: Envelope,
    /// Overall amplitude (ramps on `set_gain`).
    amp: Ramp,
    /// Equal-power L/R gains (ramp together on `set_pan`).
    pan_l: Ramp,
    pan_r: Ramp,
    /// Pause gain multiplier: 1.0 = audible, 0.0 = paused. Ramps on
    /// pause/resume so playback doesn't click. While `paused` and the ramp has
    /// settled at 0, the voice is "frozen" — it produces no output and its
    /// playback position stops advancing until resumed.
    pause: Ramp,
    paused: bool,
}

impl Voice {
    /// True once a pause has fully faded out: the voice is silent and its
    /// position should stop advancing until resumed.
    fn is_frozen(&self) -> bool {
        self.paused && self.pause.is_settled()
    }
}

/// Equal-power pan law. `pan` in `[-1, 1]` (-1 = full left, 0 = center,
/// +1 = full right). Returns `(gain_l, gain_r)`; center yields ~0.707 on
/// each side so total power stays constant as the source moves across.
fn pan_gains(pan: f32) -> (f32, f32) {
    let p = pan.clamp(-1.0, 1.0);
    let angle = (p + 1.0) * std::f32::consts::FRAC_PI_4;
    (angle.cos(), angle.sin())
}

/// Transposed Direct Form II biquad with independent state per stereo channel.
#[derive(Clone, Copy)]
struct Biquad {
    b0: f32,
    b1: f32,
    b2: f32,
    a1: f32,
    a2: f32,
    z1: [f32; 2],
    z2: [f32; 2],
}

impl Biquad {
    fn low_pass(cutoff: f32, q: f32) -> Self {
        let mut bq = Biquad { b0: 1.0, b1: 0.0, b2: 0.0, a1: 0.0, a2: 0.0, z1: [0.0; 2], z2: [0.0; 2] };
        bq.set_low_pass(cutoff, q);
        bq
    }

    /// Recompute low-pass coefficients (RBJ cookbook), preserving filter state
    /// so a live cutoff change doesn't reset the delay line.
    fn set_low_pass(&mut self, cutoff: f32, q: f32) {
        let cutoff = cutoff.clamp(10.0, SAMPLE_RATE as f32 * 0.49);
        let q = q.max(0.01);
        let w0 = TAU * cutoff / SAMPLE_RATE as f32;
        let (sin, cos) = w0.sin_cos();
        let alpha = sin / (2.0 * q);
        let a0 = 1.0 + alpha;
        self.b0 = (1.0 - cos) / 2.0 / a0;
        self.b1 = (1.0 - cos) / a0;
        self.b2 = (1.0 - cos) / 2.0 / a0;
        self.a1 = (-2.0 * cos) / a0;
        self.a2 = (1.0 - alpha) / a0;
    }

    #[inline]
    fn process_sample(&mut self, x: f32, ch: usize) -> f32 {
        let y = self.b0 * x + self.z1[ch];
        self.z1[ch] = self.b1 * x - self.a1 * y + self.z2[ch];
        self.z2[ch] = self.b2 * x - self.a2 * y;
        y
    }
}

/// A per-bus DSP effect. Processes an interleaved stereo buffer in place.
enum Effect {
    LowPass { biquad: Biquad, cutoff: Ramp, q: f32, fading_out: bool },
}

impl Effect {
    fn process(&mut self, buffer: &mut [f32]) {
        match self {
            Effect::LowPass { biquad, cutoff, q, .. } => {
                let frames = buffer.len() / 2;
                let cur = cutoff.advance(frames as u64).exp();
                biquad.set_low_pass(cur, *q);
                for f in 0..frames {
                    buffer[f * 2] = biquad.process_sample(buffer[f * 2], 0);
                    buffer[f * 2 + 1] = biquad.process_sample(buffer[f * 2 + 1], 1);
                }
            }
        }
    }

    /// Ramp toward a passthrough state. The effect removes itself once settled.
    fn fade_out(&mut self, ramp_samples: u64) {
        match self {
            Effect::LowPass { cutoff, fading_out, .. } => {
                cutoff.set_target((SAMPLE_RATE as f32 * 0.49).ln(), ramp_samples);
                *fading_out = true;
            }
        }
    }

    fn is_faded_out(&self) -> bool {
        match self {
            Effect::LowPass { fading_out, cutoff, .. } => *fading_out && cutoff.is_settled(),
        }
    }
}

/// A mix bus: voices routed here sum into `accum`, the `effects` chain processes
/// it, then `gain` is applied as it folds into the master mix.
struct Bus {
    /// Ramped output gain (default 1.0).
    gain: Ramp,
    /// Pause gain multiplier applied on top of `gain`: 1.0 = audible, 0.0 =
    /// paused. Ramps on pause/resume so the bus doesn't click. While `paused`
    /// and settled at 0, voices routed here are frozen.
    pause: Ramp,
    paused: bool,
    /// Reused interleaved stereo accumulator for this buffer.
    accum: Vec<f32>,
    effects: Vec<Effect>,
    /// Set each buffer if this bus has any input/effects to process.
    active: bool,
}

impl Default for Bus {
    fn default() -> Self {
        Self {
            gain: Ramp::new(1.0),
            pause: Ramp::new(1.0),
            paused: false,
            accum: Vec::new(),
            effects: Vec::new(),
            active: false,
        }
    }
}

impl Bus {
    /// True once a bus pause has fully faded out, meaning voices routed here
    /// should freeze until the bus is resumed.
    fn is_frozen(&self) -> bool {
        self.paused && self.pause.is_settled()
    }
}

impl Bus {
    fn prepare(&mut self, frames: usize) {
        self.accum.clear();
        self.accum.resize(frames * 2, 0.0);
        self.active = true;
    }
}

struct Mixer {
    voices: Vec<Voice>,
    bank: HashMap<u32, SampleBuffer>,
    /// Reused interleaved master accumulator so `mix` never allocates per call.
    master: Vec<f32>,
    buses: Vec<Bus>,
    /// Global pause gain multiplier applied to the master: 1.0 = audible,
    /// 0.0 = paused. Ramps on pause/resume so the whole mix doesn't click.
    /// While `master_paused` and settled at 0, every voice is frozen.
    master_pause: Ramp,
    master_paused: bool,
}

impl Default for Mixer {
    fn default() -> Self {
        Self {
            voices: Vec::new(),
            bank: HashMap::new(),
            master: Vec::new(),
            buses: (0..NUM_BUSES).map(|_| Bus::default()).collect(),
            master_pause: Ramp::new(1.0),
            master_paused: false,
        }
    }
}

/// Lock-free single-producer/single-consumer ring. The game thread is the sole
/// producer (`submit`); the audio thread is the sole consumer (`mix`). Slots
/// hold owned `SoundCmd`s — moving one in/out is a memcpy, so no allocation or
/// drop ever happens on the audio thread (a `CreateSample`'s `Vec` is moved
/// straight into the bank).
struct SpscRing<T> {
    slots: Box<[UnsafeCell<MaybeUninit<T>>]>,
    mask: usize,
    /// Producer-owned write index; consumer reads it with Acquire.
    head: AtomicUsize,
    /// Consumer-owned read index; producer reads it with Acquire.
    tail: AtomicUsize,
}

impl<T> SpscRing<T> {
    fn with_capacity(cap: usize) -> Self {
        assert!(cap.is_power_of_two());
        let mut slots = Vec::with_capacity(cap);
        slots.resize_with(cap, || UnsafeCell::new(MaybeUninit::uninit()));
        Self {
            slots: slots.into_boxed_slice(),
            mask: cap - 1,
            head: AtomicUsize::new(0),
            tail: AtomicUsize::new(0),
        }
    }

    /// Producer only. Returns `Err(item)` if the ring is full.
    fn push(&self, item: T) -> Result<(), T> {
        let head = self.head.load(Ordering::Relaxed);
        let tail = self.tail.load(Ordering::Acquire);
        if head.wrapping_sub(tail) == self.slots.len() {
            return Err(item);
        }
        unsafe { (*self.slots[head & self.mask].get()).write(item) };
        self.head.store(head.wrapping_add(1), Ordering::Release);
        Ok(())
    }

    /// Consumer only.
    fn pop(&self) -> Option<T> {
        let tail = self.tail.load(Ordering::Relaxed);
        let head = self.head.load(Ordering::Acquire);
        if head == tail {
            return None;
        }
        let item = unsafe { (*self.slots[tail & self.mask].get()).assume_init_read() };
        self.tail.store(tail.wrapping_add(1), Ordering::Release);
        Some(item)
    }
}

struct AudioState {
    /// Game thread → audio thread: play and control commands.
    ring: SpscRing<SoundCmd>,
    /// Audio thread → game thread: ids of voices that finished playing.
    finished: SpscRing<u32>,
    /// Owned exclusively by the audio thread (`mix`); never touched elsewhere.
    mixer: UnsafeCell<Mixer>,
}

// Safe because the rings enforce the SPSC discipline and `mixer` is only ever
// accessed from the single audio-callback thread.
unsafe impl Sync for AudioState {}

fn state() -> &'static AudioState {
    static STATE: OnceLock<AudioState> = OnceLock::new();
    STATE.get_or_init(|| AudioState {
        ring: SpscRing::with_capacity(1024),
        finished: SpscRing::with_capacity(1024),
        mixer: UnsafeCell::new(Mixer::default()),
    })
}

/// Push commands from the game thread into the ring. Lock-free; if the ring is
/// full the command is dropped (freed here on the producer thread).
pub fn submit<I: IntoIterator<Item = SoundCmd>>(cmds: I) {
    let ring = &state().ring;
    for cmd in cmds {
        if ring.push(cmd).is_err() {
            break;
        }
    }
}

/// Drain ids of voices that have finished playing. Consumer side of the
/// finished ring; must be called from a single thread (the same game thread
/// that calls `submit`).
pub fn drain_finished(out: &mut Vec<u32>) {
    let ring = &state().finished;
    while let Some(id) = ring.pop() {
        out.push(id);
    }
}

fn secs_to_samples(secs: f32) -> u64 {
    (secs.max(0.0) * SAMPLE_RATE as f32) as u64
}

fn make_pan_ramps(pan: f32) -> (Ramp, Ramp) {
    let (l, r) = pan_gains(pan);
    (Ramp::new(l), Ramp::new(r))
}

fn apply_command(m: &mut Mixer, cmd: SoundCmd) {
    match cmd {
        SoundCmd::CreateSample { sample_id, sample_rate, samples } => {
            m.bank.insert(sample_id, SampleBuffer { sample_rate, samples });
        }
        SoundCmd::Sine { voice_id, freq, amp, duration, attack, decay, sustain, release, delay, pan, bus } => {
            let total = secs_to_samples(duration);
            if total == 0 {
                return;
            }
            let (pan_l, pan_r) = make_pan_ramps(pan);
            m.voices.push(Voice {
                id: voice_id,
                bus,
                source: VoiceSource::Sine { phase_inc: freq / SAMPLE_RATE as f32 },
                delay_remaining: secs_to_samples(delay),
                position: 0,
                total_samples: total,
                envelope: make_adsr(attack, decay, sustain, release, total),
                amp: Ramp::new(amp),
                pan_l,
                pan_r,
                pause: Ramp::new(1.0),
                paused: false,
            });
        }
        SoundCmd::Square { voice_id, freq, duty, amp, duration, attack, decay, sustain, release, delay, pan, bus } => {
            let total = secs_to_samples(duration);
            if total == 0 {
                return;
            }
            let (pan_l, pan_r) = make_pan_ramps(pan);
            m.voices.push(Voice {
                id: voice_id,
                bus,
                source: VoiceSource::Square {
                    phase_inc: freq / SAMPLE_RATE as f32,
                    duty: duty.clamp(0.0, 1.0),
                },
                delay_remaining: secs_to_samples(delay),
                position: 0,
                total_samples: total,
                envelope: make_adsr(attack, decay, sustain, release, total),
                amp: Ramp::new(amp),
                pan_l,
                pan_r,
                pause: Ramp::new(1.0),
                paused: false,
            });
        }
        SoundCmd::PlaySample { voice_id, sample_id, amp, speed, fade_in, fade_out, delay, pan, bus } => {
            let (sample_rate, sample_len) = match m.bank.get(&sample_id) {
                Some(b) => (b.sample_rate, b.samples.len()),
                None => return,
            };
            let speed = speed.max(0.0001);
            let source_step = speed * sample_rate as f32 / SAMPLE_RATE as f32;
            let total = (sample_len as f32 / source_step).floor().max(0.0) as u64;
            if total == 0 {
                return;
            }
            let (pan_l, pan_r) = make_pan_ramps(pan);
            m.voices.push(Voice {
                id: voice_id,
                bus,
                source: VoiceSource::Sample { sample_id, source_step },
                delay_remaining: secs_to_samples(delay),
                position: 0,
                total_samples: total,
                envelope: Envelope::FadeInOut {
                    fade_in: secs_to_samples(fade_in).min(total),
                    fade_out: secs_to_samples(fade_out).min(total),
                },
                amp: Ramp::new(amp),
                pan_l,
                pan_r,
                pause: Ramp::new(1.0),
                paused: false,
            });
        }
        SoundCmd::StopVoice { voice_id } => {
            if let Some(v) = m.voices.iter_mut().find(|v| v.id == voice_id) {
                if v.delay_remaining > 0 {
                    // Not yet audible: end immediately.
                    v.total_samples = 0;
                    v.position = 0;
                } else {
                    // Ramp out over the remaining (or a minimum) fade window.
                    let fade = match &v.envelope {
                        Envelope::FadeInOut { fade_out, .. } => *fade_out,
                        Envelope::Adsr { release, .. } => *release,
                    }.max(PROC_FADE);
                    match &mut v.envelope {
                        Envelope::FadeInOut { fade_out, .. } => *fade_out = fade,
                        Envelope::Adsr { release, .. } => *release = fade,
                    }
                    v.total_samples = v.position + fade;
                }
            }
        }
        SoundCmd::SetVoiceGain { voice_id, amp, ramp } => {
            if let Some(v) = m.voices.iter_mut().find(|v| v.id == voice_id) {
                v.amp.set_target(amp, secs_to_samples(ramp));
            }
        }
        SoundCmd::SetVoicePan { voice_id, pan, ramp } => {
            if let Some(v) = m.voices.iter_mut().find(|v| v.id == voice_id) {
                let (l, r) = pan_gains(pan);
                let ramp_samples = secs_to_samples(ramp);
                v.pan_l.set_target(l, ramp_samples);
                v.pan_r.set_target(r, ramp_samples);
            }
        }
        SoundCmd::SetBusGain { bus, gain, ramp } => {
            if let Some(b) = m.buses.get_mut(bus as usize) {
                b.gain.set_target(gain, secs_to_samples(ramp));
            }
        }
        SoundCmd::SetBusLowPass { bus, cutoff, q, ramp } => {
            if let Some(b) = m.buses.get_mut(bus as usize) {
                let ramp_samples = secs_to_samples(ramp);
                match b.effects.iter_mut().find_map(|e| match e {
                    Effect::LowPass { cutoff, q, fading_out, .. } => Some((cutoff, q, fading_out)),
                }) {
                    Some((cr, qr, fo)) => {
                        cr.set_target(cutoff.ln(), ramp_samples);
                        *qr = q;
                        *fo = false;
                    }
                    None => {
                        let start = if ramp_samples > 0 { SAMPLE_RATE as f32 * 0.49 } else { cutoff };
                        let mut cutoff_ramp = Ramp::new(start.ln());
                        cutoff_ramp.set_target(cutoff.ln(), ramp_samples);
                        b.effects.push(Effect::LowPass {
                            biquad: Biquad::low_pass(start, q),
                            cutoff: cutoff_ramp,
                            q,
                            fading_out: false,
                        });
                    }
                }
            }
        }
        SoundCmd::ClearBusEffects { bus } => {
            if let Some(b) = m.buses.get_mut(bus as usize) {
                b.effects.clear();
            }
        }
        SoundCmd::SetVoicePaused { voice_id, paused, fade } => {
            if let Some(v) = m.voices.iter_mut().find(|v| v.id == voice_id) {
                let fade = secs_to_samples(fade).max(PROC_FADE);
                v.paused = paused;
                v.pause.set_target(if paused { 0.0 } else { 1.0 }, fade);
            }
        }
        SoundCmd::SetBusPaused { bus, paused, fade } => {
            if let Some(b) = m.buses.get_mut(bus as usize) {
                let fade = secs_to_samples(fade).max(PROC_FADE);
                b.paused = paused;
                b.pause.set_target(if paused { 0.0 } else { 1.0 }, fade);
            }
        }
        SoundCmd::SetAudioPaused { paused, fade } => {
            let fade = secs_to_samples(fade).max(PROC_FADE);
            m.master_paused = paused;
            m.master_pause.set_target(if paused { 0.0 } else { 1.0 }, fade);
        }
        SoundCmd::ResetAudio { fade_secs } => {
            let fade = secs_to_samples(fade_secs).max(PROC_FADE);
            // Clear any pause so frozen voices unfreeze and fade out normally
            // rather than lingering silent.
            m.master_paused = false;
            m.master_pause.set_target(1.0, 0);
            for voice in m.voices.iter_mut() {
                voice.paused = false;
                voice.pause.set_target(1.0, 0);
                if voice.delay_remaining > 0 {
                    voice.total_samples = 0;
                    voice.position = 0;
                } else {
                    match &mut voice.envelope {
                        Envelope::FadeInOut { fade_out, .. } => *fade_out = fade,
                        Envelope::Adsr { release, .. } => *release = fade,
                    }
                    voice.total_samples = voice.position + fade;
                }
            }
            for bus in m.buses.iter_mut() {
                bus.paused = false;
                bus.pause.set_target(1.0, 0);
                bus.gain.set_target(1.0, fade);
                for fx in bus.effects.iter_mut() {
                    fx.fade_out(fade);
                }
            }
        }
    }
}

/// Mix all active voices into an interleaved L/R `i16` output buffer. Runs on
/// the audio thread: drains pending commands from the ring, then renders. No
/// locks; the only steady-state allocation risk is the bank's hashmap growing
/// on a `CreateSample`, which happens at load time, not during playback.
pub fn mix(stream: &mut [i16]) {
    let state = state();
    // Sole accessor of `mixer` is this thread, upholding the SPSC invariant.
    let m = unsafe { &mut *state.mixer.get() };

    while let Some(cmd) = state.ring.pop() {
        apply_command(m, cmd);
    }

    let Mixer { voices, bank, master, buses, master_pause, master_paused } = m;
    let frames = stream.len() / 2;
    master.clear();
    master.resize(frames * 2, 0.0);

    // Buses with effects always process so their tails ring out even with no
    // input this buffer; zero their accumulators up front.
    for bus in buses.iter_mut() {
        bus.active = false;
        if !bus.effects.is_empty() {
            bus.prepare(frames);
        }
    }

    let finished = &state.finished;
    // Once a pause (voice, bus, or global) has fully faded to silence, the
    // voice is frozen: it produces nothing and its position holds until resume.
    let global_frozen = *master_paused && master_pause.is_settled();
    voices.retain_mut(|voice| {
        if global_frozen || voice.is_frozen() || buses[voice.bus as usize].is_frozen() {
            return true;
        }
        let skip = voice.delay_remaining.min(frames as u64) as usize;
        voice.delay_remaining -= skip as u64;
        let bus = &mut buses[voice.bus as usize];
        if !bus.active {
            bus.prepare(frames);
        }
        render_voice(voice, &mut bus.accum, skip, frames, bank);
        let alive = voice.position < voice.total_samples;
        if !alive {
            // Best-effort notify the game thread; dropping a notification only
            // delays handle cleanup, never corrupts state.
            let _ = finished.push(voice.id);
        }
        alive
    });

    // Process each active bus's effect chain, then fold it into the master with
    // a per-sample ramped gain. Idle buses still advance their gain ramp so it
    // tracks wall-clock time.
    for bus in buses.iter_mut() {
        if !bus.active {
            bus.gain.advance(frames as u64);
            bus.pause.advance(frames as u64);
            continue;
        }
        bus.effects.retain_mut(|fx| {
            fx.process(&mut bus.accum);
            !fx.is_faded_out()
        });
        for f in 0..frames {
            let g = bus.gain.next() * bus.pause.next();
            master[f * 2] += bus.accum[f * 2] * g;
            master[f * 2 + 1] += bus.accum[f * 2 + 1] * g;
        }
    }

    // Apply the global pause ramp as the master folds down to the output.
    for f in 0..frames {
        let gp = master_pause.next();
        let l = master[f * 2] * gp;
        let r = master[f * 2 + 1] * gp;
        stream[f * 2] = ((l / (1.0 + l.abs())) * i16::MAX as f32) as i16;
        stream[f * 2 + 1] = ((r / (1.0 + r.abs())) * i16::MAX as f32) as i16;
    }
}

/// 5ms attack/release applied to procedural voices to mask clicks.
const PROC_FADE: u64 = SAMPLE_RATE as u64 / 200;

/// Build an ADSR `Envelope`, clamping A/D/R to fit within `total` and
/// enforcing `PROC_FADE` as a minimum for attack and release.
fn make_adsr(attack: f32, decay: f32, sustain: f32, release: f32, total: u64) -> Envelope {
    let a = secs_to_samples(attack).max(PROC_FADE);
    let d = secs_to_samples(decay);
    let r = secs_to_samples(release).max(PROC_FADE);
    // Scale proportionally if A+D+R would exceed the total duration.
    let sum = a + d + r;
    let (a, d, r) = if sum > total && sum > 0 {
        let s = total as f64 / sum as f64;
        ((a as f64 * s) as u64, (d as f64 * s) as u64, (r as f64 * s) as u64)
    } else {
        (a, d, r)
    };
    Envelope::Adsr { attack: a, decay: d, sustain, release: r }
}

/// Linear fade-in/out envelope. A zero fade length disables that edge.
fn eval_envelope(env: &Envelope, pos: u64, total: u64) -> f32 {
    match env {
        Envelope::FadeInOut { fade_in, fade_out } => {
            let mut e = 1.0f32;
            if pos < *fade_in {
                e = e.min(pos as f32 / *fade_in as f32);
            }
            let tail = total.saturating_sub(pos);
            if tail < *fade_out {
                e = e.min(tail as f32 / *fade_out as f32);
            }
            e
        }
        Envelope::Adsr { attack, decay, sustain, release } => {
            if *attack > 0 && pos < *attack {
                return pos as f32 / *attack as f32;
            }
            let pos2 = pos.saturating_sub(*attack);
            if *decay > 0 && pos2 < *decay {
                let t = pos2 as f32 / *decay as f32;
                return 1.0 - t * (1.0 - sustain);
            }
            let tail = total.saturating_sub(pos);
            if *release > 0 && tail < *release {
                return sustain * tail as f32 / *release as f32;
            }
            *sustain
        }
    }
}

/// Render one voice's contribution into the interleaved `accum` buffer for the
/// frame range `[start_frame, frames)`, advancing `voice.position`. The enum
/// dispatch and bank lookup happen once per call rather than per sample.
fn render_voice(
    voice: &mut Voice,
    accum: &mut [f32],
    start_frame: usize,
    frames: usize,
    bank: &HashMap<u32, SampleBuffer>,
) {
    let total = voice.total_samples;

    // Number of frames to render: limited by both the output buffer space and
    // the voice's remaining samples.
    let available = (frames - start_frame) as u64;
    let remaining = total - voice.position;
    let end_frame = start_frame + available.min(remaining) as usize;

    // Sample buffers are looked up once; an unknown id ends the voice.
    let buffer = if let VoiceSource::Sample { sample_id, .. } = voice.source {
        match bank.get(&sample_id) {
            Some(b) => Some(b),
            None => {
                voice.position = total;
                return;
            }
        }
    } else {
        None
    };

    for f in start_frame..end_frame {
        let raw = match voice.source {
            VoiceSource::Sine { phase_inc } => {
                (TAU * phase_inc * voice.position as f32).sin()
            }
            VoiceSource::Square { phase_inc, duty } => {
                let phase = (phase_inc * voice.position as f32).fract();
                if phase < duty { 1.0 } else { -1.0 }
            }
            VoiceSource::Sample { source_step, .. } => {
                let buffer = buffer.unwrap();
                let pos = voice.position as f32 * source_step;
                let i0 = pos.floor() as usize;
                let frac = pos - pos.floor();
                let s0 = buffer.samples[i0];
                // i0+1 can be one past the end on the final frame; hold the
                // last sample rather than wrapping or reading OOB.
                let s1 = buffer.samples.get(i0 + 1).copied().unwrap_or(s0);
                s0 + (s1 - s0) * frac
            }
        };

        let env = eval_envelope(&voice.envelope, voice.position, total);
        let amp = voice.amp.next();
        let pause = voice.pause.next();
        let l = voice.pan_l.next();
        let r = voice.pan_r.next();
        let s = raw * env * amp * pause;
        accum[f * 2] += s * l;
        accum[f * 2 + 1] += s * r;
        voice.position += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::script_bridge::SoundCmd;

    fn long_sine(voice_id: u32, bus: u32) -> SoundCmd {
        SoundCmd::Sine {
            voice_id,
            freq: 440.0,
            amp: 1.0,
            duration: 100.0,
            attack: 0.0,
            decay: 0.0,
            sustain: 1.0,
            release: 0.0,
            delay: 0.0,
            pan: 0.0,
            bus,
        }
    }

    #[test]
    fn pause_voice_ramps_then_freezes() {
        let mut m = Mixer::default();
        apply_command(&mut m, long_sine(1, 0));

        apply_command(&mut m, SoundCmd::SetVoicePaused { voice_id: 1, paused: true, fade: 0.01 });
        let v = &mut m.voices[0];
        assert!(v.paused);
        // Mid-ramp it is not yet frozen (still audible, fading out).
        assert!(!v.is_frozen());
        assert_eq!(v.pause.target, 0.0);
        // Once the ramp settles at zero it is frozen.
        v.pause.advance(SAMPLE_RATE as u64);
        assert!(v.is_frozen());

        // Resuming clears the freeze and targets full volume.
        apply_command(&mut m, SoundCmd::SetVoicePaused { voice_id: 1, paused: false, fade: 0.01 });
        let v = &m.voices[0];
        assert!(!v.paused);
        assert!(!v.is_frozen());
        assert_eq!(v.pause.target, 1.0);
    }

    #[test]
    fn frozen_voice_holds_position() {
        let mut m = Mixer::default();
        apply_command(&mut m, long_sine(1, 0));
        // Pause with no fade so it freezes immediately.
        apply_command(&mut m, SoundCmd::SetVoicePaused { voice_id: 1, paused: true, fade: 0.0 });
        // The minimum PROC_FADE still applies, so settle the ramp.
        m.voices[0].pause.advance(SAMPLE_RATE as u64);
        assert!(m.voices[0].is_frozen());

        let mut stream = [0i16; 256];
        let before = m.voices[0].position;
        render_through_global_is_not_used(&mut m, &mut stream);
        // A frozen voice does not advance and the stream stays silent.
        assert_eq!(m.voices[0].position, before);
        assert!(stream.iter().all(|&s| s == 0));
    }

    /// Mimics the per-buffer work `mix` does, but against a local `Mixer` so the
    /// test doesn't touch the global singleton. Keeps the freeze/advance logic
    /// in sync with `mix`.
    fn render_through_global_is_not_used(m: &mut Mixer, stream: &mut [i16]) {
        let Mixer { voices, bank, master, buses, master_pause, master_paused } = m;
        let frames = stream.len() / 2;
        master.clear();
        master.resize(frames * 2, 0.0);
        for bus in buses.iter_mut() {
            bus.active = false;
            if !bus.effects.is_empty() {
                bus.prepare(frames);
            }
        }
        let global_frozen = *master_paused && master_pause.is_settled();
        voices.retain_mut(|voice| {
            if global_frozen || voice.is_frozen() || buses[voice.bus as usize].is_frozen() {
                return true;
            }
            let skip = voice.delay_remaining.min(frames as u64) as usize;
            voice.delay_remaining -= skip as u64;
            let bus = &mut buses[voice.bus as usize];
            if !bus.active {
                bus.prepare(frames);
            }
            render_voice(voice, &mut bus.accum, skip, frames, bank);
            voice.position < voice.total_samples
        });
        for bus in buses.iter_mut() {
            if !bus.active {
                bus.gain.advance(frames as u64);
                bus.pause.advance(frames as u64);
                continue;
            }
            for f in 0..frames {
                let g = bus.gain.next() * bus.pause.next();
                master[f * 2] += bus.accum[f * 2] * g;
                master[f * 2 + 1] += bus.accum[f * 2 + 1] * g;
            }
        }
        for f in 0..frames {
            let gp = master_pause.next();
            let l = master[f * 2] * gp;
            let r = master[f * 2 + 1] * gp;
            stream[f * 2] = ((l / (1.0 + l.abs())) * i16::MAX as f32) as i16;
            stream[f * 2 + 1] = ((r / (1.0 + r.abs())) * i16::MAX as f32) as i16;
        }
    }

    #[test]
    fn pause_bus_freezes_routed_voice() {
        let mut m = Mixer::default();
        apply_command(&mut m, long_sine(1, 3));
        apply_command(&mut m, SoundCmd::SetBusPaused { bus: 3, paused: true, fade: 0.0 });
        m.buses[3].pause.advance(SAMPLE_RATE as u64);
        assert!(m.buses[3].is_frozen());

        let mut stream = [0i16; 256];
        let before = m.voices[0].position;
        render_through_global_is_not_used(&mut m, &mut stream);
        assert_eq!(m.voices[0].position, before, "voice on a paused bus should freeze");

        apply_command(&mut m, SoundCmd::SetBusPaused { bus: 3, paused: false, fade: 0.01 });
        render_through_global_is_not_used(&mut m, &mut stream);
        assert!(m.voices[0].position > before, "voice should advance once the bus resumes");
    }

    #[test]
    fn pause_all_freezes_every_voice() {
        let mut m = Mixer::default();
        apply_command(&mut m, long_sine(1, 0));
        apply_command(&mut m, long_sine(2, 5));
        apply_command(&mut m, SoundCmd::SetAudioPaused { paused: true, fade: 0.0 });
        m.master_pause.advance(SAMPLE_RATE as u64);

        let mut stream = [0i16; 256];
        let p0 = m.voices[0].position;
        let p1 = m.voices[1].position;
        render_through_global_is_not_used(&mut m, &mut stream);
        assert_eq!(m.voices[0].position, p0);
        assert_eq!(m.voices[1].position, p1);
        assert!(stream.iter().all(|&s| s == 0));
    }
}
