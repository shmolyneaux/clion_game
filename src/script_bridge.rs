use shimlang::{ArgBundle, Ident};
use shimlang::runtime::{ArgUnpacker, CallResult, NativeFn, StructDef, StructAttribute, ShimFn};
use shimlang::{Environment, Interpreter, ShimNative, ShimValue, debug_u8s};
use std::collections::HashSet;
use std::ffi::CString;
use std::mem;
use tinyjson::JsonValue;

use crate::shimlang_imgui;
//use crate::test_mocks::igBegin;
use crate::*;

#[cfg(not(target_arch = "wasm32"))]
use std::fs;
#[cfg(not(target_arch = "wasm32"))]
use std::sync::mpsc::{Receiver, Sender, channel};

// TODO: Probably shouldn't be cloned?
#[derive(Debug, Clone)]
pub struct TextureHandle {
    pub texture_id: u32,
}

impl ShimNative for TextureHandle {
    fn gc_vals(&self) -> Vec<ShimValue> {
        Vec::new()
    }
}

/// A UV rectangle with all values in the range [0.0, 1.0], (0,0) = top-left.
/// (x1, y1) is the top-left corner; (x2, y2) is the bottom-right corner.
#[derive(Debug, Clone)]
pub struct Rect {
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
}

impl ShimNative for Rect {
    fn gc_vals(&self) -> Vec<ShimValue> {
        Vec::new()
    }
}

#[derive(Debug, Clone)]
pub struct DrawRect {
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
    pub texture: Option<TextureHandle>,
    pub modulate: [u8; 4],
    pub region: Option<[f32; 4]>,
}

#[derive(Debug, Clone)]
pub struct DrawText {
    pub x: f32,
    pub y: f32,
    pub size: f32,
    pub text: String,
}

pub enum DrawListItem {
    Rect(DrawRect),
    Text(DrawText),
    CreateTexture(u32, u32, u32, Vec<u8>, bool),
}

/// A reference to a sample registered with the audio system
#[derive(Debug, Clone)]
pub struct SoundSampleHandle {
    pub sample_id: u32,
}

impl ShimNative for SoundSampleHandle {
    fn get_attr(
        &self,
        self_as_val: &ShimValue,
        interpreter: &mut Interpreter,
        ident: &[u8],
    ) -> Result<ShimValue, String> {
        if ident == b"play" {
            fn shim_sample_play(
                interpreter: &mut Interpreter,
                args: &ArgBundle,
            ) -> Result<ShimValue, String> {
                if args.args.len() != 1 {
                    return Err("SoundSampleHandle.play takes no positional args".to_string());
                }
                let sample_id = args.args[0]
                    .as_native::<SoundSampleHandle>(interpreter)?
                    .sample_id;
                let mut amp = 1.0f32;
                let mut speed = 1.0f32;
                let mut fade_in = 0.005f32;
                let mut fade_out = 0.005f32;
                let mut delay = 0.0f32;
                let mut pan = 0.0f32;
                let mut bus = 0u32;
                for (k, v) in args.kwargs.iter() {
                    let f = match v {
                        ShimValue::Float(f) => *f,
                        ShimValue::Integer(i) => *i as f32,
                        _ => {
                            return Err(format!(
                                "play kwarg '{}' must be numeric",
                                debug_u8s(k)
                            ));
                        }
                    };
                    match k.as_slice() {
                        b"amp" => amp = f,
                        b"speed" => speed = f,
                        b"fade_in" => fade_in = f,
                        b"fade_out" => fade_out = f,
                        b"delay" => delay = f,
                        b"pan" => pan = f,
                        b"bus" => bus = parse_bus(Some(*v))?,
                        _ => {
                            return Err(format!(
                                "Unknown play kwarg '{}'",
                                debug_u8s(k)
                            ));
                        }
                    }
                }
                let sl = interpreter.fetch_mut::<SoundList>();
                let voice_id = sl.alloc_voice_id();
                sl.items.push(SoundCmd::PlaySample {
                    voice_id,
                    sample_id,
                    amp,
                    speed,
                    fade_in,
                    fade_out,
                    delay,
                    pan: pan.clamp(-1.0, 1.0),
                    bus,
                });
                Ok(interpreter.mem.alloc_native(VoiceHandle { voice_id }))
            }
            Ok(interpreter.mem.alloc_bound_native_fn(self_as_val, shim_sample_play))
        } else {
            Err(format!(
                "SoundSampleHandle has no attribute '{}'",
                debug_u8s(ident)
            ))
        }
    }

    fn gc_vals(&self) -> Vec<ShimValue> {
        Vec::new()
    }
}

/// Default gain/pan ramp (seconds) when a script doesn't pass `ramp`.
const DEFAULT_RAMP_SECS: f32 = 0.01;

/// A reference to a playing voice. Holds only the id; `.stop()`, `.set_gain()`
/// and `.set_pan()` push control commands. Targeting a finished or unknown
/// voice is a harmless no-op on the audio side.
#[derive(Debug, Clone)]
pub struct VoiceHandle {
    pub voice_id: u32,
}

impl ShimNative for VoiceHandle {
    fn get_attr(
        &self,
        self_as_val: &ShimValue,
        interpreter: &mut Interpreter,
        ident: &[u8],
    ) -> Result<ShimValue, String> {
        match ident {
            b"finished" => {
                let done = interpreter
                    .fetch_mut::<SoundList>()
                    .finished
                    .contains(&self.voice_id);
                Ok(ShimValue::Bool(done))
            }
            b"stop" => {
                fn shim_stop(
                    interpreter: &mut Interpreter,
                    args: &ArgBundle,
                ) -> Result<ShimValue, String> {
                    let voice_id = args.args[0]
                        .as_native::<VoiceHandle>(interpreter)?
                        .voice_id;
                    interpreter
                        .fetch_mut::<SoundList>()
                        .items
                        .push(SoundCmd::StopVoice { voice_id });
                    Ok(ShimValue::None)
                }
                Ok(interpreter.mem.alloc_bound_native_fn(self_as_val, shim_stop))
            }
            b"set_gain" => {
                fn shim_set_gain(
                    interpreter: &mut Interpreter,
                    args: &ArgBundle,
                ) -> Result<ShimValue, String> {
                    let voice_id = args.args[0]
                        .as_native::<VoiceHandle>(interpreter)?
                        .voice_id;
                    let mut unpacker = ArgUnpacker::new(args);
                    let _ = unpacker.required(b"self")?;
                    let amp = unpacker.required_number(b"amp")?;
                    let ramp = unpacker.optional_number(b"ramp", DEFAULT_RAMP_SECS)?;
                    unpacker.end()?;
                    interpreter.fetch_mut::<SoundList>().items.push(SoundCmd::SetVoiceGain {
                        voice_id,
                        amp,
                        ramp: ramp.max(0.0),
                    });
                    Ok(ShimValue::None)
                }
                Ok(interpreter.mem.alloc_bound_native_fn(self_as_val, shim_set_gain))
            }
            b"set_pan" => {
                fn shim_set_pan(
                    interpreter: &mut Interpreter,
                    args: &ArgBundle,
                ) -> Result<ShimValue, String> {
                    let voice_id = args.args[0]
                        .as_native::<VoiceHandle>(interpreter)?
                        .voice_id;
                    let mut unpacker = ArgUnpacker::new(args);
                    let _ = unpacker.required(b"self")?;
                    let pan = unpacker.required_number(b"pan")?;
                    let ramp = unpacker.optional_number(b"ramp", DEFAULT_RAMP_SECS)?;
                    unpacker.end()?;
                    interpreter.fetch_mut::<SoundList>().items.push(SoundCmd::SetVoicePan {
                        voice_id,
                        pan: pan.clamp(-1.0, 1.0),
                        ramp: ramp.max(0.0),
                    });
                    Ok(ShimValue::None)
                }
                Ok(interpreter.mem.alloc_bound_native_fn(self_as_val, shim_set_pan))
            }
            _ => Err(format!("VoiceHandle has no attribute '{}'", debug_u8s(ident))),
        }
    }

    fn gc_vals(&self) -> Vec<ShimValue> {
        Vec::new()
    }
}

pub enum SoundCmd {
    Square {
        voice_id: u32,
        freq: f32,
        duty: f32,
        amp: f32,
        duration: f32,
        attack: f32,
        decay: f32,
        sustain: f32,
        release: f32,
        delay: f32,
        pan: f32,
        bus: u32,
    },
    Sine {
        voice_id: u32,
        freq: f32,
        amp: f32,
        duration: f32,
        attack: f32,
        decay: f32,
        sustain: f32,
        release: f32,
        delay: f32,
        pan: f32,
        bus: u32,
    },
    CreateSample {
        sample_id: u32,
        sample_rate: u32,
        samples: Vec<f32>,
    },
    PlaySample {
        voice_id: u32,
        sample_id: u32,
        amp: f32,
        speed: f32,
        fade_in: f32,
        fade_out: f32,
        delay: f32,
        pan: f32,
        bus: u32,
    },
    StopVoice {
        voice_id: u32,
    },
    SetVoiceGain {
        voice_id: u32,
        amp: f32,
        ramp: f32,
    },
    SetVoicePan {
        voice_id: u32,
        pan: f32,
        ramp: f32,
    },
    SetBusGain {
        bus: u32,
        gain: f32,
        ramp: f32,
    },
    SetBusLowPass {
        bus: u32,
        cutoff: f32,
        q: f32,
        ramp: f32,
    },
    ClearBusEffects {
        bus: u32,
    },
    ResetAudio {
        fade_secs: f32,
    },
}

/// Number of mixer buses. Buses are addressed 0..NUM_BUSES.
pub const NUM_BUSES: u32 = 32;

fn parse_bus(value: Option<ShimValue>) -> Result<u32, String> {
    let bus = match value {
        None => return Ok(0),
        Some(ShimValue::Integer(i)) => i,
        Some(ShimValue::Float(f)) => f as i32,
        Some(_) => return Err("bus must be an integer".to_string()),
    };
    if (0..NUM_BUSES as i32).contains(&bus) {
        Ok(bus as u32)
    } else {
        Err(format!("bus must be 0..={}, got {}", NUM_BUSES - 1, bus))
    }
}

#[derive(Default)]
pub struct SoundList {
    next_sample_id: u32,
    next_voice_id: u32,
    pub items: Vec<SoundCmd>,
    /// Voice ids that have finished playing, drained from the audio thread.
    /// Grows unbounded for now; entries will be released once `ShimNative`
    /// gains drop-like behavior.
    pub finished: HashSet<u32>,
}

impl SoundList {
    fn push_create_sample(&mut self, sample_rate: u32, samples: Vec<f32>) -> SoundSampleHandle {
        let sample_id = self.next_sample_id;
        self.next_sample_id += 1;
        self.items.push(SoundCmd::CreateSample {
            sample_id,
            sample_rate,
            samples,
        });
        SoundSampleHandle { sample_id }
    }

    fn alloc_voice_id(&mut self) -> u32 {
        let id = self.next_voice_id;
        self.next_voice_id += 1;
        id
    }
}

const KEY_REPEAT_INITIAL_DELAY: f32 = 0.4;
const KEY_REPEAT_RATE: f32 = 0.08;

/// Keyboard state for the current and previous frame, indexed by SDL scancode.
///
/// - `keys` — pressed state this frame (1 = pressed, 0 = released)
/// - `last_keys` — pressed state last frame; diff against `keys` gives just-pressed/released
/// - `held_time` — seconds each key has been continuously held; resets to 0 on release
/// - `delta` — frame time in seconds, mirrored here so `KeyValue::get_attr` can compute repeat
///   boundaries without a separate lookup
#[derive(Default)]
pub struct KeyState {
    pub keys: Vec<u8>,
    pub last_keys: Vec<u8>,
    pub held_time: Vec<f32>,
    pub delta: f32,
}

/// Mouse button state for the current and previous frame, as SDL button bitmasks.
/// Button N (1 = left, 2 = middle, 3 = right, 4 = x1, 5 = x2) is bit `1 << (N - 1)`.
#[derive(Default)]
pub struct MouseState {
    pub buttons: u32,
    pub last_buttons: u32,
}

fn mouse_button_mask(button: i32) -> Result<u32, String> {
    if !(1..=5).contains(&button) {
        return Err(format!(
            "mouse button must be 1 (left), 2 (middle), 3 (right), 4 (x1), or 5 (x2); got {}",
            button
        ));
    }
    Ok(1u32 << (button - 1))
}


fn scancode_from_name(name: &[u8]) -> Option<usize> {
    match name {
        b"A" => Some(4),
        b"B" => Some(5),
        b"C" => Some(6),
        b"D" => Some(7),
        b"E" => Some(8),
        b"F" => Some(9),
        b"G" => Some(10),
        b"H" => Some(11),
        b"I" => Some(12),
        b"J" => Some(13),
        b"K" => Some(14),
        b"L" => Some(15),
        b"M" => Some(16),
        b"N" => Some(17),
        b"O" => Some(18),
        b"P" => Some(19),
        b"Q" => Some(20),
        b"R" => Some(21),
        b"S" => Some(22),
        b"T" => Some(23),
        b"U" => Some(24),
        b"V" => Some(25),
        b"W" => Some(26),
        b"X" => Some(27),
        b"Y" => Some(28),
        b"Z" => Some(29),
        b"Key1" => Some(30),
        b"Key2" => Some(31),
        b"Key3" => Some(32),
        b"Key4" => Some(33),
        b"Key5" => Some(34),
        b"Key6" => Some(35),
        b"Key7" => Some(36),
        b"Key8" => Some(37),
        b"Key9" => Some(38),
        b"Key0" => Some(39),
        b"Enter" => Some(40),
        b"Escape" => Some(41),
        b"Backspace" => Some(42),
        b"Tab" => Some(43),
        b"Space" => Some(44),
        b"Minus" => Some(45),
        b"Equal" => Some(46),
        b"LeftBracket" => Some(47),
        b"RightBracket" => Some(48),
        b"Backslash" => Some(49),
        b"Semicolon" => Some(51),
        b"Apostrophe" => Some(52),
        b"GraveAccent" => Some(53),
        b"Comma" => Some(54),
        b"Period" => Some(55),
        b"Slash" => Some(56),
        b"CapsLock" => Some(57),
        b"F1" => Some(58),
        b"F2" => Some(59),
        b"F3" => Some(60),
        b"F4" => Some(61),
        b"F5" => Some(62),
        b"F6" => Some(63),
        b"F7" => Some(64),
        b"F8" => Some(65),
        b"F9" => Some(66),
        b"F10" => Some(67),
        b"F11" => Some(68),
        b"F12" => Some(69),
        b"PrintScreen" => Some(70),
        b"ScrollLock" => Some(71),
        b"Pause" => Some(72),
        b"Insert" => Some(73),
        b"Home" => Some(74),
        b"PageUp" => Some(75),
        b"Delete" => Some(76),
        b"End" => Some(77),
        b"PageDown" => Some(78),
        b"Right" => Some(79),
        b"Left" => Some(80),
        b"Down" => Some(81),
        b"Up" => Some(82),
        b"NumLock" => Some(83),
        b"KpDivide" => Some(84),
        b"KpMultiply" => Some(85),
        b"KpMinus" => Some(86),
        b"KpPlus" => Some(87),
        b"KpEnter" => Some(88),
        b"Kp1" => Some(89),
        b"Kp2" => Some(90),
        b"Kp3" => Some(91),
        b"Kp4" => Some(92),
        b"Kp5" => Some(93),
        b"Kp6" => Some(94),
        b"Kp7" => Some(95),
        b"Kp8" => Some(96),
        b"Kp9" => Some(97),
        b"Kp0" => Some(98),
        b"LeftCtrl" => Some(224),
        b"LeftShift" => Some(225),
        b"LeftAlt" => Some(226),
        b"LeftSuper" => Some(227),
        b"RightCtrl" => Some(228),
        b"RightShift" => Some(229),
        b"RightAlt" => Some(230),
        b"RightSuper" => Some(231),
        _ => None,
    }
}

#[derive(Debug)]
struct KeyMap;

impl ShimNative for KeyMap {
    fn get_attr(
        &self,
        _self_as_val: &ShimValue,
        interpreter: &mut Interpreter,
        ident: &[u8],
    ) -> Result<ShimValue, String> {
        match scancode_from_name(ident) {
            Some(scancode) => Ok(interpreter.mem.alloc_native(KeyValue { scancode })),
            None => Err(format!("Unknown key: {}", debug_u8s(ident))),
        }
    }

    fn gc_vals(&self) -> Vec<ShimValue> {
        Vec::new()
    }
}

#[derive(Debug)]
struct KeyValue {
    scancode: usize,
}

impl ShimNative for KeyValue {
    fn get_attr(
        &self,
        _self_as_val: &ShimValue,
        interpreter: &mut Interpreter,
        ident: &[u8],
    ) -> Result<ShimValue, String> {
        let ks = interpreter.fetch_mut::<KeyState>();
        let cur = ks.keys.get(self.scancode).copied().unwrap_or(0);
        let last = ks.last_keys.get(self.scancode).copied().unwrap_or(0);
        match ident {
            b"pressed" => Ok(ShimValue::Bool(cur == 1)),
            b"released" => Ok(ShimValue::Bool(cur == 0)),
            b"just_pressed" => Ok(ShimValue::Bool(cur == 1 && cur != last)),
            b"just_released" => Ok(ShimValue::Bool(cur == 0 && cur != last)),
            b"just_pressed_with_repeat" => {
                let just_pressed = cur == 1 && cur != last;
                let repeat = if cur == 1 && !just_pressed {
                    let held = ks.held_time.get(self.scancode).copied().unwrap_or(0.0);
                    let delta = ks.delta;
                    let prev = held - delta;
                    let repeat_idx = |t: f32| -> u32 {
                        if t > KEY_REPEAT_INITIAL_DELAY {
                            ((t - KEY_REPEAT_INITIAL_DELAY) / KEY_REPEAT_RATE) as u32 + 1
                        } else {
                            0
                        }
                    };
                    repeat_idx(held) > repeat_idx(prev)
                } else {
                    false
                };
                Ok(ShimValue::Bool(just_pressed || repeat))
            }
            _ => Err(format!(
                "KeyValue has pressed/released/just_pressed/just_released/just_pressed_with_repeat, not '{}'",
                debug_u8s(ident)
            )),
        }
    }

    fn gc_vals(&self) -> Vec<ShimValue> {
        Vec::new()
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub struct PerfTimer {
    pub script: f32,
    pub gc: f32,
    pub render: f32,
    pub vsync: f32,
}

#[derive(Debug)]
struct PerfModule;

impl ShimNative for PerfModule {
    fn get_attr(
        &self,
        _self_as_val: &ShimValue,
        interpreter: &mut Interpreter,
        ident: &[u8],
    ) -> Result<ShimValue, String> {
        let times = interpreter.fetch_mut::<PerfTimer>();
        let (script, gc, render, vsync) = match times {
            PerfTimer { script, gc, render, vsync } => (script, gc, render, vsync)
        };
        Ok(ShimValue::Float(match ident {
            b"total" => *script + *gc + *render + *vsync,
            b"script" => *script,
            b"gc" => *gc,
            b"render" => *render,
            b"vsync" => *vsync,
            _ => return Err(format!(
                "PerfModule has no attribute '{}'",
                debug_u8s(ident)
            )),
        }))
    }

    fn gc_vals(&self) -> Vec<ShimValue> {
        Vec::new()
    }
}

pub struct DrawList {
    next_texture_handle: u32,
    items: Vec<DrawListItem>,
}

impl Default for DrawList {
    fn default() -> Self {
        Self::new()
    }
}

impl DrawList {
    fn new() -> Self {
        Self {
            next_texture_handle: 1,
            items: Vec::new(),
        }
    }

    fn push_rect(&mut self, x: f32, y: f32, w: f32, h: f32, texture: Option<TextureHandle>, modulate: [u8; 4], region: Option<[f32; 4]>) {
        self.items.push(DrawListItem::Rect(DrawRect { x, y, w, h, texture, modulate, region }));
    }

    fn push_text(&mut self, x: f32, y: f32, size: f32, text: String) {
        self.items.push(DrawListItem::Text(DrawText { x, y, size, text }));
    }

    fn push_texture(&mut self, w: u32, h: u32, data: Vec<u8>, nearest: bool) -> TextureHandle {
        let id = self.next_texture_handle;
        self.next_texture_handle += 1;
        self.items.push(DrawListItem::CreateTexture(id, w, h, data, nearest));
        TextureHandle { texture_id: id }
    }
}

/// Messages sent from the Engine (Main Thread) to the Script Thread
#[cfg(not(target_arch = "wasm32"))]
pub enum ScriptRequest {
    ExecuteLoop(Interpreter, Environment, ShimValue, Vec<u8>, Vec<u8>, f32, PerfTimer, Vec<u32>),
}

/// Messages sent from the Script Thread to the Engine
#[cfg(not(target_arch = "wasm32"))]
pub enum ScriptResponse {
    LoopComplete(
        Interpreter,
        Environment,
        ShimValue,
        Vec<DrawListItem>,
        Vec<SoundCmd>,
        f32,
    ),
    Error(Interpreter, Environment, ShimValue, String),
}

/// Messages sent from the file watcher thread to the Engine
#[cfg(not(target_arch = "wasm32"))]
enum ScriptWatcherMessage {
    Loaded(Interpreter, Environment, ShimValue),
    Error(String),
}

enum BridgeState {
    #[cfg(not(target_arch = "wasm32"))]
    Running,
    Paused(Box<Interpreter>, Environment, ShimValue),
}

pub struct ScriptBridge {
    state: BridgeState,
    pub interpreter_errors: Vec<String>,
    pub draw_list: Vec<DrawListItem>,
    pub sound_list: Vec<SoundCmd>,
    pub last_gc_time: f32,
    script_path: String,
    #[cfg(not(target_arch = "wasm32"))]
    tx: Sender<ScriptRequest>,
    #[cfg(not(target_arch = "wasm32"))]
    rx: Receiver<ScriptResponse>,
    #[cfg(not(target_arch = "wasm32"))]
    watcher_rx: Receiver<ScriptWatcherMessage>,
}

fn shim_ig_begin(interpreter: &mut Interpreter, args: &ArgBundle) -> Result<ShimValue, String> {
    let mut unpacker = ArgUnpacker::new(args);
    let title_val = unpacker.required(b"title")?;
    unpacker.end()?;
    let title_bytes = title_val.to_string(interpreter);
    let c_title =
        CString::new(title_bytes).map_err(|_| "igBegin: title contains null byte".to_string())?;
    #[cfg(not(test))]
    let result = unsafe { super::igBegin(c_title.as_ptr(), std::ptr::null_mut(), 0) };
    #[cfg(test)]
    let result = false;
    Ok(ShimValue::Bool(result))
}

fn shim_ig_end(_interpreter: &mut Interpreter, args: &ArgBundle) -> Result<ShimValue, String> {
    let unpacker = ArgUnpacker::new(args);
    unpacker.end()?;
    #[cfg(not(test))]
    unsafe {
        super::igEnd();
    }
    Ok(ShimValue::None)
}

fn shim_ig_text(interpreter: &mut Interpreter, args: &ArgBundle) -> Result<ShimValue, String> {
    let mut unpacker = ArgUnpacker::new(args);
    let text_val = unpacker.required(b"text")?;
    unpacker.end()?;
    let text_bytes = text_val.to_string(interpreter);
    let c_text =
        CString::new(text_bytes).map_err(|_| "igText: text contains null byte".to_string())?;
    super::igText(c_text.as_ptr());
    Ok(ShimValue::None)
}

fn shim_draw_rect(interpreter: &mut Interpreter, args: &ArgBundle) -> Result<ShimValue, String> {
    let mut unpacker = ArgUnpacker::new(args);
    let mut x = unpacker.required_number(b"x")?;
    let mut y = unpacker.required_number(b"y")?;
    let w = unpacker.required_number(b"w")?;
    let h = unpacker.required_number(b"h")?;
    let texture: Option<TextureHandle> = match unpacker.optional(b"texture") {
        Some(val) => Some(val.as_native::<TextureHandle>(interpreter)?.clone()),
        None => None,
    };
    let center = matches!(unpacker.optional(b"center"), Some(ShimValue::Bool(true)));
    fn optional_channel(val: Option<ShimValue>) -> Result<u8, String> {
        match val {
            None => Ok(255),
            Some(ShimValue::Integer(i)) => Ok(i.clamp(0, 255) as u8),
            Some(ShimValue::Float(f)) => Ok(if f <= 0.0 {
                0
            } else if f >= 1.0 {
                255
            } else {
                (f * 255.0).round() as u8
            }),
            Some(_) => Err("Color channel must be a number".to_string()),
        }
    }
    let r = optional_channel(unpacker.optional(b"r"))?;
    let g = optional_channel(unpacker.optional(b"g"))?;
    let b = optional_channel(unpacker.optional(b"b"))?;
    let a = optional_channel(unpacker.optional(b"a"))?;
    let region: Option<[f32; 4]> = match unpacker.optional(b"region") {
        Some(val) => {
            let rect = val.as_native::<Rect>(interpreter)?;
            Some([rect.x1, rect.y1, rect.x2, rect.y2])
        }
        None => None,
    };
    unpacker.end()?;

    if center {
        x -= w/2.0;
        y -= h/2.0;
    }

    interpreter.fetch_mut::<DrawList>().push_rect(x, y, w, h, texture, [r, g, b, a], region);
    Ok(ShimValue::None)
}

fn shim_rect(interpreter: &mut Interpreter, args: &ArgBundle) -> Result<ShimValue, String> {
    let mut unpacker = ArgUnpacker::new(args);
    let x1 = unpacker.required_number(b"x1")?;
    let y1 = unpacker.required_number(b"y1")?;
    let x2 = unpacker.required_number(b"x2")?;
    let y2 = unpacker.required_number(b"y2")?;
    unpacker.end()?;

    fn clamp01(v: f32) -> f32 { v.clamp(0.0, 1.0) }
    Ok(interpreter.mem.alloc_native(Rect { x1: clamp01(x1), y1: clamp01(y1), x2: clamp01(x2), y2: clamp01(y2) }))
}

fn text_size(text: &[u8], font_size: f32) -> (f32, f32) {
    let mut line_count = 1;
    let mut longest_line = 0;
    let mut current_line_length = 0;
    for byte in text {
        current_line_length += 1;
        if *byte == b'\n' {
            line_count += 1;
            longest_line = longest_line.max(current_line_length - 1);
            current_line_length = 0;
        }
    }
    longest_line = longest_line.max(current_line_length);

    (
        longest_line as f32 * font_size,
        // Each char is 7px, but with line spacing of 8px
        (7.0/8.0 + (line_count as f32 - 1.0)) * font_size,
    )
}

fn shim_text_size(interpreter: &mut Interpreter, args: &ArgBundle) -> Result<ShimValue, String> {
    let mut unpacker = ArgUnpacker::new(args);
    let text_val = unpacker.required(b"text")?;
    let size = unpacker.optional_number(b"size", 1.0)?;
    let center = matches!(unpacker.optional(b"center"), Some(ShimValue::Bool(true)));
    unpacker.end()?;

    let text = text_val.to_string(interpreter);

    let size = text_size(text.as_bytes(), size*8.0);

    Ok(interpreter.mem.alloc_tuple(&[ShimValue::Float(size.0), ShimValue::Float(size.1)]))
}

fn shim_draw_text(interpreter: &mut Interpreter, args: &ArgBundle) -> Result<ShimValue, String> {
    let mut unpacker = ArgUnpacker::new(args);
    let mut x = unpacker.required_number(b"x")?;
    let mut y = unpacker.required_number(b"y")?;
    let text_val = unpacker.required(b"text")?;
    let size = unpacker.optional_number(b"size", 1.0)?;
    let center = matches!(unpacker.optional(b"center"), Some(ShimValue::Bool(true)));
    unpacker.end()?;

    let text = text_val.to_string(interpreter);

    if center {
        let dim = text_size(text.as_bytes(), size*8.0);
        x -= dim.0/2.0;
        // Offset by size since the top row of pixels in the font is transparent
        y -= dim.1/2.0 + size;
    } else {
        // Offset by size since the top row of pixels in the font is transparent
        y -= size;
    }

    interpreter
        .fetch_mut::<DrawList>()
        .push_text(x, y, size, text);
    Ok(ShimValue::None)
}

fn shim_create_texture(
    interpreter: &mut Interpreter,
    args: &ArgBundle,
) -> Result<ShimValue, String> {
    let mut unpacker = ArgUnpacker::new(args);
    let w = unpacker.required_int(b"w")?;
    let h = unpacker.required_int(b"h")?;
    let data = unpacker.required_list(interpreter, b"rgba_bytes")?;
    let nearest = matches!(unpacker.optional(b"nearest"), Some(ShimValue::Bool(true)));
    unpacker.end()?;

    let w: u32 = if w < 0 {
        return Err(format!("Int w {} must be non-negative", w));
    } else {
        w as u32
    };

    let h: u32 = if h < 0 {
        return Err(format!("Int h {} must be non-negative", h));
    } else {
        h as u32
    };

    if data.len() != (w * h * 4) as usize {
        return Err(format!(
            "Expected a w*h*4={} length array but got array length {}",
            w * h * 4,
            data.len()
        ));
    }

    // TODO: The ShimList should provide an iterable the yields ShimValues
    let shimvalues = data.raw_data(&interpreter.mem);

    let mut rgba_bytes: Vec<u8> = Vec::with_capacity((w * h * 4) as usize);
    for val in shimvalues.iter() {
        rgba_bytes.push(match unsafe { ShimValue::from_u64(*val) } {
            ShimValue::Integer(i) => {
                if i <= 0 {
                    0
                } else if i >= 255 {
                    255
                } else {
                    i as u8
                }
            }
            ShimValue::Float(f) => {
                if f <= 0.0 {
                    0
                } else if f >= 1.0 {
                    255
                } else {
                    (f * 255.0).round() as u8
                }
            }
            _ => {
                return Err(format!(
                    "Non-numeric passed to create_texture {}",
                    val
                ));
            }
        });
    }

    let handle = interpreter.fetch_mut::<DrawList>().push_texture(w, h, rgba_bytes, nearest);

    Ok(interpreter.mem.alloc_native(handle))
}

fn shim_mouse_pos(
    interpreter: &mut Interpreter,
    args: &ArgBundle,
) -> Result<ShimValue, String> {
    ArgUnpacker::new(args).end()?;

    let mut x = 0;
    let mut y = 0;
    unsafe {
        SDL_GetMouseState(&mut x as *mut i32, &mut y as *mut i32);
    }

    let new_lst_val = interpreter.mem.alloc_list();
    let new_lst = new_lst_val.list_mut(interpreter)?;
    new_lst.push(&mut interpreter.mem, ShimValue::Integer(x));
    new_lst.push(&mut interpreter.mem, ShimValue::Integer(y));
    Ok(new_lst_val)
}

fn shim_mouse_pressed(
    interpreter: &mut Interpreter,
    args: &ArgBundle,
) -> Result<ShimValue, String> {
    let mut unpacker = ArgUnpacker::new(args);
    let button = unpacker.required_int(b"button")?;
    unpacker.end()?;
    let mask = mouse_button_mask(button)?;
    let ms = interpreter.fetch_mut::<MouseState>();
    Ok(ShimValue::Bool(ms.buttons & mask != 0))
}

fn shim_mouse_just_pressed(
    interpreter: &mut Interpreter,
    args: &ArgBundle,
) -> Result<ShimValue, String> {
    let mut unpacker = ArgUnpacker::new(args);
    let button = unpacker.required_int(b"button")?;
    unpacker.end()?;
    let mask = mouse_button_mask(button)?;
    let ms = interpreter.fetch_mut::<MouseState>();
    Ok(ShimValue::Bool(
        ms.buttons & mask != 0 && ms.last_buttons & mask == 0,
    ))
}

fn shim_mouse_just_released(
    interpreter: &mut Interpreter,
    args: &ArgBundle,
) -> Result<ShimValue, String> {
    let mut unpacker = ArgUnpacker::new(args);
    let button = unpacker.required_int(b"button")?;
    unpacker.end()?;
    let mask = mouse_button_mask(button)?;
    let ms = interpreter.fetch_mut::<MouseState>();
    Ok(ShimValue::Bool(
        ms.buttons & mask == 0 && ms.last_buttons & mask != 0,
    ))
}

fn shim_show_cursor(
    _interpreter: &mut Interpreter,
    args: &ArgBundle,
) -> Result<ShimValue, String> {
    ArgUnpacker::new(args).end()?;
    unsafe { SHM_SetCursorVisible(true); }
    Ok(ShimValue::None)
}

fn shim_hide_cursor(
    _interpreter: &mut Interpreter,
    args: &ArgBundle,
) -> Result<ShimValue, String> {
    ArgUnpacker::new(args).end()?;
    unsafe { SHM_SetCursorVisible(false); }
    Ok(ShimValue::None)
}

fn shim_set_window_title(
    interpreter: &mut Interpreter,
    args: &ArgBundle,
) -> Result<ShimValue, String> {
    let mut unpacker = ArgUnpacker::new(args);
    let title_val = unpacker.required(b"title")?;
    unpacker.end()?;
    let title_bytes = title_val.to_string(interpreter);
    let c_title = CString::new(title_bytes).map_err(|_| "set_window_title: title contains null byte".to_string())?;
    unsafe { SHM_SetWindowTitle(c_title.as_ptr()); }
    Ok(ShimValue::None)
}

fn shim_mouse_focus(
    _interpreter: &mut Interpreter,
    args: &ArgBundle,
) -> Result<ShimValue, String> {
    ArgUnpacker::new(args).end()?;
    let flags = unsafe { SHM_GetWindowFlags() };
    Ok(ShimValue::Bool(flags & 0x00000400 != 0))
}

fn shim_input_focus(
    _interpreter: &mut Interpreter,
    args: &ArgBundle,
) -> Result<ShimValue, String> {
    ArgUnpacker::new(args).end()?;
    let flags = unsafe { SHM_GetWindowFlags() };
    Ok(ShimValue::Bool(flags & 0x00000200 != 0))
}

/// xorshift64 PRNG state, seeded from system time on first fetch.
pub struct RandomState {
    state: u64,
}

impl Default for RandomState {
    fn default() -> Self {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0x9e3779b97f4a7c15);
        let seed = nanos ^ 0x9e3779b97f4a7c15;
        Self { state: if seed == 0 { 0xdeadbeefcafebabe } else { seed } }
    }
}

impl RandomState {
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    /// Uniform float in [0.0, 1.0).
    pub fn next_f32_unit(&mut self) -> f32 {
        // Top 24 bits — matches f32 mantissa width.
        (self.next_u64() >> 40) as f32 / ((1u32 << 24) as f32)
    }
}

fn shim_rand(interpreter: &mut Interpreter, args: &ArgBundle) -> Result<ShimValue, String> {
    let mut unpacker = ArgUnpacker::new(args);
    let n = args.args.len();
    let result = match n {
        0 => {
            unpacker.end()?;
            interpreter.fetch_mut::<RandomState>().next_f32_unit()
        }
        1 => {
            let hi = unpacker.required_number(b"hi")?;
            unpacker.end()?;
            interpreter.fetch_mut::<RandomState>().next_f32_unit() * hi
        }
        2 => {
            let lo = unpacker.required_number(b"lo")?;
            let hi = unpacker.required_number(b"hi")?;
            unpacker.end()?;
            let r = interpreter.fetch_mut::<RandomState>().next_f32_unit();
            lo + (hi - lo) * r
        }
        _ => return Err(format!("rand expects 0, 1, or 2 arguments; got {}", n)),
    };
    Ok(ShimValue::Float(result))
}

fn shim_randi(interpreter: &mut Interpreter, args: &ArgBundle) -> Result<ShimValue, String> {
    let mut unpacker = ArgUnpacker::new(args);
    let n = args.args.len();
    let result = match n {
        0 => {
            unpacker.end()?;
            (interpreter.fetch_mut::<RandomState>().next_u64() & 0x7fff_ffff) as i32
        }
        1 => {
            let hi = unpacker.required_int(b"hi")?;
            unpacker.end()?;
            if hi < 0 {
                return Err(format!("randi: hi {} must be non-negative", hi));
            }
            let span = hi as u64;
            (interpreter.fetch_mut::<RandomState>().next_u64() % span) as i32
        }
        2 => {
            let lo = unpacker.required_int(b"lo")?;
            let hi = unpacker.required_int(b"hi")?;
            unpacker.end()?;
            if lo > hi {
                return Err(format!("randi: lo {} must be <= hi {}", lo, hi));
            }
            let span = (hi as i64 - lo as i64) as u64;
            let r = interpreter.fetch_mut::<RandomState>().next_u64() % span;
            (lo as i64 + r as i64) as i32
        }
        _ => return Err(format!("randi expects 0, 1, or 2 arguments; got {}", n)),
    };
    Ok(ShimValue::Integer(result))
}

fn shim_play_square(interpreter: &mut Interpreter, args: &ArgBundle) -> Result<ShimValue, String> {
    let mut unpacker = ArgUnpacker::new(args);
    let freq = unpacker.required_number(b"freq")?;
    let duration = unpacker.required_number(b"duration")?;
    let duty = unpacker.optional_number(b"duty", 0.5)?;
    let amp = unpacker.optional_number(b"amp", 0.5)?;
    let attack = unpacker.optional_number(b"attack", 0.005)?;
    let decay = unpacker.optional_number(b"decay", 0.0)?;
    let sustain = unpacker.optional_number(b"sustain", 1.0)?;
    let release = unpacker.optional_number(b"release", 0.005)?;
    let delay = unpacker.optional_number(b"delay", 0.0)?;
    let pan = unpacker.optional_number(b"pan", 0.0)?;
    let bus = parse_bus(unpacker.optional(b"bus"))?;
    unpacker.end()?;
    let sl = interpreter.fetch_mut::<SoundList>();
    let voice_id = sl.alloc_voice_id();
    sl.items.push(SoundCmd::Square {
        voice_id,
        freq,
        duty: duty.clamp(0.0, 1.0),
        amp,
        duration,
        attack,
        decay,
        sustain: sustain.clamp(0.0, 1.0),
        release,
        delay,
        pan: pan.clamp(-1.0, 1.0),
        bus,
    });
    Ok(interpreter.mem.alloc_native(VoiceHandle { voice_id }))
}

fn shim_play_sine(interpreter: &mut Interpreter, args: &ArgBundle) -> Result<ShimValue, String> {
    let mut unpacker = ArgUnpacker::new(args);
    let freq = unpacker.required_number(b"freq")?;
    let duration = unpacker.required_number(b"duration")?;
    let amp = unpacker.optional_number(b"amp", 0.5)?;
    let attack = unpacker.optional_number(b"attack", 0.005)?;
    let decay = unpacker.optional_number(b"decay", 0.0)?;
    let sustain = unpacker.optional_number(b"sustain", 1.0)?;
    let release = unpacker.optional_number(b"release", 0.005)?;
    let delay = unpacker.optional_number(b"delay", 0.0)?;
    let pan = unpacker.optional_number(b"pan", 0.0)?;
    let bus = parse_bus(unpacker.optional(b"bus"))?;
    unpacker.end()?;
    let sl = interpreter.fetch_mut::<SoundList>();
    let voice_id = sl.alloc_voice_id();
    sl.items.push(SoundCmd::Sine {
        voice_id,
        freq,
        amp,
        duration,
        attack,
        decay,
        sustain: sustain.clamp(0.0, 1.0),
        release,
        delay,
        pan: pan.clamp(-1.0, 1.0),
        bus,
    });
    Ok(interpreter.mem.alloc_native(VoiceHandle { voice_id }))
}

fn shim_create_sample(
    interpreter: &mut Interpreter,
    args: &ArgBundle,
) -> Result<ShimValue, String> {
    let mut unpacker = ArgUnpacker::new(args);
    let samples_list = unpacker.required_list(interpreter, b"samples")?;
    let sample_rate = unpacker.required_int(b"sample_rate")?;
    unpacker.end()?;

    if sample_rate <= 0 {
        return Err(format!(
            "create_sample: sample_rate {} must be positive",
            sample_rate
        ));
    }

    let raw = samples_list.raw_data(&interpreter.mem);
    let mut samples: Vec<f32> = Vec::with_capacity(raw.len());
    for val in raw.iter() {
        samples.push(match unsafe { ShimValue::from_u64(*val) } {
            ShimValue::Float(f) => f,
            ShimValue::Integer(i) => i as f32,
            other => {
                return Err(format!(
                    "create_sample: non-numeric sample {:?}",
                    other
                ));
            }
        });
    }

    let handle = interpreter
        .fetch_mut::<SoundList>()
        .push_create_sample(sample_rate as u32, samples);
    Ok(interpreter.mem.alloc_native(handle))
}

fn shim_set_bus_gain(interpreter: &mut Interpreter, args: &ArgBundle) -> Result<ShimValue, String> {
    let mut unpacker = ArgUnpacker::new(args);
    let bus = parse_bus(Some(unpacker.required(b"bus")?))?;
    let gain = unpacker.required_number(b"gain")?;
    let ramp = unpacker.optional_number(b"ramp", DEFAULT_RAMP_SECS)?;
    unpacker.end()?;
    interpreter.fetch_mut::<SoundList>().items.push(SoundCmd::SetBusGain {
        bus,
        gain,
        ramp: ramp.max(0.0),
    });
    Ok(ShimValue::None)
}

fn shim_set_bus_lowpass(interpreter: &mut Interpreter, args: &ArgBundle) -> Result<ShimValue, String> {
    let mut unpacker = ArgUnpacker::new(args);
    let bus = parse_bus(Some(unpacker.required(b"bus")?))?;
    let cutoff = unpacker.required_number(b"cutoff")?;
    let q = unpacker.optional_number(b"q", std::f32::consts::FRAC_1_SQRT_2)?;
    let ramp = unpacker.optional_number(b"ramp", 0.01)?.max(0.0);
    unpacker.end()?;
    interpreter.fetch_mut::<SoundList>().items.push(SoundCmd::SetBusLowPass {
        bus,
        cutoff,
        q,
        ramp,
    });
    Ok(ShimValue::None)
}

fn shim_clear_bus_effects(interpreter: &mut Interpreter, args: &ArgBundle) -> Result<ShimValue, String> {
    let mut unpacker = ArgUnpacker::new(args);
    let bus = parse_bus(Some(unpacker.required(b"bus")?))?;
    unpacker.end()?;
    interpreter
        .fetch_mut::<SoundList>()
        .items
        .push(SoundCmd::ClearBusEffects { bus });
    Ok(ShimValue::None)
}

fn shim_reset_audio(interpreter: &mut Interpreter, args: &ArgBundle) -> Result<ShimValue, String> {
    let mut unpacker = ArgUnpacker::new(args);
    let fade_secs = unpacker.optional_number(b"fade", 0.05)?.max(0.0);
    unpacker.end()?;
    interpreter
        .fetch_mut::<SoundList>()
        .items
        .push(SoundCmd::ResetAudio { fade_secs });
    Ok(ShimValue::None)
}

fn shim_window_size(
    interpreter: &mut Interpreter,
    args: &ArgBundle,
) -> Result<ShimValue, String> {
    ArgUnpacker::new(args).end()?;

    let mut display_w = 0;
    let mut display_h = 0;
    unsafe {
        SHM_GetDrawableSize(&mut display_w as *mut i32, &mut display_h as *mut i32);
    }

    let new_lst_val = interpreter.mem.alloc_list();
    let new_lst = new_lst_val.list_mut(interpreter)?;

    new_lst.push(&mut interpreter.mem, ShimValue::Integer(display_w));
    new_lst.push(&mut interpreter.mem, ShimValue::Integer(display_h));

    Ok(new_lst_val)
}

fn shim_save_data(
    interpreter: &mut Interpreter,
    args: &ArgBundle,
) -> Result<ShimValue, String> {
    let mut unpacker = ArgUnpacker::new(args);
    let data = unpacker.required(b"data")?;
    let path = unpacker.required(b"path")?.to_string(interpreter);
    unpacker.end()?;

    let json = value_to_json(interpreter, data)?;
    let data = json.stringify().unwrap(); // This only fails for NaN and Inf?
    fs::write(&path, &data).map_err(|e| format!("{e:?}"))?;

    Ok(ShimValue::None)
}

fn value_to_json(
    interpreter: &Interpreter,
    data: ShimValue,
) -> Result<JsonValue, String> {
    Ok(
        JsonValue::Object(
            match data {
                ShimValue::None => [
                    ("type".to_string(), JsonValue::String("None".to_string())),
                    ("data".to_string(), JsonValue::Null),
                ].into_iter().collect(),
                ShimValue::Integer(i) => [
                    ("type".to_string(), JsonValue::String("int".to_string())),
                    ("data".to_string(), JsonValue::Number(i as f64)),
                ].into_iter().collect(),
                ShimValue::Float(f) => [
                    ("type".to_string(), JsonValue::String("float".to_string())),
                    ("data".to_string(), JsonValue::Number(f as f64)),
                ].into_iter().collect(),
                ShimValue::Bool(b) => [
                    ("type".to_string(), JsonValue::String("bool".to_string())),
                    ("data".to_string(), JsonValue::Boolean(b)),
                ].into_iter().collect(),
                s @ ShimValue::String(..) => [
                    ("type".to_string(), JsonValue::String("str".to_string())),
                    ("data".to_string(), JsonValue::String(String::from_utf8_lossy(s.string(interpreter)?).into_owned())),
                ].into_iter().collect(),
                ShimValue::Tuple(len, pos) => {
                    let pos = usize::from(pos);
                    let len = usize::from(len);
                    let mut lst = Vec::new();

                    for idx in pos..(pos+len) {
                        unsafe {
                            lst.push(
                                value_to_json(
                                    interpreter,
                                    ShimValue::from_u64(interpreter.mem.mem()[idx])
                                )?
                            );
                        }
                    }
                    [
                        ("type".to_string(), JsonValue::String("tuple".to_string())),
                        ("data".to_string(), JsonValue::Array(lst)),
                    ].into_iter().collect()
                },
                lst @ ShimValue::List(u24) => {
                    let lst = lst.list(interpreter)?;
                    let mut out = Vec::new();

                    for word in lst.raw_data(&interpreter.mem) {
                        unsafe {
                            out.push(
                                value_to_json(
                                    interpreter,
                                    ShimValue::from_u64(*word),
                                )?
                            );
                        }
                    }
                    [
                        ("type".to_string(), JsonValue::String("list".to_string())),
                        ("data".to_string(), JsonValue::Array(out)),
                    ].into_iter().collect()
                },
                d @ ShimValue::Dict(u24) => {
                    let d = d.dict(interpreter)?;
                    // We use a Vec since we want to preserve insertion order and we
                    // want to have non-string keys
                    let mut attrs: Vec<JsonValue> = Vec::new();
                    for entry in d.entries_array(interpreter) {
                        attrs.push(
                            JsonValue::Array(
                                vec![
                                    value_to_json(interpreter, entry.key)?,
                                    value_to_json(interpreter, entry.value)?,
                                ]
                            )
                        )
                    }
                    [
                        ("type".to_string(), JsonValue::String("dict".to_string())),
                        ("data".to_string(), JsonValue::Array(attrs)),
                    ].into_iter().collect()
                },
                ShimValue::Struct(def_pos, pos) => {
                    unsafe {
                        let mut attrs = HashMap::new();
                        let def: &StructDef = interpreter.mem.get(def_pos);
                        for (attr, loc) in def.lookup.iter() {
                            match loc {
                                StructAttribute::MemberInstanceOffset(offset) => {
                                    let val: ShimValue = *interpreter.mem.get(pos + *offset as u32);
                                    attrs.insert(String::from_utf8_lossy(attr).to_string(), value_to_json(interpreter, val)?);
                                }
                                StructAttribute::MethodDef(fn_pos) => (),
                            };
                        }
                        [
                            ("type".to_string(), JsonValue::String(String::from_utf8_lossy(&def.name).to_string())),
                            ("data".to_string(), JsonValue::Object(attrs)),
                        ].into_iter().collect()
                    }
                },
                other => return Err(format!("Can't JSON-ify {other:?}")),
            }
        )
    )
}

fn value_from_json(
    interpreter: &mut Interpreter,
    env: &mut Environment,
    data: &JsonValue,
) -> Result<ShimValue, String> {
    let obj: &HashMap<_, _> = data.get().ok_or(format!("JSON not an object"))?;
    let type_name: &String = obj.get("type").ok_or(format!("JSON has no type"))?.get::<String>().ok_or(format!("type not a string"))?;
    let data: &JsonValue = obj.get("data").ok_or(format!("JSON has no data"))?;
    match type_name.as_str() {
        "int" => {
            let val: &f64 = data.get().ok_or(format!("JSON data not a float(int)"))?;
            Ok(
                ShimValue::Integer(*val as i32)
            )
        },
        "float" => {
            let val: &f64 = data.get().ok_or(format!("JSON data not a float"))?;
            Ok(
                ShimValue::Float(*val as f32)
            )
        },
        "bool" => {
            let val: &bool = data.get().ok_or(format!("JSON data not a bool"))?;
            Ok(
                ShimValue::Bool(*val)
            )
        },
        "str" => {
            let val: &String = data.get().ok_or(format!("JSON data not a str"))?;
            Ok(
                interpreter.mem.alloc_str(val.as_bytes())
            )
        },
        "None" => {
            Ok(ShimValue::None)
        },
        "list" => {
            let val: &Vec<_> = data.get().ok_or(format!("JSON data not a list"))?;
            let items: Vec<ShimValue> = val.iter()
                .map(|item| value_from_json(interpreter, env, item))
                .collect::<Result<_, _>>()?;
            let lst = interpreter.mem.alloc_list();
            let lst_inner = lst.list_mut(interpreter)?;
            for item in items {
                lst_inner.push(&mut interpreter.mem, item);
            }
            Ok(lst)
        },
        "dict" => {
            let val: &Vec<_> = data.get().ok_or(format!("JSON data not an list(dict)"))?;
            let pairs: Vec<(ShimValue, ShimValue)> = val.iter()
                .map(|pair| {
                    let pair: &Vec<JsonValue> = pair.get().ok_or(format!("JSON dict item not a pair"))?;
                    if pair.len() != 2 {
                        return Err(format!("Got unexpected length {} for dict item", pair.len()));
                    }
                    let shim_key = value_from_json(interpreter, env, &pair[0])?;
                    let shim_value = value_from_json(interpreter, env, &pair[1])?;
                    Ok((shim_key, shim_value))
                })
                .collect::<Result<_, String>>()?;
            let dict = interpreter.mem.alloc_dict();
            let dict_inner = dict.dict_mut(interpreter)?;
            for (shim_key, shim_value) in pairs {
                dict_inner.set(interpreter, shim_key, shim_value)?;
            }
            Ok(dict)
        },
        "tuple" => {
            // TODO: It seems like the size of the tuple should be encoded in the type?
            let val: &Vec<_> = data.get().ok_or(format!("JSON data not a list(tuple)"))?;
            let mut vec_of_shimvals = Vec::new();
            for item in val.iter() {
                vec_of_shimvals.push(
                    value_from_json(interpreter, env, item)?
                );
            }
            Ok(interpreter.mem.alloc_tuple(&vec_of_shimvals))
        },
        _ => {
            let val: &HashMap<_, _> = data.get().ok_or(format!("JSON data not an object"))?;
            let ty = env.get(interpreter, type_name.as_bytes()).ok_or(format!("env has no type_name {type_name}"))?;
            let kwargs: Vec<(Ident, ShimValue)> = val.iter()
                .map(|(key, value)| {
                    let shim_key = key.as_bytes().to_vec();
                    let shim_value = value_from_json(interpreter, env, value)?;
                    Ok((shim_key, shim_value))
                })
                .collect::<Result<_, String>>()?;
            let mut args = ArgBundle {
                kwargs,
                args: Vec::new(),
            };
            match ty.call(interpreter, &mut args) {
                Ok(CallResult::ReturnValue(val)) => Ok(val),
                Ok(CallResult::PC(pc, captured_scope)) => {
                    let mut new_env = Environment::with_scope(captured_scope);
                    match interpreter.execute_bytecode_extended(
                        &mut (pc as usize),
                        args,
                        &mut new_env,
                    ) {
                        Ok(val) => Ok(val),
                        Err(e) => Err(e),
                    }
                }
                Err(e) => Err(e),
            }

        }
    }
}

fn shim_load_data(
    interpreter: &mut Interpreter,
    args: &ArgBundle,
) -> Result<ShimValue, String> {
    let mut unpacker = ArgUnpacker::new(args);
    // NOTE: It feels like we shouldn't need to specify the type, but this
    // gives us the captured environment we need to get the type definitions.
    // We could have `struct Foo` defined in multiple places, but that captured
    // environment gives us the exact type to use.
    let mut env = match unpacker.required(b"type")? {
        ShimValue::StructDef(struct_def_pos) => {
            let struct_def: &StructDef = unsafe { interpreter.mem.get(struct_def_pos) };
            if let Some(StructAttribute::MethodDef(fn_pos)) = struct_def.find(b"__init__") {
                let shim_fn: &ShimFn = unsafe { interpreter.mem.get(fn_pos) };
                Environment::with_scope(shim_fn.captured_scope)
            } else {
                return Err("INTERNAL: no __init__ on StructDef".to_string());
            }
        },
        otherwise => return Err(format!("Can't `load_data` for value {otherwise:?}")),
    };
    let path = unpacker.required(b"path")?.to_string(interpreter);
    let default = unpacker.optional(b"default");
    let default_fn = unpacker.optional(b"default_fn");
    unpacker.end()?;

    if default.is_some() && default_fn.is_some() {
        return Err(format!("Can't provide 'default' and 'default_fn' to load_data"));
    }

    match (fs::read_to_string(&path), default, default_fn) {
        (Ok(contents), _, _) => {
            let data: JsonValue = contents.parse().map_err(|e| format!("Could not parse JSON: {e:?}"))?;
            value_from_json(interpreter, &mut env, &data)
        },
        (Err(_), Some(val), _) => Ok(val),
        (Err(_), _, Some(func)) => {
            let mut args = ArgBundle::new();
            match func.call(interpreter, &mut args)? {
                CallResult::ReturnValue(val) => Ok(val),
                CallResult::PC(pc, captured_scope) => {
                    let mut new_env = Environment::with_scope(captured_scope);
                    interpreter.execute_bytecode_extended(&mut (pc as usize), args, &mut new_env)
                }
            }
        },
        (Err(e), None, None) => Err(format!("Could not read path {path}: {e:?}")),
    }
}

fn load_script(bytes: &[u8]) -> Result<(Interpreter, Environment, ShimValue), String> {
    let ast = shimlang::ast_from_text(bytes)?;
    let program = shimlang::compile_ast(&ast)?;
    let mut interpreter = shimlang::Interpreter::create(&shimlang::Config::default(), program);
    let mut env = shimlang::Environment::new_with_builtins(&mut interpreter);

    let builtins: &[(&[u8], NativeFn)] = &[
        (b"ig_begin", shim_ig_begin),
        (b"ig_end", shim_ig_end),
        (b"ig_text", shim_ig_text),
        (b"draw_rect", shim_draw_rect),
        (b"draw_text", shim_draw_text),
        (b"text_size", shim_text_size),
        (b"create_texture", shim_create_texture),
        (b"Rect", shim_rect),
        (b"window_size", shim_window_size),
        (b"mouse_pos", shim_mouse_pos),
        (b"mouse_pressed", shim_mouse_pressed),
        (b"mouse_just_pressed", shim_mouse_just_pressed),
        (b"mouse_just_released", shim_mouse_just_released),
        (b"show_cursor", shim_show_cursor),
        (b"hide_cursor", shim_hide_cursor),
        (b"set_window_title", shim_set_window_title),
        (b"mouse_focus", shim_mouse_focus),
        (b"input_focus", shim_input_focus),
        (b"rand", shim_rand),
        (b"randi", shim_randi),
        (b"play_square", shim_play_square),
        (b"play_sine", shim_play_sine),
        (b"create_sample", shim_create_sample),
        (b"set_bus_gain", shim_set_bus_gain),
        (b"set_bus_lowpass", shim_set_bus_lowpass),
        (b"clear_bus_effects", shim_clear_bus_effects),
        (b"reset_audio", shim_reset_audio),
        (b"save_data", shim_save_data),
        (b"load_data", shim_load_data),
    ];
    for (name, func) in builtins {
        let position = interpreter.mem.alloc_and_set(
            *func,
            &format!("builtin imgui::{}", std::str::from_utf8(name).unwrap()),
        );
        env.insert_new(
            &mut interpreter,
            name.to_vec(),
            ShimValue::NativeFn(position),
        );
    }

    let key_val = interpreter.mem.alloc_native(KeyMap);
    env.insert_new(&mut interpreter, b"key".to_vec(), key_val);
    env.insert_new(&mut interpreter, b"delta".to_vec(), ShimValue::Float(0.0));

    let perf_module = interpreter.mem.alloc_native(PerfModule);
    env.insert_new(&mut interpreter, b"perf".to_vec(), perf_module);

    let mouse_buttons: &[(&[u8], i32)] = &[
        (b"MOUSE_LEFT", 1),
        (b"MOUSE_MIDDLE", 2),
        (b"MOUSE_RIGHT", 3),
        (b"MOUSE_X1", 4),
        (b"MOUSE_X2", 5),
    ];
    for (name, value) in mouse_buttons {
        env.insert_new(&mut interpreter, name.to_vec(), ShimValue::Integer(*value));
    }

    let mut pc = 0;
    interpreter.execute_bytecode_extended(&mut pc, shimlang::ArgBundle::new(), &mut env)?;
    interpreter.gc(&env);

    let loop_fn = match env.get(&mut interpreter, b"loop") {
        Some(func @ ShimValue::Fn(_)) => func,
        None => return Err("No loop function found".to_string()),
        _ => return Err("Identifier 'loop' is not a function".to_string()),
    };

    Ok((interpreter, env, loop_fn))
}

fn call_loop_fn(
    interpreter: &mut Interpreter,
    env: &mut Environment,
    loop_fn: ShimValue,
) -> Result<f32, String> {
    match loop_fn.call(interpreter, &mut shimlang::ArgBundle::new()) {
        Ok(CallResult::ReturnValue(_)) => {
            let gc_start = std::time::Instant::now();
            interpreter.gc(env);
            Ok(gc_start.elapsed().as_secs_f32())
        }
        Ok(CallResult::PC(pc, captured_scope)) => {
            let mut new_env = Environment::with_scope(captured_scope);
            match interpreter.execute_bytecode_extended(
                &mut (pc as usize),
                shimlang::ArgBundle::new(),
                &mut new_env,
            ) {
                Err(msg) => Err(msg),
                Ok(_) => {
                    let gc_start = std::time::Instant::now();
                    interpreter.gc(env);
                    Ok(gc_start.elapsed().as_secs_f32())
                }
            }
        }
        Err(msg) => Err(msg),
    }
}

impl ScriptBridge {
    pub fn new() -> Self {
        let (interpreter, env, loop_fn) =
            load_script(b"fn loop() {}").expect("Should be able to load hardcoded script");
        let script_path = "game.shm".to_string();

        #[cfg(not(target_arch = "wasm32"))]
        {
            let (request_tx, request_rx) = channel();
            let (response_tx, response_rx) = channel();
            let (watcher_tx, watcher_rx) = channel();

            std::thread::spawn(move || {
                script_thread_logic(request_rx, response_tx);
            });

            std::thread::spawn({
                let path = script_path.clone();
                move || file_watcher_logic(path, watcher_tx)
            });

            Self {
                state: BridgeState::Paused(Box::new(interpreter), env, loop_fn),
                interpreter_errors: Vec::new(),
                draw_list: Vec::new(),
                sound_list: Vec::new(),
                last_gc_time: 0.0,
                script_path,
                tx: request_tx,
                rx: response_rx,
                watcher_rx,
            }
        }

        #[cfg(target_arch = "wasm32")]
        {
            Self {
                state: BridgeState::Paused(Box::new(interpreter), env, loop_fn),
                interpreter_errors: Vec::new(),
                draw_list: Vec::new(),
                sound_list: Vec::new(),
                last_gc_time: 0.0,
                script_path,
            }
        }
    }

    pub fn step(&mut self, keys: &[u8], last_keys: &[u8], delta: f32, perf: PerfTimer, finished: Vec<u32>) {
        let _zone = zone_scoped!("Run interpreter");
        #[cfg(not(target_arch = "wasm32"))]
        {
            // Apply any script reloads from the watcher thread
            while let Ok(msg) = self.watcher_rx.try_recv() {
                match msg {
                    ScriptWatcherMessage::Loaded(interpreter, env, loop_fn) => {
                        self.state = BridgeState::Paused(Box::new(interpreter), env, loop_fn);
                        self.interpreter_errors = Vec::new();
                        crate::audio::submit(std::iter::once(SoundCmd::ResetAudio { fade_secs: 0.05 }));
                    }
                    ScriptWatcherMessage::Error(msg) => {
                        self.interpreter_errors.push(msg);
                    }
                }
            }

            if self.interpreter_errors.is_empty() {
                let state = mem::replace(&mut self.state, BridgeState::Running);
                match state {
                    BridgeState::Running => panic!("Somehow the interpreter is running"),
                    BridgeState::Paused(interpreter, env, loop_fn) => {
                        self.tx
                            .send(ScriptRequest::ExecuteLoop(
                                *interpreter,
                                env,
                                loop_fn,
                                keys.to_vec(),
                                last_keys.to_vec(),
                                delta,
                                perf,
                                finished,
                            ))
                            .unwrap();
                    }
                }

                {
                    let _zone = zone_scoped!("Wait for interpreter response");
                    match self.rx.recv() {
                        Ok(ScriptResponse::Error(interpreter, env, loop_fn, msg)) => {
                            self.state = BridgeState::Paused(Box::new(interpreter), env, loop_fn);
                            self.interpreter_errors.push(msg);
                        }
                        Ok(ScriptResponse::LoopComplete(interpreter, env, loop_fn, draw_list, sound_list, gc_time)) => {
                            self.state = BridgeState::Paused(Box::new(interpreter), env, loop_fn);
                            self.draw_list = draw_list;
                            self.sound_list = sound_list;
                            self.last_gc_time = gc_time;
                        }
                        Err(e) => {
                            dbg!(e);
                            panic!("Received error for recv");
                        }
                    }
                }
            }
        }

        if !self.interpreter_errors.is_empty() {
            let mut open = true;
            unsafe {
                if super::igBegin(
                    c"Shimlang Errors".as_ptr(),
                    &mut open as *mut bool,
                    IMGUI_WINDOW_FLAGS_NO_FOCUS_ON_APPEARING,
                ) {
                    for err in self.interpreter_errors.iter() {
                        super::igTextColoredBC(
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            0.5,
                            0.5,
                            0.5,
                            0.5,
                            CString::new(err.to_string()).unwrap().as_ptr(),
                        );
                    }
                    super::igEnd();
                }
            }
        }

        #[cfg(target_arch = "wasm32")]
        {
            if self.interpreter_errors.is_empty() {
                if let BridgeState::Paused(ref mut interpreter, ref mut env, loop_fn) = self.state {
                    let ks = interpreter.fetch_mut::<KeyState>();
                    ks.held_time.resize(keys.len(), 0.0);
                    for (i, &pressed) in keys.iter().enumerate() {
                        if pressed == 1 {
                            ks.held_time[i] += delta;
                        } else {
                            ks.held_time[i] = 0.0;
                        }
                    }
                    ks.delta = delta;
                    ks.keys = keys.to_vec();
                    ks.last_keys = last_keys.to_vec();
                    let buttons = unsafe {
                        SDL_GetMouseState(std::ptr::null_mut(), std::ptr::null_mut())
                    };
                    let ms = interpreter.fetch_mut::<MouseState>();
                    ms.last_buttons = ms.buttons;
                    ms.buttons = buttons;
                    *interpreter.fetch_mut::<PerfTimer>() = perf;
                    interpreter.fetch_mut::<SoundList>().finished.extend(finished.iter().copied());
                    match call_loop_fn(interpreter, env, loop_fn) {
                        Ok(gc_time) => self.last_gc_time = gc_time,
                        Err(msg) => self.interpreter_errors.push(msg),
                    }
                    self.draw_list = interpreter
                        .fetch_mut::<DrawList>()
                        .items
                        .drain(..)
                        .collect();
                    self.sound_list = interpreter
                        .fetch_mut::<SoundList>()
                        .items
                        .drain(..)
                        .collect();
                }
            }
        }
    }

    pub fn errors(&self) -> &[String] {
        &self.interpreter_errors
    }

    pub fn debug_window(&mut self, shimlang_debug_window: &mut shimlang_imgui::Navigation) {
        let _zone = zone_scoped!("ShimLang Debug");
        match &mut self.state {
            #[cfg(not(target_arch = "wasm32"))]
            BridgeState::Running => {}
            BridgeState::Paused(interpreter, env, _loop_fn) => {
                shimlang_debug_window.debug_window(interpreter, env);
            }
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn file_watcher_logic(script_path: String, tx: Sender<ScriptWatcherMessage>) {
    use std::time::{Duration, SystemTime};
    let mut mtime = SystemTime::UNIX_EPOCH;
    loop {
        {
            let _zone = zone_scoped!("Script update poll wait");
            std::thread::sleep(Duration::from_millis(200));
        }
        {
            let _zone = zone_scoped!("Get script update");
            match fs::metadata(&script_path) {
                Ok(metadata) => match metadata.modified() {
                    Ok(time) => {
                        if mtime != time {
                            mtime = time;
                            match fs::read(&script_path) {
                                Ok(bytes) => {
                                    println!("{}", debug_u8s(&bytes));
                                    let msg = match load_script(&bytes) {
                                        Ok((interp, env, loop_fn)) => {
                                            ScriptWatcherMessage::Loaded(interp, env, loop_fn)
                                        }
                                        Err(msg) => ScriptWatcherMessage::Error(msg),
                                    };
                                    if tx.send(msg).is_err() {
                                        break;
                                    }
                                }
                                Err(_) => {
                                    if tx
                                        .send(ScriptWatcherMessage::Error(format!(
                                            "Could not read {}",
                                            script_path
                                        )))
                                        .is_err()
                                    {
                                        break;
                                    }
                                }
                            }
                        }
                    }
                    Err(e) => {
                        if tx
                            .send(ScriptWatcherMessage::Error(format!(
                                "Could not get modification time for {}: {}",
                                script_path, e
                            )))
                            .is_err()
                        {
                            break;
                        }
                    }
                },
                Err(_) => {} // File not found yet; silently wait
            }
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn script_thread_logic(rx: Receiver<ScriptRequest>, tx: Sender<ScriptResponse>) {
    loop {
        if let Ok(request) = rx.recv() {
            match request {
                ScriptRequest::ExecuteLoop(mut interpreter, mut env, loop_fn, keys, last_keys, delta, perf, finished) => {
                    let ks = interpreter.fetch_mut::<KeyState>();
                    ks.held_time.resize(keys.len(), 0.0);
                    for (i, &pressed) in keys.iter().enumerate() {
                        if pressed == 1 {
                            ks.held_time[i] += delta;
                        } else {
                            ks.held_time[i] = 0.0;
                        }
                    }
                    ks.delta = delta;
                    ks.keys = keys;
                    ks.last_keys = last_keys;
                    let buttons = unsafe {
                        SDL_GetMouseState(std::ptr::null_mut(), std::ptr::null_mut())
                    };
                    let ms = interpreter.fetch_mut::<MouseState>();
                    ms.last_buttons = ms.buttons;
                    ms.buttons = buttons;
                    *interpreter.fetch_mut::<PerfTimer>() = perf;
                    interpreter.fetch_mut::<SoundList>().finished.extend(finished.iter().copied());
                    env.update(&mut interpreter, b"delta", ShimValue::Float(delta)).expect("delta should be in env");
                    tx.send(match call_loop_fn(&mut interpreter, &mut env, loop_fn) {
                        Ok(gc_time) => {
                            let draw_list = interpreter
                                .fetch_mut::<DrawList>()
                                .items
                                .drain(..)
                                .collect();
                            let sound_list = interpreter
                                .fetch_mut::<SoundList>()
                                .items
                                .drain(..)
                                .collect();
                            ScriptResponse::LoopComplete(interpreter, env, loop_fn, draw_list, sound_list, gc_time)
                        }
                        Err(msg) => ScriptResponse::Error(interpreter, env, loop_fn, msg),
                    })
                    .unwrap();
                }
            }
        }
    }
}
