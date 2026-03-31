#![allow(static_mut_refs)]
use macroquad::prelude::*;
use macroquad::audio::Sound;
use std::mem;
use std::sync::mpsc::{channel, Receiver, Sender};
use std::fs;
use std::time::SystemTime;
use std::any::Any;

use shimlang::{ShimValue, Environment, Interpreter, ShimNative};
use shimlang::runtime::{ArgUnpacker, NativeFn, CallResult};
use shimlang::ArgBundle;
use shimlang::lex::debug_u8s;

use macroquad::audio::{load_sound_from_bytes, play_sound_once};

use std::collections::HashMap;
use std::mem::MaybeUninit;
use std::sync::Once;


// "Manager" is a bad name...
struct SoundManager {
    next_id: u32,
    map: HashMap<u32, Sound>,
    queue: Vec<SoundManagerQueueItem>,
}

enum SoundManagerQueueItem {
    NewBloop(u32, f32, f32),
    Play(u32),
}

impl SoundManager {
    fn new() -> Self {
        Self {
            next_id: 0,
            map: HashMap::new(),
            queue: Vec::new(),
        }
    }

    fn play(&mut self, handle: &SoundHandle) {
        if let Some(sound) = self.map.get(&handle.id) {
            // We've already loaded the sound
            play_sound_once(sound);
            return;
        } else {
            // Check if the load is enqueued
            for item in self.queue.iter() {
                if let SoundManagerQueueItem::NewBloop(id, _start_freq, _end_freq) = item {
                    // Only add the `Play` action if the id is in the process of being loaded
                    self.queue.push(SoundManagerQueueItem::Play(handle.id));
                    return;
                }
            }
        }
        eprintln!("No sound for handle {:?}", handle);
    }

    fn new_bloop(&mut self, start_freq: f32, end_freq: f32) -> SoundHandle {
        let id = self.next_id;
        self.next_id += 1;

        self.queue.push(SoundManagerQueueItem::NewBloop(id, start_freq, end_freq));
        SoundHandle { id }
    }

}

#[inline(always)]
fn sound_manager() -> &'static mut SoundManager {
    unsafe {
        SOUND_MANAGER.assume_init_mut()
    }
}

static mut SOUND_MANAGER: MaybeUninit<SoundManager> = MaybeUninit::uninit();

#[derive(Debug)]
struct SoundHandle {
    id: u32,
}

impl ShimNative for SoundHandle {
    fn get_attr(&self, self_as_val: &ShimValue, interpreter: &mut Interpreter, ident: &[u8]) -> Result<ShimValue, String> {
        if ident == b"play" {
            fn shim_sound_handle_play(interpreter: &mut Interpreter, args: &ArgBundle) -> Result<ShimValue, String> {
                if args.args.len() != 1 {
                    return Err(format!("Can't provide positional args to SoundHandle.next()"));
                }
                let handle: &mut SoundHandle = args.args[0].as_native(interpreter)?;
                sound_manager().play(handle);
                Ok(ShimValue::None)
            }

            Ok(interpreter.mem.alloc_bound_native_fn(self_as_val, shim_sound_handle_play))
        } else {
            Err("Can only play on a SoundHandle".to_string())
        }
    }

    fn as_any_mut(&mut self) -> &mut dyn Any where Self: Sized {
        self
    }

    fn gc_vals(&self) -> Vec<ShimValue> {
        Vec::new()
    }
}

fn shim_new_bloop(interpreter: &mut Interpreter, args: &ArgBundle) -> Result<ShimValue, String> {
    let mut unpacker = ArgUnpacker::new(args);
    Ok(interpreter.mem.alloc_native(
        sound_manager().new_bloop(
            unpacker.required_number(b"start_freq")?,
            unpacker.required_number(b"end_freq")?,
        )
    ))
}

fn shim_draw_circle(_interpreter: &mut Interpreter, args: &ArgBundle) -> Result<ShimValue, String> {
    let mut unpacker = ArgUnpacker::new(args);
    draw_circle(
        unpacker.required_number(b"x")?,
        unpacker.required_number(b"y")?,
        unpacker.required_number(b"radius")?,
        Color::new(
            unpacker.required_number(b"r")?,
            unpacker.required_number(b"g")?,
            unpacker.required_number(b"b")?,
            unpacker.required_number(b"a")?,
        ),
    );
    unpacker.end()?;
    Ok(ShimValue::None)
}

fn shim_clear_background(_interpreter: &mut Interpreter, args: &ArgBundle) -> Result<ShimValue, String> {
    let mut unpacker = ArgUnpacker::new(args);
    clear_background(
        Color::new(
            unpacker.required_number(b"r")?,
            unpacker.required_number(b"g")?,
            unpacker.required_number(b"b")?,
            unpacker.required_number(b"a")?,
        ),
    );
    unpacker.end()?;
    Ok(ShimValue::None)
}

fn shim_draw_line(_interpreter: &mut Interpreter, args: &ArgBundle) -> Result<ShimValue, String> {
    let mut unpacker = ArgUnpacker::new(args);
    draw_line(
        unpacker.required_number(b"x0")?,
        unpacker.required_number(b"y0")?,
        unpacker.required_number(b"x1")?,
        unpacker.required_number(b"y1")?,
        unpacker.required_number(b"thickness")?,
        Color::new(
            unpacker.required_number(b"r")?,
            unpacker.required_number(b"g")?,
            unpacker.required_number(b"b")?,
            unpacker.required_number(b"a")?,
        ),
    );
    unpacker.end()?;
    Ok(ShimValue::None)
}

fn key_code_from_name(name: &[u8]) -> Option<KeyCode> {
    match name {
        b"Space" => Some(KeyCode::Space),
        b"Apostrophe" => Some(KeyCode::Apostrophe),
        b"Comma" => Some(KeyCode::Comma),
        b"Minus" => Some(KeyCode::Minus),
        b"Period" => Some(KeyCode::Period),
        b"Slash" => Some(KeyCode::Slash),
        b"Key0" => Some(KeyCode::Key0),
        b"Key1" => Some(KeyCode::Key1),
        b"Key2" => Some(KeyCode::Key2),
        b"Key3" => Some(KeyCode::Key3),
        b"Key4" => Some(KeyCode::Key4),
        b"Key5" => Some(KeyCode::Key5),
        b"Key6" => Some(KeyCode::Key6),
        b"Key7" => Some(KeyCode::Key7),
        b"Key8" => Some(KeyCode::Key8),
        b"Key9" => Some(KeyCode::Key9),
        b"Semicolon" => Some(KeyCode::Semicolon),
        b"Equal" => Some(KeyCode::Equal),
        b"A" => Some(KeyCode::A),
        b"B" => Some(KeyCode::B),
        b"C" => Some(KeyCode::C),
        b"D" => Some(KeyCode::D),
        b"E" => Some(KeyCode::E),
        b"F" => Some(KeyCode::F),
        b"G" => Some(KeyCode::G),
        b"H" => Some(KeyCode::H),
        b"I" => Some(KeyCode::I),
        b"J" => Some(KeyCode::J),
        b"K" => Some(KeyCode::K),
        b"L" => Some(KeyCode::L),
        b"M" => Some(KeyCode::M),
        b"N" => Some(KeyCode::N),
        b"O" => Some(KeyCode::O),
        b"P" => Some(KeyCode::P),
        b"Q" => Some(KeyCode::Q),
        b"R" => Some(KeyCode::R),
        b"S" => Some(KeyCode::S),
        b"T" => Some(KeyCode::T),
        b"U" => Some(KeyCode::U),
        b"V" => Some(KeyCode::V),
        b"W" => Some(KeyCode::W),
        b"X" => Some(KeyCode::X),
        b"Y" => Some(KeyCode::Y),
        b"Z" => Some(KeyCode::Z),
        b"LeftBracket" => Some(KeyCode::LeftBracket),
        b"Backslash" => Some(KeyCode::Backslash),
        b"RightBracket" => Some(KeyCode::RightBracket),
        b"GraveAccent" => Some(KeyCode::GraveAccent),
        b"World1" => Some(KeyCode::World1),
        b"World2" => Some(KeyCode::World2),
        b"Escape" => Some(KeyCode::Escape),
        b"Enter" => Some(KeyCode::Enter),
        b"Tab" => Some(KeyCode::Tab),
        b"Backspace" => Some(KeyCode::Backspace),
        b"Insert" => Some(KeyCode::Insert),
        b"Delete" => Some(KeyCode::Delete),
        b"Right" => Some(KeyCode::Right),
        b"Left" => Some(KeyCode::Left),
        b"Down" => Some(KeyCode::Down),
        b"Up" => Some(KeyCode::Up),
        b"PageUp" => Some(KeyCode::PageUp),
        b"PageDown" => Some(KeyCode::PageDown),
        b"Home" => Some(KeyCode::Home),
        b"End" => Some(KeyCode::End),
        b"CapsLock" => Some(KeyCode::CapsLock),
        b"ScrollLock" => Some(KeyCode::ScrollLock),
        b"NumLock" => Some(KeyCode::NumLock),
        b"PrintScreen" => Some(KeyCode::PrintScreen),
        b"Pause" => Some(KeyCode::Pause),
        b"F1" => Some(KeyCode::F1),
        b"F2" => Some(KeyCode::F2),
        b"F3" => Some(KeyCode::F3),
        b"F4" => Some(KeyCode::F4),
        b"F5" => Some(KeyCode::F5),
        b"F6" => Some(KeyCode::F6),
        b"F7" => Some(KeyCode::F7),
        b"F8" => Some(KeyCode::F8),
        b"F9" => Some(KeyCode::F9),
        b"F10" => Some(KeyCode::F10),
        b"F11" => Some(KeyCode::F11),
        b"F12" => Some(KeyCode::F12),
        b"F13" => Some(KeyCode::F13),
        b"F14" => Some(KeyCode::F14),
        b"F15" => Some(KeyCode::F15),
        b"F16" => Some(KeyCode::F16),
        b"F17" => Some(KeyCode::F17),
        b"F18" => Some(KeyCode::F18),
        b"F19" => Some(KeyCode::F19),
        b"F20" => Some(KeyCode::F20),
        b"F21" => Some(KeyCode::F21),
        b"F22" => Some(KeyCode::F22),
        b"F23" => Some(KeyCode::F23),
        b"F24" => Some(KeyCode::F24),
        b"F25" => Some(KeyCode::F25),
        b"Kp0" => Some(KeyCode::Kp0),
        b"Kp1" => Some(KeyCode::Kp1),
        b"Kp2" => Some(KeyCode::Kp2),
        b"Kp3" => Some(KeyCode::Kp3),
        b"Kp4" => Some(KeyCode::Kp4),
        b"Kp5" => Some(KeyCode::Kp5),
        b"Kp6" => Some(KeyCode::Kp6),
        b"Kp7" => Some(KeyCode::Kp7),
        b"Kp8" => Some(KeyCode::Kp8),
        b"Kp9" => Some(KeyCode::Kp9),
        b"KpDecimal" => Some(KeyCode::KpDecimal),
        b"KpDivide" => Some(KeyCode::KpDivide),
        b"KpMultiply" => Some(KeyCode::KpMultiply),
        b"KpSubtract" => Some(KeyCode::KpSubtract),
        b"KpAdd" => Some(KeyCode::KpAdd),
        b"KpEnter" => Some(KeyCode::KpEnter),
        b"KpEqual" => Some(KeyCode::KpEqual),
        b"LeftShift" => Some(KeyCode::LeftShift),
        b"LeftControl" => Some(KeyCode::LeftControl),
        b"LeftAlt" => Some(KeyCode::LeftAlt),
        b"LeftSuper" => Some(KeyCode::LeftSuper),
        b"RightShift" => Some(KeyCode::RightShift),
        b"RightControl" => Some(KeyCode::RightControl),
        b"RightAlt" => Some(KeyCode::RightAlt),
        b"RightSuper" => Some(KeyCode::RightSuper),
        b"Menu" => Some(KeyCode::Menu),
        b"Back" => Some(KeyCode::Back),
        b"Unknown" => Some(KeyCode::Unknown),
        _ => None,
    }
}

#[derive(Debug)]
struct KeyMap;

impl ShimNative for KeyMap {
    fn get_attr(&self, _self_as_val: &ShimValue, interpreter: &mut Interpreter, ident: &[u8]) -> Result<ShimValue, String> {
        match key_code_from_name(ident) {
            Some(code) => Ok(interpreter.mem.alloc_native(KeyValue { code })),
            None => Err(format!("Unknown key: {}", debug_u8s(ident))),
        }
    }

    fn as_any_mut(&mut self) -> &mut dyn Any where Self: Sized {
        self
    }

    fn gc_vals(&self) -> Vec<ShimValue> {
        Vec::new()
    }
}

#[derive(Debug)]
struct KeyValue {
    code: KeyCode,
}

impl ShimNative for KeyValue {
    fn get_attr(&self, _self_as_val: &ShimValue, _interpreter: &mut Interpreter, ident: &[u8]) -> Result<ShimValue, String> {
        if ident == b"is_down" {
            Ok(ShimValue::Bool(is_key_down(self.code)))
        } else {
            Err(format!("KeyValue only has 'is_down', not '{}'", debug_u8s(ident)))
        }
    }

    fn as_any_mut(&mut self) -> &mut dyn Any where Self: Sized {
        self
    }

    fn gc_vals(&self) -> Vec<ShimValue> {
        Vec::new()
    }
}

fn shim_draw_rectangle(_interpreter: &mut Interpreter, args: &ArgBundle) -> Result<ShimValue, String> {
    let mut unpacker = ArgUnpacker::new(args);
    draw_rectangle(
        unpacker.required_number(b"x")?,
        unpacker.required_number(b"y")?,
        unpacker.required_number(b"w")?,
        unpacker.required_number(b"h")?,
        Color::new(
            unpacker.required_number(b"r")?,
            unpacker.required_number(b"g")?,
            unpacker.required_number(b"b")?,
            unpacker.required_number(b"a")?,
        ),
    );
    unpacker.end()?;
    Ok(ShimValue::None)
}

fn load_script(text: &[u8]) -> Result<(Interpreter, Environment, ShimValue), String> {
    let interpreter_config = shimlang::Config::default();
    let ast = shimlang::ast_from_text(text)?;

    let program = shimlang::compile_ast(&ast)?;
    let mut interpreter = shimlang::Interpreter::create(&shimlang::Config::default(), program);
    let mut env = shimlang::Environment::new_with_builtins(&mut interpreter);

    let builtins: &[(&[u8], Box<NativeFn>)] = &[
        (b"draw_circle", Box::new(shim_draw_circle)),
        (b"draw_rectangle", Box::new(shim_draw_rectangle)),
        (b"draw_line", Box::new(shim_draw_line)),
        (b"clear_background", Box::new(shim_clear_background)),
        (b"new_bloop", Box::new(shim_new_bloop)),
    ];

    for (name, func) in builtins {
        let position = interpreter.mem.alloc_and_set(**func, &format!("builtin func {}", debug_u8s(name)));
        env.insert_new(&mut interpreter, name.to_vec(), ShimValue::NativeFn(position));
    }

    let key_val = interpreter.mem.alloc_native(KeyMap);
    env.insert_new(&mut interpreter, b"key".to_vec(), key_val);

    let mut pc = 0;
    interpreter.execute_bytecode_extended(&mut pc, shimlang::ArgBundle::new(), &mut env)?;
    interpreter.gc(&env);

    // Technically this is an external reference that should be passed to the
    // gc as a root, but since it's in the `env` it should already be marked.
    let loop_fn = match env.get(&mut interpreter, b"loop") {
        Some(func @ ShimValue::Fn(_)) => {
            func
        },
        None => {
            return Err("No loop function found".to_string());
        },
        _ => {
            return Err("Identifier 'loop' is not a function".to_string());
        },
    };

    Ok((interpreter, env, loop_fn))
}

fn generate_bloop_wav(start_freq: f32, end_freq: f32) -> Vec<u8> {
    let sample_rate = 44100u32;
    let duration_secs = 0.15;
    let duration_secs_inv = (0.15_f32).recip();
    let num_samples = (sample_rate as f32 * duration_secs) as usize;
    let mut samples = Vec::with_capacity(num_samples);

    for i in 0..num_samples {
        let t = i as f32 / sample_rate as f32;
        let freq = start_freq // Starts at start_frew
            + ( // This bracket is equal to 0 at t=0
                (end_freq - start_freq) *
                t * duration_secs_inv // When t=duration_secs the bracket equals (end_freq - start_freq)
            );
        let sample = (t * freq * 2.0 * std::f32::consts::PI).sin();
        
        // Convert f32 sample to i16 (PCM 16-bit)
        let amplitude = i16::MAX as f32 * 0.5; // 50% volume
        let pcm_sample = (sample * amplitude) as i16;
        samples.extend_from_slice(&pcm_sample.to_le_bytes());
    }

    create_wav_container(samples, sample_rate)
}

fn create_wav_container(data: Vec<u8>, sample_rate: u32) -> Vec<u8> {
    let mut wav = Vec::new();
    let data_len = data.len() as u32;

    // RIFF Header
    wav.extend_from_slice(b"RIFF");
    wav.extend_from_slice(&(36 + data_len).to_le_bytes()); // Total file size - 8
    wav.extend_from_slice(b"WAVE");

    // fmt chunk (Metadata)
    wav.extend_from_slice(b"fmt ");
    wav.extend_from_slice(&16u32.to_le_bytes()); // Chunk size
    wav.extend_from_slice(&1u16.to_le_bytes());  // Audio format (1 = PCM)
    wav.extend_from_slice(&1u16.to_le_bytes());  // Num channels (1 = Mono)
    wav.extend_from_slice(&sample_rate.to_le_bytes());
    wav.extend_from_slice(&(sample_rate * 2).to_le_bytes()); // Byte rate
    wav.extend_from_slice(&2u16.to_le_bytes());  // Block align
    wav.extend_from_slice(&16u16.to_le_bytes()); // Bits per sample

    // data chunk
    wav.extend_from_slice(b"data");
    wav.extend_from_slice(&data_len.to_le_bytes());
    wav.extend(data);

    wav
}

#[macroquad::main("BasicShapes")]
async fn main() {
    unsafe {
        SOUND_MANAGER.write(SoundManager::new());
    }

    let (mut interpreter, mut env, mut loop_fn) = load_script(b"fn loop() {}").expect("Should be able to load hardcoded script");
    let mut script_errors = Vec::new();

    let script_path = "game.shm";

    // On wasm, load the script once via load_file
    #[cfg(target_arch = "wasm32")]
    match load_file(script_path).await {
        Ok(bytes) => {
            match load_script(&bytes) {
                Ok(res) => {
                    (interpreter, env, loop_fn) = res;
                    script_errors = Vec::new();
                },
                Err(msg) => {
                    script_errors.push(msg);
                }
            };
        },
        Err(_msg) => {
            script_errors.push(format!("Could not read {script_path}"));
        }
    }

    // On native, track mtime for auto-reloading
    #[cfg(not(target_arch = "wasm32"))]
    let mut mtime = SystemTime::now();

    loop {
        // Auto-reload script on native when file changes
        #[cfg(not(target_arch = "wasm32"))]
        match fs::metadata(script_path) {
            Ok(metadata) => match metadata.modified() {
                Ok(time) => {
                    if mtime != time {
                        mtime = time;
                        match fs::read(script_path) {
                            Ok(bytes) => {
                                match load_script(&bytes) {
                                    Ok(res) => {
                                        (interpreter, env, loop_fn) = res;
                                        script_errors = Vec::new();
                                    },
                                    Err(msg) => {
                                        script_errors.push(msg);
                                    }
                                };
                            },
                            Err(_msg) => {
                                script_errors.push(format!("Could not read {script_path}"));
                            }
                        }
                    }
                }
                Err(e) => {
                    script_errors.push(format!("Could not get modification time for {script_path}: {}", e));
                }
            },
            Err(e) => {
                script_errors.push(format!("Could not get modification time for {script_path}: {}", e));
            }
        }

        if script_errors.is_empty() {
            match loop_fn.call(&mut interpreter, &mut shimlang::ArgBundle::new()) {
                Ok(CallResult::ReturnValue(val)) => (),
                Ok(CallResult::PC(pc, captured_scope)) => {
                    let mut new_env = Environment::with_scope(captured_scope);
                    match interpreter.execute_bytecode_extended(
                        &mut (pc as usize),
                        shimlang::ArgBundle::new(),
                        &mut new_env,
                    ) {
                        Err(msg) => script_errors.push(msg),
                        Ok(_) => interpreter.gc(&env),
                    }
                },
                Err(msg) => {
                    script_errors.push(msg);
                }
            }
        }

        if let Some(msg) = script_errors.last() {
            for (lineno, line) in msg.split("\n").enumerate() {
                draw_text(line, 20.0, 20.0*(lineno as f32+1.0), 16.0, WHITE);
            }
        }

        let mut s = sound_manager();
        let queue = std::mem::take(&mut s.queue);
        for item in queue {
            match item {
                SoundManagerQueueItem::NewBloop(id, start_freq, end_freq) => {
                    let sound_bytes = generate_bloop_wav(start_freq, end_freq);
                    match load_sound_from_bytes(&sound_bytes).await {
                        Ok(sound) => {
                            s.map.insert(id, sound);
                        },
                        Err(e) => {
                            eprintln!("{e}");
                        }
                    }
                }
                SoundManagerQueueItem::Play(id) => {
                    // Should already be in the map since we check that ::Play isn't
                    // inserted into the queue unless the `NewBloop` is before it
                    s.play(&SoundHandle { id: id });
                }
            }
        }

        next_frame().await
    }
}

async fn error_loop(msg: &str) -> ! {
    loop {
        for (lineno, line) in msg.split("\n").enumerate() {
            draw_text(line, 20.0, 20.0*(lineno as f32+1.0), 30.0, WHITE);
        }
        next_frame().await
    }
}
