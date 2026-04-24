use shimlang::ArgBundle;
use shimlang::runtime::{ArgUnpacker, CallResult, NativeFn};
use shimlang::{Environment, Interpreter, ShimNative, ShimValue, debug_u8s};
use std::ffi::CString;
use std::mem;

use crate::shimlang_imgui;
//use crate::test_mocks::igBegin;
use crate::*;

#[cfg(not(target_arch = "wasm32"))]
use std::fs;
#[cfg(not(target_arch = "wasm32"))]
use std::sync::mpsc::{Receiver, Sender, channel};
#[cfg(not(target_arch = "wasm32"))]
use std::time::SystemTime;

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

#[derive(Debug, Clone)]
pub struct DrawRect {
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
    pub texture: Option<TextureHandle>,
    pub modulate: [u8; 4],
}

pub enum DrawListItem {
    Rect(DrawRect),
    CreateTexture(u32, u32, u32, Vec<u8>),
}

pub struct KeyState {
    pub keys: Vec<u8>,
    pub last_keys: Vec<u8>,
}

impl Default for KeyState {
    fn default() -> Self {
        Self {
            keys: Vec::new(),
            last_keys: Vec::new(),
        }
    }
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
            _ => Err(format!(
                "KeyValue has pressed/released/just_pressed/just_released, not '{}'",
                debug_u8s(ident)
            )),
        }
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

    fn push_rect(
        &mut self,
        x: f32,
        y: f32,
        w: f32,
        h: f32,
        texture: Option<TextureHandle>,
        modulate: [u8; 4],
    ) {
        self.items.push(DrawListItem::Rect(DrawRect {
            x,
            y,
            w,
            h,
            texture,
            modulate,
        }));
    }

    fn push_texture(&mut self, w: u32, h: u32, data: Vec<u8>) -> TextureHandle {
        let id = self.next_texture_handle;
        self.next_texture_handle += 1;
        self.items.push(DrawListItem::CreateTexture(id, w, h, data));
        TextureHandle { texture_id: id }
    }
}

/// Messages sent from the Engine (Main Thread) to the Script Thread
#[cfg(not(target_arch = "wasm32"))]
pub enum ScriptRequest {
    ExecuteLoop(Interpreter, Environment, ShimValue, Vec<u8>, Vec<u8>),
}

/// Messages sent from the Script Thread to the Engine
#[cfg(not(target_arch = "wasm32"))]
pub enum ScriptResponse {
    LoopComplete(Interpreter, Environment, ShimValue, Vec<DrawListItem>),
    Error(Interpreter, Environment, ShimValue, String),
}

enum BridgeState {
    #[cfg(not(target_arch = "wasm32"))]
    Running,
    Paused(Interpreter, Environment, ShimValue),
}

pub struct ScriptBridge {
    state: BridgeState,
    pub interpreter_errors: Vec<String>,
    pub draw_list: Vec<DrawListItem>,
    script_path: String,
    #[cfg(not(target_arch = "wasm32"))]
    mtime: SystemTime,
    #[cfg(not(target_arch = "wasm32"))]
    tx: Sender<ScriptRequest>,
    #[cfg(not(target_arch = "wasm32"))]
    rx: Receiver<ScriptResponse>,
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
    let x = unpacker.required_number(b"x")?;
    let y = unpacker.required_number(b"y")?;
    let w = unpacker.required_number(b"w")?;
    let h = unpacker.required_number(b"h")?;
    let texture: Option<TextureHandle> = match unpacker.optional(b"texture") {
        Some(val) => Some(val.as_native::<TextureHandle>(interpreter)?.clone()),
        None => None,
    };
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
    unpacker.end()?;

    interpreter
        .fetch_mut::<DrawList>()
        .push_rect(x, y, w, h, texture, [r, g, b, a]);
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
                    val.to_string()
                ));
            }
        });
    }

    let handle = interpreter
        .fetch_mut::<DrawList>()
        .push_texture(w, h, rgba_bytes);

    Ok(interpreter.mem.alloc_native(handle))
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
        (b"create_texture", shim_create_texture),
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
) -> Result<(), String> {
    match loop_fn.call(interpreter, &mut shimlang::ArgBundle::new()) {
        Ok(CallResult::ReturnValue(_)) => {
            interpreter.gc(env);
            Ok(())
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
                    interpreter.gc(env);
                    Ok(())
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

            std::thread::spawn(move || {
                script_thread_logic(request_rx, response_tx);
            });

            Self {
                state: BridgeState::Paused(interpreter, env, loop_fn),
                interpreter_errors: Vec::new(),
                draw_list: Vec::new(),
                script_path,
                mtime: SystemTime::UNIX_EPOCH,
                tx: request_tx,
                rx: response_rx,
            }
        }

        #[cfg(target_arch = "wasm32")]
        {
            Self {
                state: BridgeState::Paused(interpreter, env, loop_fn),
                interpreter_errors: Vec::new(),
                draw_list: Vec::new(),
                script_path,
            }
        }
    }

    pub fn step(&mut self, keys: &[u8], last_keys: &[u8]) {
        #[cfg(not(target_arch = "wasm32"))]
        {
            // Check for file changes and reload
            match fs::metadata(&self.script_path) {
                Ok(metadata) => match metadata.modified() {
                    Ok(time) => {
                        if self.mtime != time {
                            self.mtime = time;
                            match fs::read(&self.script_path) {
                                Ok(bytes) => {
                                    println!("{}", debug_u8s(&bytes));
                                    match load_script(&bytes) {
                                        Ok((interpreter, env, loop_fn)) => {
                                            self.state =
                                                BridgeState::Paused(interpreter, env, loop_fn);
                                            self.interpreter_errors = Vec::new();
                                        }
                                        Err(msg) => {
                                            self.interpreter_errors.push(msg);
                                        }
                                    }
                                }
                                Err(_) => {
                                    self.interpreter_errors
                                        .push(format!("Could not read {}", self.script_path));
                                }
                            }
                        }
                    }
                    Err(e) => {
                        self.interpreter_errors.push(format!(
                            "Could not get modification time for {}: {}",
                            self.script_path, e
                        ));
                    }
                },
                Err(_) => {
                    println!("file not found {}", &self.script_path);
                    // File not found yet; silently wait
                }
            }

            if self.interpreter_errors.is_empty() {
                let state = mem::replace(&mut self.state, BridgeState::Running);
                match state {
                    BridgeState::Running => panic!("Somehow the interpreter is running"),
                    BridgeState::Paused(interpreter, env, loop_fn) => {
                        self.tx
                            .send(ScriptRequest::ExecuteLoop(
                                interpreter,
                                env,
                                loop_fn,
                                keys.to_vec(),
                                last_keys.to_vec(),
                            ))
                            .unwrap();
                    }
                }

                match self.rx.recv().unwrap() {
                    ScriptResponse::Error(interpreter, env, loop_fn, msg) => {
                        self.state = BridgeState::Paused(interpreter, env, loop_fn);
                        self.interpreter_errors.push(msg);
                    }
                    ScriptResponse::LoopComplete(interpreter, env, loop_fn, draw_list) => {
                        self.state = BridgeState::Paused(interpreter, env, loop_fn);
                        self.draw_list = draw_list;
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
                            CString::new(format!("{}", err)).unwrap().as_ptr(),
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
                    ks.keys = keys.to_vec();
                    ks.last_keys = last_keys.to_vec();
                    if let Err(msg) = call_loop_fn(interpreter, env, loop_fn) {
                        self.interpreter_errors.push(msg);
                    }
                    self.draw_list = interpreter
                        .fetch_mut::<DrawList>()
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
        match &mut self.state {
            #[cfg(not(target_arch = "wasm32"))]
            BridgeState::Running => {}
            BridgeState::Paused(interpreter, env, _loop_fn) => {
                shimlang_debug_window.debug_window(interpreter, &env);
            }
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn script_thread_logic(rx: Receiver<ScriptRequest>, tx: Sender<ScriptResponse>) {
    loop {
        if let Ok(request) = rx.recv() {
            match request {
                ScriptRequest::ExecuteLoop(mut interpreter, mut env, loop_fn, keys, last_keys) => {
                    let ks = interpreter.fetch_mut::<KeyState>();
                    ks.keys = keys;
                    ks.last_keys = last_keys;
                    tx.send(match call_loop_fn(&mut interpreter, &mut env, loop_fn) {
                        Ok(()) => {
                            let draw_list = interpreter
                                .fetch_mut::<DrawList>()
                                .items
                                .drain(..)
                                .collect();
                            ScriptResponse::LoopComplete(interpreter, env, loop_fn, draw_list)
                        }
                        Err(msg) => ScriptResponse::Error(interpreter, env, loop_fn, msg),
                    })
                    .unwrap();
                }
            }
        }
    }
}
