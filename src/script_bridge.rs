use std::mem;
use std::ffi::CString;

use shimlang::{Environment, Interpreter, ShimValue};
use shimlang::runtime::{ArgUnpacker, NativeFn, CallResult};
use shimlang::ArgBundle;

use crate::shimlang_imgui;
//use crate::test_mocks::igBegin;
use crate::*;

#[cfg(not(target_arch = "wasm32"))]
use std::sync::mpsc::{channel, Receiver, Sender};
#[cfg(not(target_arch = "wasm32"))]
use std::fs;
#[cfg(not(target_arch = "wasm32"))]
use std::time::SystemTime;

/// Messages sent from the Engine (Main Thread) to the Script Thread
#[cfg(not(target_arch = "wasm32"))]
pub enum ScriptRequest {
    ExecuteLoop(Interpreter, Environment, ShimValue),
}

/// Messages sent from the Script Thread to the Engine
#[cfg(not(target_arch = "wasm32"))]
pub enum ScriptResponse {
    LoopComplete(Interpreter, Environment, ShimValue),
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
    let c_title = CString::new(title_bytes)
        .map_err(|_| "igBegin: title contains null byte".to_string())?;
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
    unsafe { super::igEnd(); }
    Ok(ShimValue::None)
}

fn shim_ig_text(interpreter: &mut Interpreter, args: &ArgBundle) -> Result<ShimValue, String> {
    let mut unpacker = ArgUnpacker::new(args);
    let text_val = unpacker.required(b"text")?;
    unpacker.end()?;
    let text_bytes = text_val.to_string(interpreter);
    let c_text = CString::new(text_bytes)
        .map_err(|_| "igText: text contains null byte".to_string())?;
    super::igText(c_text.as_ptr());
    Ok(ShimValue::None)
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
    ];
    for (name, func) in builtins {
        let position = interpreter.mem.alloc_and_set(*func, &format!("builtin imgui::{}", std::str::from_utf8(name).unwrap()));
        env.insert_new(&mut interpreter, name.to_vec(), ShimValue::NativeFn(position));
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

fn call_loop_fn(interpreter: &mut Interpreter, env: &mut Environment, loop_fn: ShimValue) -> Result<(), String> {
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
        let (interpreter, env, loop_fn) = load_script(b"fn loop() {}").expect("Should be able to load hardcoded script");
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
                script_path,
            }
        }
    }

    pub fn step(&mut self) {
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
                                    match load_script(&bytes) {
                                        Ok((interpreter, env, loop_fn)) => {
                                            self.state = BridgeState::Paused(interpreter, env, loop_fn);
                                            self.interpreter_errors = Vec::new();
                                        },
                                        Err(msg) => {
                                            self.interpreter_errors.push(msg);
                                        }
                                    }
                                },
                                Err(_) => {
                                    self.interpreter_errors.push(format!("Could not read {}", self.script_path));
                                }
                            }
                        }
                    }
                    Err(e) => {
                        self.interpreter_errors.push(format!("Could not get modification time for {}: {}", self.script_path, e));
                    }
                },
                Err(_) => {
                    // File not found yet; silently wait
                }
            }

            if self.interpreter_errors.is_empty() {
                let state = mem::replace(&mut self.state, BridgeState::Running);
                match state {
                    BridgeState::Running => panic!("Somehow the interpreter is running"),
                    BridgeState::Paused(interpreter, env, loop_fn) => {
                        self.tx.send(ScriptRequest::ExecuteLoop(interpreter, env, loop_fn)).unwrap();
                    },
                }

                match self.rx.recv().unwrap() {
                    ScriptResponse::Error(interpreter, env, loop_fn, msg) => {
                        self.state = BridgeState::Paused(interpreter, env, loop_fn);
                        self.interpreter_errors.push(msg);
                    }
                    ScriptResponse::LoopComplete(interpreter, env, loop_fn) => {
                        self.state = BridgeState::Paused(interpreter, env, loop_fn);
                    }
                }
            }
        }

        if !self.interpreter_errors.is_empty() {
            let mut open = true;
            unsafe {
                if super::igBegin(c"Shimlang Errors".as_ptr(), &mut open as *mut bool, IMGUI_WINDOW_FLAGS_NO_FOCUS_ON_APPEARING) {
                    for err in self.interpreter_errors.iter() {
                        super::igTextColoredBC(1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5,
                            CString::new(format!("{}", err)).unwrap().as_ptr()
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
                    if let Err(msg) = call_loop_fn(interpreter, env, loop_fn) {
                        self.interpreter_errors.push(msg);
                    }
                }
            }
        }
    }

    pub fn errors(&self) -> &[String] {
        &self.interpreter_errors
    }

    pub fn debug_window(
        &mut self,
        shimlang_debug_window: &mut shimlang_imgui::Navigation,
        shimlang_repl: &mut shimlang_imgui::Repl,
    ) {
        match &mut self.state {
            #[cfg(not(target_arch = "wasm32"))]
            BridgeState::Running => {},
            BridgeState::Paused(interpreter, env, _loop_fn) => {
                shimlang_debug_window.debug_window(interpreter, &env);
                shimlang_repl.window(interpreter);
            },
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn script_thread_logic(rx: Receiver<ScriptRequest>, tx: Sender<ScriptResponse>) {
    loop {
        if let Ok(request) = rx.recv() {
            match request {
                ScriptRequest::ExecuteLoop(mut interpreter, mut env, loop_fn) => {
                    tx.send(
                        match call_loop_fn(&mut interpreter, &mut env, loop_fn) {
                            Ok(()) => ScriptResponse::LoopComplete(interpreter, env, loop_fn),
                            Err(msg) => ScriptResponse::Error(interpreter, env, loop_fn, msg),
                        }
                    ).unwrap();
                }
            }
        }
    }
}
