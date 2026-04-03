use std::mem;
use std::ffi::CString;

use shimlang::{Environment, Interpreter, ShimValue};
use shimlang::runtime::{ArgUnpacker, NativeFn};
use shimlang::ArgBundle;

use crate::shimlang_imgui;

#[cfg(not(target_arch = "wasm32"))]
use std::sync::mpsc::{channel, Receiver, Sender};

/// Messages sent from the Engine (Main Thread) to the Script Thread
#[cfg(not(target_arch = "wasm32"))]
pub enum ScriptRequest {
    ExecuteLoop(Interpreter, Environment),
}

/// Messages sent from the Script Thread to the Engine
#[cfg(not(target_arch = "wasm32"))]
pub enum ScriptResponse {
    LoopComplete(Interpreter, Environment),
    Error(Interpreter, Environment, String),
}

enum BridgeState {
    #[cfg(not(target_arch = "wasm32"))]
    Running,
    Paused(Interpreter, Environment),
}

pub struct ScriptBridge {
    state: BridgeState,
    pub interpreter_errors: Vec<String>,
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

fn make_initial_interpreter_and_env() -> (Interpreter, Environment) {
    let ast = shimlang::ast_from_text(br#"
    struct Point {
        x,
        y
    }

    let p = Point(2, 3);

    if ig_begin("from shimlang") {
        ig_text(p);
        ig_end();
    }
    "#).unwrap();
    let program = shimlang::compile_ast(&ast).unwrap();
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

    (interpreter, env)
}

impl ScriptBridge {
    pub fn new() -> Self {
        let (interpreter, env) = make_initial_interpreter_and_env();

        #[cfg(not(target_arch = "wasm32"))]
        {
            let (request_tx, request_rx) = channel();
            let (response_tx, response_rx) = channel();

            std::thread::spawn(move || {
                script_thread_logic(request_rx, response_tx);
            });

            Self {
                state: BridgeState::Paused(interpreter, env),
                interpreter_errors: Vec::new(),
                tx: request_tx,
                rx: response_rx,
            }
        }

        #[cfg(target_arch = "wasm32")]
        {
            Self {
                state: BridgeState::Paused(interpreter, env),
                interpreter_errors: Vec::new(),
            }
        }
    }

    pub fn step(&mut self) {
        #[cfg(not(target_arch = "wasm32"))]
        {
            // Get the state, and put `Running` in its place
            let state = mem::replace(&mut self.state, BridgeState::Running);
            match state {
                // It shouldn't be possible to run .step in a overlapping way since there
                // should only be a single mutable reference
                BridgeState::Running => panic!("Somehow the interpreter is running"),
                BridgeState::Paused(interpreter, env) => {
                    self.tx.send(ScriptRequest::ExecuteLoop(interpreter, env)).unwrap();
                },
            }

            match self.rx.recv().unwrap() {
                ScriptResponse::Error(interpreter, env, msg) => {
                    self.state = BridgeState::Paused(interpreter, env);
                    self.interpreter_errors.push(msg);
                }
                ScriptResponse::LoopComplete(interpreter, env) => {
                    self.state = BridgeState::Paused(interpreter, env);
                }
            }
        }

        #[cfg(target_arch = "wasm32")]
        {
            if let BridgeState::Paused(ref mut interpreter, ref mut env) = self.state {
                let mut pc = 0;
                match interpreter.execute_bytecode_extended(&mut pc, shimlang::ArgBundle::new(), env) {
                    Ok(_) => {
                        interpreter.gc(env);
                    }
                    Err(msg) => {
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
            BridgeState::Paused(interpreter, env) => {
                shimlang_debug_window.debug_window(interpreter, &env);
                shimlang_repl.window(interpreter);
            },
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn script_thread_logic(rx: Receiver<ScriptRequest>, tx: Sender<ScriptResponse>) {
    let mut is_running = true;

    while is_running {
        if let Ok(request) = rx.recv() {
            match request {
                ScriptRequest::ExecuteLoop(mut interpreter, mut env) => {
                    let mut pc = 0;
                    tx.send(
                        match interpreter.execute_bytecode_extended(&mut pc, shimlang::ArgBundle::new(), &mut env) {
                            Ok(_) => {
                                interpreter.gc(&env);
                                ScriptResponse::LoopComplete(interpreter, env)
                            },
                            Err(msg) => {
                                ScriptResponse::Error(interpreter, env, msg)
                            }
                        }
                    ).unwrap();
                }
            }
        }
    }
}
