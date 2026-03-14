use std::mem;
use std::sync::mpsc::{channel, Receiver, Sender};
use std::thread;

use shimlang::{Environment, Interpreter};

use crate::shimlang_imgui;

/// Messages sent from the Engine (Main Thread) to the Script Thread
pub enum ScriptRequest {
    ExecuteLoop(Interpreter, Environment),
}

/// Messages sent from the Script Thread to the Engine
pub enum ScriptResponse {
    LoopComplete(Interpreter, Environment),
    Error(Interpreter, Environment, String),
}

enum BridgeState {
    Running,
    Paused(Interpreter, Environment),
}

pub struct ScriptBridge {
    state: BridgeState,
    pub interpreter_errors: Vec<String>,
    tx: Sender<ScriptRequest>,
    rx: Receiver<ScriptResponse>,
}

impl ScriptBridge {
    pub fn new() -> Self {
        let (request_tx, request_rx) = channel();
        let (response_tx, response_rx) = channel();

        thread::spawn(move || {
            script_thread_logic(request_rx, response_tx);
        });

        let interpreter_config = shimlang::Config::default();
        let ast = shimlang::ast_from_text(br#"
        struct Point {
            x,
            y
        }

        let d = dict();
        for i in 0..100 {
            d[i] = str(i);
        }

        let some_p0 = Point(0, 1);
        let some_p1 = Point(2, 3);
        let some_p2 = Point(4, 5);
        let some_p3 = Point(6, 7);
        let s = "testing a longer string";
        let i = -1;

        fn rounds() {
            let d = dict();
            for i in 0..1000 {
                d[i] = Point(i*2, i*3);
            }
        }

        rounds();
        
        print("done");
        "#).unwrap();
        let program = shimlang::compile_ast(&ast).unwrap();
        let mut interpreter = shimlang::Interpreter::create(&shimlang::Config::default(), program);
        let mut env = shimlang::Environment::new_with_builtins(&mut interpreter);

        Self {
            state: BridgeState::Paused(interpreter, env),
            interpreter_errors: Vec::new(),
            tx: request_tx,
            rx: response_rx,
        }
    }

    pub fn step(&mut self) {
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
            ScriptResponse::Error(Interpreter, env, msg) => {
                self.state = BridgeState::Paused(Interpreter, env);
                self.interpreter_errors.push(msg);
            }
            ScriptResponse::LoopComplete(Interpreter, env) => {
                self.state = BridgeState::Paused(Interpreter, env);
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
            // It shouldn't be possible to run .step in a overlapping way since there
            // should only be a single mutable reference
            BridgeState::Running => {
            },
            BridgeState::Paused(interpreter, env) => {
                shimlang_debug_window.debug_window(interpreter, &env);
                shimlang_repl.window(interpreter);
            },
        }
    }
}

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