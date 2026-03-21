use macroquad::prelude::*;
use std::mem;
use std::sync::mpsc::{channel, Receiver, Sender};
use std::fs;
use std::time::SystemTime;

use shimlang::{ShimValue, Environment, Interpreter};
use shimlang::runtime::{ArgUnpacker, NativeFn, CallResult};
use shimlang::ArgBundle;
use shimlang::lex::debug_u8s;

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
    let ast = shimlang::ast_from_text(text).unwrap();

    let program = shimlang::compile_ast(&ast).unwrap();
    let mut interpreter = shimlang::Interpreter::create(&shimlang::Config::default(), program);
    let mut env = shimlang::Environment::new_with_builtins(&mut interpreter);

    let builtins: &[(&[u8], Box<NativeFn>)] = &[
        (b"draw_circle", Box::new(shim_draw_circle)),
        (b"draw_rectangle", Box::new(shim_draw_rectangle)),
        (b"draw_line", Box::new(shim_draw_line)),
        (b"clear_background", Box::new(shim_clear_background)),
    ];

    for (name, func) in builtins {
        let position = interpreter.mem.alloc_and_set(**func, &format!("builtin func {}", debug_u8s(name)));
        env.insert_new(&mut interpreter, name.to_vec(), ShimValue::NativeFn(position));
    }

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

#[macroquad::main("BasicShapes")]
async fn main() {
    let (mut interpreter, mut env, mut loop_fn) = load_script(b"fn loop() {}").expect("Should be able to load hardcoded script");
    let mut script_errors = Vec::new();

    let mut mtime = SystemTime::now();
    let script_path = "game.shm";
    loop {
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
                            Err(msg) => {
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
                    Ok(_) => (),
                }
            },
            Err(msg) => {
                script_errors.push(msg);
            }
        }

        if let Some(msg) = script_errors.last() {
            for (lineno, line) in msg.split("\n").enumerate() {
                draw_text(line, 20.0, 20.0*(lineno as f32+1.0), 30.0, WHITE);
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