use macroquad::prelude::*;
use std::mem;
use std::sync::mpsc::{channel, Receiver, Sender};
use std::thread;

use shimlang::{ShimValue, Environment, Interpreter};
use shimlang::runtime::{ArgUnpacker, NativeFn};
use shimlang::ArgBundle;
use shimlang::lex::debug_u8s;

fn shim_draw_circle(interpreter: &mut Interpreter, args: &ArgBundle) -> Result<ShimValue, String> {
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

fn shim_clear_background(interpreter: &mut Interpreter, args: &ArgBundle) -> Result<ShimValue, String> {
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

fn shim_draw_line(interpreter: &mut Interpreter, args: &ArgBundle) -> Result<ShimValue, String> {
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

fn shim_draw_rectangle(interpreter: &mut Interpreter, args: &ArgBundle) -> Result<ShimValue, String> {
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

#[macroquad::main("BasicShapes")]
async fn main() {
    let interpreter_config = shimlang::Config::default();
    let ast = shimlang::ast_from_text(br#"
        fn loop() {
            draw_circle(
                120.0,
                120.0,
                50.0,
                0.7,
                0.4,
                0.2,
                1.0,
            );
        }
    "#).unwrap();

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

    let mut script_errors = Vec::new();

    let mut pc = 0;
    match interpreter.execute_bytecode_extended(&mut pc, shimlang::ArgBundle::new(), &mut env) {
        Ok(_) => {
            interpreter.gc(&env);
        },
        Err(msg) => {
            script_errors.push(msg);
        }
    }

    // Technically this is an external reference that should be passed to the
    // gc as a root, but since it's in the `env` it should already be marked.
    let loop_fn = match env.get(&mut interpreter, b"loop") {
        Some(func @ ShimValue::Fn(_)) => {
            func
        },
        None => {
            error_loop("No loop function found").await;
        },
        _ => {
            error_loop("Identifier 'loop' is not a function").await;
        },
    };

    loop {
        match loop_fn.call(interpreter, &mut args) {
            Ok(CallResult::ReturnValue(val)) => val,
            Ok(CallResult::PC(pc, captured_scope)) => {
                let mut new_env = Environment::with_scope(captured_scope);
                interpreter.execute_bytecode_extended(
                    &mut (pc as usize),
                    args,
                    &mut new_env,
                )?
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