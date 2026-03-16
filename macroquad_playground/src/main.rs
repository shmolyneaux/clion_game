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

    // TODO: unpacker should support type checking right here
    let x = unpacker.required(b"x")?;
    let y = unpacker.required(b"y")?;
    let radius = unpacker.required(b"radius")?;
    let r = unpacker.required(b"r")?;
    let g = unpacker.required(b"g")?;
    let b = unpacker.required(b"b")?;
    let a = unpacker.required(b"a")?;
    unpacker.end()?;

    match (x, y, radius, r, g, b, a) {
        (ShimValue::Float(x), ShimValue::Float(y), ShimValue::Float(radius), ShimValue::Float(r), ShimValue::Float(g), ShimValue::Float(b), ShimValue::Float(a), ) => {
            draw_circle(
                x,
                y,
                radius,
                Color::new(r, g, b, a),
            );
            Ok(ShimValue::None)
        },
        _ => Err(format!("Bad type for draw_circle {:?} {:?} {:?} {:?} {:?} {:?} {:?}", x, y, radius, r, g, b, a))
    }

}

#[macroquad::main("BasicShapes")]
async fn main() {
    let interpreter_config = shimlang::Config::default();
    let ast = shimlang::ast_from_text(br#"
        draw_circle(
            120.0,
            120.0,
            50.0,
            0.7,
            0.4,
            0.2,
            1.0,
        );
    "#).unwrap();

    let program = shimlang::compile_ast(&ast).unwrap();
    let mut interpreter = shimlang::Interpreter::create(&shimlang::Config::default(), program);
    let mut env = shimlang::Environment::new_with_builtins(&mut interpreter);


    let builtins: &[(&[u8], Box<NativeFn>)] = &[
        (b"draw_circle", Box::new(shim_draw_circle)),
    ];

    for (name, func) in builtins {
        let position = interpreter.mem.alloc_and_set(**func, &format!("builtin func {}", debug_u8s(name)));
        env.insert_new(&mut interpreter, name.to_vec(), ShimValue::NativeFn(position));
    }

    let mut script_errors = Vec::new();

    loop {
        //clear_background(RED);
        let mut pc = 0;

        match interpreter.execute_bytecode_extended(&mut pc, shimlang::ArgBundle::new(), &mut env) {
            Ok(_) => {
                interpreter.gc(&env);
            },
            Err(msg) => {
                script_errors.push(msg);
            }
        }

        //draw_line(40.0, 40.0, 100.0, 200.0, 15.0, BLUE);
        //draw_rectangle(screen_width() / 2.0 - 60.0, 100.0, 120.0, 60.0, GREEN);
        //draw_circle(screen_width() - 30.0, screen_height() - 30.0, 15.0, YELLOW);

        if let Some(msg) = script_errors.last() {
            for (lineno, line) in msg.split("\n").enumerate() {
                draw_text(line, 20.0, 20.0*(lineno as f32+1.0), 30.0, WHITE);
            }
        }

        next_frame().await
    }
}