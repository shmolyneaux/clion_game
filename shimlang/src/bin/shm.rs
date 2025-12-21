use std::env;
use std::fs::File;
use std::io::Read;

use shimlang;

#[derive(Debug, PartialEq)]
enum Command {
    Parse,
    Execute,
    Spans
}

impl Default for Command {
    fn default() -> Self {
        Command::Execute
    }
}

#[derive(Debug, Default)]
struct Args {
    path: Option<String>,
    command: Command,
}

fn parse_args() -> Result<Args, String> {
    let mut args = Args::default();

    for (idx, arg) in env::args().enumerate() {
        // Skip the name of the executable, which is the first arg
        if idx == 0 {
            continue;
        } else if !arg.starts_with('-') {
            if let Some(existing_positional_arg) = args.path {
                return Err(format!("Found multiple positional arguments {} and {}", existing_positional_arg, arg));
            } else {
                args.path = Some(arg.clone());
            }
        } else if arg == "--parse" {
            if args.command != Command::default() {
                return Err(format!("Attempted to set command multiple times! {}", arg));
            }
            args.command = Command::Parse;
        } else if arg == "--spans" {
            if args.command != Command::default() {
                return Err(format!("Attempted to set command multiple times! {}", arg));
            }
            args.command = Command::Spans;
        } else {
            return Err(format!("Unknown args {}", arg));
        }
    }

    Ok(args)
}

fn main() -> Result<(), Box<dyn std::error::Error>>{
    let args = parse_args()?;

    if let Some(script_path) = args.path {
        let mut file = File::open(&script_path)?;
        let mut contents = Vec::new();
        file.read_to_end(&mut contents)?;

        match std::str::from_utf8(&contents) {
            Ok(_) => (),
            Err(e) => return Err((format!("Script is not utf8 {:?}", e)).into())
        }

        match args.command {
            Command::Execute => {
                let ast = match shimlang::ast_from_text(&contents) {
                    Ok(ast) => ast,
                    Err(msg) => {
                        eprintln!(
                            "Parse Error:\n{msg}"
                        );
                        return Err((format!("Failed to parse script")).into());
                    }
                };
                let bytecode = shimlang::compile_ast(&ast)?;
                let mut interpreter = shimlang::Interpreter::default();
                interpreter.execute_bytecode(&bytecode)?;
            },
            Command::Parse => {
                match shimlang::ast_from_text(&contents) {
                    Ok(program) => program,
                    Err(msg) => {
                        eprintln!(
                            "Parse Error:\n{msg}"
                        );
                        return Err((format!("Failed to parse script")).into());
                    }
                };
            },
            Command::Spans => {
                let tokens = shimlang::lex(&contents)?;
                for span in tokens.spans() {
                    println!(
                        "{}",
                        std::str::from_utf8(&contents[(span.0 as usize)..(span.1 as usize)])?
                    );
                }
            },
        }
    } else {
        return Err("Expected script path".into());
    }

    Ok(())
}
