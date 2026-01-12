use std::env;
use std::fs::File;
use std::io::Read;
use std::process::ExitCode;

use shimlang;

#[derive(Debug, PartialEq)]
enum Command {
    Parse,
    Execute,
    Spans,
    Compile,
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
                return Err(format!(
                    "Found multiple positional arguments {} and {}",
                    existing_positional_arg, arg
                ));
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
        } else if arg == "--compile" {
            if args.command != Command::default() {
                return Err(format!("Attempted to set command multiple times! {}", arg));
            }
            args.command = Command::Compile;
        } else {
            return Err(format!("Unknown args {}", arg));
        }
    }

    Ok(args)
}

fn run() -> Result<(), String> {
    let args = parse_args()?;

    if let Some(script_path) = args.path {
        let mut file = File::open(&script_path).map_err(|e| format!("{:?}", e))?;
        let mut contents = Vec::new();
        file.read_to_end(&mut contents)
            .map_err(|e| format!("{:?}", e))?;

        match std::str::from_utf8(&contents) {
            Ok(_) => (),
            Err(e) => return Err((format!("Script is not utf8 {:?}", e)).into()),
        }

        match args.command {
            Command::Execute => {
                let ast = match shimlang::ast_from_text(&contents) {
                    Ok(ast) => ast,
                    Err(msg) => {
                        eprintln!("Parse Error:\n{msg}");
                        return Err((format!("Failed to parse script")).into());
                    }
                };
                let program = shimlang::compile_ast(&ast)?;
                let mut interpreter = shimlang::Interpreter::default();
                match interpreter.execute_bytecode(&program) {
                    Ok(()) => (),
                    Err(msg) => {
                        eprintln!("{msg}");
                        return Err((format!("")).into());
                    }
                };
            }
            Command::Parse => {
                match shimlang::ast_from_text(&contents) {
                    Ok(program) => program,
                    Err(msg) => {
                        eprintln!("Parse Error:\n{msg}");
                        return Err((format!("Failed to parse script")).into());
                    }
                };
            }
            Command::Spans => {
                let tokens = shimlang::lex(&contents)?;
                for span in tokens.spans() {
                    println!(
                        "{}",
                        std::str::from_utf8(&contents[(span.start as usize)..(span.end as usize)])
                            .map_err(|e| format!("{:?}", e))?
                    );
                }
            }
            Command::Compile => {
                let ast = match shimlang::ast_from_text(&contents) {
                    Ok(ast) => ast,
                    Err(msg) => {
                        eprintln!("Parse Error:\n{msg}");
                        return Err((format!("Failed to parse script")).into());
                    }
                };
                let program = shimlang::compile_ast(&ast)?;
                shimlang::print_asm(&program.bytecode);
            }
        }
    } else {
        return Err("Expected script path".into());
    }

    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(msg) => {
            if msg != "" {
                eprintln!("{msg}");
            }
            ExitCode::from(1)
        }
    }
}
