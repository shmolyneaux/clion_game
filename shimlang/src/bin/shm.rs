use std::env;
use std::fs::File;
use std::io::Read;

use shimlang;

#[derive(Debug, Default)]
struct Args {
    path: Option<String>,
}

fn parse_args() -> Result<Args, String> {
    let mut args = Args::default();

    let mut first_arg = true;
    for arg in env::args() {
        // Skip the name of the executable, which is the first arg
        if first_arg {
            first_arg = false;
            continue;
        }
        if !arg.starts_with('-') {
            if let Some(existing_positional_arg) = args.path {
                return Err(format!("Found multiple positional arguments {} and {}", existing_positional_arg, arg));
            } else {
                args.path = Some(arg.clone());
            }
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

        let program = match shimlang::ast_from_text(&contents) {
            Ok(program) => program,
            Err(msg) => {
                eprintln!(
                    "Parse Error:\n{msg}"
                );
                return Err(msg.into());
            }
        };
        let mut interpreter = shimlang::Interpreter::default();
        interpreter.execute(&program)?;
    } else {
        return Err("Expected script path".into());
    }

    Ok(())
}