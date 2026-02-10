use shimlang;
use shimlang::*;

fn main() {
    let script = br#"
struct Point {
    x,
    y
}

fn rounds() {
    let d = dict();
    for i in 0..1000 {
        d[i] = Point(i*2, i*3);
    }
}

rounds();

print("done");
"#;

    let ast = shimlang::ast_from_text(script).expect("Failed to parse");
    let program = shimlang::compile_ast(&ast).expect("Failed to compile");

    let mut interpreter = shimlang::Interpreter::create(&Config::default(), program);
    let mut env = shimlang::Environment::new_with_builtins(&mut interpreter);

    for round in 0..5 {
        let mut pc = 0;
        match interpreter.execute_bytecode_extended(&mut pc, shimlang::ArgBundle::new(), &mut env) {
            Ok(_) => {
                eprintln!("Round {} executed successfully", round);
            }
            Err(msg) => {
                eprintln!("Round {} failed: {}", round, msg);
                std::process::exit(1);
            }
        }

        eprintln!("Running GC after round {}...", round);
        interpreter.gc(&env);
        eprintln!("GC after round {} completed", round);
    }

    eprintln!("All rounds completed successfully");
}
