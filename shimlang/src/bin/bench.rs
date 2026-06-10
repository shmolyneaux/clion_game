// Microbenchmark harness for the interpreter dispatch loop.
//
// Times only `Interpreter::execute()` (compilation is excluded), running several
// trials with a fresh interpreter each time and reporting min/median/mean. The
// harness only uses the stable `ast_from_text` / `compile_ast` / `Interpreter`
// surface so it builds against the pre-debugger runtime too.
use std::time::Instant;

use shimlang::*;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path = args
        .get(1)
        .expect("usage: bench <script> [trials]")
        .clone();
    let trials: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(9);
    let src = std::fs::read(&path).expect("read script");

    let mut times_ms: Vec<f64> = Vec::new();
    for t in 0..trials {
        let ast = ast_from_text(&src).expect("parse");
        let program = compile_ast(&ast).expect("compile");
        let mut interp = Interpreter::create(&Config::default(), program);

        let start = Instant::now();
        let result = interp.execute();
        let elapsed = start.elapsed();
        result.expect("execute");

        let ms = elapsed.as_secs_f64() * 1000.0;
        times_ms.push(ms);
        eprintln!("  trial {t}: {ms:.2} ms");
    }

    times_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let min = times_ms[0];
    let median = times_ms[times_ms.len() / 2];
    let mean = times_ms.iter().sum::<f64>() / times_ms.len() as f64;
    println!("min={min:.2}ms  median={median:.2}ms  mean={mean:.2}ms  (trials={trials})");
}
