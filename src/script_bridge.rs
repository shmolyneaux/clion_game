use crate::shimlang_imgui;

pub struct ScriptBridge {
    pub interpreter: shimlang::Interpreter,
    pub env: shimlang::Environment,
    pub interpreter_errors: Vec<String>,
}

impl ScriptBridge {
    pub fn new() -> Self {
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
        let env = shimlang::Environment::new_with_builtins(&mut interpreter);

        Self {
            interpreter,
            env,
            interpreter_errors: Vec::new(),
        }
    }

    pub fn step(&mut self) {
        let mut pc = 0;
        match self.interpreter.execute_bytecode_extended(&mut pc, shimlang::ArgBundle::new(), &mut self.env) {
            Ok(_) => {},
            Err(msg) => {
                eprintln!("{msg}");
                self.interpreter_errors.push(msg);
            }
        };
        self.interpreter.gc(&self.env);
    }

    pub fn errors(&self) -> &[String] {
        &self.interpreter_errors
    }

    pub fn debug_window(
        &mut self, 
        shimlang_debug_window: &mut shimlang_imgui::Navigation,
        shimlang_repl: &mut shimlang_imgui::Repl,
    ) {
        shimlang_debug_window.debug_window(&mut self.interpreter, &self.env);
        shimlang_repl.window(&mut self.interpreter);
    }
}