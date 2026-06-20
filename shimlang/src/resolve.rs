//! Shadow resolution pass.
//!
//! Shimlang resolves variable names dynamically: a closure captures its
//! enclosing scope by reference and looks names up by string at call time,
//! walking the scope chain upward. This is what makes forward references and
//! mutual recursion work — a function can call another function declared later
//! because, by the time it is actually called, that function exists in a shared
//! enclosing scope.
//!
//! That same late-binding behaviour means a `let` that re-declares (shadows) a
//! name already declared in the same scope would, naively, just overwrite the
//! existing slot. Closures created *before* the shadowing `let` would then
//! observe the new value instead of the one that was in scope when they were
//! created.
//!
//! This pass fixes that without disturbing dynamic resolution: every generation
//! of a shadowed `let` binding is rewritten to a distinct internal name
//! (`x` -> `x#1`, `x#2`, ...), and every reference is rewritten to the
//! generation that is lexically in scope at that point. The first generation
//! keeps the bare name, so anything that is *not* shadowed (functions, structs,
//! ordinary variables) is left completely untouched and continues to resolve
//! dynamically. Because nothing is moved into a nested scope, forward
//! references and mutual recursion keep working.
//!
//! The mangling separator `#` cannot appear in a source identifier (which is
//! `[A-Za-z0-9_]`), so a rewritten name can never collide with a real one.

use crate::parse::{Ast, Block, Expression, Fn, Primary, Statement, UnaryOp};
use std::collections::HashMap;

/// Per-scope record of declared names. For each name we track how many times it
/// has been declared (its next generation index) and the name that references
/// should currently resolve to (bare for generation 0, mangled afterwards).
#[derive(Default)]
struct Scope {
    names: HashMap<Vec<u8>, NameState>,
}

struct NameState {
    /// Number of declarations seen so far; also the next generation index.
    count: u32,
    /// The name references in this scope should currently resolve to.
    current: Vec<u8>,
}

#[derive(Default)]
struct ShadowResolver {
    scopes: Vec<Scope>,
}

/// Build the internal name for `generation` (>= 1) of `name`.
fn mangle(name: &[u8], generation: u32) -> Vec<u8> {
    let mut out = Vec::with_capacity(name.len() + 4);
    out.extend_from_slice(name);
    out.push(b'#');
    out.extend_from_slice(generation.to_string().as_bytes());
    out
}

impl ShadowResolver {
    fn push_scope(&mut self) {
        self.scopes.push(Scope::default());
    }

    fn pop_scope(&mut self) {
        self.scopes.pop();
    }

    /// Register a name as occupying generation 0 of the current scope without
    /// ever mangling it. Used for declarations whose shadowing behaviour we
    /// intentionally leave unchanged (function/struct names, parameters, loop
    /// variables), while still letting a later `let` of the same name shadow
    /// them correctly.
    fn seed(&mut self, name: &[u8]) {
        let scope = self.scopes.last_mut().expect("a scope to seed into");
        scope
            .names
            .entry(name.to_vec())
            .or_insert_with(|| NameState {
                count: 1,
                current: name.to_vec(),
            });
    }

    /// Declare a `let` binding in the current scope, returning the (possibly
    /// mangled) name to emit. The first generation keeps the bare name; each
    /// subsequent generation gets a fresh mangled name so earlier closures keep
    /// the previous binding.
    fn declare_let(&mut self, name: &[u8]) -> Vec<u8> {
        let scope = self.scopes.last_mut().expect("a scope to declare into");
        match scope.names.get_mut(name) {
            Some(state) => {
                let generation = state.count;
                state.count += 1;
                let mangled = mangle(name, generation);
                state.current = mangled.clone();
                mangled
            }
            None => {
                scope.names.insert(
                    name.to_vec(),
                    NameState {
                        count: 1,
                        current: name.to_vec(),
                    },
                );
                name.to_vec()
            }
        }
    }

    /// Resolve a reference to the name currently in scope. Walks scopes from
    /// innermost outward; the first scope that declared the name decides which
    /// generation is visible. Names not found in any scope (globals, builtins,
    /// or genuine forward references) are left bare for dynamic resolution.
    fn resolve_ref(&self, name: &[u8]) -> Vec<u8> {
        for scope in self.scopes.iter().rev() {
            if let Some(state) = scope.names.get(name) {
                return state.current.clone();
            }
        }
        name.to_vec()
    }

    fn resolve_block_scoped(&mut self, block: &mut Block) {
        self.push_scope();
        self.resolve_block_contents(block);
        self.pop_scope();
    }

    fn resolve_block_contents(&mut self, block: &mut Block) {
        for stmt in block.stmts.iter_mut() {
            self.resolve_stmt(&mut stmt.data);
        }
        if let Some(last) = block.last_expr.as_mut() {
            self.resolve_expr(&mut last.data);
        }
    }

    /// Resolve a function: parameters live in the same runtime scope as the
    /// body (mirroring how the compiler binds them), default-argument
    /// expressions are evaluated in that scope, and the body shares it too.
    fn resolve_fn(&mut self, func: &mut Fn) {
        self.push_scope();
        for param in func.pos_args_required.iter() {
            self.seed(param);
        }
        for (param, _) in func.pos_args_optional.iter() {
            self.seed(param);
        }
        for (_param, default) in func.pos_args_optional.iter_mut() {
            self.resolve_expr(&mut default.data);
        }
        self.resolve_block_contents(&mut func.body);
        self.pop_scope();
    }

    fn resolve_stmt(&mut self, stmt: &mut Statement) {
        match stmt {
            Statement::Let(name, expr) => {
                // The initializer is resolved before the new binding takes
                // effect, so `let x = x + 1` still reads the previous `x`.
                self.resolve_expr(&mut expr.data);
                *name = self.declare_let(name);
            }
            Statement::Assignment(name, expr) => {
                self.resolve_expr(&mut expr.data);
                *name = self.resolve_ref(name);
            }
            Statement::CompoundAssignment(name, _op, expr) => {
                self.resolve_expr(&mut expr.data);
                *name = self.resolve_ref(name);
            }
            Statement::AttributeAssignment(obj, _attr, expr) => {
                self.resolve_expr(&mut obj.data);
                self.resolve_expr(&mut expr.data);
            }
            Statement::CompoundAttributeAssignment(obj, _attr, _op, expr) => {
                self.resolve_expr(&mut obj.data);
                self.resolve_expr(&mut expr.data);
            }
            Statement::IndexAssignment(obj, idx, expr) => {
                self.resolve_expr(&mut obj.data);
                self.resolve_expr(&mut idx.data);
                self.resolve_expr(&mut expr.data);
            }
            Statement::CompoundIndexAssignment(obj, idx, _op, expr) => {
                self.resolve_expr(&mut obj.data);
                self.resolve_expr(&mut idx.data);
                self.resolve_expr(&mut expr.data);
            }
            Statement::If(cond, if_body, else_body) => {
                self.resolve_expr(&mut cond.data);
                self.resolve_block_scoped(if_body);
                self.resolve_block_scoped(else_body);
            }
            Statement::For(idents, iter, body) => {
                self.resolve_expr(&mut iter.data);
                self.push_scope();
                for ident in idents.iter() {
                    self.seed(ident);
                }
                self.resolve_block_contents(body);
                self.pop_scope();
            }
            Statement::While(cond, body) => {
                self.resolve_expr(&mut cond.data);
                self.resolve_block_scoped(body);
            }
            Statement::Break | Statement::Continue => {}
            Statement::Fn(func) => {
                // Bind the function name in the current scope (generation 0,
                // never mangled) so recursion and forward references resolve it
                // dynamically, but a later `let` of the same name can shadow it.
                if let Some(ident) = func.ident.clone() {
                    self.seed(&ident);
                }
                self.resolve_fn(func);
            }
            Statement::Struct(s) => {
                let ident = s.ident.clone();
                self.seed(&ident);
                for (_name, default) in s.members_optional.iter_mut() {
                    self.resolve_expr(&mut default.data);
                }
                for method in s.methods.iter_mut() {
                    self.resolve_fn(method);
                }
            }
            Statement::Expression(expr) => self.resolve_expr(&mut expr.data),
            Statement::Return(opt) => {
                if let Some(expr) = opt.as_mut() {
                    self.resolve_expr(&mut expr.data);
                }
            }
        }
    }

    fn resolve_expr(&mut self, expr: &mut Expression) {
        match expr {
            Expression::Primary(p) => self.resolve_primary(p),
            Expression::BooleanOp(op) => {
                let (a, b) = op.exprs_mut();
                self.resolve_expr(&mut a.data);
                self.resolve_expr(&mut b.data);
            }
            Expression::BinaryOp(op) => {
                let (a, b) = op.exprs_mut();
                self.resolve_expr(&mut a.data);
                self.resolve_expr(&mut b.data);
            }
            Expression::Compare(operands, _) => {
                for e in operands.iter_mut() {
                    self.resolve_expr(&mut e.data);
                }
            }
            Expression::UnaryOp(op) => match op {
                UnaryOp::Not(e) | UnaryOp::Negate(e) => self.resolve_expr(&mut e.data),
            },
            Expression::Call(callee, args, kwargs) => {
                self.resolve_expr(&mut callee.data);
                for arg in args.iter_mut() {
                    self.resolve_expr(&mut arg.data);
                }
                // Keyword-argument names are parameter names, not variables.
                for (_name, value) in kwargs.iter_mut() {
                    self.resolve_expr(&mut value.data);
                }
            }
            Expression::Index(obj, index) => {
                self.resolve_expr(&mut obj.data);
                self.resolve_expr(&mut index.data);
            }
            Expression::Attribute(obj, _attr) => self.resolve_expr(&mut obj.data),
            Expression::Block(block) => self.resolve_block_scoped(block),
            Expression::If(cond, if_body, else_body) => {
                self.resolve_expr(&mut cond.data);
                self.resolve_block_scoped(if_body);
                self.resolve_block_scoped(else_body);
            }
            Expression::Fn(func) => self.resolve_fn(func),
        }
    }

    fn resolve_primary(&mut self, primary: &mut Primary) {
        match primary {
            Primary::Identifier(name) => {
                *name = self.resolve_ref(name.as_slice());
            }
            Primary::List(items) | Primary::Tuple(items) => {
                for e in items.iter_mut() {
                    self.resolve_expr(&mut e.data);
                }
            }
            Primary::Expression(boxed) => self.resolve_expr(&mut boxed.data),
            Primary::None
            | Primary::Integer(_)
            | Primary::Float(_)
            | Primary::Bool(_)
            | Primary::String(_) => {}
        }
    }
}

/// Rewrite shadowed `let` bindings (and references to them) to distinct
/// internal names so that closures created before a shadowing `let` keep the
/// original binding. See the module documentation for details.
pub fn resolve_shadowing(ast: &mut Ast) {
    let mut resolver = ShadowResolver::default();
    resolver.push_scope();
    resolver.resolve_block_contents(&mut ast.block);
    resolver.pop_scope();
}
