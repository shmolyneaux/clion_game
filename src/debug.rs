//! Debugger support for the shimlang interpreter.
//!
//! The interpreter runs on its own thread (see [`crate::script_bridge`]). When
//! debugging is enabled, the interpreter carries a [`ChannelDebugHook`] that the VM
//! invokes at every source-line boundary. On hitting a breakpoint (or while stepping)
//! the hook *blocks the interpreter thread* and services commands from a separate
//! debug-UI thread over channels. Because inspection runs on the interpreter thread —
//! where the live `Interpreter` state lives — no interpreter memory is shared and no
//! locking of interpreter internals is required.
//!
//! Wire protocol:
//! - the UI thread sends [`DebugCommand`]s (set/clear breakpoints, pause, continue, step);
//! - the hook sends [`DebugEvent`]s (paused with a snapshot, or resumed).

#[cfg(not(target_arch = "wasm32"))]
use std::collections::HashSet;
#[cfg(not(target_arch = "wasm32"))]
use std::sync::mpsc::{Receiver, Sender, TryRecvError};

#[cfg(not(target_arch = "wasm32"))]
use shimlang::{DebugContext, DebugHook};

/// A command sent from the debug UI to the interpreter thread.
#[derive(Debug, Clone)]
pub enum DebugCommand {
    /// Set a breakpoint on a 1-based source line.
    SetBreakpoint(u32),
    /// Clear a breakpoint on a 1-based source line.
    ClearBreakpoint(u32),
    /// Pause at the next source line reached.
    Pause,
    /// Resume until the next breakpoint.
    Continue,
    /// Run until the next source line, descending into called functions.
    StepInto,
    /// Run until the next source line in the current function (skip over calls).
    StepOver,
    /// Run until the current function returns.
    StepOut,
}

/// A snapshot of interpreter state captured when execution pauses.
#[derive(Debug, Clone)]
pub struct PauseInfo {
    /// 1-based source line where execution is paused.
    pub line: u32,
    /// Variables in scope as `(name, formatted value)` pairs.
    pub locals: Vec<(String, String)>,
    /// Source lines of the active call frames (outermost first, current line last).
    pub call_stack: Vec<u32>,
}

/// An event sent from the interpreter thread to the debug UI.
#[derive(Debug, Clone)]
pub enum DebugEvent {
    /// Execution paused; carries a snapshot of the current state.
    Paused(PauseInfo),
    /// Execution resumed after a pause.
    Resumed,
}

/// How the interpreter should proceed past the current line.
#[cfg(not(target_arch = "wasm32"))]
#[derive(Debug, Clone, Copy)]
enum StepMode {
    /// Run freely, stopping only at breakpoints.
    Run,
    /// Stop at the very next line boundary.
    Pause,
    /// Stop at the next line, including inside called functions.
    StepInto,
    /// Stop at the next line at or above this call depth.
    StepOver(usize),
    /// Stop once the call depth drops below this value.
    StepOut(usize),
}

/// A [`DebugHook`] that bridges the interpreter to a debug UI over channels.
///
/// Owned by the `Interpreter`, it travels with the interpreter as it moves between
/// threads. On hot-reload the hook is moved onto the fresh interpreter (see
/// `ScriptBridge::step`) so breakpoints and channel endpoints survive.
#[cfg(not(target_arch = "wasm32"))]
pub struct ChannelDebugHook {
    cmd_rx: Receiver<DebugCommand>,
    event_tx: Sender<DebugEvent>,
    breakpoints: HashSet<u32>,
    mode: StepMode,
}

#[cfg(not(target_arch = "wasm32"))]
impl ChannelDebugHook {
    pub fn new(cmd_rx: Receiver<DebugCommand>, event_tx: Sender<DebugEvent>) -> Self {
        Self {
            cmd_rx,
            event_tx,
            breakpoints: HashSet::new(),
            mode: StepMode::Run,
        }
    }

    /// Apply a breakpoint/pause command to the local state. Returns false for commands
    /// that are not breakpoint edits (so the caller can decide how to handle them).
    fn apply_breakpoint_cmd(&mut self, cmd: &DebugCommand) -> bool {
        match cmd {
            DebugCommand::SetBreakpoint(line) => {
                self.breakpoints.insert(*line);
                true
            }
            DebugCommand::ClearBreakpoint(line) => {
                self.breakpoints.remove(line);
                true
            }
            DebugCommand::Pause => {
                self.mode = StepMode::Pause;
                true
            }
            _ => false,
        }
    }

    /// Decide whether execution should stop at the current line.
    fn should_stop(&self, line: u32, depth: usize) -> bool {
        if self.breakpoints.contains(&line) {
            return true;
        }
        match self.mode {
            StepMode::Run => false,
            StepMode::Pause => true,
            StepMode::StepInto => true,
            StepMode::StepOver(start_depth) => depth <= start_depth,
            StepMode::StepOut(start_depth) => depth < start_depth,
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl DebugHook for ChannelDebugHook {
    fn at_line(&mut self, ctx: &DebugContext) {
        // Drain any commands that arrived while running (breakpoint edits / async pause).
        loop {
            match self.cmd_rx.try_recv() {
                Ok(cmd) => {
                    self.apply_breakpoint_cmd(&cmd);
                }
                Err(TryRecvError::Empty) => break,
                // UI gone: detach behavior — keep running freely.
                Err(TryRecvError::Disconnected) => {
                    self.mode = StepMode::Run;
                    return;
                }
            }
        }

        let line = ctx.line();
        let depth = ctx.call_depth();
        if !self.should_stop(line, depth) {
            return;
        }

        // Report the pause with a fresh snapshot.
        let info = PauseInfo {
            line,
            locals: ctx.locals(),
            call_stack: ctx.call_stack(),
        };
        if self.event_tx.send(DebugEvent::Paused(info)).is_err() {
            // UI gone while we were about to pause; run free.
            self.mode = StepMode::Run;
            return;
        }

        // Block the interpreter thread, servicing commands until told to resume.
        loop {
            match self.cmd_rx.recv() {
                Ok(cmd) => {
                    if self.apply_breakpoint_cmd(&cmd) {
                        // Breakpoint edit / pause: stay paused, keep waiting.
                        continue;
                    }
                    self.mode = match cmd {
                        DebugCommand::Continue => StepMode::Run,
                        DebugCommand::StepInto => StepMode::StepInto,
                        DebugCommand::StepOver => StepMode::StepOver(depth),
                        DebugCommand::StepOut => StepMode::StepOut(depth),
                        // Handled above by apply_breakpoint_cmd.
                        DebugCommand::SetBreakpoint(_)
                        | DebugCommand::ClearBreakpoint(_)
                        | DebugCommand::Pause => unreachable!(),
                    };
                    let _ = self.event_tx.send(DebugEvent::Resumed);
                    return;
                }
                // UI gone: resume and run free.
                Err(_) => {
                    self.mode = StepMode::Run;
                    return;
                }
            }
        }
    }
}

/// Spawn a minimal console debug UI on its own thread.
///
/// This is a placeholder driver demonstrating the protocol end-to-end: it reads
/// breakpoints from the `SHIM_DEBUG_BREAK` env var (comma-separated line numbers) at
/// startup, prints a snapshot whenever execution pauses, and reads single-line
/// commands from stdin to drive stepping:
///
/// - `c` / `continue`  — continue
/// - `s` / `step`      — step into
/// - `n` / `next`      — step over
/// - `o` / `out`       — step out
/// - `b <line>`        — set breakpoint
/// - `d <line>`        — clear breakpoint
///
/// A real UI (e.g. an imgui panel on the main thread) can be built against the same
/// `Sender<DebugCommand>` / `Receiver<DebugEvent>` pair instead of using this.
#[cfg(not(target_arch = "wasm32"))]
pub fn spawn_console_debug_ui(cmd_tx: Sender<DebugCommand>, event_rx: Receiver<DebugEvent>) {
    use std::io::Write;

    std::thread::spawn(move || {
        // Seed breakpoints from the environment so a run can stop without prior input.
        if let Ok(spec) = std::env::var("SHIM_DEBUG_BREAK") {
            for part in spec.split(',') {
                if let Ok(line) = part.trim().parse::<u32>() {
                    let _ = cmd_tx.send(DebugCommand::SetBreakpoint(line));
                }
            }
        }

        let stdin = std::io::stdin();
        while let Ok(event) = event_rx.recv() {
            let info = match event {
                DebugEvent::Resumed => continue,
                DebugEvent::Paused(info) => info,
            };

            println!("\n── shimdbg: paused at line {} ──", info.line);
            println!("call stack (lines): {:?}", info.call_stack);
            if info.locals.is_empty() {
                println!("locals: <none>");
            } else {
                println!("locals:");
                for (name, value) in &info.locals {
                    println!("  {name} = {value}");
                }
            }

            // Read commands until one resumes execution.
            loop {
                print!("(shimdbg) ");
                let _ = std::io::stdout().flush();
                let mut line = String::new();
                if stdin.read_line(&mut line).unwrap_or(0) == 0 {
                    // EOF: detach and let the interpreter run free.
                    let _ = cmd_tx.send(DebugCommand::Continue);
                    return;
                }
                let mut parts = line.split_whitespace();
                let resumed = match parts.next() {
                    Some("c") | Some("continue") => {
                        let _ = cmd_tx.send(DebugCommand::Continue);
                        true
                    }
                    Some("s") | Some("step") => {
                        let _ = cmd_tx.send(DebugCommand::StepInto);
                        true
                    }
                    Some("n") | Some("next") => {
                        let _ = cmd_tx.send(DebugCommand::StepOver);
                        true
                    }
                    Some("o") | Some("out") => {
                        let _ = cmd_tx.send(DebugCommand::StepOut);
                        true
                    }
                    Some("b") => {
                        if let Some(line) = parts.next().and_then(|n| n.parse::<u32>().ok()) {
                            let _ = cmd_tx.send(DebugCommand::SetBreakpoint(line));
                            println!("breakpoint set at line {line}");
                        } else {
                            println!("usage: b <line>");
                        }
                        false
                    }
                    Some("d") => {
                        if let Some(line) = parts.next().and_then(|n| n.parse::<u32>().ok()) {
                            let _ = cmd_tx.send(DebugCommand::ClearBreakpoint(line));
                            println!("breakpoint cleared at line {line}");
                        } else {
                            println!("usage: d <line>");
                        }
                        false
                    }
                    Some(other) => {
                        println!("unknown command: {other} (c/s/n/o/b <line>/d <line>)");
                        false
                    }
                    None => false,
                };
                if resumed {
                    break;
                }
            }
        }
    });
}
