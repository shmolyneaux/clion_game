use std::str::FromStr;

use crate::*;
use ::shimlang::{Word, FreeBlock, Interpreter};

#[derive(Facet, Default)]
pub struct Repl {
    lines: Vec<Vec<u8>>,
    input: Vec<u8>,
    sent_last_frame: bool,
}

impl Repl {
    pub fn window(&mut self, vm: &mut Interpreter) {
        let _zone = zone_scoped!("Repl::window");
        // Set the capacity of the vec here since we want to derive Default for convenience

        unsafe {
            if self.input.capacity() == 0 {
                self.input.reserve(4096);
                self.input.set_len(self.input.capacity());
                // Fill will nulls
                for c in self.input.iter_mut() {
                    *c = 0;
                }
            }

            let mut open = true;
            igBegin(c"Shimlang REPL".as_ptr(), &mut open as *mut bool, IMGUI_WINDOW_FLAGS_NO_FOCUS_ON_APPEARING);
            for line in self.lines.iter() {
                igText(CString::new(line.clone()).unwrap().as_ptr());
            }

            if self.sent_last_frame {
                igSetKeyboardFocusHere();
                self.sent_last_frame = false;
            }
            igSeparator();
            let height = shmConsoleFooterHeight();
            igText(CString::new(format!("Footer height {}", height)).unwrap().as_ptr());
            let sent = igInputText(
                c"REPL Input Line".as_ptr(), 
                self.input.as_mut_ptr() as *mut i8, 
                self.input.capacity() as i32,
                IMGUI_INPUT_TEXT_FLAGS_ENTER_RETURNS_TRUE
            );
            if sent {
                if let Some(index) = self.input.iter().position(|&x| x == 0) {
                    self.lines.push(self.input[..index].to_vec());
                    self.input[..index].fill(0);
                } else {
                    // Since we couldn't find the null byte, the len being the capacity is okay
                    self.lines.push(b"Could not find null byte".to_vec())
                }
                self.sent_last_frame = true;
            }
            igEnd();
        }
    }
}

#[derive(Facet, Default)]
pub struct Navigation {
    memory_page: u32,
}

fn idx_in_free_block(idx: Word, free_list: &[FreeBlock]) -> bool {
    for block in free_list {
        if block.pos <= idx && idx < block.pos + block.size {
            return true;
        }
    }
    false
}

impl Navigation {
    pub fn debug_window(&mut self, interpreter: &mut shimlang::Interpreter) {
        let _zone = zone_scoped!("Navigation::debug_window");
        let mut open = true;
        unsafe {
            igBegin(c"Shimlang Debug".as_ptr(), &mut open as *mut bool, IMGUI_WINDOW_FLAGS_NO_FOCUS_ON_APPEARING);

            igText(CString::new(format!("Mem size is {}", interpreter.mem.mem.len())).unwrap().as_ptr());
            igText(CString::new(format!("Mask size is {}", usize::from(interpreter.mem.free_list[interpreter.mem.free_list.len()-1].pos))).unwrap().as_ptr());


            igText(CString::new(format!("Interpreter source: {:#?}", interpreter.source)).unwrap().as_ptr());
            igText(cformat!("ANother test {}", 42).as_ptr());
            igText(CString::new(format!("Disassembly:\n{}",
                                        shimlang::format_asm(&interpreter.program.bytecode)
                                        )).unwrap().as_ptr());

            // Memory viewer

            igText(CString::new(format!("Mem size is {}", interpreter.mem.mem.len())).unwrap().as_ptr());




            if self.memory_page == 0 {
                igBeginDisabled();
                igButton(CString::new(format!("Prev")).unwrap().as_ptr());
                igEndDisabled();
            } else {
                if igButton(
                    CString::new(format!("Prev")).unwrap().as_ptr()
                ) {
                    self.memory_page -= 1;
                }
            }

            igSameLine();

            if igButton(
                CString::new(format!("Next")).unwrap().as_ptr()
            ) {
                // TODO: logic for disabling on last page
                self.memory_page += 1;
            }

            let page_size: usize = 128;
            let item_offset: usize = page_size * self.memory_page as usize;
            for (relative_idx, x) in interpreter.mem.mem.iter().skip(item_offset).take(page_size).enumerate() {
                let idx = relative_idx + item_offset;
                if !(idx % 4 == 0) {
                    // We show 4 memory locations per line, so if we're not at the start of the line
                    // we know the text should continue on the same line
                    igSameLine();
                } else {
                    // Start of a new line, show the memory address
                    igTextColored(0.6, 0.6, 0.6, 1.0, CString::new(format!("{:08X}", idx)).unwrap().as_ptr());
                    igSameLine();
                }
                let mut is_free_memory = false;
                for block in interpreter.mem.free_list.iter() {
                    if usize::from(block.pos) > idx {
                        break;
                    }
                    if usize::from(block.pos) <= idx && idx < block.end().into() {
                        is_free_memory = true;
                    }
                }
                if is_free_memory {
                    igTextColored(0.3, 0.3, 0.3, 0.3, CString::new(format!("{:016X}", x)).unwrap().as_ptr());
                } else {
                    igTextColored(1.0, 1.0, 1.0, 1.0, CString::new(format!("{:016X}", x)).unwrap().as_ptr());
                }
            }
            igText(CString::new(format!("Did that work?")).unwrap().as_ptr());

            // TODO: Add a source code viewer
            // TODO: Add support for executing single statements
            igEnd();
        }
    }
}
