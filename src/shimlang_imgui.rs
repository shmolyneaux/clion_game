use std::str::FromStr;

use crate::{shimlang::*, *};

#[derive(Facet, Default)]
pub struct Repl {
    lines: Vec<Vec<u8>>,
    input: Vec<u8>,
}

impl Repl {
    pub fn window(&mut self, vm: &mut Interpreter) {
        let mut open = true;
        unsafe {
            igBegin(c"Shimlang REPL".as_ptr(), &mut open as *mut bool, IMGUI_WINDOW_FLAGS_NO_FOCUS_ON_APPEARING);
            for line in self.lines.iter() {
                igText(CString::new(line.clone()).unwrap().as_ptr());
            }

            let starting_len = self.input.len();
            if self.input.capacity() < self.input.len() + 4096 {
                self.input.reserve(4096);
                if starting_len == 0 {
                    self.input.push(0);
                }
            }
            let sent: = igInputText(c"REPL Input Line".as_ptr(), self.input.as_mut_ptr() as *mut i8, self.input.capacity() as i32, ImGuiInputTextFlags_EnterReturnsTrue);
            if changed {
                // Set the len to the capacity so that we search the whole vec
                self.input.set_len(self.input.capacity());
                if let Some(index) = self.input.iter().position(|&x| x == 0) {
                    // Set the len
                    self.input.set_len(index);
                } else {
                    // Since we couldn't find the null byte, the len being the capacity is okay
                    self.lines.push(b"Could not find null byte".to_vec())
                }
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
        let mut open = true;
        unsafe {
            igBegin(c"Shimlang Debug".as_ptr(), &mut open as *mut bool, IMGUI_WINDOW_FLAGS_NO_FOCUS_ON_APPEARING);
            igText(CString::new(format!("Interpreter source: {:#?}", interpreter.source)).unwrap().as_ptr());
            igText(cformat!("ANother test {}", 42).as_ptr());
            igText(CString::new(format!("Free list: {:#?}", interpreter.mem.free_list)).unwrap().as_ptr());

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
            for (idx, x) in interpreter.mem.mem.iter().skip(item_offset).take(page_size).enumerate() {
                if !(idx % 4 == 0) {
                    igSameLine();
                } else {
                    igTextColored(0.6, 0.6, 0.6, 1.0, CString::new(format!("{:08X}", idx + item_offset as usize)).unwrap().as_ptr());
                    igSameLine();
                }
                igTextColored(1.0, 1.0, 1.0, 1.0, CString::new(format!("{:016X}", x)).unwrap().as_ptr());
            }
            igText(CString::new(format!("Did that work?")).unwrap().as_ptr());

            // TODO: Add a source code viewer
            // TODO: Add support for executing single statements
            igEnd();
        }
    }
}