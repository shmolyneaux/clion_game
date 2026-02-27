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
    pub fn debug_window(&mut self, interpreter: &mut shimlang::Interpreter, env: &shimlang::Environment) {
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


            draw_colored_grid(5, 5, 8.0);


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
            let index_description = interpreter.describe_memory(env);
            let mut origin = ImVec2 { x: 0.0, y: 0.0 };
            igGetCursorScreenPos(&mut origin);
            let cell_size = 8.0;
            for (relative_idx, x) in interpreter.mem.mem.iter().skip(item_offset).take(page_size).enumerate() {
                let idx = relative_idx + item_offset;

                let x = relative_idx % 32;
                let y = relative_idx / 32;

                let mut cell_min = ImVec2 {
                    x: origin.x + (x as f32 * cell_size),
                    y: origin.y + (y as f32 * (cell_size+1.0)),
                };
                let mut cell_max = ImVec2 {
                    x: cell_min.x + cell_size,
                    y: cell_min.y + cell_size,
                };

                if let Some(s) = index_description.get(&idx) {
                    if idx == s.start {
                        cell_min.x += 1.0;
                    };
                    if idx == s.end - 1 {
                        cell_max.x -= 1.0;
                    };
                    // Draw the cell
                    let color = match &s.t {
                        shimlang::MemDescriptorType::EnvHeader(_) => im_col32(0, 255, 0, 255),
                        shimlang::MemDescriptorType::EnvData(_) => im_col32(0, 200, 0, 255),
                        shimlang::MemDescriptorType::Struct(..) => im_col32(0, 0, 220, 255),
                        _ => im_col32(255, 0, 255, 255),
                    };
                    igDrawRectFilled(cell_min, cell_max, color);

                    // Tooltip logic
                    if igIsMouseHoveringRect(cell_min, cell_max, true) {
                        igBeginTooltip();
                        igText(CString::new(format!("{}", s.to_string())).unwrap().as_ptr());
                        igEndTooltip();
                    }
                } else {
                    let color = im_col32(100, 100, 100, 255);
                    igDrawRectFilled(cell_min, cell_max, color);
                }
            }
            igDummy(ImVec2 {
                x: 0.0,
                y: 32.0 * cell_size,
            });
            igText(CString::new(format!("Did that work?")).unwrap().as_ptr());

            // TODO: Add a source code viewer
            // TODO: Add support for executing single statements
            igEnd();
        }
    }
}



fn draw_colored_grid(rows: i32, cols: i32, cell_size: f32) {
    unsafe {

        for y in 0..rows {
            for x in 0..cols {
            }
        }

    }
}