use std::str::FromStr;

use crate::*;
use ::shimlang::{u24, FreeBlock, Interpreter, fnv1a_hash};

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

fn idx_in_free_block(idx: u24, free_list: &[FreeBlock]) -> bool {
    for block in free_list {
        if block.pos <= idx && idx < block.pos + block.size {
            return true;
        }
    }
    false
}

pub fn i32_to_rgb(i: i32) -> [u8; 3] {
    // 1. Convert to u32 to handle negative IDs and treat as a continuous scale
    let val = i as u32;

    // 2. Use the Golden Ratio conjugate (0.6180339887...) 
    // This spreads the "Hue" evenly around the 360-degree wheel.
    let golden_ratio_conjugate = 0.618033988749895_f64;
    let hue = (val as f64 * golden_ratio_conjugate).fract(); 

    // 3. Set constant Saturation (0.8) and Lightness (0.6) for a "clean" UI look
    hsl_to_rgb(hue, 0.8, 0.6)
}

fn hsl_to_rgb(h: f64, s: f64, l: f64) -> [u8; 3] {
    let q = if l < 0.5 { l * (1.0 + s) } else { l + s - l * s };
    let p = 2.0 * l - q;

    let mut rgb = [0u8; 3];
    let transitions = [h + 1.0/3.0, h, h - 1.0/3.0];

    for (i, &t) in transitions.iter().enumerate() {
        let mut tc = t;
        if tc < 0.0 { tc += 1.0; }
        if tc > 1.0 { tc -= 1.0; }

        let color_val = if tc < 1.0/6.0 { p + (q - p) * 6.0 * tc }
            else if tc < 1.0/2.0 { q }
            else if tc < 2.0/3.0 { p + (q - p) * (2.0/3.0 - tc) * 6.0 }
            else { p };
        
        rgb[i] = (color_val * 255.0).round() as u8;
    }
    rgb
}

pub fn f32_to_rgb(f: f32) -> [u8; 3] {
    // 1. Normalize 0.0 and -0.0 so they produce the same color
    let normalized = if f == 0.0 { 0.0 } else { f };
    
    // 2. Bit-cast to u32 (this is a no-op in assembly, just a type change)
    let bits = normalized.to_bits();
    
    // 3. Use your existing vibrant function
    i32_to_rgb(bits as i32)
}

pub fn f64_to_rgb(f: f64) -> [u8; 3] {
    let normalized = if f == 0.0 { 0.0 } else { f };
    let bits = normalized.to_bits();
    
    // Fold the 64 bits into 32 bits using XOR
    // This ensures both the exponent and mantissa contribute to the color
    let folded = (bits ^ (bits >> 32)) as u32;
    
    i32_to_rgb(folded as i32)
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

            let page_size: usize = 128;

            let mem_end: usize = if let Some(block) = interpreter.mem.free_list.last() {
                block.pos.into()
            } else {
                interpreter.mem.mem.len()
            };

            if (self.memory_page + 1) * (page_size as u32) < mem_end as u32 {
                if igButton(
                    CString::new(format!("Next")).unwrap().as_ptr()
                ) {
                    self.memory_page += 1;
                }
            } else {
                igBeginDisabled();
                igButton(CString::new(format!("Next")).unwrap().as_ptr());
                igEndDisabled();
            }

            igSameLine();


            // TODO: disable "Next" button if it goes past the free_list last position
            igText(CString::new(format!("Last block: {:?}", interpreter.mem.free_list.last())).unwrap().as_ptr());

            let item_offset: usize = page_size * self.memory_page as usize;
            let index_description = interpreter.describe_memory(env);
            let mut origin = ImVec2 { x: 0.0, y: 0.0 };
            igGetCursorScreenPos(&mut origin);
            let cell_size = 16.0;
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
                        _ => im_col32(200, 100, 200, 255),
                    };
                    igDrawRectFilled(cell_min, cell_max, color);

                    // Tooltip logic
                    if igIsMouseHoveringRect(cell_min, cell_max, true) {
                        igBeginTooltip();
                        igText(CString::new(format!("{}: {}", s.start, s.to_string(&interpreter.mem))).unwrap().as_ptr());
                        igEndTooltip();
                    }

                    let inner_cell_min = ImVec2 {
                        x: cell_min.x + 4.0,
                        y: cell_min.y + 4.0,
                    };
                    let inner_cell_max = ImVec2 {
                        x: cell_max.x - 4.0,
                        y: cell_max.y - 4.0,
                    };

                    let inner_cell_border_min = ImVec2 {
                        x: cell_min.x + 2.0,
                        y: cell_min.y + 2.0,
                    };
                    let inner_cell_border_max = ImVec2 {
                        x: cell_max.x - 2.0,
                        y: cell_max.y - 2.0,
                    };

                    if let shimlang::MemDescriptorType::Struct(_, members) = &s.t {
                        let member_idx = idx - s.start;
                        let val = members[member_idx];
                        let (inner_color, border_color) = if let shimlang::ShimValue::Integer(i) = val {
                            let rgb = i32_to_rgb(i);
                            (
                                im_col32(
                                    rgb[0].into(),
                                    rgb[1].into(),
                                    rgb[2].into(),
                                    255
                                ),
                                im_col32(
                                    128,
                                    128,
                                    128,
                                    255
                                ),
                            )
                        } else if let shimlang::ShimValue::Float(f) = val {
                            let rgb = f32_to_rgb(f);
                            (
                                im_col32(
                                    rgb[0].into(),
                                    rgb[1].into(),
                                    rgb[2].into(),
                                    255,
                                ),
                                im_col32(
                                    100,
                                    200,
                                    100,
                                    255,
                                ),
                            )
                        } else {
                            (
                                im_col32(0, 0, 0, 255),
                                im_col32(0, 0, 0, 255),
                            )
                        };
                        igDrawRectFilled(inner_cell_border_min, inner_cell_border_max, border_color);
                        igDrawRectFilled(inner_cell_min, inner_cell_max, inner_color);
                    }
                } else if mem_end <= idx {
                    let color = im_col32(10, 10, 10, 255);
                    igDrawRectFilled(cell_min, cell_max, color);
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