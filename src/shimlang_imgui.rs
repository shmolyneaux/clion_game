use crate::{shimlang::*, *};


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
            igBegin(c"Shimlang Interpreter".as_ptr(), &mut open as *mut bool, IMGUI_WINDOW_FLAGS_NO_FOCUS_ON_APPEARING);
            igText(CString::new(format!("Interpreter source: {:#?}", interpreter.source)).unwrap().as_ptr());
            igText(CString::new(format!("Free list: {:#?}", interpreter.mem.free_list)).unwrap().as_ptr());

            // Memory viewer
            igText(CString::new(format!("Mem size is {}", interpreter.mem.mem.len())).unwrap().as_ptr());
            for (idx, x) in interpreter.mem.mem.iter().take(128).enumerate() {
                if !(idx % 4 == 0) {
                    igSameLine();
                } else {
                    igTextColored(0.6, 0.6, 0.6, 1.0, CString::new(format!("{:08X}", idx)).unwrap().as_ptr());
                    igSameLine();
                }
                igTextColored(1.0, 1.0, 1.0, 1.0, CString::new(format!("{:016X}", x)).unwrap().as_ptr());
            }
            igText(CString::new(format!("Did that work?")).unwrap().as_ptr());



            // TODO: Fill out the basic properties here
            // TODO: Add a source code viewer
            // TODO: Add support for executing single statements
            igEnd();
        }
    }
}