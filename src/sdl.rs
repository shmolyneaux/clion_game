#[cfg(not(test))]
unsafe extern "C" {
    pub fn SDL_GL_GetProcAddress(proc: *const i8) -> *mut std::ffi::c_void;
    pub fn SDL_GetKeyboardState(numkeys: *mut i32) -> *const u8;
    pub fn SDL_GetRelativeMouseState(x: *mut i32, y: *mut i32) -> u32;
    pub fn SDL_SetRelativeMouseMode(enabled: bool) -> i32;
    pub fn SHM_GetDrawableSize(display_w: *mut i32, display_h: *mut i32);
}

#[cfg(test)]
mod test_mocks {
    pub fn SDL_GL_GetProcAddress(proc: *const i8) -> *mut std::ffi::c_void {
        panic!("Can't call SDL_GL_GetProcAddress in test context")
    }
    pub fn SDL_GetKeyboardState(numkeys: *mut i32) -> *const u8 {
        panic!("Can't call SDL_GetKeyboardState in test context")
    }
    pub fn SDL_GetRelativeMouseState(x: *mut i32, y: *mut i32) -> u32 {
        panic!("Can't call SDL_GetRelativeMouseState in test context")
    }
    pub fn SDL_SetRelativeMouseMode(enabled: bool) -> i32 {
        panic!("Can't call SDL_SetRelativeMouseMode in test context")
    }
    pub fn SHM_GetDrawableSize(display_w: *mut i32, display_h: *mut i32) {
        panic!("Can't call SHM_GetDrawableSize in test context")
    }
}

#[cfg(test)]
pub use test_mocks::*;

pub static SDL_SCANCODE_A: usize = 4;
pub static SDL_SCANCODE_B: usize = 5;
pub static SDL_SCANCODE_C: usize = 6;
pub static SDL_SCANCODE_D: usize = 7;
pub static SDL_SCANCODE_E: usize = 8;
pub static SDL_SCANCODE_F: usize = 9;
pub static SDL_SCANCODE_G: usize = 10;
pub static SDL_SCANCODE_H: usize = 11;
pub static SDL_SCANCODE_I: usize = 12;
pub static SDL_SCANCODE_J: usize = 13;
pub static SDL_SCANCODE_K: usize = 14;
pub static SDL_SCANCODE_L: usize = 15;
pub static SDL_SCANCODE_M: usize = 16;
pub static SDL_SCANCODE_N: usize = 17;
pub static SDL_SCANCODE_O: usize = 18;
pub static SDL_SCANCODE_P: usize = 19;
pub static SDL_SCANCODE_Q: usize = 20;
pub static SDL_SCANCODE_R: usize = 21;
pub static SDL_SCANCODE_S: usize = 22;
pub static SDL_SCANCODE_T: usize = 23;
pub static SDL_SCANCODE_U: usize = 24;
pub static SDL_SCANCODE_V: usize = 25;
pub static SDL_SCANCODE_W: usize = 26;
pub static SDL_SCANCODE_X: usize = 27;
pub static SDL_SCANCODE_Y: usize = 28;
pub static SDL_SCANCODE_Z: usize = 29;
pub static SDL_SCANCODE_ESCAPE: usize = 41;
pub static SDL_SCANCODE_LCTRL: usize = 224;
pub static SDL_SCANCODE_LSHIFT: usize = 225;
