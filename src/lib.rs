#![allow(unused_variables, unused_mut, unused_imports, unused_attributes, unused_unsafe, dead_code, unsafe_op_in_unsafe_fn, non_snake_case)]

#[macro_use]
use gl;
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::rc::Rc;
use gl::types::*;
use glam::{Vec2, Vec3, Vec4, Mat4};
use std::ffi::CString;
use std::ffi::CStr;

use std::os::raw::{c_char, c_int, c_uint};

use std::ops::{Add, Mul};

use std::slice;

use std::mem::size_of;

use std::cell::RefCell;

use glam::Vec3Swizzles;

use facet::*;
use facet_reflect::*;

use std::fmt::Formatter;
use std::borrow::Cow;

mod gpu;
mod mesh_gen;
mod debug_draw;
mod sdf;

mod shimlang_imgui;

use crate::sdf::*;
use crate::gpu::*;
use crate::debug_draw::*;
use crate::mesh_gen::{box_mesh, quad_mesh};

use shm_tracy::*;

type ImGuiWindowFlags = c_int;

#[cfg(not(test))]
unsafe extern "C" {
    fn SDL_GL_GetProcAddress(proc: *const i8) -> *mut std::ffi::c_void;
    fn SDL_GetKeyboardState(numkeys: *mut i32) -> *const u8;
    fn SDL_GetRelativeMouseState(x: *mut i32, y: *mut i32) -> u32;
    fn SDL_SetRelativeMouseMode(enabled: bool) -> i32;
    fn SHM_GetDrawableSize(display_w: *mut i32, display_h: *mut i32);
    fn igBegin(name: *const core::ffi::c_char, p_open: *mut bool, flags: ImGuiWindowFlags) -> bool;
    fn igEnd();
    fn igBeginDisabled();
    fn igEndDisabled();
    fn igIsItemHovered(flags: i32) -> bool;
    fn igSetTooltip(text: *const core::ffi::c_char);
    fn igInputText(label: *const core::ffi::c_char, buffer: *mut core::ffi::c_char, buffer_size: i32, flags: i32) -> bool;
    fn igTextC(fmt: *const core::ffi::c_char, ...);
    fn igTextColoredC(r: f32, g: f32, b: f32, a: f32, fmt: *const core::ffi::c_char, ...);
    fn igButton(label: *const core::ffi::c_char) -> bool;
    fn igSliderFloat(label: *const core::ffi::c_char, v: *mut f32, v_min: f32, v_max: f32, format: *const core::ffi::c_char);
    fn igCheckbox(label: *const core::ffi::c_char, value: *mut bool) -> bool;
    fn igWantCaptureKeyboard() -> bool;
    fn igWantCaptureMouse() -> bool;
    fn igTreeNode(label: *const core::ffi::c_char) -> bool;
    fn igTreePop();
    fn igSHMNextItemOpenOnce();
    fn igSameLine();
    fn igSetKeyboardFocusHere();

    fn igSeparator();
    fn shmConsoleFooterHeight() -> f32;

    fn igBeginTable(label: *const core::ffi::c_char, columns: i32) -> bool;
    fn igTableSetupColumn(label: *const core::ffi::c_char);
    fn igTableHeadersRow();
    fn igTableNextRow();
    fn igTableSetColumnIndex(index: i32);
    fn igEndTable();

    fn igFrameRate() -> f32;
}

fn igText(fmt: *const core::ffi::c_char) {
    #[cfg(not(test))]
    unsafe { igTextC(fmt); }
}

fn igTextColored(r: f32, g: f32, b: f32, a: f32, fmt: *const core::ffi::c_char) {
    #[cfg(not(test))]
    unsafe { igTextColoredC(r, g, b, a, fmt); }
}

#[cfg(test)]
mod test_mocks {
    use crate::ImGuiWindowFlags;
    pub fn SDL_GL_GetProcAddress(proc: *const i8) -> *mut std::ffi::c_void {panic!("Can't call SDL_GL_GetProcAddress in test context")}
    pub fn SDL_GetKeyboardState(numkeys: *mut i32) -> *const u8 {panic!("Can't call SDL_GetKeyboardState in test context")}
    pub fn SDL_GetRelativeMouseState(x: *mut i32, y: *mut i32) -> u32 {panic!("Can't call SDL_GetRelativeMouseState in test context")}
    pub fn SDL_SetRelativeMouseMode(enabled: bool) -> i32 {panic!("Can't call SDL_SetRelativeMouseMode in test context")}
    pub fn SHM_GetDrawableSize(display_w: *mut i32, display_h: *mut i32) {panic!("Can't call SHM_GetDrawableSize in test context")}
    pub fn igBegin(name: *const core::ffi::c_char, p_open: *mut bool, flags: ImGuiWindowFlags) -> bool {panic!("Can't call igBegin in test context")}
    pub fn igEnd() {panic!("Can't call igEnd in test context")}
    pub fn igBeginDisabled() {panic!("Can't call igBeginDisabled in test context")}
    pub fn igEndDisabled() {panic!("Can't call igEndDisabled in test context")}
    pub fn igIsItemHovered(flags: i32) -> bool {panic!("Can't call igIsItemHovered in test context")}
    pub fn igSetTooltip(text: *const core::ffi::c_char) {panic!("Can't call igSetTooltip in test context")}
    pub fn igInputText(label: *const core::ffi::c_char, buffer: *mut core::ffi::c_char, buffer_size: i32, flags: i32) -> bool {panic!("Can't call igInputText in test context")}
    pub fn igButton(label: *const core::ffi::c_char) -> bool {panic!("Can't call igButton in test context")}
    pub fn igSliderFloat(label: *const core::ffi::c_char, v: *mut f32, v_min: f32, v_max: f32, format: *const core::ffi::c_char) {panic!("Can't call igSliderFloat in test context")}
    pub fn igCheckbox(label: *const core::ffi::c_char, value: *mut bool) -> bool {panic!("Can't call igCheckbox in test context")}
    pub fn igWantCaptureKeyboard() -> bool {panic!("Can't call igWantCaptureKeyboard in test context")}
    pub fn igWantCaptureMouse() -> bool {panic!("Can't call igWantCaptureMouse in test context")}
    pub fn igTreeNode(label: *const core::ffi::c_char) -> bool {panic!("Can't call fn  in test context")}
    pub fn igTreePop() {panic!("Can't call igTreePop in test context")}
    pub fn igSHMNextItemOpenOnce() {panic!("Can't call igSHMNextItemOpenOnce in test context")}
    pub fn igSameLine() {panic!("Can't call igSameLine in test context")}
    pub fn igSetKeyboardFocusHere() {panic!("Can't call igSetKeyboardFocusHere in test context")}
    pub fn igSeparator() {panic!("Can't call igSeparator in test context")}
    pub fn shmConsoleFooterHeight() -> f32 {panic!("Can't call shmConsoleFooterHeight in test context")}

    pub fn igBeginTable(label: *const core::ffi::c_char, columns: i32) -> bool {panic!("Can't call fn  in test context")}
    pub fn igTableSetupColumn(label: *const core::ffi::c_char) {panic!("Can't call igTableSetupColumn in test context")}
    pub fn igTableHeadersRow() {panic!("Can't call igTableHeadersRow in test context")}
    pub fn igTableNextRow() {panic!("Can't call igTableNextRow in test context")}
    pub fn igTableSetColumnIndex(index: i32) {panic!("Can't call igTableSetColumnIndex in test context")}
    pub fn igEndTable() {panic!("Can't call igEndTable in test context")}

    pub fn igFrameRate() -> f32 {panic!("Can't call igFrameRate in test context")}
}

#[cfg(test)]
fn tracy_zone_begin_n(name: *const c_char, active: c_int) -> TracyCZoneCtx {
    let id = 0;
    let active = 0;
    TracyCZoneCtx {
        id,
        active,
    }
}

#[cfg(test)]
fn tracy_zone_begin_ns(name: *const c_char, depth: c_int, active: c_int) -> TracyCZoneCtx {
    let id = 0;
    let active = 0;
    TracyCZoneCtx {
        id,
        active,
    }
}

#[cfg(test)]
fn tracy_zone_end(ctx: TracyCZoneCtx) {}
#[cfg(test)]
fn tracy_zone_text(ctx: TracyCZoneCtx, txt: *const c_char, len: c_uint) {}
#[cfg(test)]
fn tracy_zone_name(ctx: TracyCZoneCtx, txt: *const c_char, len: c_uint) {}
#[cfg(test)]
fn tracy_zone_color(ctx: TracyCZoneCtx, color: c_uint) {}

#[cfg(test)]
unsafe fn ___tracy_emit_zone_begin(loc: *const ___tracy_source_location_data, active: i32) -> TracyCZoneCtx {
    let id = 0;
    let active = 0;
    TracyCZoneCtx {
        id,
        active,
    }
}

#[cfg(test)]
unsafe fn ___tracy_emit_zone_end(ctx: TracyCZoneCtx) {}

#[cfg(test)]
use test_mocks::*;

#[macro_export]
macro_rules! cformat {
    ($($arg:tt)*) => {{
        std::ffi::CString::new(format!($($arg)*)).unwrap()
    }};
}

const fn u32size_of<T>() -> u32 {
    size_of::<T>() as u32
}

#[derive(Facet)]
pub struct KeyState {
    keys: Vec<u8>,
    last_keys: Vec<u8>,
}

impl KeyState {
    fn new() -> Self {
        unsafe {
            let mut numkeys: i32 = 0;
            SDL_GetKeyboardState(&mut numkeys as *mut i32);

            let keys = Vec::with_capacity(numkeys as usize);
            let last_keys = Vec::with_capacity(numkeys as usize);
            Self {
                keys,
                last_keys,
            }
        }
    }

    pub fn pressed(&self, sdl_key: usize) -> bool {
        self.keys[sdl_key] == 1
    }

    pub fn released(&self, sdl_key: usize) -> bool {
        self.keys[sdl_key] == 0
    }

    pub fn just_pressed(&self, sdl_key: usize) -> bool {
        self.keys[sdl_key] == 1 && self.keys[sdl_key] != self.last_keys[sdl_key]
    }

    pub fn just_released(&self, sdl_key: usize) -> bool {
        self.keys[sdl_key] == 0 && self.keys[sdl_key] != self.last_keys[sdl_key]
    }
}

#[derive(Facet, Default)]
pub struct DebugState {
    sphere_radius: f32,
    normal_shift: f32,
    sdf_test_draw_normals: bool,
    sdf_test_draw_hoop: bool,
    sdf_test_draw_wireframe: bool,
    sdf_box_size: Vec3,
    shimlang_debug_window: shimlang_imgui::Navigation,
    shimlang_repl: shimlang_imgui::Repl,
}


//#[derive(Facet)]
/// -180.0 180.0
pub struct State {
    interpreter: shimlang::Interpreter,
    env: shimlang::Environment,

    frame_num: u64,
    view: Mat4,
    projection: Mat4,

    debug_shader_program: Rc<ShaderProgram>,
    default_shader_program: Rc<ShaderProgram>,

    debug_view_loc: gl::types::GLint,
    debug_projection_loc: gl::types::GLint,

    debug_vao: VertexArray,
    debug_vbo: VertexBufferObject,
    debug_ebo: ElementBufferObject,

    debug_verts: Vec<(Vec3, Vec3)>,
    debug_vert_indices: Vec<u32>,

    //#[facet(range = (-90.0, 90.0))]
    pitch: f32,

    //#[facet(range = (-180.0, 180.0))]
    yaw: f32,

    camera_pos: Vec3,
    camera_front: Vec3,
    camera_up: Vec3,

    //#[facet(readonly)]
    mouse_captured: bool,

    test_mesh: StaticMesh,
    meshes: HashMap<String, StaticMesh>,

    keys: KeyState,

    debug_state: DebugState,
}

/// Parses a string of the form `"range =(-180.0, 180.0)"` into a tuple of two `f32` values.
///
/// This function expects the input to start with the exact prefix `"range = ("`,
/// followed by two floating-point numbers separated by a comma and ending with a closing parenthesis.
/// It trims any surrounding whitespace around the numbers, but does not tolerate malformed input.
///
/// # Examples
///
/// ```
/// let s = "range = (-180.0, 180.0)";
/// assert_eq!(parse_range(s), Some((-180.0, 180.0)));
/// ```
///
/// Returns `Some((f32, f32))` if parsing succeeds, or `None` if the format is invalid.
fn parse_facet_range(s: &str) -> Option<(f32, f32)> {
    let prefix = "range = (";

    let rest = s.strip_prefix(prefix)?.trim_start();

    // Find the comma separating the two numbers
    let comma_index = rest.find(',')?;

    // Split around the comma
    let (first_part, second_part_with_paren) = rest.split_at(comma_index);
    let first = first_part.trim().parse::<f32>().ok()?;

    // Skip comma and trim
    let second_part = second_part_with_paren[1..].trim(); // skip the comma

    // Should end with ')'
    if !second_part.ends_with(')') {
        return None;
    }

    let second_str = &second_part[..second_part.len() - 1]; // remove ')'
    let second = second_str.trim().parse::<f32>().ok()?;

    Some((first, second))
}


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

pub static IMGUI_WINDOW_FLAGS_NONE: c_int                         = 0;
pub static IMGUI_WINDOW_FLAGS_NO_TITLE_BAR: c_int                 = 1 << 0;   // Disable title-bar
pub static IMGUI_WINDOW_FLAGS_NO_RESIZE: c_int                    = 1 << 1;   // Disable user resizing with the lower-right grip
pub static IMGUI_WINDOW_FLAGS_NO_MOVE: c_int                      = 1 << 2;   // Disable user moving the window
pub static IMGUI_WINDOW_FLAGS_NO_SCROLLBAR: c_int                 = 1 << 3;   // Disable scrollbars (window can still scroll with mouse or programmatically)
pub static IMGUI_WINDOW_FLAGS_NO_SCROLL_WITH_MOUSE: c_int         = 1 << 4;   // Disable user vertically scrolling with mouse wheel. On child window, mouse wheel will be forwarded to the parent unless NoScrollbar is also set.
pub static IMGUI_WINDOW_FLAGS_NO_COLLAPSE: c_int                  = 1 << 5;   // Disable user collapsing window by double-clicking on it. Also referred to as Window Menu Button (e.g. within a docking node).
pub static IMGUI_WINDOW_FLAGS_ALWAYS_AUTO_RESIZE: c_int           = 1 << 6;   // Resize every window to its content every frame
pub static IMGUI_WINDOW_FLAGS_NO_BACKGROUND: c_int                = 1 << 7;   // Disable drawing background color (WindowBg, etc.) and outside border. Similar as using SetNextWindowBgAlpha(0.0f).
pub static IMGUI_WINDOW_FLAGS_NO_SAVED_SETTINGS: c_int            = 1 << 8;   // Never load/save settings in .ini file
pub static IMGUI_WINDOW_FLAGS_NO_MOUSE_INPUTS: c_int              = 1 << 9;   // Disable catching mouse, hovering test with pass through.
pub static IMGUI_WINDOW_FLAGS_MENU_BAR: c_int                     = 1 << 10;  // Has a menu-bar
pub static IMGUI_WINDOW_FLAGS_HORIZONTAL_SCROLLBAR: c_int         = 1 << 11;  // Allow horizontal scrollbar to appear (off by default). You may use SetNextWindowContentSize(ImVec2(width,0.0f)); prior to calling Begin() to specify width. Read code in imgui_demo in the "Horizontal Scrolling" section.
pub static IMGUI_WINDOW_FLAGS_NO_FOCUS_ON_APPEARING: c_int        = 1 << 12;  // Disable taking focus when transitioning from hidden to visible state
pub static IMGUI_WINDOW_FLAGS_NO_BRING_TO_FRONT_ON_FOCUS: c_int   = 1 << 13;  // Disable bringing window to front when taking focus (e.g. clicking on it or programmatically giving it focus)
pub static IMGUI_WINDOW_FLAGS_ALWAYS_VERTICAL_SCROLLBAR: c_int    = 1 << 14;  // Always show vertical scrollbar (even if ContentSize.y < Size.y)
pub static IMGUI_WINDOW_FLAGS_ALWAYS_HORIZONTAL_SCROLLBAR: c_int  = 1 << 15;  // Always show horizontal scrollbar (even if ContentSize.x < Size.x)
pub static IMGUI_WINDOW_FLAGS_NO_NAV_INPUTS: c_int                = 1 << 16;  // No keyboard/gamepad navigation within the window
pub static IMGUI_WINDOW_FLAGS_NO_NAV_FOCUS: c_int                 = 1 << 17;  // No focusing toward this window with keyboard/gamepad navigation (e.g. skipped by CTRL+TAB)
pub static IMGUI_WINDOW_FLAGS_UNSAVED_DOCUMENT: c_int             = 1 << 18;  // Display a dot next to the title. When used in a tab/docking context, tab is selected when clicking the X + closure is not assumed (will wait for user to stop submitting the tab). Otherwise closure is assumed when pressing the X, so if you keep submitting the tab may reappear at end of tab bar.
pub static IMGUI_WINDOW_FLAGS_NO_DOCKING: c_int                   = 1 << 19;  // Disable docking of this window

pub static IMGUI_HOVERED_FLAGS_ALLOW_WHEN_DISABLED: c_int         = 1 << 10;  // IsItemHovered() only: Return true even if the item is disabled

pub static IMGUI_INPUT_TEXT_FLAGS_NONE: c_int                = 0;
pub static IMGUI_INPUT_TEXT_FLAGS_CHARS_DECIMAL: c_int        = 1 << 0;   // Allow 0123456789.+-*/
pub static IMGUI_INPUT_TEXT_FLAGS_CHARS_HEXADECIMAL: c_int    = 1 << 1;   // Allow 0123456789ABCDEFabcdef
pub static IMGUI_INPUT_TEXT_FLAGS_CHARS_SCIENTIFIC: c_int     = 1 << 2;   // Allow 0123456789.+-*/eE (Scientific notation input)
pub static IMGUI_INPUT_TEXT_FLAGS_CHARS_UPPERCASE: c_int      = 1 << 3;   // Turn a..z into A..Z
pub static IMGUI_INPUT_TEXT_FLAGS_CHARS_NOBLANK: c_int        = 1 << 4;   // Filter out spaces, tabs

// Inputs
pub static IMGUI_INPUT_TEXT_FLAGS_ALLOW_TAB_INPUT: c_int       = 1 << 5;   // Pressing TAB input a '\t' character into the text field
pub static IMGUI_INPUT_TEXT_FLAGS_ENTER_RETURNS_TRUE: c_int    = 1 << 6;   // Return 'true' when Enter is pressed (as opposed to every time the value was modified). Consider using IsItemDeactivatedAfterEdit() instead!
pub static IMGUI_INPUT_TEXT_FLAGS_ESCAPE_CLEARS_ALL: c_int     = 1 << 7;   // Escape key clears content if not empty, and deactivate otherwise (contrast to default behavior of Escape to revert)
pub static IMGUI_INPUT_TEXT_FLAGS_CTRL_ENTER_FOR_NEWLINE: c_int = 1 << 8;   // In multi-line mode, validate with Enter, add new line with Ctrl+Enter (default is opposite: validate with Ctrl+Enter, add line with Enter).

// Other options
pub static IMGUI_INPUT_TEXT_FLAGS_READONLY: c_int            = 1 << 9;   // Read-only mode
pub static IMGUI_INPUT_TEXT_FLAGS_PASSWORD: c_int            = 1 << 10;  // Password mode, display all characters as '*', disable copy
pub static IMGUI_INPUT_TEXT_FLAGS_ALWAYSOVERWRITE: c_int     = 1 << 11;  // Overwrite mode
pub static IMGUI_INPUT_TEXT_FLAGS_AUTOSELECTALL: c_int       = 1 << 12;  // Select entire text when first taking mouse focus
pub static IMGUI_INPUT_TEXT_FLAGS_PARSEEMPTYREFVAL: c_int    = 1 << 13;  // InputFloat(), InputInt(), InputScalar() etc. only: parse empty string as zero value.
pub static IMGUI_INPUT_TEXT_FLAGS_DISPLAYEMPTYREFVAL: c_int  = 1 << 14;  // InputFloat(), InputInt(), InputScalar() etc. only: when value is zero, do not display it. Generally used with ImGuiInputTextFlags_ParseEmptyRefVal.
pub static IMGUI_INPUT_TEXT_FLAGS_NOHORIZONTALSCROLL: c_int  = 1 << 15;  // Disable following the cursor horizontally
pub static IMGUI_INPUT_TEXT_FLAGS_NOUNDOREDO: c_int          = 1 << 16;  // Disable undo/redo. Note that input text owns the text data while active, if you want to provide your own undo/redo stack you need e.g. to call ClearActiveID().

// Elide display / Alignment
pub static IMGUI_INPUT_TEXT_FLAGS_ELIDELEFT: c_int           = 1 << 17;  // When text doesn't fit, elide left side to ensure right side stays visible. Useful for path/filenames. Single-line only!

// Callback features
pub static IMGUI_INPUT_TEXT_FLAGS_CALLBACKCOMPLETION: c_int  = 1 << 18;  // Callback on pressing TAB (for completion handling)
pub static IMGUI_INPUT_TEXT_FLAGS_CALLBACKHISTORY: c_int     = 1 << 19;  // Callback on pressing Up/Down arrows (for history handling)
pub static IMGUI_INPUT_TEXT_FLAGS_CALLBACKALWAYS: c_int      = 1 << 20;  // Callback on each iteration. User code may query cursor position, modify text buffer.
pub static IMGUI_INPUT_TEXT_FLAGS_CALLBACKCHARFILTER: c_int  = 1 << 21;  // Callback on character inputs to replace or discard them. Modify 'EventChar' to replace or discard, or return 1 in callback to discard.
pub static IMGUI_INPUT_TEXT_FLAGS_CALLBACKRESIZE: c_int      = 1 << 22;  // Callback on buffer capacity changes request (beyond 'buf_size' parameter value), allowing the string to grow. Notify when the string wants to be resized (for string types which hold a cache of their Size). You will be provided a new BufSize in the callback and NEED to honor it. (see misc/cpp/imgui_stdlib.h for an example of using this)
pub static IMGUI_INPUT_TEXT_FLAGS_CALLBACKEDIT: c_int        = 1 << 23;  // Callback on any edit (note that InputText() already returns true on edit, the callback is useful mainly to manipulate the underlying buffer while focus is active)

thread_local! {
    static STATE_REFCELL: RefCell<Option<State>> = RefCell::default();
}

thread_local! {
    static DEBUG_LOG: RefCell<Vec<CString>> = RefCell::default();
}

#[macro_export]
macro_rules! log_opengl_errors {
    () => {
        unsafe {
            loop {
                let err = {
                    let _zone = zone_scoped!("gl::GetError()");
                    gl::GetError()
                };
                match err {
                    gl::NO_ERROR => break,
                    gl::INVALID_ENUM => log(format!("[{}:{}] OpenGL Error INVALID_ENUM: An unacceptable value is specified for an enumerated argument.", file!(), line!())),
                    gl::INVALID_VALUE => log(format!("[{}:{}] OpenGL Error INVALID_VALUE: A numeric argument is out of range.", file!(), line!())),
                    gl::INVALID_OPERATION => log(format!("[{}:{}] OpenGL Error INVALID_OPERATION: The specified operation is not allowed in the current state.", file!(), line!())),
                    gl::INVALID_FRAMEBUFFER_OPERATION => log(format!("[{}:{}] OpenGL Error INVALID_FRAMEBUFFER_OPERATION: The framebuffer object is not complete.", file!(), line!())),
                    gl::OUT_OF_MEMORY => log(format!("[{}:{}] OpenGL Error OUT_OF_MEMORY: Not enough memory to execute the command.", file!(), line!())),
                    gl::STACK_UNDERFLOW => log(format!("[{}:{}] OpenGL Error STACK_UNDERFLOW: Stack underflow detected.", file!(), line!())),
                    gl::STACK_OVERFLOW => log(format!("[{}:{}] OpenGL Error STACK_OVERFLOW: Stack overflow detected.", file!(), line!())),
                    _ => log(format!("[{}:{}] OpenGL Error: Unknown error code {}", file!(), line!(), err)),
                }
            }
        }
    };
}

trait BitCheck {
    fn bit(&self, n: u32) -> bool;
}

impl BitCheck for u32 {
    fn bit(&self, n: u32) -> bool {
        (self >> n) & 1 != 0
    }
}

fn log<T: AsRef<str>>(s: T) {
    logc(CString::new(s.as_ref().to_string()).unwrap());
}

fn logc(s: CString) {
    println!("{:?}", s);
    DEBUG_LOG.with_borrow_mut(|logs| logs.push(s));
}

fn draw_log_window() {
    DEBUG_LOG.with_borrow(|logs| {
        unsafe {
            let mut open = true;
            igBegin(
                c"Rust Log Window".as_ptr(),
                &mut open as *mut bool,
                IMGUI_WINDOW_FLAGS_NO_FOCUS_ON_APPEARING,
            );
            for line in logs.iter() {
                igText(line.as_ptr());
            }
            igEnd();
        }
    });
}

fn format_shape_typename(shape: &Shape) -> String {
    let mut output = String::new();
    fmt::write(
        &mut output,
        format_args!("{}", FormatWrapper { func: shape.vtable.type_name()})
    ).unwrap();
    output
}

struct FormatWrapper {
    func: TypeNameFn,
}

impl fmt::Display for FormatWrapper {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        (self.func)(f, TypeNameOpts::infinite())
    }
}

fn imgui_debug<'a, T: Facet<'a>>(obj: &mut T) {
    let poke = Peek::new(obj);

    let mut open = true;
    unsafe {
        igBegin(c"State".as_ptr(), &mut open as *mut bool, IMGUI_WINDOW_FLAGS_NO_FOCUS_ON_APPEARING);
        imgui_debug_inner(&poke, &[], "");
        igEnd();
    }
}

unsafe fn imgui_debug_inner(peek: &Peek, attributes: &[FieldAttribute], path: &str) {
    let shape = peek.shape();
    match (peek.scalar_type(), shape.def) {
        (Some(ScalarType::Unit), _) => igText(CString::new(format!("()")).unwrap().as_ptr()),
        (Some(ScalarType::Bool), _) => {
            let mut readonly = false;
            for attr in attributes.iter() {
                let attr_str = match attr {
                    FieldAttribute::Arbitrary(s) => s,
                    _ => continue,
                };
                if *attr_str == "readonly" {
                    readonly = true;
                    break;
                }
            }

            if readonly {
                igText(CString::new(format!("{}", peek.get::<bool>().unwrap())).unwrap().as_ptr());
            } else {
                let ptr = unsafe { peek.data().thin().unwrap().as_ptr::<bool>() as *mut bool };
                igCheckbox(
                    CString::new(path).unwrap().as_ptr(),
                    ptr,
                );
            }
        },
        (Some(ScalarType::Char), _) => igText(CString::new(format!("{}", peek.get::<char>().unwrap())).unwrap().as_ptr()),
        (Some(ScalarType::Str), _) => igText(CString::new(format!("{}", peek.get::<&str>().unwrap())).unwrap().as_ptr()),
        (Some(ScalarType::String), _) => igText(CString::new(format!("{}", peek.get::<String>().unwrap())).unwrap().as_ptr()),
        (Some(ScalarType::CowStr), _) => igText(CString::new(format!("{}", peek.get::<Cow<str>>().unwrap())).unwrap().as_ptr()),
        (Some(ScalarType::F32), _) => {
            let ptr = unsafe { peek.data().thin().unwrap().as_ptr::<f32>() as *mut f32 };
            let mut min = 0.0;
            let mut max = 1.0;

            for attr in attributes.iter() {
                let attr_str = match attr {
                    FieldAttribute::Arbitrary(s) => s,
                    _ => continue,
                };
                if let Some(v) = parse_facet_range(attr_str) {
                    min = v.0;
                    max = v.1;
                }
            }
            igSliderFloat(
                CString::new(path).unwrap().as_ptr(),
                ptr,
                min,
                max,
                c"%.3f".as_ptr(),
            );
        },
        (Some(ScalarType::F64), _) => igText(CString::new(format!("{}f64", peek.get::<f64>().unwrap())).unwrap().as_ptr()),
        (Some(ScalarType::U8), _) => igText(CString::new(format!("{}u8", peek.get::<u8>().unwrap())).unwrap().as_ptr()),
        (Some(ScalarType::U16), _) => igText(CString::new(format!("{}u16", peek.get::<u16>().unwrap())).unwrap().as_ptr()),
        (Some(ScalarType::U32), _) => igText(CString::new(format!("{}u32", peek.get::<u32>().unwrap())).unwrap().as_ptr()),
        (Some(ScalarType::U64), _) => igText(CString::new(format!("{}u64", peek.get::<u64>().unwrap())).unwrap().as_ptr()),
        (Some(ScalarType::U128), _) => igText(CString::new(format!("{}u128", peek.get::<u128>().unwrap())).unwrap().as_ptr()),
        (Some(ScalarType::USize), _) => igText(CString::new(format!("{}usize", peek.get::<usize>().unwrap())).unwrap().as_ptr()),
        (Some(ScalarType::I8), _) => igText(CString::new(format!("{}i8", peek.get::<i8>().unwrap())).unwrap().as_ptr()),
        (Some(ScalarType::I16), _) => igText(CString::new(format!("{}i16", peek.get::<i16>().unwrap())).unwrap().as_ptr()),
        (Some(ScalarType::I32), _) => {
            let ptr = unsafe { peek.data().thin().unwrap().as_ptr::<i32>() as *mut i32 };
            igText(CString::new(format!("{}i32", peek.get::<i32>().unwrap())).unwrap().as_ptr());
            // unsafe { *ptr = *ptr - 1 };
        },
        (Some(ScalarType::I64), _) => igText(CString::new(format!("{}i64", peek.get::<i64>().unwrap())).unwrap().as_ptr()),
        (Some(ScalarType::I128), _) => igText(CString::new(format!("{}128", peek.get::<i128>().unwrap())).unwrap().as_ptr()),
        (Some(ScalarType::ISize), _) => igText(CString::new(format!("{}isize", peek.get::<isize>().unwrap())).unwrap().as_ptr()),
        (_, Def::Scalar(def)) => igText(CString::new(format!("Def::Scalar")).unwrap().as_ptr()),
        (_, Def::List(def)) => {
            let shape_typename = format_shape_typename(peek.shape());
            let peek = peek.into_list_like().unwrap();
            if peek.len() == 0 {
                igText(CString::new(format!("[]")).unwrap().as_ptr());
            }
            else {
                if igTreeNode(
                    CString::new(
                        format!(
                            "{}##{}",
                            shape_typename,
                            path,
                        )
                    ).unwrap().as_ptr()
                ) {
                    for (idx, item) in peek.iter().enumerate() {
                        imgui_debug_inner(
                            &item,
                            &[],
                            &format!("{}[{}]", path, idx),
                        );
                    }
                    igTreePop();
                }
            }
        },
        (_, Def::Map(def)) => igText(CString::new(format!("Def::Map")).unwrap().as_ptr()),
        (_, Def::Set(def)) => igText(CString::new(format!("Def::Set")).unwrap().as_ptr()),
        (_, Def::Array(def)) => {
            let shape_typename = format_shape_typename(peek.shape());
            let peek = peek.into_list_like().unwrap();
            if peek.len() == 0 {
                igText(CString::new(format!("[]")).unwrap().as_ptr());
            }
            else {
                if igTreeNode(
                    CString::new(
                        format!(
                            "{}##{}",
                            shape_typename,
                            path,
                        )
                    ).unwrap().as_ptr()
                ) {
                    for (idx, item) in peek.iter().enumerate() {
                        imgui_debug_inner(
                            &item,
                            &[],
                            &format!("{}[{}]", path, idx),
                        );
                    }
                    igTreePop();
                }
            }
        },
        (_, Def::Slice(def)) => igText(CString::new(format!("Def::Slice")).unwrap().as_ptr()),
        (_, Def::Option(def)) => igText(CString::new(format!("Def::Option")).unwrap().as_ptr()),
        (_, Def::SmartPointer(def)) => igText(CString::new(format!("Def::SmartPointer")).unwrap().as_ptr()),
        (_, Def::Undefined) => {
            let ty = shape.ty;
            match ty {
                Type::Primitive(ty) => igText(CString::new(format!("Type::Primitive")).unwrap().as_ptr()),
                Type::Sequence(ty) => igText(CString::new(format!("Type::Sequence")).unwrap().as_ptr()),
                Type::User(UserType::Struct(ty)) => {
                    // TODO: If this is a tuple struct with a single element, don't print the field
                    if path == "" {
                        igSHMNextItemOpenOnce();
                    }
                    if igTreeNode(
                        CString::new(
                            format!(
                                "{}##{}",
                                format_shape_typename(peek.shape()),
                                path,
                            )).unwrap().as_ptr()
                    ) {
                        let peek = peek.into_struct().unwrap();

                        if igBeginTable(CString::new(path).unwrap().as_ptr(), 2) {
                            // TODO:
                            //     ImGui::TableSetupColumn("Key", ImGuiTableColumnFlags_WidthFixed);
                            //     ImGui::TableSetupColumn("Value", ImGuiTableColumnFlags_WidthStretch);
                            igTableSetupColumn(c"Attribute".as_ptr());
                            igTableSetupColumn(c"Value".as_ptr());

                            // Don't show the header row (we only keep it for specifying column info)
                            // igTableHeadersRow();

                            for field_idx in 0..peek.field_count() {
                                igTableNextRow();

                                igTableSetColumnIndex(0);
                                let field = peek.field(field_idx).unwrap();
                                let field_shape = ty.fields[field_idx];
                                let field_name = field_shape.name;
                                igText(CString::new(format!("{field_name}: ")).unwrap().as_ptr());

                                igTableSetColumnIndex(1);
                                imgui_debug_inner(
                                    &field,
                                    &field_shape.attributes,
                                    &format!("{}.{}", path, field_name),
                                );
                            }

                            igEndTable();
                        }
                        igTreePop();
                    }
                },
                Type::User(UserType::Enum(ty)) => igText(CString::new(format!("UserType::Enum")).unwrap().as_ptr()),
                Type::User(UserType::Union(ty)) => igText(CString::new(format!("UserType::Union")).unwrap().as_ptr()),
                Type::User(UserType::Opaque) => igText(CString::new(format!("UserType::Opaque")).unwrap().as_ptr()),
                Type::Pointer(ty) => igText(CString::new(format!("Type::Pointer")).unwrap().as_ptr()),
                _ => igText(CString::new(format!("Can't debug {}", format_shape_typename(peek.shape()))).unwrap().as_ptr()),
            }
        },
        _ => igText(CString::new(format!("Can't debug {}", format_shape_typename(peek.shape()))).unwrap().as_ptr()),
    }
}

fn gen_cpu_texture() -> GLuint {
    let mut texture: GLuint = 0;

    let width: i32 = 128;
    let height: i32 = 128;

    let mut my_data: Vec<u8> = vec![
        0xff; (width*height*3) as usize
    ];

    for x in 0..width {
        for y in 0..height {
            my_data[((y*height + x)*3) as usize] = 0xff;//x as u8;
            my_data[((y*height + x)*3+1) as usize] = 0x0;//y as u8;
            my_data[((y*height + x)*3+2) as usize] = 0xff;
        }
    }

    unsafe {
        gl::GenTextures(1, &mut texture as *mut u32);
        gl::BindTexture(gl::TEXTURE_2D, texture);

        // set the texture wrapping parameters
        gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_WRAP_S, gl::CLAMP_TO_EDGE as i32);
        gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_WRAP_T, gl::CLAMP_TO_EDGE as i32);
        // set texture filtering parameters
        gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_MIN_FILTER, gl::LINEAR as i32);
        gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_MAG_FILTER, gl::LINEAR as i32);

        gl::TexImage2D(gl::TEXTURE_2D, 0, gl::RGB as i32, width, height, 0, gl::RGB, gl::UNSIGNED_BYTE, my_data.as_ptr().cast());
        gl::GenerateMipmap(gl::TEXTURE_2D);
    }

    log(format!("Created texture in Rust {}", texture));

    texture
}

fn fbo_test_mesh(size: Vec3) -> MeshDataRaw {
    let mut my_box = box_mesh(Vec3::new(2.0,2.0,2.0));
    let len = my_box.verts.get("aPos").unwrap().len();

    let mut uvs = vec![Vec2::new(0.0, 0.0); len];
    match my_box.verts.get("aPos").unwrap() {
        VertVec::Vec3(positions) => {
            for (uv, pos) in uvs.iter_mut().zip(positions.iter()) {
                uv.x = pos.x;
                uv.y = pos.y;
            }
        },
        _ => panic!("Why is aPos not a Vec3?"),
    }
    my_box.verts.insert("aUV".to_string(), VertVec::Vec2(uvs));

    let mut color = vec![Vec3::new(1.0, 0.0, 1.0); len];
    for (idx, vert_color) in color.iter_mut().enumerate() {
        vert_color.x = idx as f32 / len as f32;
        vert_color.y = idx as f32 / len as f32;
        vert_color.z = idx as f32 / len as f32;
    }
    my_box.verts.insert("aColor".to_string(), VertVec::Vec3(color));

    my_box
}

fn state_test_mesh() -> MeshDataRaw {
    let mut my_box = box_mesh(Vec3::new(1.0, 1.0, 1.0));
    let len = my_box.verts.get("aPos").unwrap().len();

    let mut uvs = vec![Vec2::new(0.0, 0.0); len];
    match my_box.verts.get("aPos").unwrap() {
        VertVec::Vec3(positions) => {
            for (uv, pos) in uvs.iter_mut().zip(positions.iter()) {
                uv.x = pos.x + 0.5;
                uv.y = pos.y + 0.5;
            }
        },
        _ => panic!("Why is aPos not a Vec3?"),
    }
    my_box.verts.insert("aUV".to_string(), VertVec::Vec2(uvs));

    let mut color = vec![Vec3::new(1.0, 0.0, 1.0); len];
    for (idx, vert_color) in color.iter_mut().enumerate() {
        vert_color.x = idx as f32 / len as f32;
        vert_color.y = idx as f32 / len as f32;
        vert_color.z = idx as f32 / len as f32;
    }
    my_box.verts.insert("aColor".to_string(), VertVec::Vec3(color));

    my_box
}

const fn compile_time_checks() {
    assert!(2 + 2 == 4);
}

fn gen_fbo_texture(code: &str) -> u32 {
     let view = Mat4::IDENTITY;
     let projection = Mat4::IDENTITY;

     let texture_shader = ShaderProgram::create(
         ShaderBuilder::new()
             .with_input(ShaderSymbol::new(ShaderDataType::Vec3, "aPos"))
             .with_input(ShaderSymbol::new(ShaderDataType::Vec2, "aUV"))
             .with_output(ShaderSymbol::new(ShaderDataType::Vec2, "uv"))
             .with_code(
                 r#"
                     void main() {
                         gl_Position = vec4(aPos, 1.0);
                         uv = aUV;
                     }
                 "#.to_string()
             ).build_vertex_shader().unwrap(),
         ShaderBuilder::new()
             .with_input(ShaderSymbol::new(ShaderDataType::Vec2, "uv"))
             .with_output(ShaderSymbol::new(ShaderDataType::Vec4, "FragColor"))
             .with_code(code.to_string())
             .build_fragment_shader().unwrap()
         ).expect("Could not build texture shader");
     let texture_shader = Rc::new(texture_shader);

     println!("Creating test mesh");
     let mut test_mesh = StaticMesh::create(
         texture_shader.clone(),
         Rc::new(
             Mesh::create(
                 &fbo_test_mesh(Vec3::new(1.0, 1.0, 1.0))
             ).unwrap()),
     ).expect("Can't create the test mesh");

     let mut fbo: GLuint = 0;
     let mut texture: GLuint = 0;

     unsafe {
         // Step 1: Create a framebuffer
         gl::GenFramebuffers(1, &mut fbo as *mut u32);
         gl::BindFramebuffer(gl::FRAMEBUFFER, fbo);

         // Step 2: Create a texture to render into
         gl::GenTextures(1, &mut texture as *mut u32);
         gl::BindTexture(gl::TEXTURE_2D, texture);
         gl::TexImage2D(gl::TEXTURE_2D, 0, gl::RGB as i32, 128, 128, 0, gl::RGB, gl::UNSIGNED_BYTE, std::ptr::null_mut());

         // Setup texture parameters
         gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_MIN_FILTER, gl::LINEAR as i32);
         gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_MAG_FILTER, gl::LINEAR as i32);

         // Step 3: Attach the texture to the framebuffer
         gl::FramebufferTexture2D(gl::FRAMEBUFFER, gl::COLOR_ATTACHMENT0, gl::TEXTURE_2D, texture, 0);

         // Step 4: Check if framebuffer is complete
         if gl::CheckFramebufferStatus(gl::FRAMEBUFFER) != gl::FRAMEBUFFER_COMPLETE {
             log("Error: Framebuffer is not complete!");
         }

         // Step 5: Render to the framebuffer
         gl::BindFramebuffer(gl::FRAMEBUFFER, fbo);
         gl::Viewport(0, 0, 128, 128); // Set viewport size to match the texture

         // Clear and render the scene
         gl::Clear(gl::COLOR_BUFFER_BIT | gl::DEPTH_BUFFER_BIT);

         test_mesh.draw(&mut HashMap::new());

         // Step 6: Switch back to the default framebuffer
         gl::BindFramebuffer(gl::FRAMEBUFFER, 0);
     }

     log(format!("Created FBO texture {}", texture));

     texture
}

fn create_default_shader() -> Rc<ShaderProgram> {
    let default_vertex_shader = ShaderBuilder::new()
        .with_input(ShaderSymbol::new(ShaderDataType::Vec3, "aPos"))
        .with_input(ShaderSymbol::new(ShaderDataType::Vec2, "aUV"))
        .with_uniform(ShaderSymbol::new(ShaderDataType::Mat4, "projection"))
        .with_uniform(ShaderSymbol::new(ShaderDataType::Mat4, "view"))
        .with_output(ShaderSymbol::new(ShaderDataType::Vec2, "uv"))
        .with_code(
            r#"
                void main() {
                    gl_Position = projection * view * vec4(aPos, 1.0);
                    uv = aUV;
                }
            "#.to_string()
        )
        .build_vertex_shader()
        .expect("Could not compile default vertex shader");
     log_opengl_errors!();

    let default_fragment_shader = ShaderBuilder::new()
        .with_input(ShaderSymbol::new(ShaderDataType::Vec2, "uv"))
        .with_output(ShaderSymbol::new(ShaderDataType::Vec4, "FragColor"))
        .with_code(
            r#"
                void main() {
                    FragColor = vec4(uv, 0.0, 1.0);
                }
            "#.to_string()
        )
        .build_fragment_shader()
        .expect("Could not compiled default fragment shader");

    println!("Starting default shader compilation");
    let default_shader_program = ShaderProgram::create(default_vertex_shader, default_fragment_shader).expect("Could not link default shader");
    println!("Linked default shader");

    let default_shader_program = Rc::new(default_shader_program);
    println!("Shader program now in RC");

    default_shader_program
}

fn init_state() -> State {
    log_opengl_errors!();
    println!("Starting Rust state initialization");

    println!("Creating interpreter");
    let interpreter_config = shimlang::Config::default();
    let ast = shimlang::ast_from_text(br#"

    struct Point {
        x,
        y
    }

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

    println!("Generating arrays/buffers");
    let debug_vao = VertexArray::create();
    let debug_vbo = VertexBufferObject::create();
    let debug_ebo = ElementBufferObject::create();

    let view = Mat4::IDENTITY;
    let projection = Mat4::IDENTITY;

    let debug_verts = Vec::new();
    let debug_vert_indices = Vec::new();

     log_opengl_errors!();
    let debug_vertex_shader = ShaderBuilder::new()
        .with_input(ShaderSymbol::new(ShaderDataType::Vec3, "aPos"))
        .with_input(ShaderSymbol::new(ShaderDataType::Vec3, "aColor"))
        .with_uniform(ShaderSymbol::new(ShaderDataType::Mat4, "projection"))
        .with_uniform(ShaderSymbol::new(ShaderDataType::Mat4, "view"))
        .with_output(ShaderSymbol::new(ShaderDataType::Vec3, "color"))
        .with_code(
            r#"
                void main() {
                    gl_Position = projection * view * vec4(aPos, 1.0);
                    color = aColor;
                }
            "#.to_string()
        )
        .build_vertex_shader()
        .expect("Could not compile debug vertex shader");
     log_opengl_errors!();

    let debug_fragment_shader = ShaderBuilder::new()
        .with_input(ShaderSymbol::new(ShaderDataType::Vec3, "color"))
        .with_output(ShaderSymbol::new(ShaderDataType::Vec4, "FragColor"))
        .with_code(
            r#"
                void main() {
                    FragColor = vec4(color, 1.0);
                }
            "#.to_string()
        )
        .build_fragment_shader()
        .expect("Could not compiled debug fragment shader");

    println!("Starting debug shader compilation");
    let debug_shader_program = ShaderProgram::create(debug_vertex_shader, debug_fragment_shader).expect("Could not link debug shader");
    println!("Linked debug shader");

    let debug_shader_program = Rc::new(debug_shader_program);
    println!("Shader program now in RC");

    log_opengl_errors!();

    let debug_view_loc = debug_shader_program.uniform_location(c"view");
    let debug_projection_loc = debug_shader_program.uniform_location(c"projection");

    println!("Successfully initialized Rust State");

    let frame_num = 0;

    let camera_pos = Vec3::new(0.0f32, 2.8f32, 7.7f32);
    let camera_front = Vec3::new(0.0f32, 0.0f32, -1.0f32);
    let camera_up = Vec3::new(0.0f32, 1.0f32,  0.0f32);

    let pitch = -20.;
    let yaw = -90.;

    let mouse_captured = true;

    let default_vertex_shader = ShaderBuilder::new()
        .with_input(ShaderSymbol::new(ShaderDataType::Vec3, "aPos"))
        .with_uniform(ShaderSymbol::new(ShaderDataType::Mat4, "model"))
        .with_uniform(ShaderSymbol::new(ShaderDataType::Mat4, "projection"))
        .with_uniform(ShaderSymbol::new(ShaderDataType::Mat4, "view"))
        .with_code(
            r#"
                void main() {
                    gl_Position = projection * view * model * vec4(aPos, 1.0);
                }
            "#.to_string()
        )
        .build_vertex_shader()
        .expect("Could not compile default vertex shader");

    let default_fragment_shader = ShaderBuilder::new()
        .with_output(ShaderSymbol::new(ShaderDataType::Vec4, "FragColor"))
        .with_code(
            r#"
                void main() {
                    FragColor = vec4(0.0, 1.0, 0.0, 1.0);
                }
            "#.to_string()
        )
        .build_fragment_shader()
        .expect("Could not compiled default fragment shader");
     log_opengl_errors!();

    println!("Starting default shader compilation");
    let default_shader = ShaderProgram::create(default_vertex_shader, default_fragment_shader).expect("Could not link default shader");
    println!("Default shader created");
    let _default_shader = Rc::new(default_shader);

    println!("Creating vert color shader");
    let vert_color_shader = ShaderProgram::create(
        ShaderBuilder::new()
            .with_input(ShaderSymbol::new(ShaderDataType::Vec3, "aPos"))
            .with_input(ShaderSymbol::new(ShaderDataType::Vec3, "aColor"))
            .with_uniform(ShaderSymbol::new(ShaderDataType::Mat4, "model"))
            .with_uniform(ShaderSymbol::new(ShaderDataType::Mat4, "projection"))
            .with_uniform(ShaderSymbol::new(ShaderDataType::Mat4, "view"))
            .with_output(ShaderSymbol::new(ShaderDataType::Vec3, "color"))
            .with_code(
                r#"
                    void main() {
                        gl_Position = projection * view * model * vec4(aPos, 1.0);
                        color = aColor;
                    }
                "#.to_string()
            ).build_vertex_shader().unwrap(),
        ShaderBuilder::new()
            .with_input(ShaderSymbol::new(ShaderDataType::Vec3, "color"))
            .with_output(ShaderSymbol::new(ShaderDataType::Vec4, "FragColor"))
            .with_code(
                r#"
                    void main() {
                        FragColor = vec4(color, 1.0);
                    }
                "#.to_string()
            ).build_fragment_shader().unwrap()
        ).expect("Could not build color shader");
    let vert_color_shader = Rc::new(vert_color_shader);

    let texture_shader = ShaderProgram::create(
        ShaderBuilder::new()
            .with_input(ShaderSymbol::new(ShaderDataType::Vec3, "aPos"))
            .with_input(ShaderSymbol::new(ShaderDataType::Vec2, "aUV"))
            .with_uniform(ShaderSymbol::new(ShaderDataType::Mat4, "model"))
            .with_uniform(ShaderSymbol::new(ShaderDataType::Mat4, "projection"))
            .with_uniform(ShaderSymbol::new(ShaderDataType::Mat4, "view"))
            .with_output(ShaderSymbol::new(ShaderDataType::Vec2, "uv"))
            .with_code(
                r#"
                    void main() {
                        gl_Position = projection * view * model * vec4(aPos, 1.0);
                        uv = aUV;
                    }
                "#.to_string()
            ).build_vertex_shader().unwrap(),
        ShaderBuilder::new()
            .with_input(ShaderSymbol::new(ShaderDataType::Vec2, "uv"))
            .with_output(ShaderSymbol::new(ShaderDataType::Vec4, "FragColor"))
            .with_uniform(ShaderSymbol::new(ShaderDataType::Sampler2D, "texture1"))
            .with_code(
                r#"
                    void main() {
                        FragColor = texture(texture1, uv);
                    }
                "#.to_string()
            ).build_fragment_shader().unwrap()
        ).expect("Could not build texture shader");
    let texture_shader = Rc::new(texture_shader);

    println!("Creating test mesh");
    let mut test_mesh = StaticMesh::create(
        texture_shader.clone(),
        Rc::new(Mesh::create(&state_test_mesh()).unwrap()),
    ).expect("Can't create the test mesh");


    let texture: GLuint = gen_fbo_texture(
         r#"
             void main() {
                 float value = smoothstep(0.8, 1.0, uv.x*uv.x + uv.y*uv.y);
                 FragColor = vec4(value, value, value, 1.0);
             }
         "#
    );
    log_opengl_errors!();

    test_mesh.uniform_override.insert("texture1".to_string(), gpu::ShaderValue::Sampler2D(texture));

    let mut numkeys: i32 = 0;
    unsafe { SDL_GetKeyboardState(&mut numkeys as *mut i32) };
    let keys = KeyState::new();

    let mut meshes = HashMap::new();

    let red_shader = ShaderProgram::create(
        ShaderBuilder::new()
            .with_input(ShaderSymbol::new(ShaderDataType::Vec3, "aPos"))
            .with_input(ShaderSymbol::new(ShaderDataType::Vec3, "aNormal"))
            .with_uniform(ShaderSymbol::new(ShaderDataType::Mat4, "model"))
            .with_uniform(ShaderSymbol::new(ShaderDataType::Mat4, "projection"))
            .with_uniform(ShaderSymbol::new(ShaderDataType::Mat4, "view"))
            .with_output(ShaderSymbol::new(ShaderDataType::Vec3, "normal"))
            .with_code(
                r#"
                    void main() {
                        gl_Position = projection * view * model * vec4(aPos, 1.0);
                        normal = aNormal;
                    }
                "#.to_string()
            ).build_vertex_shader().unwrap(),
        ShaderBuilder::new()
            .with_input(ShaderSymbol::new(ShaderDataType::Vec3, "normal"))
            .with_output(ShaderSymbol::new(ShaderDataType::Vec4, "FragColor"))
            .with_uniform(ShaderSymbol::new(ShaderDataType::Vec3, "color"))
            .with_code(
                r#"
                    void main() {
                        FragColor = vec4(color, 1.0);
                    }
                "#.to_string()
            ).build_fragment_shader().unwrap()
        ).expect("Could not build red shader");
    let red_shader = Rc::new(red_shader);

    let mut my_box = box_mesh(Vec3::new(1.0, 1.0, 1.0));
    let mut box_static_mesh = StaticMesh::create(
        red_shader.clone(),
        Rc::new(Mesh::create(&my_box).unwrap()),
    ).expect("Can't create the test mesh");

    meshes.insert("red box".to_string(), box_static_mesh);


    let mut debug_state = DebugState::default();
    debug_state.sphere_radius = 1.0;
    debug_state.sdf_box_size = Vec3::new(0.5, 0.7, 1.0);

    let default_shader_program = create_default_shader();
    log_opengl_errors!();

    println!("State initialized!");
    let mut state = State {
        interpreter,
        env,
        frame_num,
        debug_vao,
        debug_vbo,
        debug_ebo,
        view,
        projection,
        debug_view_loc,
        debug_projection_loc,
        debug_shader_program,
        default_shader_program,
        debug_verts,
        debug_vert_indices,
        camera_pos,
        camera_front,
        camera_up,
        pitch,
        yaw,
        mouse_captured,
        test_mesh,
        keys,
        meshes,
        debug_state,
    };

    state
}

#[unsafe(no_mangle)]
pub extern "C" fn rust_init() -> i32 {

    unsafe {
        gl::load_with(|s| SDL_GL_GetProcAddress(CString::new(s).unwrap().as_ptr()));
    }

    STATE_REFCELL.with_borrow_mut(|value| {
        let initial_state = init_state();
        *value = Some(initial_state);
    });
    0
}

fn update_keys(state: &mut State) {
    // Move the keys from the last frame to `last_keys`
    std::mem::swap(&mut state.keys.keys, &mut state.keys.last_keys);

    // Copy the SDL key state to state.keys
    unsafe {
        let mut numkeys: i32 = 0;
        let key_state = SDL_GetKeyboardState(&mut numkeys as *mut i32);
        let src_slice = slice::from_raw_parts(key_state, numkeys as usize);
        state.keys.keys.clear();
        state.keys.keys.reserve(numkeys as usize);
        state.keys.keys.extend_from_slice(src_slice);
    }
}

fn frame(state: &mut State, delta: f32) {
    if state.frame_num == 00 {
        println!("Starting first frame");
    }

    {
        let _zone = zone_scoped!("imgui_debug");
        //imgui_debug(state);
    }

    {
        let _zone = zone_scoped!("Run interpreter");
        let mut pc = 0;
        match state.interpreter.execute_bytecode_extended(&mut pc, shimlang::ArgBundle::new(), &mut state.env) {
            Ok(_) => {},
            Err(msg) => {
                eprintln!("{msg}");
            }
        };
        state.interpreter.gc(&state.env);
    }

    unsafe {
        let _zone = zone_scoped!("rust frame unsafe block");
        update_keys(state);

        let test = vec![0.1, 0.1, 0.12, 1.0];
        gl::ClearColor(
            test[0],
            test[1],
            test[2],
            test[3],
        );

        debug_line_loop(
            state,
            &[
                Vec3::new(-1.0f32, 1.0f32, 0.0f32),
                Vec3::new( 1.0f32, 1.0f32, 0.0f32),
                Vec3::new( 1.0f32,-1.0f32, 0.0f32),
                Vec3::new(-1.0f32,-1.0f32, 0.0f32),
            ],
        );

        let base_camera_speed = 2.0f32;
        let camera_speed = if state.keys.pressed(SDL_SCANCODE_LCTRL) {
            base_camera_speed*0.2
        } else if state.keys.pressed(SDL_SCANCODE_LSHIFT) {
            base_camera_speed*5.0
        } else {
            base_camera_speed
        };

        let mut xrel: i32 = 0;
        let mut yrel: i32 = 0;

        let button_bitmask = SDL_GetRelativeMouseState(&mut xrel as *mut i32, &mut yrel as *mut i32);
        if state.keys.pressed(SDL_SCANCODE_ESCAPE) {
            SDL_SetRelativeMouseMode(false);
            state.mouse_captured = false;
        }

        if button_bitmask.bit(0) && !igWantCaptureMouse() {
            SDL_SetRelativeMouseMode(true);
            state.mouse_captured = true;
        }

        let camera_sensitivity = 0.1;
        if state.mouse_captured {
            state.yaw += (xrel as f32)*camera_sensitivity;
            state.pitch -= (yrel as f32)*camera_sensitivity;
            state.pitch = state.pitch.clamp(-89.0, 89.0);

            if state.keys.pressed(SDL_SCANCODE_S) {
                state.camera_pos -= delta*camera_speed*state.camera_front;
            }
            if state.keys.pressed(SDL_SCANCODE_W)  {
                state.camera_pos += delta*camera_speed*state.camera_front;
            }
            if state.keys.pressed(SDL_SCANCODE_D) {
                state.camera_pos += state.camera_front.cross(state.camera_up).normalize() * delta*camera_speed;
            }
            if state.keys.pressed(SDL_SCANCODE_A) {
                state.camera_pos -= state.camera_front.cross(state.camera_up).normalize() * delta*camera_speed;
            }
        }

        let pitch = state.pitch;
        let yaw = state.yaw;
        let direction = Vec3::new(
            yaw.to_radians().cos() * pitch.to_radians().cos(),
            pitch.to_radians().sin(),
            yaw.to_radians().sin() * pitch.to_radians().cos(),
        );
        state.camera_front = direction.normalize();

        state.view = Mat4::look_at_rh(state.camera_pos, state.camera_pos + state.camera_front, state.camera_up);

        let mut display_w: i32 = 0;
        let mut display_h: i32 = 0;
        SHM_GetDrawableSize(&mut display_w as *mut i32, &mut display_h as *mut i32);
        state.projection = Mat4::perspective_rh_gl(
            f32::to_radians(45.0), display_w as f32 / display_h as f32, 0.1f32, 100.0f32
        );

        let mut ctx = HashMap::new();
        ctx.insert("view".to_string(), ShaderValue::Mat4(state.view));
        ctx.insert("projection".to_string(), ShaderValue::Mat4(state.projection));

        let mut open = true;
        igBegin(c"From Rust".as_ptr(), &mut open as *mut bool, IMGUI_WINDOW_FLAGS_NO_FOCUS_ON_APPEARING);
        igText(CString::new(format!("Display size: {}x{}", display_w, display_h)).unwrap().as_ptr());

        #[cfg(target_arch = "wasm32")]
        {
            igBeginDisabled();
        }

        igText(CString::new(format!("Frame Rate: {:.2}ms/frame", 1000.0 / igFrameRate())).unwrap().as_ptr());

        igCheckbox(
            c"Show SDF Wireframe".as_ptr(),
            &mut state.debug_state.sdf_test_draw_wireframe as *mut bool,
        );

        #[cfg(target_arch = "wasm32")]
        {
            igEndDisabled();
            if igIsItemHovered(IMGUI_HOVERED_FLAGS_ALLOW_WHEN_DISABLED) {
                igSetTooltip(c"WebGL 2 does not support LINE polygon mode".as_ptr())
            }
        }

        igEnd();

        {
            let zone = zone_scoped!("IMGUI Debug Windows");
            {
                let _zone = zone_scoped!("ShimLang Debug");

                state.debug_state.shimlang_debug_window.debug_window(&mut state.interpreter);
                state.debug_state.shimlang_repl.window(&mut state.interpreter);

                draw_log_window();
            }
        }

        {
            let _zone = zone_scoped!("Draw Test Mesh");
            state.test_mesh.transform = Mat4::from_translation(Vec3::new(1.2, 0.2, -0.2));
            {
                let _zone = zone_scoped!("state.test_mesh.draw");
                state.test_mesh.draw(&mut ctx);
            }

            let my_box = {
                let _zone = zone_scoped!("Create box mesh");
                box_mesh(Vec3::new(1.0, 1.0, 1.0))
            };
            let positions = match my_box.verts.get("aPos").unwrap() {
                VertVec::Vec3(v) => v,
                _ => panic!("Expected aPos to be a Vec3"),
            };
            let normals = match my_box.verts.get("aNormal").unwrap() {
                VertVec::Vec3(v) => v,
                _ => panic!("Expected aNormal to be a Vec3"),
            };
            for (pos, norm) in positions.iter().zip(normals.iter()) {
                let start = state.test_mesh.transform.transform_point3(*pos);
                let end = state.test_mesh.transform.transform_point3(pos + norm*0.2);
                debug_line(state, &[start, end]);
            }
        }

        {
            let _zone = zone_scoped!("draw_debug_shapes");
            draw_debug_shapes(state);
        }

        {
            // let _zone = zone_scoped!("Log OpenGL Error Macro");
            // // Log errors each frame. This call can be copied to wherever necessary to trace back to the bad call.
            // log_opengl_errors!();
        }

        {
            let _zone = zone_scoped!("Increment frame counter");
            if state.frame_num == 0 {
                println!("Finished first frame");
            }
            state.frame_num += 1;
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn rust_frame(delta: f32) -> i32 {
    let _zone = zone_scoped!("rust_frame_inner");
    STATE_REFCELL.with_borrow_mut(|value| {
        frame(value.as_mut().unwrap(), delta)
    });

    42
}

const _: () = compile_time_checks();
