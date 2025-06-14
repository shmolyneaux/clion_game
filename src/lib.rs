#![allow(unused_variables, unused_mut, unused_imports, unused_attributes, unused_unsafe)]

#[macro_use]
use gl;
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::rc::Rc;
use gl::types::*;
use glam::{Vec2, Vec3, Vec4, Mat4};
use std::ffi::CString;
use std::ffi::CStr;


use std::slice;

use std::mem::size_of;

use std::cell::RefCell;
use core::ffi::c_int;

use glam::Vec3Swizzles;

mod gpu;
mod mesh_gen;
mod debug_draw;

use crate::gpu::*;
use crate::debug_draw::*;
use crate::mesh_gen::{box_mesh, quad_mesh};

type ImGuiWindowFlags = c_int;
unsafe extern "C" {
    fn SDL_GL_GetProcAddress(proc: *const i8) -> *mut std::ffi::c_void;
    fn SDL_GetKeyboardState(numkeys: *mut i32) -> *const u8;
    fn SDL_GetRelativeMouseState(x: *mut i32, y: *mut i32) -> u32;
    fn SDL_SetRelativeMouseMode(enabled: bool) -> i32;
    fn SHM_GetDrawableSize(display_w: *mut i32, display_h: *mut i32);
    fn igBegin(name: *const core::ffi::c_char, p_open: *mut bool, flags: ImGuiWindowFlags) -> bool;
    fn igEnd();
    fn igText(fmt: *const core::ffi::c_char, ...);
    fn igSliderFloat(label: *const core::ffi::c_char, v: *mut f32, v_min: f32, v_max: f32, format: *const core::ffi::c_char);
    fn igWantCaptureKeyboard() -> bool;
    fn igWantCaptureMouse() -> bool;
}

const fn u32size_of<T>() -> u32 {
    size_of::<T>() as u32
}

struct KeyState {
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

    fn pressed(&self, sdl_key: usize) -> bool {
        self.keys[sdl_key] == 1
    }

    fn released(&self, sdl_key: usize) -> bool {
        self.keys[sdl_key] == 0
    }

    fn just_pressed(&self, sdl_key: usize) -> bool {
        self.keys[sdl_key] == 1 && self.keys[sdl_key] != self.last_keys[sdl_key]
    }

    fn just_released(&self, sdl_key: usize) -> bool {
        self.keys[sdl_key] == 0 && self.keys[sdl_key] != self.last_keys[sdl_key]
    }
}

#[derive(Default)]
pub struct DebugState {
    sphere_radius: f32,
}

pub struct State {
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

    pitch: f32,
    yaw: f32,

    camera_pos: Vec3,
    camera_front: Vec3,
    camera_up: Vec3,

    mouse_captured: bool,

    test_mesh: StaticMesh,
    meshes: HashMap<String, StaticMesh>,

    keys: KeyState,

    debug_state: DebugState,
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
                let err = gl::GetError();
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

    let default_shader_program = create_default_shader();
    log_opengl_errors!();

    println!("State initialized!");
    State {
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
    }
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

fn sdf_sphere(pos: Vec3) -> f32 {
    (pos.x*pos.x + pos.y*pos.y + pos.z*pos.z).sqrt() - 1.0
}

fn sdf_vert_set(grid: &Vec<Vec<Vec<f32>>>, pos: (usize, usize, usize)) -> bool {
    let x = pos.0;
    let y = pos.1;
    let z = pos.2;

    let sign = grid[x][y][z].is_sign_positive();

    (
        sign != grid[x][y][z+1].is_sign_positive() ||
        sign != grid[x][y+1][z].is_sign_positive() ||
        sign != grid[x][y+1][z+1].is_sign_positive() ||
        sign != grid[x+1][y][z].is_sign_positive() ||
        sign != grid[x+1][y][z+1].is_sign_positive() ||
        sign != grid[x+1][y+1][z].is_sign_positive() ||
        sign != grid[x+1][y+1][z+1].is_sign_positive()
    )
}

struct SdfCache<F>
where
    F: Fn(Vec3) -> f32
{
    sdf: F,
    cache: RefCell<HashMap<(i32, i32, i32), f32>>,
    scale: f32,
    resolution: u32,
    bounds: (Vec3, Vec3),
}

impl<F: Fn(Vec3) -> f32> SdfCache<F> {
    fn new(sdf: F, scale: f32, resolution: u32) -> Self {
        let cache = RefCell::new(HashMap::new());
        let bounds = (Vec3::new(-1.0, -1.0, -1.0), Vec3::new(1.0, 1.0, 1.0));
        Self {
            sdf,
            scale,
            resolution,
            bounds,
            cache,
        }
    }

    fn get(&self, coord: (i32, i32, i32)) -> f32 {
        let mut cache = self.cache.borrow_mut();
        match cache.get(&coord) {
            Some(dist) => *dist,
            None => {
                let posf = Vec3::new(coord.0 as f32, coord.1 as f32, coord.2 as f32) * self.scale;
                let dist = (self.sdf)(posf);
                cache.insert(coord, dist);
                dist
            }
        }
    }
}

fn frame(state: &mut State, delta: f32) {
    if state.frame_num == 0 {
        println!("Starting first frame");
    }
    unsafe {
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


        // Some boxes
        //debug_box(
        //    state,
        //    Vec3::new(-2.0f32, 1.0f32, 0.0f32),
        //    Vec3::new(1.0f32, 1.0f32, 1.0f32),
        //    Vec3::new(1.0f32, 1.0f32, 1.0f32),
        //);
        //debug_box(
        //    state,
        //    Vec3::new(0.0f32, 1.0f32, 0.0f32),
        //    Vec3::new(1.0f32, 1.0f32, 1.0f32),
        //    Vec3::new(1.0f32, 1.0f32, 1.0f32),
        //);
        //debug_box(
        //    state,
        //    Vec3::new(2.0f32, 1.0f32, 0.0f32),
        //    Vec3::new(1.0f32, 1.0f32, 1.0f32),
        //    Vec3::new(1.0f32, 1.0f32, 1.0f32),
        //);

        // Grid lines
        //for x in -10..=10 {
        //    debug_line(
        //        state,
        //        &[
        //            Vec3::new(x as f32, 0.0f32, -10.0f32),
        //            Vec3::new(x as f32, 0.0f32, 10.0f32),
        //        ]
        //    );
        //    debug_line(
        //        state,
        //        &[
        //            Vec3::new(10.0f32, 0.0f32, x as f32),
        //            Vec3::new(-10.0f32, 0.0f32, x as f32),
        //        ]
        //    );
        //}

        let base_camera_speed = 2.0f32;
        let camera_speed = if state.keys.pressed(SDL_SCANCODE_LCTRL) {
            base_camera_speed*0.2
        } else if state.keys.pressed(SDL_SCANCODE_LSHIFT) {
            base_camera_speed*5.0
        } else {
            base_camera_speed
        };

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

        let recip_radius = state.debug_state.sphere_radius.recip();
        let mut sdf = SdfCache::new(
            |pos| sdf_sphere(pos*recip_radius),
            0.1,
            20,
       );

        let mut ctx = HashMap::new();
        ctx.insert("view".to_string(), ShaderValue::Mat4(state.view));
        ctx.insert("projection".to_string(), ShaderValue::Mat4(state.projection));

        let white = ShaderValue::Vec3(Vec3::new(1.0, 1.0, 1.0));
        let black = ShaderValue::Vec3(Vec3::new(0.0, 0.0, 0.0));
        let magenta = ShaderValue::Vec3(Vec3::new(1.0, 0.0, 1.0));

        let mut index_count = 0;
        let mut positions = Vec::new();
        let mut uvs = Vec::new();
        let mut normals = Vec::new();
        let mut indices = Vec::new();

        // Vert coord to index
        let mut vert_lookup = HashMap::new();
        //let mut edges = HashH

        for x in -10..10 {
            for y in -10..10 {
                for z in -10..10 {
                    let coord = (x, y, z);
                    let dist = sdf.get(coord);
                    let mut is_vert = false;

                    // Logically, the vert is somewhere in the cube represented by 8 sample points. We know there's a
                    // vert in the cube if there's a transition from negative to positive somewhere within the volume
                    // of the cube
                    for (offx, offy, offz) in [(0, 0, 1), (0, 1, 0), (0, 1, 1), (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1)] {
                        if dist.is_sign_positive() != sdf.get((coord.0+offx, coord.1+offy, coord.2+offz)).is_sign_positive() {
                            is_vert = true;
                        }
                    }

                    if is_vert {
                        let dx = sdf.get(
                            (
                                coord.0+1,
                                coord.1,
                                coord.2
                            )
                        ) - dist;

                        let dy = sdf.get(
                            (
                                coord.0,
                                coord.1+1,
                                coord.2
                            )
                        ) - dist;

                        let dz = sdf.get(
                            (
                                coord.0,
                                coord.1,
                                coord.2+1
                            )
                        ) - dist;

                        let idx = positions.len() as u32;
                        vert_lookup.insert(coord, idx);

                        let posf = Vec3::new(coord.0 as f32, coord.1 as f32, coord.2 as f32);
                        positions.push(posf * 0.1);

                        let norm = Vec3::new(dx, dy, dz).normalize_or_zero();
                        normals.push(norm);

                        uvs.push(posf.xy() * 0.1);
                    }
                }
            }
        }

        let mut x_edges: HashSet<(i32, i32, i32)> = HashSet::new();
        let mut y_edges: HashSet<(i32, i32, i32)> = HashSet::new();
        let mut z_edges: HashSet<(i32, i32, i32)> = HashSet::new();

        for (c0, _) in vert_lookup.iter() {
            let cx = ((c0.0+1), (c0.1+0), (c0.2+0));
            let cxz = ((c0.0+1), (c0.1+0), (c0.2+1));
            let cxy = ((c0.0+1), (c0.1+1), (c0.2+0));
            let cxyz = ((c0.0+1), (c0.1+1), (c0.2+1));
            let cz = ((c0.0+0), (c0.1+0), (c0.2+1));
            let cy = ((c0.0+0), (c0.1+1), (c0.2+0));
            let cyz = ((c0.0+0), (c0.1+1), (c0.2+1));

            // We only want an edge if the isosurface intersects with the
            // face of the cube this edge is passing through.

            if vert_lookup.get(&cx).is_some() {
                let dist = sdf.get(cx);
                if dist.is_sign_positive() != sdf.get(cxz).is_sign_positive() ||
                   dist.is_sign_positive() != sdf.get(cxy).is_sign_positive() ||
                   dist.is_sign_positive() != sdf.get(cxyz).is_sign_positive()
                {
                    x_edges.insert(*c0);
                }
            }

            if vert_lookup.get(&cy).is_some() {
                let dist = sdf.get(cy);
                if dist.is_sign_positive() != sdf.get(cxy).is_sign_positive() ||
                   dist.is_sign_positive() != sdf.get(cyz).is_sign_positive() ||
                   dist.is_sign_positive() != sdf.get(cxyz).is_sign_positive()
                {
                    y_edges.insert(*c0);
                }
            }

            if vert_lookup.get(&cz).is_some() {
                let dist = sdf.get(cz);
                if dist.is_sign_positive() != sdf.get(cxz).is_sign_positive() ||
                   dist.is_sign_positive() != sdf.get(cyz).is_sign_positive() ||
                   dist.is_sign_positive() != sdf.get(cxyz).is_sign_positive()
                {
                    z_edges.insert(*c0);
                }
            }
        }

        for coord in x_edges.iter() {
            if x_edges.contains(&(coord.0, coord.1+1, coord.2)) &&
                y_edges.contains(coord) &&
                y_edges.contains(&(coord.0+1, coord.1, coord.2))
            {
                let coord0 = (coord.0+0, coord.1+0, coord.2+0);
                let coord1 = (coord.0+0, coord.1+1, coord.2+0);
                let coord2 = (coord.0+1, coord.1+0, coord.2+0);

                let coord3 = (coord.0+1, coord.1+1, coord.2+0);
                let coord4 = (coord.0+0, coord.1+1, coord.2+0);
                let coord5 = (coord.0+1, coord.1+0, coord.2+0);

                let idx0 = vert_lookup.get(&coord0).unwrap();
                let idx1 = vert_lookup.get(&coord1).unwrap();
                let idx2 = vert_lookup.get(&coord2).unwrap();
                let idx3 = vert_lookup.get(&coord3).unwrap();
                let idx4 = vert_lookup.get(&coord4).unwrap();
                let idx5 = vert_lookup.get(&coord5).unwrap();

                indices.push(*idx0);
                indices.push(*idx1);
                indices.push(*idx2);

                indices.push(*idx3);
                indices.push(*idx4);
                indices.push(*idx5);
            }

            if x_edges.contains(&(coord.0, coord.1, coord.2+1)) &&
                z_edges.contains(coord) &&
                z_edges.contains(&(coord.0+1, coord.1, coord.2))
            {
                let coord0 = (coord.0+0, coord.1+0, coord.2+0);
                let coord1 = (coord.0+0, coord.1+0, coord.2+1);
                let coord2 = (coord.0+1, coord.1+0, coord.2+0);

                let coord3 = (coord.0+1, coord.1+0, coord.2+1);
                let coord4 = (coord.0+0, coord.1+0, coord.2+1);
                let coord5 = (coord.0+1, coord.1+0, coord.2+0);

                let idx0 = vert_lookup.get(&coord0).unwrap();
                let idx1 = vert_lookup.get(&coord1).unwrap();
                let idx2 = vert_lookup.get(&coord2).unwrap();
                let idx3 = vert_lookup.get(&coord3).unwrap();
                let idx4 = vert_lookup.get(&coord4).unwrap();
                let idx5 = vert_lookup.get(&coord5).unwrap();

                indices.push(*idx0);
                indices.push(*idx1);
                indices.push(*idx2);

                indices.push(*idx3);
                indices.push(*idx4);
                indices.push(*idx5);
            }
        }

        for coord in y_edges.iter() {
            if y_edges.contains(&(coord.0, coord.1, coord.2+1)) &&
                z_edges.contains(coord) &&
                z_edges.contains(&(coord.0, coord.1+1, coord.2))
            {
                let coord0 = (coord.0+0, coord.1+0, coord.2+0);
                let coord1 = (coord.0+0, coord.1+1, coord.2+0);
                let coord2 = (coord.0+0, coord.1+0, coord.2+1);

                let coord3 = (coord.0+0, coord.1+1, coord.2+1);
                let coord4 = (coord.0+0, coord.1+1, coord.2+0);
                let coord5 = (coord.0+0, coord.1+0, coord.2+1);

                let idx0 = vert_lookup.get(&coord0).unwrap();
                let idx1 = vert_lookup.get(&coord1).unwrap();
                let idx2 = vert_lookup.get(&coord2).unwrap();
                let idx3 = vert_lookup.get(&coord3).unwrap();
                let idx4 = vert_lookup.get(&coord4).unwrap();
                let idx5 = vert_lookup.get(&coord5).unwrap();

                indices.push(*idx0);
                indices.push(*idx1);
                indices.push(*idx2);

                indices.push(*idx3);
                indices.push(*idx4);
                indices.push(*idx5);
            }
        }


        let mut verts = HashMap::new();
        verts.insert("aPos".to_string(), VertVec::Vec3(positions));
        verts.insert("aUV".to_string(), VertVec::Vec2(uvs));

        let primitive_type = Primitive::Triangles;

        let meshdata = MeshDataRaw {
            verts,
            indices,
            primitive_type,
        };

        let mut generated = StaticMesh::create(
            state.default_shader_program.clone(),
            Rc::new(Mesh::create(&meshdata).unwrap()),
        ).expect("Can't create the test mesh");

        generated.draw(&mut ctx);

        let mut open = true;
        igBegin(c"From Rust".as_ptr(), &mut open as *mut bool, IMGUI_WINDOW_FLAGS_NO_FOCUS_ON_APPEARING);
        igText(CString::new(format!("Display size: {}x{}", display_w, display_h)).unwrap().as_ptr());

        igSliderFloat(
            c"My Slider".as_ptr(),
            &mut state.debug_state.sphere_radius as *mut f32,
            0.001,
            1.0,
            c"%.3f".as_ptr(),
        );

        igEnd();

        draw_log_window();

        state.test_mesh.transform = Mat4::from_translation(Vec3::new(1.2, 0.2, -0.2));

        state.test_mesh.draw(&mut ctx);

        let my_box = box_mesh(Vec3::new(1.0, 1.0, 1.0));
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

        draw_debug_shapes(state);

        // Log errors each frame. This call can be copied to wherever necessary to trace back to the bad call.
        log_opengl_errors!();

        if state.frame_num == 0 {
            println!("Finished first frame");
        }
        state.frame_num += 1;
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn rust_frame(delta: f32) -> i32 {
    STATE_REFCELL.with_borrow_mut(|value| {
        frame(value.as_mut().unwrap(), delta)
    });

    42
}

const _: () = compile_time_checks();