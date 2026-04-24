#![allow(
    unused_variables,
    unused_mut,
    unused_imports,
    unused_attributes,
    unused_unsafe,
    dead_code,
    unsafe_op_in_unsafe_fn,
    non_snake_case
)]

#[macro_use]
use gl::types::*;
use glam::{Mat4, Vec2, Vec3, Vec4};
use std::collections::{HashMap, HashSet, VecDeque};
use std::ffi::CStr;
use std::ffi::CString;
use std::fmt;
use std::rc::Rc;

use std::os::raw::{c_char, c_int, c_uint};

use std::ops::{Add, Mul};

use std::slice;

use std::mem::size_of;

use std::cell::RefCell;

use glam::Vec3Swizzles;

use facet::*;
use facet_reflect::*;

use std::borrow::Cow;
use std::fmt::Formatter;

mod debug_draw;
mod gpu;
mod imgui;
mod mesh_gen;
mod script_bridge;
mod sdf;
mod sdl;

mod shimlang_imgui;

use crate::debug_draw::*;
use crate::gpu::*;
use crate::imgui::*;
use crate::mesh_gen::{box_mesh, quad_mesh, screen_quad_mesh};
use crate::script_bridge::*;
use crate::sdf::*;
use crate::sdl::*;

use shm_tracy::*;

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
            Self { keys, last_keys }
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
}

//#[derive(Facet)]
/// -180.0 180.0
pub struct State {
    script_bridge: ScriptBridge,

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

    meshes: HashMap<String, StaticMesh>,

    keys: KeyState,

    frame_captures: VecDeque<GLuint>,
    frame_capture_shader: Rc<ShaderProgram>,
    frame_capture_mesh: StaticMesh,
    frame_capture_fbo: GLuint,
    screen_quad_shader: Rc<ShaderProgram>,
    screen_quad_mesh: StaticMesh,

    debug_state: DebugState,

    shimlang_texture_handle_to_gl_texture: HashMap<u32, i32>,
    default_texture: GLuint,
}

thread_local! {
    static STATE_REFCELL: RefCell<Option<State>> = RefCell::default();
}

thread_local! {
    pub(crate) static DEBUG_LOG: RefCell<Vec<CString>> = RefCell::default();
}

trait BitCheck {
    fn bit(&self, n: u32) -> bool;
}

impl BitCheck for u32 {
    fn bit(&self, n: u32) -> bool {
        (self >> n) & 1 != 0
    }
}

pub(crate) fn log<T: AsRef<str>>(s: T) {
    logc(CString::new(s.as_ref().to_string()).unwrap());
}

pub(crate) fn logc(s: CString) {
    println!("{:?}", s);
    DEBUG_LOG.with_borrow_mut(|logs| logs.push(s));
}

const fn compile_time_checks() {
    assert!(size_of::<u8>() == 1);
}

fn draw_frame_captures(
    frame_captures: &VecDeque<GLuint>,
    frame_capture_mesh: &mut StaticMesh,
    ctx: &mut HashMap<String, ShaderValue>,
) {
    for (idx, frame_texture) in frame_captures.iter().enumerate() {
        if (idx as isize) < frame_captures.len() as isize - 10 {
            continue;
        }
        let local_idx = (idx + 10 - frame_captures.len()) as f32;
        let pos = local_idx / 5.0f32 - 0.9f32;

        frame_capture_mesh.uniform_override.insert(
            "texture1".to_string(),
            ShaderValue::Sampler2D(*frame_texture),
        );
        frame_capture_mesh
            .uniform_override
            .insert("x".to_string(), ShaderValue::Float(pos));
        frame_capture_mesh
            .uniform_override
            .insert("y".to_string(), ShaderValue::Float(0.1));
        frame_capture_mesh
            .uniform_override
            .insert("w".to_string(), ShaderValue::Float(0.1));
        frame_capture_mesh
            .uniform_override
            .insert("h".to_string(), ShaderValue::Float(0.1));
        frame_capture_mesh.draw(ctx);
    }
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
            "#
            .to_string(),
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
            "#
            .to_string(),
        )
        .build_fragment_shader()
        .expect("Could not compiled default fragment shader");

    println!("Starting default shader compilation");
    let default_shader_program =
        ShaderProgram::create(default_vertex_shader, default_fragment_shader)
            .expect("Could not link default shader");
    println!("Linked default shader");

    let default_shader_program = Rc::new(default_shader_program);
    println!("Shader program now in RC");

    default_shader_program
}

fn init_state() -> State {
    log_opengl_errors!();
    println!("Starting Rust state initialization");

    println!("Creating interpreter");
    let script_bridge = ScriptBridge::new();

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
            "#
            .to_string(),
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
            "#
            .to_string(),
        )
        .build_fragment_shader()
        .expect("Could not compiled debug fragment shader");

    println!("Starting debug shader compilation");
    let debug_shader_program = ShaderProgram::create(debug_vertex_shader, debug_fragment_shader)
        .expect("Could not link debug shader");
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
    let camera_up = Vec3::new(0.0f32, 1.0f32, 0.0f32);

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
            "#
            .to_string(),
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
            "#
            .to_string(),
        )
        .build_fragment_shader()
        .expect("Could not compiled default fragment shader");
    log_opengl_errors!();

    println!("Starting default shader compilation");
    let default_shader = ShaderProgram::create(default_vertex_shader, default_fragment_shader)
        .expect("Could not link default shader");
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
                "#
                .to_string(),
            )
            .build_vertex_shader()
            .unwrap(),
        ShaderBuilder::new()
            .with_input(ShaderSymbol::new(ShaderDataType::Vec3, "color"))
            .with_output(ShaderSymbol::new(ShaderDataType::Vec4, "FragColor"))
            .with_code(
                r#"
                    void main() {
                        FragColor = vec4(color, 1.0);
                    }
                "#
                .to_string(),
            )
            .build_fragment_shader()
            .unwrap(),
    )
    .expect("Could not build color shader");
    let vert_color_shader = Rc::new(vert_color_shader);

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
                "#
                .to_string(),
            )
            .build_vertex_shader()
            .unwrap(),
        ShaderBuilder::new()
            .with_input(ShaderSymbol::new(ShaderDataType::Vec3, "normal"))
            .with_output(ShaderSymbol::new(ShaderDataType::Vec4, "FragColor"))
            .with_uniform(ShaderSymbol::new(ShaderDataType::Vec3, "color"))
            .with_code(
                r#"
                    void main() {
                        FragColor = vec4(color, 1.0);
                    }
                "#
                .to_string(),
            )
            .build_fragment_shader()
            .unwrap(),
    )
    .expect("Could not build red shader");
    let red_shader = Rc::new(red_shader);

    let mut my_box = box_mesh(Vec3::new(1.0, 1.0, 1.0));
    let mut box_static_mesh =
        StaticMesh::create(red_shader.clone(), Rc::new(Mesh::create(&my_box).unwrap()))
            .expect("Can't create the test mesh");

    meshes.insert("red box".to_string(), box_static_mesh);

    let screen_quad_shader = ShaderProgram::create(
        ShaderBuilder::new()
            .with_input(ShaderSymbol::new(ShaderDataType::Vec3, "aPos"))
            .with_input(ShaderSymbol::new(ShaderDataType::Vec2, "aUV"))
            .with_uniform(ShaderSymbol::new(ShaderDataType::Vec2, "uResolution"))
            .with_uniform(ShaderSymbol::new(ShaderDataType::Vec2, "uRectPos"))
            .with_uniform(ShaderSymbol::new(ShaderDataType::Vec2, "uRectSize"))
            .with_output(ShaderSymbol::new(ShaderDataType::Vec2, "uv"))
            .with_code(
                r#"
                    void main() {
                        vec2 pixel_pos = uRectPos + aPos.xy * uRectSize;
                        gl_Position = vec4(
                            pixel_pos.x * (2.0/uResolution.x) - 1.0,
                            pixel_pos.y * (-2.0/uResolution.y) + 1.0,
                            aPos.z,
                            1.0
                        );
                        uv = aUV;
                    }
                "#
                .to_string(),
            )
            .build_vertex_shader()
            .unwrap(),
        ShaderBuilder::new()
            .with_input(ShaderSymbol::new(ShaderDataType::Vec2, "uv"))
            .with_output(ShaderSymbol::new(ShaderDataType::Vec4, "FragColor"))
            .with_uniform(ShaderSymbol::new(ShaderDataType::Sampler2D, "texture1"))
            .with_uniform(ShaderSymbol::new(ShaderDataType::Vec4, "uModulate"))
            .with_code(
                r#"
                    void main() {
                        FragColor = texture(texture1, uv) * uModulate;
                    }
                "#
                .to_string(),
            )
            .build_fragment_shader()
            .unwrap(),
    )
    .expect("Could not build screen quad shader");
    let screen_quad_shader = Rc::new(screen_quad_shader);

    let screen_quad_mesh = StaticMesh::create(
        screen_quad_shader.clone(),
        Rc::new(Mesh::create(&screen_quad_mesh(0, 0, 1, 1)).unwrap()),
    )
    .expect("Can't create screen quad mesh");

    let debug_state = DebugState {
        sphere_radius: 1.0,
        sdf_box_size: Vec3::new(0.5, 0.7, 1.0),
        ..Default::default()
    };

    let default_shader_program = create_default_shader();
    log_opengl_errors!();

    let frame_captures = VecDeque::new();

    let frame_capture_shader = ShaderProgram::create(
        ShaderBuilder::new()
            .with_input(ShaderSymbol::new(ShaderDataType::Vec3, "aPos"))
            .with_input(ShaderSymbol::new(ShaderDataType::Vec2, "aUV"))
            .with_output(ShaderSymbol::new(ShaderDataType::Vec2, "uv"))
            .with_uniform(ShaderSymbol::new(ShaderDataType::Float, "x"))
            .with_uniform(ShaderSymbol::new(ShaderDataType::Float, "y"))
            .with_uniform(ShaderSymbol::new(ShaderDataType::Float, "w"))
            .with_uniform(ShaderSymbol::new(ShaderDataType::Float, "h"))
            .with_code(
                r#"
                    void main() {
                        gl_Position = vec4(aPos.x*w, aPos.y*h, aPos.z, 1.0) + vec4(x, y, 0.0, 0.0);
                        uv = aUV;
                    }
                "#
                .to_string(),
            )
            .build_vertex_shader()
            .unwrap(),
        ShaderBuilder::new()
            .with_input(ShaderSymbol::new(ShaderDataType::Vec2, "uv"))
            .with_output(ShaderSymbol::new(ShaderDataType::Vec4, "FragColor"))
            .with_uniform(ShaderSymbol::new(ShaderDataType::Sampler2D, "texture1"))
            .with_code(
                r#"
                    void main() {
                        FragColor = texture(texture1, uv);
                    }
                "#
                .to_string(),
            )
            .build_fragment_shader()
            .unwrap(),
    )
    .expect("Could not build frame capture shader");
    let frame_capture_shader = Rc::new(frame_capture_shader);

    let frame_capture_mesh = StaticMesh::create(
        frame_capture_shader.clone(),
        Rc::new(Mesh::create(&quad_mesh()).unwrap()),
    )
    .expect("Can't create the frame capture mesh");

    let mut frame_capture_fbo: GLuint = 0;
    unsafe {
        gl::GenFramebuffers(1, &mut frame_capture_fbo as *mut u32);
    }

    let default_texture = gen_cpu_texture(128, 128, |x, y| [0xff, 0xff, 0xff, 0xff]);

    println!("State initialized!");
    

    State {
        script_bridge,
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
        keys,
        meshes,
        frame_captures,
        frame_capture_shader,
        frame_capture_mesh,
        frame_capture_fbo,
        screen_quad_shader,
        screen_quad_mesh,
        debug_state,
        shimlang_texture_handle_to_gl_texture: HashMap::new(),
        default_texture,
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

fn frame(state: &mut State, delta: f32) {
    if state.frame_num == 00 {
        println!("Starting first frame");
    }

    {
        let _zone = zone_scoped!("imgui_debug");
        //imgui_debug(state);
    }

    unsafe {
        let mut open = true;
        igBegin(
            c"From Rust".as_ptr(),
            &mut open as *mut bool,
            IMGUI_WINDOW_FLAGS_NO_FOCUS_ON_APPEARING,
        );

        {
            let _zone = zone_scoped!("Run interpreter");
            state
                .script_bridge
                .step(&state.keys.keys, &state.keys.last_keys);
        }

        for err in state.script_bridge.errors() {
            igTextColoredBC(
                0.7,
                0.0,
                0.0,
                1.0,
                0.5,
                0.5,
                0.5,
                0.5,
                CString::new(err.to_string()).unwrap().as_ptr(),
            );
        }

        igTextColoredBC(
            1.0,
            1.0,
            1.0,
            1.0,
            0.5,
            0.5,
            0.5,
            0.5,
            CString::new("BG text".to_string()).unwrap().as_ptr(),
        );

        let _zone = zone_scoped!("rust frame unsafe block");
        update_keys(state);

        let test = [0.1, 0.1, 0.12, 1.0];
        gl::ClearColor(test[0], test[1], test[2], test[3]);

        debug_line_loop(
            state,
            &[
                Vec3::new(-1.0f32, 1.0f32, 0.0f32),
                Vec3::new(1.0f32, 1.0f32, 0.0f32),
                Vec3::new(1.0f32, -1.0f32, 0.0f32),
                Vec3::new(-1.0f32, -1.0f32, 0.0f32),
            ],
        );

        let base_camera_speed = 2.0f32;
        let camera_speed = if state.keys.pressed(SDL_SCANCODE_LCTRL) {
            base_camera_speed * 0.2
        } else if state.keys.pressed(SDL_SCANCODE_LSHIFT) {
            base_camera_speed * 5.0
        } else {
            base_camera_speed
        };

        let mut xrel: i32 = 0;
        let mut yrel: i32 = 0;

        let button_bitmask =
            SDL_GetRelativeMouseState(&mut xrel as *mut i32, &mut yrel as *mut i32);
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
            state.yaw += (xrel as f32) * camera_sensitivity;
            state.pitch -= (yrel as f32) * camera_sensitivity;
            state.pitch = state.pitch.clamp(-89.0, 89.0);

            if state.keys.pressed(SDL_SCANCODE_S) {
                state.camera_pos -= delta * camera_speed * state.camera_front;
            }
            if state.keys.pressed(SDL_SCANCODE_W) {
                state.camera_pos += delta * camera_speed * state.camera_front;
            }
            if state.keys.pressed(SDL_SCANCODE_D) {
                state.camera_pos +=
                    state.camera_front.cross(state.camera_up).normalize() * delta * camera_speed;
            }
            if state.keys.pressed(SDL_SCANCODE_A) {
                state.camera_pos -=
                    state.camera_front.cross(state.camera_up).normalize() * delta * camera_speed;
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

        state.view = Mat4::look_at_rh(
            state.camera_pos,
            state.camera_pos + state.camera_front,
            state.camera_up,
        );

        let mut display_w: i32 = 0;
        let mut display_h: i32 = 0;
        SHM_GetDrawableSize(&mut display_w as *mut i32, &mut display_h as *mut i32);
        state.projection = Mat4::perspective_rh_gl(
            f32::to_radians(45.0),
            display_w as f32 / display_h as f32,
            0.1f32,
            100.0f32,
        );

        let mut ctx = HashMap::new();
        ctx.insert("view".to_string(), ShaderValue::Mat4(state.view));
        ctx.insert(
            "projection".to_string(),
            ShaderValue::Mat4(state.projection),
        );

        igText(
            CString::new(format!("Display size: {}x{}", display_w, display_h))
                .unwrap()
                .as_ptr(),
        );

        #[cfg(target_arch = "wasm32")]
        {
            igBeginDisabled();
        }

        igText(
            CString::new(format!("Frame Rate: {:.2}ms/frame", 1000.0 / igFrameRate()))
                .unwrap()
                .as_ptr(),
        );

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
                state
                    .script_bridge
                    .debug_window(&mut state.debug_state.shimlang_debug_window);

                draw_log_window();
            }
        }

        {
            let _zone = zone_scoped!("draw_debug_shapes");
            draw_debug_shapes(state);
        }

        gl::Disable(gl::DEPTH_TEST);

        gl::Enable(gl::BLEND);
        gl::BlendFunc(gl::SRC_ALPHA, gl::ONE_MINUS_SRC_ALPHA);
        for item in state.script_bridge.draw_list.iter() {
            match item {
                DrawListItem::Rect(rect) => {
                    let texture: u32 = match &rect.texture {
                        Some(id) => *state
                            .shimlang_texture_handle_to_gl_texture
                            .get(&id.texture_id)
                            .unwrap() as u32,
                        None => state.default_texture,
                    };
                    state.screen_quad_mesh.uniform_override.insert(
                        "uResolution".to_string(),
                        ShaderValue::Vec2(Vec2::new(display_w as f32, display_h as f32)),
                    );
                    state.screen_quad_mesh.uniform_override.insert(
                        "uRectPos".to_string(),
                        ShaderValue::Vec2(Vec2::new(rect.x.round(), rect.y.round())),
                    );
                    state.screen_quad_mesh.uniform_override.insert(
                        "uRectSize".to_string(),
                        ShaderValue::Vec2(Vec2::new(rect.w.round(), rect.h.round())),
                    );
                    state
                        .screen_quad_mesh
                        .uniform_override
                        .insert("texture1".to_string(), ShaderValue::Sampler2D(texture));
                    let [mr, mg, mb, ma] = rect.modulate;
                    state.screen_quad_mesh.uniform_override.insert(
                        "uModulate".to_string(),
                        ShaderValue::Vec4(Vec4::new(
                            mr as f32 / 255.0,
                            mg as f32 / 255.0,
                            mb as f32 / 255.0,
                            ma as f32 / 255.0,
                        )),
                    );
                    state.screen_quad_mesh.draw(&mut ctx);
                }
                DrawListItem::CreateTexture(shimlang_texture_handle, w, h, rgba_bytes) => {
                    let gl_texture_id = gen_cpu_texture(*w, *h, |x, y| {
                        let i = ((y * w + x) * 4) as usize;
                        [
                            rgba_bytes[i],
                            rgba_bytes[i + 1],
                            rgba_bytes[i + 2],
                            rgba_bytes[i + 3],
                        ]
                    });
                    state
                        .shimlang_texture_handle_to_gl_texture
                        .insert(*shimlang_texture_handle, gl_texture_id as i32);
                }
            }
        }
        gl::Disable(gl::BLEND);

        {
            let _zone = zone_scoped!("capture frame");
            const MAX_FRAME_CAPTURES: usize = 1000;
            let existing_texture = if state.frame_captures.len() >= MAX_FRAME_CAPTURES {
                Some(state.frame_captures.pop_front().unwrap())
            } else {
                None
            };
            let texture = capture_frame_texture(
                display_w,
                display_h,
                state.frame_capture_fbo,
                existing_texture,
            );
            state.frame_captures.push_back(texture);
        }
        draw_frame_captures(
            &state.frame_captures,
            &mut state.frame_capture_mesh,
            &mut ctx,
        );

        gl::Enable(gl::DEPTH_TEST);

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
    STATE_REFCELL.with_borrow_mut(|value| frame(value.as_mut().unwrap(), delta));

    42
}

const _: () = compile_time_checks();
