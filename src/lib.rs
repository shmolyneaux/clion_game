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

const MAX_FRAME_CAPTURES: usize = 1000;
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

    display_w: i32,
    display_h: i32,

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

    edit_mode: bool,

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
    font_texture: GLuint,
}

impl State {
    fn update_camera_matrices(&mut self) {
        let pitch = self.pitch;
        let yaw = self.yaw;
        let direction = Vec3::new(
            yaw.to_radians().cos() * pitch.to_radians().cos(),
            pitch.to_radians().sin(),
            yaw.to_radians().sin() * pitch.to_radians().cos(),
        );
        self.camera_front = direction.normalize();

        self.view = Mat4::look_at_rh(
            self.camera_pos,
            self.camera_pos + self.camera_front,
            self.camera_up,
        );
        self.projection = Mat4::perspective_rh_gl(
            f32::to_radians(45.0),
            self.display_w as f32 / self.display_h as f32,
            0.1f32,
            100.0f32,
        );
    }

    fn move_camera_and_update_matrices(&mut self, delta: f32) {
        let base_camera_speed = 2.0f32;
        let camera_speed = if self.keys.pressed(SDL_SCANCODE_LCTRL) {
            base_camera_speed * 0.2
        } else if self.keys.pressed(SDL_SCANCODE_LSHIFT) {
            base_camera_speed * 5.0
        } else {
            base_camera_speed
        };

        let mut xrel: i32 = 0;
        let mut yrel: i32 = 0;

        let camera_sensitivity = 0.1;
        if self.mouse_captured {
            self.yaw += (xrel as f32) * camera_sensitivity;
            self.pitch -= (yrel as f32) * camera_sensitivity;
            self.pitch = self.pitch.clamp(-89.0, 89.0);

            if self.keys.pressed(SDL_SCANCODE_S) {
                self.camera_pos -= delta * camera_speed * self.camera_front;
            }
            if self.keys.pressed(SDL_SCANCODE_W) {
                self.camera_pos += delta * camera_speed * self.camera_front;
            }
            if self.keys.pressed(SDL_SCANCODE_D) {
                self.camera_pos +=
                    self.camera_front.cross(self.camera_up).normalize() * delta * camera_speed;
            }
            if self.keys.pressed(SDL_SCANCODE_A) {
                self.camera_pos -=
                    self.camera_front.cross(self.camera_up).normalize() * delta * camera_speed;
            }
        }

        self.update_camera_matrices();
    }

    fn capture_frame(&mut self) {
        {
            let _zone = zone_scoped!("capture frame");
            let recycled = self.frame_captures.pop_front().unwrap();
            let texture = capture_frame_texture(
                self.display_w,
                self.display_h,
                self.frame_capture_fbo,
                Some(recycled),
            );
            self.frame_captures.push_back(texture);
        }
    }
    fn handle_draw_list(
        &mut self,
        ctx: &mut HashMap<String, ShaderValue>,
    ) {
        unsafe {
            gl::Disable(gl::DEPTH_TEST);
            gl::Enable(gl::BLEND);
            gl::BlendFunc(gl::SRC_ALPHA, gl::ONE_MINUS_SRC_ALPHA);
        }
        for item in self.script_bridge.draw_list.iter() {
            match item {
                DrawListItem::Rect(rect) => {
                    let texture: u32 = match &rect.texture {
                        Some(id) => *self
                            .shimlang_texture_handle_to_gl_texture
                            .get(&id.texture_id)
                            .unwrap() as u32,
                        None => self.default_texture,
                    };
                    self.screen_quad_mesh.uniform_override.insert(
                        "uResolution".to_string(),
                        ShaderValue::Vec2(Vec2::new(self.display_w as f32, self.display_h as f32)),
                    );
                    self.screen_quad_mesh.uniform_override.insert(
                        "uRectPos".to_string(),
                        ShaderValue::Vec2(Vec2::new(rect.x.round(), rect.y.round())),
                    );
                    self.screen_quad_mesh.uniform_override.insert(
                        "uRectSize".to_string(),
                        ShaderValue::Vec2(Vec2::new(rect.w.round(), rect.h.round())),
                    );
                    self.screen_quad_mesh.uniform_override.insert(
                        "uUVOffset".to_string(),
                        ShaderValue::Vec2(Vec2::new(0.0, 0.0)),
                    );
                    self.screen_quad_mesh.uniform_override.insert(
                        "uUVScale".to_string(),
                        ShaderValue::Vec2(Vec2::new(1.0, 1.0)),
                    );
                    self
                        .screen_quad_mesh
                        .uniform_override
                        .insert("texture1".to_string(), ShaderValue::Sampler2D(texture));
                    let [mr, mg, mb, ma] = rect.modulate;
                    self.screen_quad_mesh.uniform_override.insert(
                        "uModulate".to_string(),
                        ShaderValue::Vec4(Vec4::new(
                            mr as f32 / 255.0,
                            mg as f32 / 255.0,
                            mb as f32 / 255.0,
                            ma as f32 / 255.0,
                        )),
                    );
                    let (uv_offset, uv_scale) = match rect.region {
                        Some([x1, y1, x2, y2]) => (Vec2::new(x1, y1), Vec2::new(x2 - x1, y2 - y1)),
                        None => (Vec2::new(0.0, 0.0), Vec2::new(1.0, 1.0)),
                    };
                    self.screen_quad_mesh.uniform_override.insert("uUVOffset".to_string(), ShaderValue::Vec2(uv_offset));
                    self.screen_quad_mesh.uniform_override.insert("uUVScale".to_string(), ShaderValue::Vec2(uv_scale));
                    self.screen_quad_mesh.draw(ctx);
                }
                DrawListItem::Text(text_item) => {
                    const FONT_COLS: f32 = 16.0;
                    const FONT_ROWS: f32 = 6.0;
                    const CHAR_PX: f32 = 8.0;

                    let char_w = (CHAR_PX * text_item.size).round();
                    let char_h = (CHAR_PX * text_item.size).round();
                    let origin_x = text_item.x;
                    let mut cur_x = text_item.x;
                    let mut cur_y = text_item.y;

                    self.screen_quad_mesh.uniform_override.insert(
                        "uResolution".to_string(),
                        ShaderValue::Vec2(Vec2::new(self.display_w as f32, self.display_h as f32)),
                    );
                    self.screen_quad_mesh.uniform_override.insert(
                        "uRectSize".to_string(),
                        ShaderValue::Vec2(Vec2::new(char_w, char_h)),
                    );
                    self.screen_quad_mesh.uniform_override.insert(
                        "texture1".to_string(),
                        ShaderValue::Sampler2D(self.font_texture),
                    );
                    self.screen_quad_mesh.uniform_override.insert(
                        "uModulate".to_string(),
                        ShaderValue::Vec4(Vec4::new(1.0, 1.0, 1.0, 1.0)),
                    );
                    self.screen_quad_mesh.uniform_override.insert(
                        "uUVScale".to_string(),
                        ShaderValue::Vec2(Vec2::new(1.0 / FONT_COLS, 1.0 / FONT_ROWS)),
                    );

                    for byte in text_item.text.bytes() {
                        if byte == b'\n' {
                            cur_y += CHAR_PX * text_item.size;
                            cur_x = origin_x;
                            continue;
                        }
                        let glyph = if byte >= 32 && byte <= 127 { byte } else { 127 };
                        let index = (glyph - 32) as f32;
                        let col = (index as u32 % 16) as f32;
                        let row = (index as u32 / 16) as f32;
                        self.screen_quad_mesh.uniform_override.insert(
                            "uRectPos".to_string(),
                            ShaderValue::Vec2(Vec2::new(cur_x.round(), cur_y.round())),
                        );
                        self.screen_quad_mesh.uniform_override.insert(
                            "uUVOffset".to_string(),
                            ShaderValue::Vec2(Vec2::new(col / FONT_COLS, row / FONT_ROWS)),
                        );
                        self.screen_quad_mesh.draw(ctx);
                        cur_x += char_w;
                    }
                }
                DrawListItem::CreateTexture(shimlang_texture_handle, w, h, rgba_bytes, nearest) => {
                    let gl_texture_id = gen_cpu_texture(*w, *h, *nearest, |x, y| {
                        let i = ((y * w + x) * 4) as usize;
                        [
                            rgba_bytes[i],
                            rgba_bytes[i + 1],
                            rgba_bytes[i + 2],
                            rgba_bytes[i + 3],
                        ]
                    });
                    self
                        .shimlang_texture_handle_to_gl_texture
                        .insert(*shimlang_texture_handle, gl_texture_id as i32);
                }
            }
        }
        unsafe {
            gl::Disable(gl::BLEND);
            gl::Enable(gl::DEPTH_TEST);
        }
    }
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

use png::{Decoder, Transformations};
use std::io::Cursor;

static FONT_BYTES: &[u8] = include_bytes!("../font_atlas_1bit.png");

fn load_png_from_bytes(bytes: &[u8]) -> (Vec<(u8, u8, u8, u8)>, u32, u32) {
    let mut decoder = Decoder::new(Cursor::new(bytes));
    decoder.set_transformations(Transformations::EXPAND | Transformations::STRIP_16 | Transformations::ALPHA);
    let mut reader = decoder.read_info().expect("Failed to read PNG info");

    let mut buf = vec![0u8; reader.output_buffer_size().expect("Failed to read output buffer size")];

    let info = reader.next_frame(&mut buf).expect("Failed to read PNG frame");
    let width = info.width;
    let height = info.height;

    let pixels = buf[..info.buffer_size()]
        .chunks_exact(4)
        .map(|c| (c[0], c[1], c[2], c[3]))
        .collect();

    (pixels, width, height)
}

fn draw_frame_captures(
    frame_captures: &VecDeque<GLuint>,
    frame_capture_mesh: &mut StaticMesh,
    ctx: &mut HashMap<String, ShaderValue>,
) {
    unsafe {
        gl::Disable(gl::DEPTH_TEST);
    }
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
    unsafe {
        gl::Enable(gl::DEPTH_TEST);
    }
}

fn create_screen_quad_shader() -> Rc<ShaderProgram> {
    let screen_quad_shader = ShaderProgram::create(
        ShaderBuilder::new()
            .with_input(ShaderSymbol::new(ShaderDataType::Vec3, "aPos"))
            .with_input(ShaderSymbol::new(ShaderDataType::Vec2, "aUV"))
            .with_uniform(ShaderSymbol::new(ShaderDataType::Vec2, "uResolution"))
            .with_uniform(ShaderSymbol::new(ShaderDataType::Vec2, "uRectPos"))
            .with_uniform(ShaderSymbol::new(ShaderDataType::Vec2, "uRectSize"))
            .with_uniform(ShaderSymbol::new(ShaderDataType::Vec2, "uUVOffset"))
            .with_uniform(ShaderSymbol::new(ShaderDataType::Vec2, "uUVScale"))
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
                        uv = uUVOffset + aUV * uUVScale;
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
    Rc::new(screen_quad_shader)
}

fn create_frame_capture_shader() -> Rc<ShaderProgram> {
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

    Rc::new(frame_capture_shader)
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
    let (font_pixels, font_w, font_h) = load_png_from_bytes(FONT_BYTES);
    let font_texture = gen_cpu_texture(font_w, font_h, true, |x, y| {
        let (r, g, b, a) = font_pixels[(y * font_w + x) as usize];
        [r, g, b, if r == 255 { 255 } else { 0 }]
    });
    println!("Starting Rust state initialization");

    println!("Creating interpreter");
    let script_bridge = ScriptBridge::new();

    println!("Generating arrays/buffers");
    let debug_vao = VertexArray::create();
    let debug_vbo = VertexBufferObject::create();
    let debug_ebo = ElementBufferObject::create();

    let view = Mat4::IDENTITY;
    let projection = Mat4::IDENTITY;

    let mut display_w = 0;
    let mut display_h = 0;
    unsafe {
        SHM_GetDrawableSize(&mut display_w as *mut i32, &mut display_h as *mut i32);
    }

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

    let mut numkeys: i32 = 0;
    unsafe { SDL_GetKeyboardState(&mut numkeys as *mut i32) };
    let keys = KeyState::new();

    let mut meshes = HashMap::new();

    let screen_quad_shader = create_screen_quad_shader();

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

    let frame_captures: VecDeque<GLuint> = (0..MAX_FRAME_CAPTURES)
        .map(|_| alloc_capture_texture())
        .collect();
    let frame_capture_shader = create_frame_capture_shader();

    let frame_capture_mesh = StaticMesh::create(
        frame_capture_shader.clone(),
        Rc::new(Mesh::create(&quad_mesh()).unwrap()),
    )
    .expect("Can't create the frame capture mesh");

    let mut frame_capture_fbo: GLuint = 0;
    unsafe {
        gl::GenFramebuffers(1, &mut frame_capture_fbo as *mut u32);
    }

    let default_texture = gen_cpu_texture(128, 128, false, |x, y| [0xff, 0xff, 0xff, 0xff]);

    println!("State initialized!");
    
    State {
        script_bridge,
        frame_num,
        debug_vao,
        debug_vbo,
        debug_ebo,
        view,
        projection,
        display_w,
        display_h,
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
        edit_mode: false,
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
        font_texture,
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
    if state.frame_num == 0 {
        println!("Starting first frame");
    }

    unsafe {
        let _zone = zone_scoped!("rust frame unsafe block");

        if !state.edit_mode {
            state
                .script_bridge
                .step(&state.keys.keys, &state.keys.last_keys);
        }

        update_keys(state);

        gl::ClearColor(
            0.1,
            0.1,
            0.12,
            1.0,
        );

        unsafe {
            let mut xrel = 0;
            let mut yrel = 0;
            let button_bitmask =
                SDL_GetRelativeMouseState(&mut xrel as *mut i32, &mut yrel as *mut i32);
            if state.keys.pressed(SDL_SCANCODE_GRAVE) {
                SDL_SetRelativeMouseMode(false);
                state.mouse_captured = false;
                state.edit_mode = true;
            }

            if button_bitmask.bit(0) && !igWantCaptureMouse() {
                SDL_SetRelativeMouseMode(true);
                state.mouse_captured = true;
                state.edit_mode = false;
            }
        }

        //state.move_camera_and_update_matrices(delta);
        state.update_camera_matrices();

        // Draw RGB axes
        if state.edit_mode {
            for axis in [
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
                Vec3::new(0.0, 0.0, 1.0),
            ] {
                debug_line_color(
                    state,
                    &[
                        Vec3::new(0.0, 0.0, 0.0),
                        axis,
                    ],
                    axis,
                );
            }
        }

        let mut ctx = HashMap::new();
        ctx.insert("view".to_string(), ShaderValue::Mat4(state.view));
        ctx.insert(
            "projection".to_string(),
            ShaderValue::Mat4(state.projection),
        );

        draw_debug_shapes(state);

        // TODO: I feel like this doesn't need to use the context since it's
        // not using the camera
        if !state.edit_mode {
            state.handle_draw_list(&mut ctx);
            state.capture_frame();
        }

        if state.edit_mode {
            draw_frame_captures(
                &state.frame_captures,
                &mut state.frame_capture_mesh,
                &mut ctx,
            );
        }

        log_opengl_errors!();

        // The log window only displays if it has content
        draw_log_window();

        if state.frame_num == 0 {
            println!("Finished first frame");
        }
        state.frame_num += 1;
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn rust_frame(delta: f32) -> i32 {
    let _zone = zone_scoped!("rust_frame_inner");
    STATE_REFCELL.with_borrow_mut(|value| frame(value.as_mut().unwrap(), delta));

    42
}

const _: () = compile_time_checks();
