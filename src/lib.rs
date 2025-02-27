use gl;
use glam::{Vec3, Mat4};
use std::ffi::CString;
use std::ffi::CStr;

//use std::ffi::{CString, c_void};
//use std::ptr;

use std::mem::size_of;

use std::cell::RefCell;
//use std::thread;

type ImGuiWindowFlags = core::ffi::c_int;
unsafe extern "C" {
    fn SDL_GL_GetProcAddress(proc: *const i8) -> *mut std::ffi::c_void;
    fn SDL_GetKeyboardState(numkeys: *const i32) -> *const u8;
    fn SDL_GetRelativeMouseState(x: *mut i32, y: *mut i32) -> u32;
    fn igBegin(name: *const core::ffi::c_char, p_open: *mut bool, flags: ImGuiWindowFlags) -> bool;
    fn igEnd();
    fn igText(fmt: *const core::ffi::c_char, ...);
}

const fn u32size_of<T>() -> u32 {
    size_of::<T>() as u32
}

#[derive(Default, Debug)]
pub struct State {
    frame_num: u64,
    view: Mat4,
    projection: Mat4,

    debug_shader_program: gl::types::GLuint,

    debug_view_loc: gl::types::GLint,
    debug_projection_loc: gl::types::GLint,

    box_vao: gl::types::GLuint,
    box_vbo: gl::types::GLuint,
    box_ebo: gl::types::GLuint,

    debug_verts: Vec<Vec3>,
    debug_vert_indices: Vec<u32>,

    pitch: f32,
    yaw: f32,

    camera_pos: Vec3,
    camera_front: Vec3,
    camera_up: Vec3,

    mouse_captured: bool,
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
pub static SDL_SCANCODE_LSHIFT: usize = 225;

thread_local! {
    static STATE_REFCELL: RefCell<State> = RefCell::default();
}

thread_local! {
    static DEBUG_LOG: RefCell<Vec<CString>> = RefCell::default();
}

fn log(s: String) {
    DEBUG_LOG.with_borrow_mut(|logs| logs.push(CString::new(s).unwrap()));
}

fn log_window() {
    DEBUG_LOG.with_borrow(|logs| {
        unsafe {
            let mut open = true;
            igBegin(c"Rust Log Window".as_ptr(), &mut open as *mut bool, 0);
            for line in logs.iter() {
                igText(line.as_ptr());
            }
            igEnd();
        }
    });
}

static END_PRIMITIVE: u32 = 0xFFFF_FFFF;

fn draw_debug_shapes(state: &mut State) {
    unsafe {
        gl::BindVertexArray(state.box_vao);
        gl::BindBuffer(gl::ARRAY_BUFFER, state.box_vbo);
        gl::BufferData(
            gl::ARRAY_BUFFER,
            (state.debug_verts.len() * size_of::<Vec3>()).try_into().unwrap(),
            state.debug_verts.as_ptr().cast(),
            gl::DYNAMIC_DRAW);
        gl::BindBuffer(gl::ELEMENT_ARRAY_BUFFER, state.box_ebo);
        gl::BufferData(gl::ELEMENT_ARRAY_BUFFER, (state.debug_vert_indices.len() * size_of::<u32>()) as isize, state.debug_vert_indices.as_ptr().cast(), gl::DYNAMIC_DRAW);

        // Vertex Position Attribute
        gl::VertexAttribPointer(0, 3, gl::FLOAT, gl::FALSE, (3 * u32size_of::<f32>()) as i32, std::ptr::null());
        gl::EnableVertexAttribArray(0);

        gl::UseProgram(state.debug_shader_program);

        gl::UniformMatrix4fv(state.debug_view_loc, 1, gl::FALSE, &state.view.to_cols_array() as *const gl::types::GLfloat);
        gl::UniformMatrix4fv(state.debug_projection_loc, 1, gl::FALSE, &state.projection.to_cols_array() as *const gl::types::GLfloat);

        gl::DrawElements(gl::LINE_STRIP, state.debug_vert_indices.len().try_into().unwrap(), gl::UNSIGNED_INT, std::ptr::null());

        state.debug_verts.clear();
        state.debug_vert_indices.clear();
    }
}

pub fn debug_line_loop(state: &mut State, points: &[Vec3]) {
    let start_offset: u32 = state.debug_verts.len().try_into().unwrap();
    let mut offset: u32 = state.debug_verts.len().try_into().unwrap();
    for p in points {
        state.debug_vert_indices.push(offset);
        state.debug_verts.push(*p);
        offset += 1;
    }
    state.debug_vert_indices.push(start_offset);
    state.debug_vert_indices.push(END_PRIMITIVE);
}

pub fn debug_line(state: &mut State, points: &[Vec3]) {
    let mut offset: u32 = state.debug_verts.len().try_into().unwrap();
    for p in points {
        state.debug_vert_indices.push(offset);
        state.debug_verts.push(*p);
        offset += 1;
    }
    state.debug_vert_indices.push(END_PRIMITIVE);
}

fn debug_box(state: &mut State, position: Vec3, size: Vec3, _color: Vec3) {
    let offset = state.debug_verts.len() as u32;

    let debug_verts_here = [
        position + Vec3::new( size.x,  size.y,  size.z)/2.0f32,
        position + Vec3::new( size.x,  size.y, -size.z)/2.0f32,
        position + Vec3::new( size.x, -size.y,  size.z)/2.0f32,
        position + Vec3::new( size.x, -size.y, -size.z)/2.0f32,
        position + Vec3::new(-size.x,  size.y,  size.z)/2.0f32,
        position + Vec3::new(-size.x,  size.y, -size.z)/2.0f32,
        position + Vec3::new(-size.x, -size.y,  size.z)/2.0f32,
        position + Vec3::new(-size.x, -size.y, -size.z)/2.0f32,
    ];

    for vert in debug_verts_here {
        state.debug_verts.push(vert)
    }

    let debug_vert_indices_here = [
        0, 1, 3, 2, 0,
        4, 5, 7, 6, 4,
        END_PRIMITIVE,
        1, 5,
        END_PRIMITIVE,
        3, 7,
        END_PRIMITIVE,
        2, 6,
        END_PRIMITIVE,
    ];

    for index in debug_vert_indices_here {
        state.debug_vert_indices.push(
            if index == END_PRIMITIVE {
                index
            } else {
                index + offset
            }
        );
    }
}

fn check_shader_compilation(shader: gl::types::GLuint) {
    unsafe {
        let mut success: gl::types::GLint = 0;
        gl::GetShaderiv(shader, gl::COMPILE_STATUS, &mut success as *mut gl::types::GLint);
        if success != gl::TRUE.into() {
            let mut info_log: [gl::types::GLchar; 512] = [0; 512]; // All elements set to 0
            gl::GetShaderInfoLog(shader, 512, std::ptr::null_mut(), &mut info_log as *mut gl::types::GLchar);
            println!("Error: Shader Compilation Failed");
            println!("{:?}", info_log);
        }
    }
}

fn check_program_linking(program: gl::types::GLuint) {
    unsafe {
        let mut success: gl::types::GLint = 0;
        gl::GetProgramiv(program, gl::LINK_STATUS, &mut success as *mut gl::types::GLint);
        if success != gl::TRUE.into() {
            let mut info_log: [gl::types::GLchar; 512] = [0; 512]; // All elements set to 0
            gl::GetProgramInfoLog(program, 512, std::ptr::null_mut(), &mut info_log as *mut gl::types::GLchar);
            println!("Error: Shader Compilation Failed");
            println!("{:?}", info_log);
        }
    }
}


fn init_state() -> State {
    unsafe {
        println!("Starting Rust state initialization");

        let mut box_vao = 0;
        let mut box_vbo = 0;
        let mut box_ebo = 0;

        println!("Generating arrays/buffers");
        gl::GenVertexArrays(1, &mut box_vao as *mut u32);
        gl::GenBuffers(1, &mut box_vbo as *mut u32);
        gl::GenBuffers(1, &mut box_ebo as *mut u32);

        let view = Mat4::IDENTITY;
        let projection = Mat4::IDENTITY;

        let debug_verts = Vec::new();
        let debug_vert_indices = Vec::new();

        let debug_vertex_shader_source: &CStr = cr#"#version 300 es
        precision highp float;
        layout (location = 0) in vec3 aPos;

        uniform mat4 projection;
        uniform mat4 view;

        void main() {
            gl_Position = projection * view * vec4(aPos, 1.0);
        }
        "#;

        // Fragment Shader Source Code
        let debug_fragment_shader_source: &CStr = cr#"#version 300 es
        precision highp float;
        out vec4 FragColor;

        void main() {
            FragColor = vec4(0.0, 1.0, 0.0, 1.0);
        }
        "#;

        println!("Starting shader compilation");

        let debug_vertex_shader = gl::CreateShader(gl::VERTEX_SHADER);


        let ptr: *const i8 = debug_vertex_shader_source.as_ptr(); // Convert &CStr to *const i8
        let ptr_array: [*const i8; 1] = [ptr]; // Create an array with one element
        let ptr_to_ptr: *const *const i8 = ptr_array.as_ptr(); // Get pointer to the array
        println!("Vertex shader source");
        gl::ShaderSource(debug_vertex_shader, 1, ptr_to_ptr, std::ptr::null());
        gl::CompileShader(debug_vertex_shader);
        println!("Rust: Compiling debug vertex shader");
        check_shader_compilation(debug_vertex_shader);

        let ptr: *const i8 = debug_fragment_shader_source.as_ptr(); // Convert &CStr to *const i8
        let ptr_array: [*const i8; 1] = [ptr]; // Create an array with one element
        let ptr_to_ptr: *const *const i8 = ptr_array.as_ptr(); // Get pointer to the array
        let debug_fragment_shader = gl::CreateShader(gl::FRAGMENT_SHADER);
        println!("Fragment shader source");
        gl::ShaderSource(debug_fragment_shader, 1, ptr_to_ptr, std::ptr::null());
        gl::CompileShader(debug_fragment_shader);
        println!("Rust: Compiling debug fragment shader");
        check_shader_compilation(debug_fragment_shader);

        let debug_shader_program = gl::CreateProgram();
        gl::AttachShader(debug_shader_program, debug_vertex_shader);
        gl::AttachShader(debug_shader_program, debug_fragment_shader);
        gl::LinkProgram(debug_shader_program);
        println!("Rust: Linking debug shader program");
        check_program_linking(debug_shader_program);

        let debug_view_loc = gl::GetUniformLocation(debug_shader_program, c"view".as_ptr());
        let debug_projection_loc = gl::GetUniformLocation(debug_shader_program, c"projection".as_ptr());

        println!("Successfully initialized Rust State");

        let frame_num = 0;

        let camera_pos = Vec3::new(0.0f32, 2.8f32, 7.7f32);
        let camera_front = Vec3::new(0.0f32, 0.0f32, -1.0f32);
        let camera_up = Vec3::new(0.0f32, 1.0f32,  0.0f32);

        let pitch = -20.;
        let yaw = -90.;

        let mouse_captured = true;

        State {
            frame_num,
            box_vao,
            box_vbo,
            box_ebo,
            view,
            projection,
            debug_view_loc,
            debug_projection_loc,
            debug_shader_program,
            debug_verts,
            debug_vert_indices,
            camera_pos,
            camera_front,
            camera_up,
            pitch,
            yaw,
            mouse_captured,
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn rust_init() -> i32 {

    unsafe {
        gl::load_with(|s| SDL_GL_GetProcAddress(CString::new(s).unwrap().as_ptr()));
    }

    STATE_REFCELL.with_borrow_mut(|value| {
        let initial_state = init_state();
        log(format!("{:#?}", initial_state));
        *value = initial_state;
    });
    0
}

fn frame(state: &mut State, delta: f32) {
    unsafe {
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

        debug_box(
            state,
            Vec3::new(-2.0f32, 1.0f32, 0.0f32),
            Vec3::new(1.0f32, 1.0f32, 1.0f32),
            Vec3::new(1.0f32, 1.0f32, 1.0f32),
        );
        debug_box(
            state,
            Vec3::new(0.0f32, 1.0f32, 0.0f32),
            Vec3::new(1.0f32, 1.0f32, 1.0f32),
            Vec3::new(1.0f32, 1.0f32, 1.0f32),
        );
        debug_box(
            state,
            Vec3::new(2.0f32, 1.0f32, 0.0f32),
            Vec3::new(1.0f32, 1.0f32, 1.0f32),
            Vec3::new(1.0f32, 1.0f32, 1.0f32),
        );

        for x in -10..=10 {
            debug_line(
                state,
                &[
                    Vec3::new(x as f32, 0.0f32, -10.0f32),
                    Vec3::new(x as f32, 0.0f32, 10.0f32),
                ]
            );
            debug_line(
                state,
                &[
                    Vec3::new(10.0f32, 0.0f32, x as f32),
                    Vec3::new(-10.0f32, 0.0f32, x as f32),
                ]
            );
        }


        let key_state = SDL_GetKeyboardState(std::ptr::null());

        let base_camera_speed = 2.0f32;
        let camera_speed = if *key_state.add(SDL_SCANCODE_LSHIFT) != 0 {
            base_camera_speed*5.0
        } else {
            base_camera_speed
        };

        if *key_state.add(SDL_SCANCODE_S) != 0 {
            state.camera_pos -= delta*camera_speed*state.camera_front;
        }
        if *key_state.add(SDL_SCANCODE_W) != 0 {
            state.camera_pos += delta*camera_speed*state.camera_front;
        }
        if *key_state.add(SDL_SCANCODE_D) != 0{
            state.camera_pos += state.camera_front.cross(state.camera_up).normalize() * delta*camera_speed;
        }
        if *key_state.add(SDL_SCANCODE_A) != 0{
            state.camera_pos -= state.camera_front.cross(state.camera_up).normalize() * delta*camera_speed;
        }

        let mut xrel: i32 = 0;
        let mut yrel: i32 = 0;

        SDL_GetRelativeMouseState(&mut xrel as *mut i32, &mut yrel as *mut i32);
        if *key_state.add(SDL_SCANCODE_ESCAPE) != 0 {
            state.mouse_captured = false;
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

        state.projection = Mat4::perspective_rh_gl(
            f32::to_radians(45.0), 1280.0f32 / 720.0f32, 0.1f32, 100.0f32
        );

        let mut open = true;
        igBegin(c"From Rust".as_ptr(), &mut open as *mut bool, 0);
        igText(c"Some text from Rust".as_ptr());
        igText(CString::new(format!("Camera Position: {} {} {}", state.camera_pos[0], state.camera_pos[1], state.camera_pos[2])).unwrap().as_ptr());
        igText(CString::new(format!("Camera Front: {} {} {}", state.camera_front[0], state.camera_front[1], state.camera_front[2])).unwrap().as_ptr());
        igText(CString::new(format!("Camera Pitch: {}", state.pitch)).unwrap().as_ptr());
        igText(CString::new(format!("Camera Yaw: {}", state.yaw)).unwrap().as_ptr());
        igEnd();

        log_window();

        draw_debug_shapes(state);
        state.frame_num += 1;
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn rust_frame(delta: f32) -> i32 {
    STATE_REFCELL.with_borrow_mut(|value| {
        frame(value, delta)
    });

    42
}