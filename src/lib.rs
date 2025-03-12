use gl;
use std::collections::HashMap;
use std::fmt;
use std::rc::Rc;
use gl::types::*;
use glam::{Vec2, Vec3, Vec4, Mat4};
use std::ffi::CString;
use std::ffi::CStr;

use std::mem::size_of;

use std::cell::RefCell;

mod gpu;
mod mesh_gen;
mod debug_draw;

use crate::gpu::*;
use crate::debug_draw::{draw_debug_shapes, debug_line, debug_line_loop, debug_box};
use crate::mesh_gen::box_mesh;

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

pub struct State {
    frame_num: u64,
    view: Mat4,
    projection: Mat4,

    debug_shader_program: Rc<ShaderProgram>,

    debug_view_loc: gl::types::GLint,
    debug_projection_loc: gl::types::GLint,

    debug_vao: VertexArray,
    debug_vbo: VertexBufferObject,
    debug_ebo: ElementBufferObject,

    debug_verts: Vec<Vec3>,
    debug_vert_indices: Vec<u32>,

    pitch: f32,
    yaw: f32,

    camera_pos: Vec3,
    camera_front: Vec3,
    camera_up: Vec3,

    mouse_captured: bool,

    test_mesh: StaticMesh,
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
    static STATE_REFCELL: RefCell<Option<State>> = RefCell::default();
}

thread_local! {
    static DEBUG_LOG: RefCell<Vec<CString>> = RefCell::default();
}

fn log<T: AsRef<str>>(s: T) {
    logc(CString::new(s.as_ref().to_string()).unwrap());
}

fn logc(s: CString) {
    println!("{:?}", s);
    DEBUG_LOG.with_borrow_mut(|logs| logs.push(s));
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


fn create_test_mesh() -> MeshDataRaw {
    let mut my_box = box_mesh(Vec3::new(1.0, 1.0, 1.0));
    let len = my_box.verts.get("aPos").unwrap().len();

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

fn init_state() -> State {
    println!("Starting Rust state initialization");

    println!("Generating arrays/buffers");
    let debug_vao = VertexArray::create();
    let debug_vbo = VertexBufferObject::create();
    let debug_ebo = ElementBufferObject::create();

    let view = Mat4::IDENTITY;
    let projection = Mat4::IDENTITY;

    let debug_verts = Vec::new();
    let debug_vert_indices = Vec::new();

    let debug_vertex_shader = ShaderBuilder::new()
        .with_input(ShaderSymbol::new(ShaderDataType::Vec3, "aPos"))
        .with_uniform(ShaderSymbol::new(ShaderDataType::Mat4, "projection"))
        .with_uniform(ShaderSymbol::new(ShaderDataType::Mat4, "view"))
        .with_code(
            r#"
                void main() {
                    gl_Position = projection * view * vec4(aPos, 1.0);
                }
            "#.to_string()
        )
        .build_vertex_shader()
        .expect("Could not compile debug vertex shader");

    let debug_fragment_shader = ShaderBuilder::new()
        .with_output(ShaderSymbol::new(ShaderDataType::Vec4, "FragColor"))
        .with_code(
            r#"
                void main() {
                    FragColor = vec4(0.0, 1.0, 0.0, 1.0);
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

    log_opengl_errors();
    println!("Creating test mesh");
    let test_mesh = StaticMesh::create(
        vert_color_shader.clone(),
        Rc::new(Mesh::create(&create_test_mesh()).unwrap()),
    ).expect("Can't create the test mesh");
    log_opengl_errors();

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
        debug_verts,
        debug_vert_indices,
        camera_pos,
        camera_front,
        camera_up,
        pitch,
        yaw,
        mouse_captured,
        test_mesh,
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

fn log_opengl_errors() {
    unsafe {
        loop {
            let err = gl::GetError();
            match err {
                gl::NO_ERROR => break,
                gl::INVALID_ENUM => log("OpenGL Error INVALID_ENUM: An unacceptable value is specified for an enumerated argument. The offending command is ignored and has no other side effect than to set the error flag."),
                gl::INVALID_VALUE => log("OpenGL Error INVALID_VALUE: A numeric argument is out of range. The offending command is ignored and has no other side effect than to set the error flag."),
                gl::INVALID_OPERATION => log("OpenGL Error INVALID_OPERATION: The specified operation is not allowed in the current state. The offending command is ignored and has no other side effect than to set the error flag."),
                gl::INVALID_FRAMEBUFFER_OPERATION => log("OpenGL Error INVALID_FRAMEBUFFER_OPERATION: The framebuffer object is not complete. The offending command is ignored and has no other side effect than to set the error flag."),
                gl::OUT_OF_MEMORY => log("OpenGL Error OUT_OF_MEMORY: There is not enough memory left to execute the command. The state of the GL is undefined, except for the state of the error flags, after this error is recorded."),
                gl::STACK_UNDERFLOW => log("OpenGL Error STACK_UNDERFLOW: An attempt has been made to perform an operation that would cause an internal stack to underflow."),
                gl::STACK_OVERFLOW => log("OpenGL Error STACK_OVERFLOW: An attempt has been made to perform an operation that would cause an internal stack to overflow. "),
                _ => log("OpenGL Error: Unknown OpenGL error"),
            }
            if err == gl::NO_ERROR {
                break;
            }
        }
    }
}

fn frame(state: &mut State, delta: f32) {
    if state.frame_num == 0 {
        println!("Starting first frame");
    }
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

        state.test_mesh.transform = Mat4::from_translation(Vec3::new(1.2, 0.2, -0.2));
        println!("Drawing test mesh");
        let mut ctx = HashMap::new();
        ctx.insert("view".to_string(), ShaderValue::Mat4(state.view));
        ctx.insert("projection".to_string(), ShaderValue::Mat4(state.projection));
        state.test_mesh.draw(&mut ctx);
        log_opengl_errors();
        println!("Drew test mesh");

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