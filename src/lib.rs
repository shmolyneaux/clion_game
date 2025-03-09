use gl;
use std::collections::HashMap;
use std::fmt;
use std::rc::Rc;
use gl::types::*;
use glam::{Vec2, Vec3, Mat4};
use std::ffi::CString;
use std::ffi::CStr;

use std::mem::size_of;

use std::cell::RefCell;

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

static END_PRIMITIVE: u32 = 0xFFFF_FFFF;

fn draw_debug_shapes(state: &mut State) {
    unsafe {
        gl::BindVertexArray(state.debug_vao.id);
        state.debug_vbo.bind_data(&state.debug_verts, gl::DYNAMIC_DRAW);
        state.debug_ebo.bind_data(&state.debug_vert_indices, gl::DYNAMIC_DRAW);

        // Vertex Position Attribute
        gl::VertexAttribPointer(0, 3, gl::FLOAT, gl::FALSE, (3 * u32size_of::<f32>()) as i32, std::ptr::null());
        gl::EnableVertexAttribArray(0);

        gl::UseProgram(state.debug_shader_program.id);

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

#[derive(Clone)]
enum ShaderDataType {
    Float,
    Vec3,
    Vec4,
    Mat4,
}

impl fmt::Display for ShaderDataType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}",
            match self {
                ShaderDataType::Float => "float",
                ShaderDataType::Vec3 => "vec3",
                ShaderDataType::Vec4 => "vec4",
                ShaderDataType::Mat4 => "mat4",
            }
        )
    }
}

#[derive(Clone)]
struct ShaderSymbol {
    data_type: ShaderDataType,
    name: String,
}

impl ShaderSymbol {
    fn new(data_type: ShaderDataType, name: &str) -> Self {
        Self {data_type, name: name.to_string()}
    }
}

#[derive(Clone)]
enum ShaderType {
    Vertex,
    Fragment,
}

struct ShaderBuilder {
    inputs: Vec<ShaderSymbol>,
    uniform_inputs: Vec<ShaderSymbol>,
    // NOTE: later this will need to be updated for multiple render targets
    outputs: Vec<ShaderSymbol>,
    code: String,
}

impl ShaderBuilder {
    fn new() -> Self {
        Self {
            inputs: Vec::new(),
            uniform_inputs: Vec::new(),
            outputs: Vec::new(),
            code: String::new(),
        }
    }

    fn with_input(mut self, input: ShaderSymbol) -> Self {
        self.inputs.push(input);
        self
    }

    fn with_uniform(mut self, uniform_input: ShaderSymbol) -> Self {
        self.uniform_inputs.push(uniform_input);
        self
    }

    fn with_output(mut self, output: ShaderSymbol) -> Self {
        self.outputs.push(output);
        self
    }

    fn with_code(mut self, code: String) -> Self {
        self.code = code;
        self
    }

    fn build_vertex_shader(&self) -> Result<VertexShader, String> {
        VertexShader::create(&self.build(ShaderType::Vertex), self.inputs.clone(), self.uniform_inputs.clone(), self.outputs.clone())
    }

    fn build_fragment_shader(&self) -> Result<FragmentShader, String> {
        FragmentShader::create(&self.build(ShaderType::Fragment), self.inputs.clone(), self.uniform_inputs.clone(), self.outputs.clone())
    }

    fn build(&self, shader_type: ShaderType) -> CString {
        let mut shader_src = "#version 300 es\nprecision highp float;\n".to_string();
        match shader_type {
            ShaderType::Vertex => {
                for (idx, vertex_attribute) in self.inputs.iter().enumerate() {
                    shader_src.push_str(&format!("layout (location = {}) in {} {};\n", idx, vertex_attribute.data_type, vertex_attribute.name));
                }
            }
            ShaderType::Fragment => {
                for attrib_info in self.inputs.iter() {
                    shader_src.push_str(&format!("in {} {};\n", attrib_info.data_type, attrib_info.name));
                }
            }
        }
        for uniform_info in self.uniform_inputs.iter() {
            shader_src.push_str(&format!("uniform {} {};\n", uniform_info.data_type, uniform_info.name));
        }
        for output in self.outputs.iter() {
            shader_src.push_str(&format!("out {} {};\n", output.data_type, output.name));
        }
        shader_src.push_str(&self.code);
        CString::new(shader_src).unwrap()
    }
}

struct VertexShader {
    id: gl::types::GLuint,
    inputs: Vec<ShaderSymbol>,
    uniform_inputs: Vec<ShaderSymbol>,
    outputs: Vec<ShaderSymbol>,
}

impl VertexShader {
    fn create(src: &CStr, inputs: Vec<ShaderSymbol>, uniform_inputs: Vec<ShaderSymbol>, outputs: Vec<ShaderSymbol>) -> Result<Self, String> {
        let id = compile_shader(src, gl::VERTEX_SHADER)?;
        Ok(Self { id, inputs, uniform_inputs, outputs })
    }
}

impl Drop for VertexShader {
    fn drop(&mut self) {
        unsafe {
            gl::DeleteShader(self.id);
        }
    }
}

struct FragmentShader {
    id: gl::types::GLuint,
    inputs: Vec<ShaderSymbol>,
    uniform_inputs: Vec<ShaderSymbol>,
    outputs: Vec<ShaderSymbol>,
}

impl FragmentShader {
    fn create(src: &CStr, inputs: Vec<ShaderSymbol>, uniform_inputs: Vec<ShaderSymbol>, outputs: Vec<ShaderSymbol>) -> Result<Self, String> {
        let id = compile_shader(src, gl::FRAGMENT_SHADER)?;
        Ok(Self { id, inputs, uniform_inputs, outputs })
    }
}

impl Drop for FragmentShader {
    fn drop(&mut self) {
        unsafe {
            gl::DeleteShader(self.id);
        }
    }
}

fn compile_shader(src: &CStr, shader_type: gl::types::GLuint) -> Result<gl::types::GLuint, String> {
    unsafe {
        let id = gl::CreateShader(shader_type);

        let ptr: *const i8 = src.as_ptr(); // Convert &CStr to *const i8
        let ptr_array: [*const i8; 1] = [ptr]; // Create an array with one element
        let ptr_to_ptr: *const *const i8 = ptr_array.as_ptr(); // Get pointer to the array
        gl::ShaderSource(id, 1, ptr_to_ptr, std::ptr::null());
        gl::CompileShader(id);

        let mut compilation_success: gl::types::GLint = 0;
        gl::GetShaderiv(id, gl::COMPILE_STATUS, &mut compilation_success as *mut gl::types::GLint);
        if compilation_success != gl::TRUE.into() {
            // Log length including the null terminator
            let mut log_length: gl::types::GLint = 0;
            gl::GetShaderiv(id, gl::INFO_LOG_LENGTH, &mut log_length as *mut gl::types::GLint);

            let mut error_text: Vec<u8> = vec![0; log_length as usize-1];
            gl::GetShaderInfoLog(id, error_text.len() as i32, std::ptr::null_mut(), error_text.as_mut_ptr() as *mut gl::types::GLchar);

            return Err(String::from_utf8_lossy(&error_text).into_owned());
        }

        Ok(id)
    }
}

struct ShaderProgram {
    id: gl::types::GLuint,
    vert: VertexShader,
    frag: FragmentShader,
}

impl ShaderProgram {
    fn create(vert: VertexShader, frag: FragmentShader) -> Result<Self, String> {
        println!("Creating shader");
        unsafe {
            let id = gl::CreateProgram();
            gl::AttachShader(id, vert.id);
            gl::AttachShader(id, frag.id);
            gl::LinkProgram(id);

            let mut link_success: gl::types::GLint = 0;
            gl::GetProgramiv(id, gl::LINK_STATUS, &mut link_success as *mut gl::types::GLint);
            if link_success != gl::TRUE.into() {
                // Log length including the null terminator
                let mut log_length: gl::types::GLint = 0;
                gl::GetProgramiv(id, gl::INFO_LOG_LENGTH, &mut log_length as *mut gl::types::GLint);

                let mut error_text: Vec<u8> = vec![0; log_length as usize-1];
                gl::GetProgramInfoLog(id, error_text.len() as i32, std::ptr::null_mut(), error_text.as_mut_ptr() as *mut gl::types::GLchar);

                return Err(String::from_utf8_lossy(&error_text).into_owned());
            }

            println!("Returning shader");
            Ok(Self {id, vert, frag})
        }
    }

    fn uniform_location(&self, uniform_name: &CStr) -> gl::types::GLint {
        unsafe {
            gl::GetUniformLocation(self.id, uniform_name.as_ptr())
        }
    }
}

#[derive(Debug)]
struct VertexArray {
    id: GLuint
}

impl VertexArray {
    fn create() -> Self {
        unsafe {
            let mut id: GLuint = 0;
            gl::GenVertexArrays(1, &mut id as *mut GLuint);
            Self { id }
        }
    }
}

#[derive(Debug)]
struct VertexBufferObject {
    id: GLuint
}

impl VertexBufferObject {
    fn create() -> Self {
        unsafe {
            let mut id: GLuint = 0;
            gl::GenBuffers(1, &mut id as *mut GLuint);
            Self { id }
        }
    }

    fn bind_data<T>(&self, data: &[T], mode: GLuint) {
        unsafe {
            gl::BindBuffer(gl::ARRAY_BUFFER, self.id);
            gl::BufferData(gl::ARRAY_BUFFER, (data.len() * size_of::<T>()) as isize, data.as_ptr().cast(), mode);
        }
    }
}

enum Primitive {
    Points,
    Lines,
    LineLoop,
    LineStrip,
    Triangles,
    TriangleStrip,
    TriangleFan,
}

#[derive(Debug)]
struct ElementBufferObject {
    id: GLuint,
}

impl ElementBufferObject {
    fn create() -> Self {
        unsafe {
            let mut id: GLuint = 0;
            gl::GenBuffers(1, &mut id as *mut GLuint);
            Self { id }
        }
    }

    fn bind_data<T>(&self, data: &[T], mode: GLuint) {
        unsafe {
            gl::BindBuffer(gl::ELEMENT_ARRAY_BUFFER, self.id);
            gl::BufferData(gl::ELEMENT_ARRAY_BUFFER, (data.len() * size_of::<T>()) as isize, data.as_ptr().cast(), mode);
        }
    }
}

enum VertVec {
    Float(Vec<f32>),
    Vec2(Vec<Vec2>),
    Vec3(Vec<Vec3>),
    Mat4(Vec<Mat4>),
}

impl VertVec {
   fn element_size(&self) -> u32 {
        match self {
            VertVec::Float(_) => u32size_of::<f32>(),
            VertVec::Vec2(_) => u32size_of::<Vec2>(),
            VertVec::Vec3(_) => u32size_of::<Vec3>(),
            VertVec::Mat4(_) => u32size_of::<Mat4>(),
        }
   }

   fn component_count(&self) -> u32 {
        match self {
            VertVec::Float(_) => 1,
            VertVec::Vec2(_) => 2,
            VertVec::Vec3(_) => 3,
            VertVec::Mat4(_) => 16,
        }
   }

   fn len(&self) -> usize {
       match self {
           VertVec::Float(v) => v.len(),
           VertVec::Vec2(v) => v.len(),
           VertVec::Vec3(v) => v.len(),
           VertVec::Mat4(v) => v.len(),
       }
   }
}


struct MeshData {
    verts: HashMap<String, VertVec>,
    indices: Vec<u32>,
}

fn create_test_mesh() -> MeshData {
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

fn box_mesh(size: Vec3) -> MeshData {
    let positions = vec![
        // POS X
        Vec3::new( size.x,  size.y,  size.z)/2.0f32,
        Vec3::new( size.x,  size.y, -size.z)/2.0f32,
        Vec3::new( size.x, -size.y,  size.z)/2.0f32,
        Vec3::new( size.x, -size.y, -size.z)/2.0f32,

        // NEG X
        Vec3::new(-size.x,  size.y,  size.z)/2.0f32,
        Vec3::new(-size.x,  size.y, -size.z)/2.0f32,
        Vec3::new(-size.x, -size.y,  size.z)/2.0f32,
        Vec3::new(-size.x, -size.y, -size.z)/2.0f32,

        // POS Y
        Vec3::new( size.x,  size.y, -size.z)/2.0f32,
        Vec3::new( size.x,  size.y,  size.z)/2.0f32,
        Vec3::new(-size.x,  size.y, -size.z)/2.0f32,
        Vec3::new(-size.x,  size.y,  size.z)/2.0f32,

        // NEG Y
        Vec3::new( size.x, -size.y, -size.z)/2.0f32,
        Vec3::new( size.x, -size.y,  size.z)/2.0f32,
        Vec3::new(-size.x, -size.y, -size.z)/2.0f32,
        Vec3::new(-size.x, -size.y,  size.z)/2.0f32,

        // POS Z
        Vec3::new( size.x, -size.y,  size.z)/2.0f32,
        Vec3::new( size.x,  size.y,  size.z)/2.0f32,
        Vec3::new(-size.x, -size.y,  size.z)/2.0f32,
        Vec3::new(-size.x,  size.y,  size.z)/2.0f32,

        // NEG Z
        Vec3::new( size.x, -size.y, -size.z)/2.0f32,
        Vec3::new( size.x,  size.y, -size.z)/2.0f32,
        Vec3::new(-size.x, -size.y, -size.z)/2.0f32,
        Vec3::new(-size.x,  size.y, -size.z)/2.0f32,
    ];
    let normals = vec![
        Vec3::X,
        Vec3::X,
        Vec3::X,
        Vec3::X,
        Vec3::NEG_X,
        Vec3::NEG_X,
        Vec3::NEG_X,
        Vec3::NEG_X,
        Vec3::Y,
        Vec3::Y,
        Vec3::Y,
        Vec3::Y,
        Vec3::NEG_Y,
        Vec3::NEG_Y,
        Vec3::NEG_Y,
        Vec3::NEG_Y,
        Vec3::Z,
        Vec3::Z,
        Vec3::Z,
        Vec3::Z,
        Vec3::NEG_Z,
        Vec3::NEG_Z,
        Vec3::NEG_Z,
        Vec3::NEG_Z,
    ];

    let indices = vec![
        0+0, 0+1, 0+2, 0+1, 0+2, 0+3,
        4+0, 4+1, 4+2, 4+1, 4+2, 4+3,
        8+0, 8+1, 8+2, 8+1, 8+2, 8+3,
       12+0,12+1,12+2,12+1,12+2,12+3,
       16+0,16+1,16+2,16+1,16+2,16+3,
       20+0,20+1,20+2,20+1,20+2,20+3,
    ];

    assert!(positions.len() == normals.len());

    let mut verts = HashMap::new();
    verts.insert("aPos".to_string(), VertVec::Vec3(positions));
    verts.insert("aNormal".to_string(), VertVec::Vec3(normals));

    MeshData {
        verts,
        indices
    }
}

struct StaticMesh {
    shader: Rc<ShaderProgram>,
    vao: VertexArray,
    vbo: VertexBufferObject,
    ebo: ElementBufferObject,
    index_count: u32,
    model_loc: GLint,
    view_loc: GLint,
    projection_loc: GLint,
    transform: Mat4,
}

impl StaticMesh {
    fn create(shader: Rc<ShaderProgram>, mesh: &MeshData) -> Result<Self, String> {
        // Check that the mesh has the vertex inputs the shader needs
        log_opengl_errors();
        println!("Getting vert count");
        let mut vert_count = 0;
        for input in shader.vert.inputs.iter() {
            match mesh.verts.get(&input.name) {
                Some(vert_attribute) => vert_count = vert_attribute.len(),
                None => return Err(format!("Shader expected vert attribute {} which is not in mesh", input.name)),
            }
        }

        if vert_count == 0 {
            return Err("No verts in mesh data!".to_string());
        }

        log_opengl_errors();
        println!("Creating buffers");
        let vao = VertexArray::create();
        let vbo = VertexBufferObject::create();
        let ebo = ElementBufferObject::create();

        unsafe { gl::BindVertexArray(vao.id) };

        log_opengl_errors();
        println!("Getting vert attribs");
        let mut vert_attribs: Vec<&VertVec> = vec![];
        for input in shader.vert.inputs.iter() {
            vert_attribs.push(mesh.verts.get(&input.name).unwrap());
        }

        // TODO: figure out how to make this bytes with properly supported endianness so that we can use non-float inputs
        log_opengl_errors();
        println!("Pushing vert data");
        let mut vert_data: Vec<f32> = vec![];
        for idx in 0..vert_count {
            for vert_attrib in vert_attribs.iter() {
                match vert_attrib {
                    VertVec::Float(v) => vert_data.push(v[idx]),
                    VertVec::Vec2(v) => {
                        let p = v[idx];
                        vert_data.push(p.x);
                        vert_data.push(p.y);
                    },
                    VertVec::Vec3(v) => {
                        let p = v[idx];
                        vert_data.push(p.x);
                        vert_data.push(p.y);
                        vert_data.push(p.z);
                    },
                    VertVec::Mat4(_v) => todo!(),
                }
            }
        }

        vbo.bind_data(&vert_data, gl::STATIC_DRAW);
        ebo.bind_data(&mesh.indices, gl::STATIC_DRAW);

        log_opengl_errors();
        println!("Setting up VAO pointers");
        unsafe {
            let stride: u32 = vert_attribs.iter().map(|a| a.element_size()).sum();

            let mut offset = 0;
            for (idx, attribute) in vert_attribs.iter().enumerate() {
                println!(
                    "gl::VertexAttribPointer({} as GLuint, {} as GLint, gl::FLOAT, gl::FALSE, {} as i32, {} as *const std::ffi::c_void)",
                    idx, attribute.component_count(), stride, offset,
                );
                gl::VertexAttribPointer(idx as GLuint, attribute.component_count() as GLint, gl::FLOAT, gl::FALSE, stride as i32, offset as *const std::ffi::c_void);
                println!(
                    "gl::EnableVertexAttribArray({} as u32)",
                    idx,
               );
                gl::EnableVertexAttribArray(idx as u32);
                log_opengl_errors();

                offset += attribute.element_size();
            }
        }

        let index_count = mesh.indices.len() as u32;

        log_opengl_errors();
        println!("Getting uniform location");
        // TODO: data-driven uniform loc retrieval
        let model_loc = shader.uniform_location(c"model");
        let view_loc = shader.uniform_location(c"view");
        let projection_loc = shader.uniform_location(c"projection");

        let transform = Mat4::IDENTITY;

        log_opengl_errors();
        println!("Returning StaticMesh");
        Ok(Self {
            shader,
            vao,
            vbo,
            ebo,
            index_count,
            model_loc,
            view_loc,
            projection_loc,
            transform,
        })
    }

    fn draw(&self, view: Mat4, projection: Mat4) {
        unsafe {
            gl::BindVertexArray(self.vao.id);

            gl::UseProgram(self.shader.id);

            gl::UniformMatrix4fv(self.model_loc, 1, gl::FALSE, &self.transform.to_cols_array() as *const gl::types::GLfloat);
            gl::UniformMatrix4fv(self.view_loc, 1, gl::FALSE, &view.to_cols_array() as *const gl::types::GLfloat);
            gl::UniformMatrix4fv(self.projection_loc, 1, gl::FALSE, &projection.to_cols_array() as *const gl::types::GLfloat);

            gl::DrawElements(gl::TRIANGLES, self.index_count as i32, gl::UNSIGNED_INT, std::ptr::null());
        }
    }
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
    let default_shader = Rc::new(default_shader);

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
        &create_test_mesh(),
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

        draw_debug_shapes(state);
        state.test_mesh.transform = Mat4::from_translation(Vec3::new(1.0, 0.0, 0.0));
        state.test_mesh.draw(state.view, state.projection);
        log_opengl_errors();

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