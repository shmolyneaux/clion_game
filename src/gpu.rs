#![allow(unused_variables, unused_mut, unused_imports)]

use crate::*;

pub static END_PRIMITIVE: u32 = 0xFFFF_FFFF;

use std::cell::RefCell;

#[derive(Facet, Copy, Clone, Debug)]
#[repr(u8)]
pub enum ShaderDataType {
    Float,
    Vec2,
    Vec3,
    Vec4,
    Mat4,
    Sampler2D,
}

impl ShaderDataType {
   fn component_count(&self) -> u32 {
        match self {
            ShaderDataType::Float => 1,
            ShaderDataType::Vec2 => 2,
            ShaderDataType::Vec3 => 3,
            ShaderDataType::Vec4 => 4,
            ShaderDataType::Mat4 => 16,
            ShaderDataType::Sampler2D => panic!("sampler2D does not have a component count"),
        }
   }
}

impl fmt::Display for ShaderDataType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}",
            match self {
                ShaderDataType::Float => "float",
                ShaderDataType::Vec2 => "vec2",
                ShaderDataType::Vec3 => "vec3",
                ShaderDataType::Vec4 => "vec4",
                ShaderDataType::Mat4 => "mat4",
                ShaderDataType::Sampler2D => "sampler2D",
            }
        )
    }
}

#[derive(Facet, Copy, Clone)]
#[repr(u8)]
pub enum ShaderValue {
    Float(f32),
    Vec2(Vec2),
    Vec3(Vec3),
    Vec4(Vec4),
    Mat4(Mat4),
    Sampler2D(GLuint),
}

#[derive(Facet, Clone)]
pub struct ShaderSymbol {
    data_type: ShaderDataType,
    name: String,
}

impl ShaderSymbol {
    pub fn new(data_type: ShaderDataType, name: &str) -> Self {
        Self {data_type, name: name.to_string()}
    }
}

#[derive(Clone)]
pub enum ShaderType {
    Vertex,
    Fragment,
}

pub struct ShaderBuilder {
    inputs: Vec<ShaderSymbol>,
    uniform_inputs: Vec<ShaderSymbol>,
    // NOTE: later this will need to be updated for multiple render targets
    outputs: Vec<ShaderSymbol>,
    code: String,
}

impl ShaderBuilder {
    pub fn new() -> Self {
        Self {
            inputs: Vec::new(),
            uniform_inputs: Vec::new(),
            outputs: Vec::new(),
            code: String::new(),
        }
    }

    pub fn with_input(mut self, input: ShaderSymbol) -> Self {
        self.inputs.push(input);
        self
    }

    pub fn with_uniform(mut self, uniform_input: ShaderSymbol) -> Self {
        self.uniform_inputs.push(uniform_input);
        self
    }

    pub fn with_output(mut self, output: ShaderSymbol) -> Self {
        self.outputs.push(output);
        self
    }

    pub fn with_code(mut self, code: String) -> Self {
        self.code = code;
        self
    }

    pub fn build_vertex_shader(&self) -> Result<VertexShader, String> {
        VertexShader::create(&self.build(ShaderType::Vertex), self.inputs.clone(), self.uniform_inputs.clone(), self.outputs.clone())
    }

    pub fn build_fragment_shader(&self) -> Result<FragmentShader, String> {
        FragmentShader::create(&self.build(ShaderType::Fragment), self.inputs.clone(), self.uniform_inputs.clone(), self.outputs.clone())
    }

    pub fn build(&self, shader_type: ShaderType) -> CString {
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

#[derive(Facet)]
pub struct VertexShader {
    id: gl::types::GLuint,
    inputs: Vec<ShaderSymbol>,
    uniform_inputs: Vec<ShaderSymbol>,
    outputs: Vec<ShaderSymbol>,
}

impl VertexShader {
    pub fn create(src: &CStr, inputs: Vec<ShaderSymbol>, uniform_inputs: Vec<ShaderSymbol>, outputs: Vec<ShaderSymbol>) -> Result<Self, String> {
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

#[derive(Facet)]
pub struct FragmentShader {
    id: gl::types::GLuint,
    inputs: Vec<ShaderSymbol>,
    uniform_inputs: Vec<ShaderSymbol>,
    outputs: Vec<ShaderSymbol>,
}

impl FragmentShader {
    pub fn create(src: &CStr, inputs: Vec<ShaderSymbol>, uniform_inputs: Vec<ShaderSymbol>, outputs: Vec<ShaderSymbol>) -> Result<Self, String> {
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

pub fn compile_shader(src: &CStr, shader_type: gl::types::GLuint) -> Result<gl::types::GLuint, String> {
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

#[derive(Facet)]
#[repr(transparent)]
pub struct SHMUnsafeCell<T: Sized> {
    value: T,
}

impl<T> SHMUnsafeCell<T> {
    #[inline(always)]
    pub const fn new(value: T) -> SHMUnsafeCell<T> {
        Self { value }
    }
}

impl<T: Sized> SHMUnsafeCell<T> {
    pub const fn get(&self) -> *mut T {
        self as *const SHMUnsafeCell<T> as *const T as *mut T
    }

    pub const unsafe fn as_mut_unchecked(&self) -> &mut T {
        unsafe { &mut *self.get() }
    }
}

#[derive(Facet)]
pub struct ShaderProgram {
    pub id: gl::types::GLuint,
    // TODO: we shouldn't need to store the vertex/fragment shader OpenGL handles after the program is linked
    pub vert: VertexShader,
    pub frag: FragmentShader,

    // TODO: This should really be a CString, but there's no derive for that for Facet
    // TODO: This should really be a RefCell rather unsafecell, but Facet
    uniform_location_cache: SHMUnsafeCell<HashMap<String, gl::types::GLint>>,
}

impl ShaderProgram {
    pub fn create(vert: VertexShader, frag: FragmentShader) -> Result<Self, String> {
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

            let uniform_location_cache = SHMUnsafeCell::new(HashMap::new());

            println!("Returning shader");
            Ok(Self {id, vert, frag, uniform_location_cache})
        }
    }

    pub fn uniforms(&self) -> impl Iterator<Item = &ShaderSymbol> {
        let zone = zone_start(c"ShaderProgram::uniforms");
        self.vert.uniform_inputs.iter().chain(self.frag.uniform_inputs.iter())
    }

    pub fn uniform_location(&self, uniform_name: &CStr) -> gl::types::GLint {
        unsafe {
            let mut cache = self.uniform_location_cache.as_mut_unchecked();
            let zone = zone_start(c"ShaderProgram::uniform_location");
            let s: &str = uniform_name.to_str().unwrap();
            if let Some(loc) = cache.get(s) {
                *loc
            } else {
                let zone = zone_start(c"uncached ShaderProgram::uniform_location");
                let loc = gl::GetUniformLocation(self.id, uniform_name.as_ptr());
                cache.insert(s.to_string(), loc);
                loc
            }
        }
    }
}

//struct AttrInfo {
//    offset,
//}

#[derive(Facet, Debug)]
pub struct VertexArray {
    pub id: GLuint,
    // TODO: should have an Rc to the vbo/ebo so they're not freed?
    //attrs: Vec::<AttrInfo>,
    //stride: u32,
}

impl VertexArray {
    pub fn create() -> Self {
        unsafe {
            let mut id: GLuint = 0;
            gl::GenVertexArrays(1, &mut id as *mut GLuint);
            Self { id }
        }
    }

    pub fn bind(&self, vbo: &VertexBufferObject, ebo: &ElementBufferObject) {
        unsafe {
            gl::BindBuffer(gl::ARRAY_BUFFER, vbo.id);
            gl::BindBuffer(gl::ELEMENT_ARRAY_BUFFER, ebo.id);
        }
    }
}

impl Drop for VertexArray {
    fn drop(&mut self) {
        unsafe {
            gl::DeleteVertexArrays(1, &mut self.id as *mut GLuint);
        }
    }
}

#[derive(Facet, Debug)]
pub struct VertexBufferObject {
    id: GLuint
}

impl VertexBufferObject {
    pub fn create() -> Self {
        unsafe {
            let mut id: GLuint = 0;
            gl::GenBuffers(1, &mut id as *mut GLuint);
            Self { id }
        }
    }

    pub fn create_with_data<T>(data: &[T], mode: GLuint) -> Self {
        let vbo = Self::create();
        vbo.bind_data(data, mode);
        vbo
    }

    pub fn bind_data<T>(&self, data: &[T], mode: GLuint) {
        unsafe {
            gl::BindBuffer(gl::ARRAY_BUFFER, self.id);
            gl::BufferData(gl::ARRAY_BUFFER, (data.len() * size_of::<T>()) as isize, data.as_ptr().cast(), mode);
        }
    }
}

impl Drop for VertexBufferObject {
    fn drop(&mut self) {
        unsafe {
            gl::DeleteBuffers(1, &mut self.id as *mut GLuint);
        }
    }
}


#[derive(Facet, Copy, Clone, Debug)]
#[repr(u8)]
pub enum Primitive {
    Points,
    Lines,
    LineLoop,
    LineStrip,
    Triangles,
    TriangleStrip,
    TriangleFan,
}

impl Primitive {
    pub fn to_gl(&self) -> GLenum {
        match self {
            Primitive::Points => gl::POINTS,
            Primitive::Lines => gl::LINES,
            Primitive::LineLoop => gl::LINE_LOOP,
            Primitive::LineStrip => gl::LINE_STRIP,
            Primitive::Triangles => gl::TRIANGLES,
            Primitive::TriangleStrip => gl::TRIANGLE_STRIP,
            Primitive::TriangleFan => gl::TRIANGLE_FAN,
       }
    }
}

#[derive(Facet, Debug)]
pub struct ElementBufferObject {
    id: GLuint,
    primitive_type: Primitive
}

impl ElementBufferObject {
    pub fn create() -> Self {
        unsafe {
            let mut id: GLuint = 0;
            gl::GenBuffers(1, &mut id as *mut GLuint);
            let primitive_type = Primitive::Triangles;
            Self { id, primitive_type }
        }
    }

    pub fn create_with_data<T>(data: &[T], primitive_type: Primitive, mode: GLuint) -> Self {
        let mut ebo = Self::create();
        ebo.bind_data(data, primitive_type, mode);
        ebo
    }

    pub fn bind_data<T>(&mut self, data: &[T], primitive_type: Primitive, mode: GLuint) {
        unsafe {
            self.primitive_type = primitive_type;
            gl::BindBuffer(gl::ELEMENT_ARRAY_BUFFER, self.id);
            gl::BufferData(gl::ELEMENT_ARRAY_BUFFER, (data.len() * size_of::<T>()) as isize, data.as_ptr().cast(), mode);
        }
    }
}

impl Drop for ElementBufferObject {
    fn drop(&mut self) {
        unsafe {
            gl::DeleteBuffers(1, &mut self.id as *mut GLuint);
        }
    }
}

pub enum VertVec {
    Float(Vec<f32>),
    Vec2(Vec<Vec2>),
    Vec3(Vec<Vec3>),
    Mat4(Vec<Mat4>),
}

impl VertVec {
   pub fn element_size(&self) -> u32 {
        match self {
            VertVec::Float(_) => u32size_of::<f32>(),
            VertVec::Vec2(_) => u32size_of::<Vec2>(),
            VertVec::Vec3(_) => u32size_of::<Vec3>(),
            VertVec::Mat4(_) => u32size_of::<Mat4>(),
        }
   }

   // pub fn component_count(&self) -> u32 {
   //      match self {
   //          VertVec::Float(_) => 1,
   //          VertVec::Vec2(_) => 2,
   //          VertVec::Vec3(_) => 3,
   //          VertVec::Mat4(_) => 16,
   //      }
   // }

   pub fn to_shader_type(&self) -> ShaderDataType {
       match self {
           VertVec::Float(_) => ShaderDataType::Float,
           VertVec::Vec2(_) => ShaderDataType::Vec2,
           VertVec::Vec3(_) => ShaderDataType::Vec3,
           VertVec::Mat4(_) => ShaderDataType::Mat4,
       }
   }

   pub fn len(&self) -> usize {
       match self {
           VertVec::Float(v) => v.len(),
           VertVec::Vec2(v) => v.len(),
           VertVec::Vec3(v) => v.len(),
           VertVec::Mat4(v) => v.len(),
       }
   }
}

#[derive(Facet, Copy, Clone, Debug)]
pub struct VertexAttribute {
    pub data_type: ShaderDataType,
    pub offset: u32,
}

impl VertexAttribute {
    pub fn new(data_type: ShaderDataType, offset: u32) -> Self {
        Self {
            offset,
            data_type,
        }
    }
}

#[derive(Facet)]
pub struct Mesh {
    // NOTE: Does not include VertexArray since vertex arrays are specific to the shader program being used
    pub vbo: VertexBufferObject,
    pub ebo: ElementBufferObject,
    pub stride: u32,
    pub attribs: HashMap<String, VertexAttribute>,
    pub index_count: u32,
}

impl Mesh {
    pub fn attribute(&self, name: &str) -> Option<VertexAttribute> {
        self.attribs.get(name).copied()
    }

    pub fn create(data: &MeshDataRaw) -> Result<Self, String> {
        log_opengl_errors!();

        let vert_count = data.verts.values().map(|v| v.len()).max().unwrap_or(0);
        // if vert_count == 0 {
        //     return Err("No verts in mesh data!".to_string());
        // }

        log_opengl_errors!();
        let mut vert_attribs: Vec<(&str, &VertVec)> = vec![];
        for (name, vert_vec) in data.verts.iter() {
            vert_attribs.push((name, vert_vec));
        }

        log_opengl_errors!();
        let mut vert_data: Vec<f32> = vec![];
        for idx in 0..vert_count {
            for (_, vert_attrib) in vert_attribs.iter() {
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

        log_opengl_errors!();

        let vbo = VertexBufferObject::create_with_data(&vert_data, gl::STATIC_DRAW);
        let ebo = ElementBufferObject::create_with_data(&data.indices, data.primitive_type, gl::STATIC_DRAW);

        let stride: u32 = vert_attribs.iter().map(|(_, a)| a.element_size()).sum();

        let mut attribs = HashMap::new();
        let mut offset = 0;
        for (attrib_name, attrib_vec) in vert_attribs.iter() {
            attribs.insert(
                attrib_name.to_string(),
                VertexAttribute::new(attrib_vec.to_shader_type(), offset),
            );
            offset += attrib_vec.element_size();
        }

        let index_count = data.indices.len() as u32;

        log_opengl_errors!();
        Ok(Self {
            vbo,
            ebo,
            stride,
            attribs,
            index_count,
        })
    }
}

pub struct MeshDataRaw {
    pub verts: HashMap<String, VertVec>,
    pub indices: Vec<u32>,
    pub primitive_type: Primitive,
}

#[derive(Facet)]
pub struct StaticMesh {
    pub shader: Rc<ShaderProgram>,
    pub vao: VertexArray,
    pub transform: Mat4,
    pub mesh: Rc<Mesh>,
    pub uniform_override: HashMap<String, ShaderValue>,
    // TODO: should model/view/projection uniform locations be cached (here or in the ShaderProgram)?
}

impl StaticMesh {
    pub fn create(shader: Rc<ShaderProgram>, mesh: Rc<Mesh>) -> Result<Self, String> {
        log_opengl_errors!();

        let vao = VertexArray::create();
        unsafe { gl::BindVertexArray(vao.id) };

        log_opengl_errors!();
        unsafe {
            let stride = mesh.stride as i32;
            for (idx, attribute) in shader.vert.inputs.iter().enumerate() {
                let idx = idx as GLuint;
                let offset = match mesh.attribute(&attribute.name) {
                    None => return Err(format!("Attribute {} doesn't exist for mesh", attribute.name)),
                    Some(mesh_attribute) => mesh_attribute.offset, // TODO: check for matching types
                };
                gl::VertexAttribPointer(idx, attribute.data_type.component_count() as GLint, gl::FLOAT, gl::FALSE, stride, offset as *const std::ffi::c_void);
                gl::EnableVertexAttribArray(idx);
                log_opengl_errors!();
            }
        }

        vao.bind(&mesh.vbo, &mesh.ebo);

        let transform = Mat4::IDENTITY;
        let uniform_override = HashMap::new();

        log_opengl_errors!();
        Ok(Self {
            shader,
            vao,
            transform,
            mesh,
            uniform_override,
        })
    }

    // TODO: "drawing" a StaticMesh shouldn't immediately draw it. Instead it should add it to the list
    // of meshes that need to be drawn. We need this if we want to do multiple passes, but we also want
    // it to do it to reduce the number of OpenGL calls from switching meshes/shaders.
    pub fn draw(&self, ctx: &mut HashMap<String, ShaderValue>) {
        unsafe {
            {
                let zone = zone_start(c"gl::BindVertexArray");
                gl::BindVertexArray(self.vao.id);
            }
            {
                let zone = zone_start(c"gl::UseProgram");
                gl::UseProgram(self.shader.id);
            }

            // TODO: This should be a chained hash map sort of thing, rather than requiring a mutable hashmap
            ctx.insert("model".to_string(), ShaderValue::Mat4(self.transform));

            let mut texture_unit = 0;
            {
                let zone = zone_start(c"iterate self.shader.uniforms");
                for uniform_info in self.shader.uniforms() {
                    {
                        let loc = {
                            let zone = zone_start(c"self.shader.uniform_location");
                            self.shader.uniform_location(&CString::new(uniform_info.name.clone()).unwrap())
                        };
                        match (uniform_info.data_type, self.uniform_override.get(&uniform_info.name).or_else(|| ctx.get(&uniform_info.name)).unwrap()) {
                            (ShaderDataType::Float, ShaderValue::Float(v)) => {
                                let zone = zone_start(c"gl::Uniform1f");
                                gl::Uniform1f(
                                    loc,
                                    *v,
                                );
                            }
                            (ShaderDataType::Vec2, ShaderValue::Vec2(v)) => {
                                let zone = zone_start(c"gl::Uniform2f");
                                gl::Uniform2f(
                                    loc,
                                    v.x,
                                    v.y,
                                );
                            }
                            (ShaderDataType::Vec3, ShaderValue::Vec3(v)) => {
                                let zone = zone_start(c"gl::Uniform3f");
                                gl::Uniform3f(
                                    loc,
                                    v.x,
                                    v.y,
                                    v.z,
                                );
                            }
                            (ShaderDataType::Vec4, ShaderValue::Vec4(v)) => {
                                let zone = zone_start(c"gl::Uniform4f");
                                gl::Uniform4f(
                                    loc,
                                    v.x,
                                    v.y,
                                    v.z,
                                    v.w,
                                );
                            }
                            (ShaderDataType::Mat4, ShaderValue::Mat4(v)) => {
                                let zone = zone_start(c"gl::UniformMatrix4fv");
                                gl::UniformMatrix4fv(
                                    loc,
                                    1,
                                    gl::FALSE,
                                    &v.to_cols_array() as *const gl::types::GLfloat
                                );
                            }
                            (ShaderDataType::Sampler2D, ShaderValue::Sampler2D(v)) => {
                                let zone = zone_start(c"gl::Sampler2D");
                                gl::ActiveTexture(
                                    match texture_unit {
                                        0 => gl::TEXTURE0,
                                        1 => gl::TEXTURE1,
                                        2 => gl::TEXTURE2,
                                        3 => gl::TEXTURE3,
                                        4 => gl::TEXTURE4,
                                        5 => gl::TEXTURE5,
                                        _ => todo!("Add the other texture unit constants"),
                                    }
                                );
                                gl::BindTexture(gl::TEXTURE_2D, *v);
                                gl::Uniform1i(
                                    loc,
                                    texture_unit,
                                );
                                texture_unit += 1;
                            }
                            _ => todo!(),
                        }
                    }
                }
            }

            {
                let zone = zone_start(c"gl::DrawElements");
                gl::DrawElements(self.mesh.ebo.primitive_type.to_gl(), self.mesh.index_count as i32, gl::UNSIGNED_INT, std::ptr::null());
            }
        }
    }
}
