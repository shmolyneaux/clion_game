use crate::*;

pub static END_PRIMITIVE: u32 = 0xFFFF_FFFF;

#[derive(Copy, Clone, Debug)]
pub enum ShaderDataType {
    Float,
    Vec2,
    Vec3,
    Vec4,
    Mat4,
}

impl ShaderDataType {
   fn component_count(&self) -> u32 {
        match self {
            ShaderDataType::Float => 1,
            ShaderDataType::Vec2 => 2,
            ShaderDataType::Vec3 => 3,
            ShaderDataType::Vec4 => 4,
            ShaderDataType::Mat4 => 16,
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
            }
        )
    }
}

pub enum ShaderValue {
    Float(f32),
    Vec2(Vec2),
    Vec3(Vec3),
    Vec4(Vec4),
    Mat4(Mat4),
}

#[derive(Clone)]
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

pub struct ShaderProgram {
    pub id: gl::types::GLuint,
    // TODO: we shouldn't need to store the vertex/fragment shader OpenGL handles after the program is linked
    pub vert: VertexShader,
    pub frag: FragmentShader,
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

            println!("Returning shader");
            Ok(Self {id, vert, frag})
        }
    }

    pub fn uniforms(&self) -> impl Iterator<Item = &ShaderSymbol> {
        self.vert.uniform_inputs.iter().chain(self.frag.uniform_inputs.iter())
    }

    pub fn uniform_location(&self, uniform_name: &CStr) -> gl::types::GLint {
        unsafe {
            gl::GetUniformLocation(self.id, uniform_name.as_ptr())
        }
    }
}

//struct AttrInfo {
//    offset,
//}

#[derive(Debug)]
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

#[derive(Debug)]
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

#[derive(Copy, Clone, Debug)]
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

#[derive(Debug)]
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

   pub fn component_count(&self) -> u32 {
        match self {
            VertVec::Float(_) => 1,
            VertVec::Vec2(_) => 2,
            VertVec::Vec3(_) => 3,
            VertVec::Mat4(_) => 16,
        }
   }

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

#[derive(Copy, Clone, Debug)]
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
        log_opengl_errors();

        println!("Getting vert count");
        let vert_count = data.verts.values().map(|v| v.len()).max().unwrap_or(0);
        if vert_count == 0 {
            return Err("No verts in mesh data!".to_string());
        }

        log_opengl_errors();
        println!("Getting vert attribs");
        let mut vert_attribs: Vec<(&str, &VertVec)> = vec![];
        for (name, vert_vec) in data.verts.iter() {
            vert_attribs.push((name, vert_vec));
        }

        log_opengl_errors();
        println!("Pushing vert data");
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

        log_opengl_errors();
        println!("Creating buffers");

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

        log_opengl_errors();
        println!("Returning Mesh");
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

pub struct StaticMesh {
    pub shader: Rc<ShaderProgram>,
    pub vao: VertexArray,
    pub transform: Mat4,
    pub mesh: Rc<Mesh>,
    // TODO: should model/view/projection uniform locations be cached (here or in the ShaderProgram)?
}

impl StaticMesh {
    pub fn create(shader: Rc<ShaderProgram>, mesh: Rc<Mesh>) -> Result<Self, String> {
        log_opengl_errors();
        println!("Creating vao");

        let vao = VertexArray::create();
        unsafe { gl::BindVertexArray(vao.id) };

        log_opengl_errors();
        println!("Setting up VAO pointers");
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
                log_opengl_errors();
            }
        }

        vao.bind(&mesh.vbo, &mesh.ebo);

        let transform = Mat4::IDENTITY;

        log_opengl_errors();
        println!("Returning StaticMesh");
        Ok(Self {
            shader,
            vao,
            transform,
            mesh,
        })
    }

    // TODO: "drawing" a StaticMesh shouldn't immediately draw it. Instead it should add it to the list
    // of meshes that need to be drawn. We need this if we want to do multiple passes, but we also want
    // it to do it to reduce the number of OpenGL calls from switching meshes/shaders.
    pub fn draw(&self, ctx: &mut HashMap<String, ShaderValue>) {
        unsafe {
            gl::BindVertexArray(self.vao.id);
            gl::UseProgram(self.shader.id);

            ctx.insert("model".to_string(), ShaderValue::Mat4(self.transform));

            for uniform_info in self.shader.uniforms() {
                match (uniform_info.data_type, ctx.get(&uniform_info.name).unwrap()) {
                    (ShaderDataType::Float, ShaderValue::Float(_v)) => todo!(),
                    (ShaderDataType::Vec2, ShaderValue::Vec2(_v)) => todo!(),
                    (ShaderDataType::Vec3, ShaderValue::Vec3(_v)) => todo!(),
                    (ShaderDataType::Vec4, ShaderValue::Vec4(_v)) => todo!(),
                    (ShaderDataType::Mat4, ShaderValue::Mat4(v)) => {
                        gl::UniformMatrix4fv(
                            self.shader.uniform_location(&CString::new(uniform_info.name.clone()).unwrap()),
                            1,
                            gl::FALSE,
                            &v.to_cols_array() as *const gl::types::GLfloat
                        );
                    }
                    _ => todo!(),
                }
            }

            gl::DrawElements(self.mesh.ebo.primitive_type.to_gl(), self.mesh.index_count as i32, gl::UNSIGNED_INT, std::ptr::null());
        }
    }
}
