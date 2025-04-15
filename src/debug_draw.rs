use glam::Vec3;
use crate::u32size_of;
use crate::State;
use crate::Primitive;
use crate::END_PRIMITIVE;

pub fn draw_debug_shapes(state: &mut State) {
    unsafe {
        gl::BindVertexArray(state.debug_vao.id);
        state.debug_vbo.bind_data(&state.debug_verts, gl::DYNAMIC_DRAW);
        state.debug_ebo.bind_data(&state.debug_vert_indices, Primitive::LineStrip, gl::DYNAMIC_DRAW);

        // Vertex Position Attribute
        gl::VertexAttribPointer(0, 3, gl::FLOAT, gl::FALSE, (6 * u32size_of::<f32>()) as i32, std::ptr::null());
        gl::EnableVertexAttribArray(0);

        // Vertex Color Attribute
        gl::VertexAttribPointer(1, 3, gl::FLOAT, gl::FALSE, (6 * u32size_of::<f32>()) as i32, (3 * u32size_of::<f32>()) as *const std::ffi::c_void);
        gl::EnableVertexAttribArray(1);

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
        state.debug_verts.push((*p, Vec3::new(0.0, 1.0, 0.0)));
        offset += 1;
    }
    state.debug_vert_indices.push(start_offset);
    state.debug_vert_indices.push(END_PRIMITIVE);
}

pub fn debug_line(state: &mut State, points: &[Vec3]) {
    debug_line_color(state, points, Vec3::new(1.0, 1.0, 1.0));
}

pub fn debug_line_color(state: &mut State, points: &[Vec3], color: Vec3) {
    let mut offset: u32 = state.debug_verts.len().try_into().unwrap();
    for p in points {
        state.debug_vert_indices.push(offset);
        state.debug_verts.push((*p, color));
        offset += 1;
    }
    state.debug_vert_indices.push(END_PRIMITIVE);
}

pub fn debug_box(state: &mut State, position: Vec3, size: Vec3, _color: Vec3) {
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
        state.debug_verts.push((vert, Vec3::new(0.0, 0.0, 1.0)))
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
