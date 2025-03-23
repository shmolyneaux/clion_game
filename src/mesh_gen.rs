use std::collections::HashMap;
use glam::{Vec2, Vec3, Mat4};

use crate::Primitive;
use crate::MeshDataRaw;
use crate::VertVec;

pub fn box_mesh(size: Vec3) -> MeshDataRaw {
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

    let primitive_type = Primitive::Triangles;

    MeshDataRaw {
        verts,
        indices,
        primitive_type,
    }
}

pub fn quad_mesh() -> MeshDataRaw {
    let positions = vec![
        Vec3::new(-1.0, -1.0, 0.0)*0.9,
        Vec3::new(-1.0,  1.0, 0.0)*0.9,
        Vec3::new( 1.0, -1.0, 0.0)*0.9,
        Vec3::new( 1.0,  1.0, 0.0)*0.9,
    ];
    let uvs = vec![
        Vec2::new( 0.0, 0.0),
        Vec2::new( 0.0, 1.0),
        Vec2::new( 1.0, 0.0),
        Vec2::new( 1.0, 1.0),
    ];

    let indices = vec![
        0+0, 0+1, 0+2, 0+1, 0+2, 0+3,
    ];

    let mut verts = HashMap::new();
    verts.insert("aPos".to_string(), VertVec::Vec3(positions));
    verts.insert("aUV".to_string(), VertVec::Vec2(uvs));

    let primitive_type = Primitive::Triangles;

    MeshDataRaw {
        verts,
        indices,
        primitive_type,
    }
}
