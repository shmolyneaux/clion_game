use glam::{Vec2, Vec3, Vec4, Mat4};
use std::collections::{HashMap, HashSet};
use std::cell::RefCell;

use glam::Vec3Swizzles;
use crate::*;

#[derive(Clone)]
pub enum Sdf {
    SdfTranslate(Vec3, Box<Sdf>),
    SdfUnion(Box<Sdf>, Box<Sdf>),
    SdfSmoothUnion(f32, Box<Sdf>, Box<Sdf>),
    SdfBox(Vec3),
    SdfSphere(f32),
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + t * (b - a)
}

impl Sdf {
    fn dist(&self, p: Vec3) -> f32 {
        match self {
            Self::SdfSphere(r) => p.length() - r,
            Self::SdfBox(b) => {
                let q = p.abs() - b;
                q.max(Vec3::ZERO).length() + q.max_element().min(0.0)
            },
            Self::SdfUnion(a, b) => f32::min(a.dist(p), b.dist(p)),
            Self::SdfSmoothUnion(r, a, b) => {
                let d1 = a.dist(p);
                let d2 = b.dist(p);

                let h: f32 = (0.5 + 0.5*(d2-d1)/r).clamp(0.0, 1.0);
                lerp( d2, d1, h ) - r*h*(1.0-h)
            }
            Self::SdfTranslate(d, sdf) => sdf.dist(p - d),
        }
    }
}

pub struct SdfCache {
    sdf: Sdf,
    cache: RefCell<HashMap<(i32, i32, i32), f32>>,
    scale: f32,
    resolution: u32,
    bounds: (Vec3, Vec3),
}

impl SdfCache {
    pub fn new(sdf: Sdf, scale: f32, resolution: u32) -> Self {
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

    pub fn get(&self, coord: (i32, i32, i32)) -> f32 {
        let mut cache = self.cache.borrow_mut();
        match cache.get(&coord) {
            Some(dist) => *dist,
            None => {
                let posf = Vec3::new(coord.0 as f32, coord.1 as f32, coord.2 as f32) * self.scale;
                let dist = self.sdf.dist(posf);
                cache.insert(coord, dist);
                dist
            }
        }
    }

    pub fn get_nocache(&self, coord: Vec3) -> f32 {
        let posf = Vec3::new(coord.x as f32, coord.y as f32, coord.z as f32) * self.scale;
        self.sdf.dist(posf)
    }

    pub fn create_mesh(&self) -> MeshDataRaw {
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
                    let dist = self.get(coord);
                    let mut is_vert = false;

                    // Logically, the vert is somewhere in the cube represented by 8 sample points. We know there's a
                    // vert in the cube if there's a transition from negative to positive somewhere within the volume
                    // of the cube
                    for (offx, offy, offz) in [(0, 0, 1), (0, 1, 0), (0, 1, 1), (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1)] {
                        if dist.is_sign_positive() != self.get((coord.0+offx, coord.1+offy, coord.2+offz)).is_sign_positive() {
                            is_vert = true;
                        }
                    }

                    if is_vert {
                        let dx = self.get_nocache(
                            Vec3::new(
                                coord.0 as f32+0.1,
                                coord.1 as f32,
                                coord.2 as f32
                            )
                        ) - dist;

                        let dy = self.get_nocache(
                            Vec3::new(
                                coord.0 as f32,
                                coord.1 as f32+0.1,
                                coord.2 as f32
                            )
                        ) - dist;

                        let dz = self.get_nocache(
                            Vec3::new(
                                coord.0 as f32,
                                coord.1 as f32,
                                coord.2 as f32+0.1
                            )
                        ) - dist;

                        let idx = positions.len() as u32;
                        vert_lookup.insert(coord, idx);

                        let norm = Vec3::new(dx, dy, dz).normalize_or_zero();
                        let raw_posf = Vec3::new(coord.0 as f32, coord.1 as f32, coord.2 as f32);
                        let pushed_posf = raw_posf - 10.0 * dist * norm;

                        let posf = pushed_posf;

                        positions.push(posf * 0.1);
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
                let dist = self.get(cx);
                if dist.is_sign_positive() != self.get(cxz).is_sign_positive() ||
                   dist.is_sign_positive() != self.get(cxy).is_sign_positive() ||
                   dist.is_sign_positive() != self.get(cxyz).is_sign_positive()
                {
                    x_edges.insert(*c0);
                }
            }

            if vert_lookup.get(&cy).is_some() {
                let dist = self.get(cy);
                if dist.is_sign_positive() != self.get(cxy).is_sign_positive() ||
                   dist.is_sign_positive() != self.get(cyz).is_sign_positive() ||
                   dist.is_sign_positive() != self.get(cxyz).is_sign_positive()
                {
                    y_edges.insert(*c0);
                }
            }

            if vert_lookup.get(&cz).is_some() {
                let dist = self.get(cz);
                if dist.is_sign_positive() != self.get(cxz).is_sign_positive() ||
                   dist.is_sign_positive() != self.get(cyz).is_sign_positive() ||
                   dist.is_sign_positive() != self.get(cxyz).is_sign_positive()
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

        MeshDataRaw {
            verts,
            indices,
            primitive_type,
        }
    }
}

pub fn sdf_box(b: Vec3) -> Box<Sdf> {
    Box::new(Sdf::SdfBox(b))
}

pub fn sdf_sphere(r: f32) -> Box<Sdf> {
    Box::new(Sdf::SdfSphere(r))
}

pub fn sdf_union(a: Box<Sdf>, b: Box<Sdf>) -> Box<Sdf> {
    Box::new(Sdf::SdfUnion(a, b))
}

pub fn sdf_smooth(r: f32, a: Box<Sdf>, b: Box<Sdf>) -> Box<Sdf> {
    Box::new(Sdf::SdfSmoothUnion(r, a, b))
}

pub fn sdf_translate(d: Vec3, sdf: Box<Sdf>) -> Box<Sdf> {
    Box::new(Sdf::SdfTranslate(d, sdf))
}
