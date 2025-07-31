use glam::{Vec2, Vec3, Vec4, Mat4};
use std::collections::HashMap;
use std::cell::RefCell;

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
}
