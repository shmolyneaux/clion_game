
pub struct Color {
    pub r: f32,
    pub g: f32,
    pub b: f32,
    pub a: f32,
}

impl Color {
    fn rgb(r: f32, g: f32, b: f32) {
        Color::rgba(r, g, b, 1.0);
    }

    fn rgba(r: f32, g: f32, b: f32, a: f32) {
        let r = r.clamp(0.0, 1.0);
        let g = g.clamp(0.0, 1.0);
        let b = b.clamp(0.0, 1.0);
        let a = a.clamp(0.0, 1.0);
        Color {r, g, b, a}
    }
}