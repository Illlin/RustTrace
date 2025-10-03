use crate::vec3::Vec3;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Material {
    pub colour: Vec3,
    pub specular: f32,
}


pub trait Sdf {
    fn distance_to(&self, p: Vec3) -> f32;
    fn get_material(&self, p: Vec3) -> Material;
}

pub struct Sphere {
    pub radius: f32,
    pub pos: Vec3,
    pub material: Material,
}

impl Sphere {
    pub fn new(radius: f32, pos: Vec3, material: Material) -> Self {
            Self { radius, pos, material }
        }
}

impl Sdf for Sphere {
    fn distance_to(&self, p: Vec3) -> f32 {
        let a = p - self.pos;
        a.magnitude() - self.radius
    }
    fn get_material(&self, p: Vec3) -> Material {
        self.material
    }
}

pub struct Box {
    pub sides: Vec3,
    pub pos: Vec3,
    pub material: Material,
}

impl Box {
    pub fn new(sides: Vec3, pos: Vec3, material: Material) -> Self {
        Self { sides, pos, material }
    }
}

impl Sdf for Box {
    fn distance_to(&self, p: Vec3) -> f32 {
        let p = p - self.pos;
        let q = p.abs() - self.sides;
        q.max(0.0).magnitude() + q.x.max(q.y.max(q.z)).min(0.0)
    }
    fn get_material(&self, p: Vec3) -> Material {
        self.material
    }
}

pub struct Union<A: Sdf, B: Sdf> {
    pub(crate) a: A,
    pub(crate) b: B,
}

impl<A: Sdf, B: Sdf> Union<A, B> {
    pub fn new(a: A, b: B) -> Self {
        Self { a, b }
    }
}

impl<A: Sdf, B: Sdf> Sdf for Union<A, B> {
    fn distance_to(&self, p: Vec3) -> f32 {
        let da = self.a.distance_to(p);
        let db = self.b.distance_to(p);
        da.min(db)
    }
    fn get_material(&self, p: Vec3) -> Material {
        let da = self.a.distance_to(p);
        let db = self.b.distance_to(p);
        if da < db {
            return self.a.get_material(p)
        }
        self.b.get_material(p)
    }
}

pub struct SmoothUnion<A: Sdf, B: Sdf> {
    pub(crate) a: A,
    pub(crate) b: B,
    pub(crate) smooth: f32,
}

impl<A: Sdf, B: Sdf> SmoothUnion<A, B> {
    pub fn new(a: A, b: B, smooth: f32) -> Self {
        Self { a, b, smooth }
    }
}

impl<A: Sdf, B: Sdf> Sdf for SmoothUnion<A, B> {
    fn distance_to(&self, p: Vec3) -> f32 {
        let da = self.a.distance_to(p);
        let db = self.b.distance_to(p);
        let k = self.smooth * 4.0;
        let h = (k-(da-db).abs()).max(0.0);
        da.min(db) - h*h*0.25/k
    }
    fn get_material(&self, p: Vec3) -> Material {
        // Todo blend mats
        let da = self.a.distance_to(p);
        let db = self.b.distance_to(p);
        if da < db {
            return self.a.get_material(p)
        }
        self.b.get_material(p)
    }
}


pub struct Mandelbulb {
    pub power: u32,
    pub iterations: u32,
    pub scale: f32,
    pub pos: Vec3,
    pub material: Material,
}

impl Mandelbulb {
    pub fn new(power: u32, iterations: u32, scale: f32, pos: Vec3, material: Material) -> Self {
        Self {
            power,
            iterations,
            scale,
            pos,
            material
        }
    }
}

impl Sdf for Mandelbulb {
    fn get_material(&self, p: Vec3) -> Material {
        self.material
    }
    fn distance_to(&self, p: Vec3) -> f32 {
        // move into fractal’s local space and scale
        let mut z = (p - self.pos) * self.scale;
        let mut dr = 1.0_f32;
        let mut r = 0.0_f32;
        let powf = self.power as f32;

        for _ in 0..self.iterations {
            r = z.magnitude();
            if r > 2.0 {
                break;
            }
            // polar coords
            let theta = (z.z / r).clamp(-1.0, 1.0).acos();
            let phi = z.y.atan2(z.x);

            // derivative of radius
            dr = r.powf(powf - 1.0) * powf * dr + 1.0;

            // scale & rotate
            let zr = r.powf(powf);
            let new_theta = theta * powf;
            let new_phi = phi * powf;

            let sin_t = new_theta.sin();
            let cos_t = new_theta.cos();
            let sin_p = new_phi.sin();
            let cos_p = new_phi.cos();

            z = Vec3::new(
                zr * sin_t * cos_p,
                zr * sin_t * sin_p,
                zr * cos_t,
            ) + (p - self.pos) * self.scale;
        }

        // distance estimate
        0.5 * (r.ln()) * r / dr / self.scale
    }
}