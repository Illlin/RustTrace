use crate::vec3::Vec3;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Material {
    pub colour: Vec3,
    pub specular: f64,
    pub mirror: bool,
    pub mirror_mix: f64,
}

impl Material {
    pub fn new(colour: Vec3, specular: f64, mirror: bool, mirror_mix: f64) -> Self {
        Self { colour, specular, mirror, mirror_mix }
    }
}


pub trait Sdf {
    fn distance_to(&self, p: Vec3) -> f64;
    fn get_material(&self, p: Vec3) -> Material;
}

pub struct Sphere {
    pub radius: f64,
    pub pos: Vec3,
    pub material: Material,
}

impl Sphere {
    pub fn new(radius: f64, pos: Vec3, material: Material) -> Self {
            Self { radius, pos, material }
        }
}

impl Sdf for Sphere {
    fn distance_to(&self, p: Vec3) -> f64 {
        let a = p - self.pos;
        a.magnitude() - self.radius
    }
    fn get_material(&self, _p: Vec3) -> Material {
        self.material
    }
}

pub struct Cuboid {
    pub sides: Vec3,
    pub pos: Vec3,
    pub material: Material,
}

impl Cuboid {
    pub fn new(sides: Vec3, pos: Vec3, material: Material) -> Self {
        Self { sides, pos, material }
    }
}

impl Sdf for Cuboid {
    fn distance_to(&self, p: Vec3) -> f64 {
        let p = p - self.pos;
        let q = p.abs() - self.sides;
        q.max(0.0).magnitude() + q.x.max(q.y.max(q.z)).min(0.0)
    }
    fn get_material(&self, _p: Vec3) -> Material {
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
    fn distance_to(&self, p: Vec3) -> f64 {
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

pub struct UnionN {
    pub subs: Vec<Box<dyn Sdf>>
}

impl UnionN {
    pub fn new<I: IntoIterator<Item = Box<dyn Sdf>>>(iter: I) -> Self {
        UnionN { subs: iter.into_iter().collect() }
    }
}

impl Sdf for UnionN {
    fn distance_to(&self, p: Vec3) -> f64 {
        // just take the min over all sub-shapes
        self.subs
            .iter()
            .map(|s| s.distance_to(p))
            .fold(f64::INFINITY, f64::min)
    }

    fn get_material(&self, p: Vec3) -> Material {
        // find the sub-shape with the smallest distance and return its material
        let (best_shape, _) = self.subs
            .iter()
            .map(|s| (s, s.distance_to(p)))
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap();
        best_shape.get_material(p)
    }
}

pub struct SmoothUnion<A: Sdf, B: Sdf> {
    pub(crate) a: A,
    pub(crate) b: B,
    pub(crate) smooth: f64,
}

impl<A: Sdf, B: Sdf> SmoothUnion<A, B> {
    pub fn new(a: A, b: B, smooth: f64) -> Self {
        Self { a, b, smooth }
    }
}

impl<A: Sdf, B: Sdf> Sdf for SmoothUnion<A, B> {
    fn distance_to(&self, p: Vec3) -> f64 {
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

pub struct Checker<A: Sdf> {
    pub a: A,
    pub scale: f64,
    pub material: Material,
}

impl<A: Sdf> Checker<A> {
    pub fn new(a: A, scale: f64, material: Material) -> Self {
        Self { a, scale, material}
    }
}

impl<A: Sdf> Sdf for Checker<A> {
    fn distance_to(&self, p: Vec3) -> f64 {
        self.a.distance_to(p)
    }
    fn get_material(&self, p: Vec3) -> Material {
        let xi = (p.x * self.scale).floor() as i32;
        let yi = (p.y * self.scale).floor() as i32;
        let zi = (p.z * self.scale).floor() as i32;
        if ((xi + yi + zi) & 1) == 0 {
            self.a.get_material(p)
        } else {
            self.material
        }
    }
}

pub struct Repeat<A: Sdf> {
    pub a: A,
    pub scale: f64,
}

impl<A: Sdf> Repeat<A> {
    pub fn new(a: A, scale: f64) -> Self {
        Self { a, scale}
    }
}

impl<A: Sdf> Sdf for Repeat<A> {
    fn distance_to(&self, p: Vec3) -> f64 {
        self.a.distance_to(
            Vec3 { x: p.x % self.scale, y: p.y % self.scale, z: p.z % self.scale }
        )
    }
    fn get_material(&self, p: Vec3) -> Material {
        self.a.get_material(p)
    }
}

pub struct SpeedField<A: Sdf> {
    pub a: A,
    pub cache: Vec<f64>,
    pub size: usize,
    pub scale: f64,
    pub centre: Vec3,
}

impl<A: Sdf> SpeedField<A> {
    pub fn new(a: A) -> Self {
        let size = 100;
        let scale = 1.0;

        let rad = scale * 2.0f64.sqrt();

        let mut cache = vec![0.0f64; size * size * size];

        // Compute known safe distance field
        let centre= Vec3 {
            x: (size/2) as f64 * scale,
            y: (size/2) as f64 * scale,
            z: (size/2) as f64 * scale
        };
        for z in 0..size {
            let z_pos = scale * z as f64;
            println!("{z}");
            for y in 0..size {
                let y_pos = scale * y as f64;
                for x in 0..size {
                    let point = Vec3 {
                        x: scale * x as f64,
                        y: y_pos,
                        z: z_pos,
                    } - centre;
                    let dist = a.distance_to(point);
                    if dist > rad {
                        cache[x + y * size + z * size * size] = dist - rad;
                    }
                }
            }
        }
        let arr_size = size*size*size;
        println!("{arr_size}");

        Self { a, cache, size, scale, centre }
    }
}

impl<A: Sdf> Sdf for SpeedField<A> {
    fn distance_to(&self, p: Vec3) -> f64 {
        let local = p + self.centre;
        let size = self.size as isize;

        let xi = (local.x / self.scale).floor() as isize;
        let yi = (local.y / self.scale).floor() as isize;
        let zi = (local.z / self.scale).floor() as isize;

        if xi < 0 || yi < 0 || zi < 0
            || xi >= size
            || yi >= size
            || zi >= size
        {
            return 10000000.0;
        }

        let s = size;
        let idx = xi + yi * s + zi * (s * s);

        let dist = self.cache[idx as usize];
        if dist > 0.5 {
            dist
        } else {
            self.a.distance_to(p)
        }
    }
    fn get_material(&self, p: Vec3) -> Material {
        self.a.get_material(p)
    }
}

pub struct Mandelbulb {
    pub power: u32,
    pub iterations: u32,
    pub scale: f64,
    pub pos: Vec3,
    pub material: Material,
}

impl Mandelbulb {
    pub fn new(power: u32, iterations: u32, scale: f64, pos: Vec3, material: Material) -> Self {
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
    fn get_material(&self, _p: Vec3) -> Material {
        self.material
    }
    fn distance_to(&self, p: Vec3) -> f64 {
        // move into fractal’s local space and scale
        let mut z = (p - self.pos) * self.scale;
        let mut dr = 1.0_f64;
        let mut r = 0.0_f64;
        let powf = self.power as f64;

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