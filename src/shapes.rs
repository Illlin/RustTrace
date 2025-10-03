use crate::vec3::Vec3;

pub trait Sdf {
    fn distance_to(&self, p: Vec3) -> f32;
}

pub struct Sphere {
    pub radius: f32,
    pub pos: Vec3,
}

impl Sphere {
    pub fn new(radius: f32, pos: Vec3) -> Self {
            Self { radius, pos }
        }
}

impl Sdf for Sphere {
    fn distance_to(&self, p: Vec3) -> f32 {
        let a = p - self.pos;
        a.magnitude() - self.radius
    }
}

pub struct Box {
    pub sides: Vec3,
    pub pos: Vec3,
}

impl Box {
    pub fn new(sides: Vec3, pos: Vec3) -> Self {
        Self { sides, pos }
    }
}

impl Sdf for Box {
    fn distance_to(&self, p: Vec3) -> f32 {
        let p = p - self.pos;
        let q = p.abs() - self.sides;
        q.max(0.0).magnitude() + q.x.max(q.y.max(q.z)).min(0.0)
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
}