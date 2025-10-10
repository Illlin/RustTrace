use std::ops;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec3 {
    pub(crate) x: f64,
    pub(crate) y: f64,
    pub(crate) z: f64,
}

impl Vec3 {
    pub(crate) fn new(x: f64, y: f64, z: f64) -> Vec3 {
        Vec3 { x, y, z }
    }

    pub(crate) fn magnitude(self) -> f64 {
        (self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }

    pub(crate) fn normalize(self) -> Vec3 {
        self / self.magnitude()
    }

    pub(crate) fn cross(self, other: Vec3) -> Vec3 {
        Vec3 {
            x: self.y * other.z - self.z * other.y,
            y: self.z * other.x - self.x * other.z,
            z: self.x * other.y - self.y * other.x,
        }
    }

    pub(crate) fn dot(self, other: Vec3) -> f64 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    pub fn abs(self) -> Vec3 {
        Vec3 {x: self.x.abs(), y: self.y.abs(), z: self.z.abs()}
    }

    pub fn max(self, x: f64) -> Vec3 {
        Vec3 {x: self.x.max(x), y: self.y.max(x), z: self.z.max(x)}
    }

    pub fn min(self, x: f64) -> Vec3 {
        Vec3 {x: self.x.min(x), y: self.y.min(x), z: self.z.min(x)}
    }

    pub fn reflect(self, normal: Vec3) -> Vec3 {
        let dot = self.dot(normal);
        self - normal * (2.0 * dot)
    }
}

impl ops::Sub<Vec3> for Vec3 {
    type Output = Vec3;
    fn sub(self, v: Vec3) -> Vec3 {
        Vec3 {
            x: self.x - v.x,
            y: self.y - v.y,
            z: self.z - v.z
        }
    }
}

impl ops::Neg for Vec3 {
    type Output = Vec3;
    fn neg(self) -> Vec3 {
        Vec3 {
            x: -self.x,
            y: -self.y,
            z: -self.z
        }
    }
}

impl ops::Add<Vec3> for Vec3 {
    type Output = Vec3;
    fn add(self, v: Vec3) -> Vec3 {
        Vec3 {
            x: self.x + v.x,
            y: self.y + v.y,
            z: self.z + v.z
        }
    }
}

impl ops::Div<f64> for Vec3 {
    type Output = Vec3;
    fn div(self, d: f64) -> Vec3 {
        Vec3 {
            x: self.x / d,
            y: self.y / d,
            z: self.z / d
        }
    }
}

impl ops::Mul<f64> for Vec3 {
    type Output = Vec3;
    fn mul(self, d: f64) -> Vec3 {
        Vec3 {
            x: self.x * d,
            y: self.y * d,
            z: self.z * d
        }
    }
}

impl ops::Mul<Vec3> for Vec3 {
    type Output = Vec3;
    fn mul(self, d: Vec3) -> Vec3 {
        Vec3 {
            x: self.x * d.x,
            y: self.y * d.y,
            z: self.z * d.z
        }
    }
}


