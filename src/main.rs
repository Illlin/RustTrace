use crate::shapes::{Checker, Mandelbulb, SpeedField};
use minifb::{Key, KeyRepeat, Window, WindowOptions};
use rayon::ThreadPoolBuilder;
use rayon::prelude::*;
use shapes::{Cuboid, Material, Sdf, SmoothUnion, Sphere, Union, UnionN};
use std::f64::consts::PI;
use std::time::{Duration, Instant};
use vec3::Vec3;

mod shapes;
mod vec3;

const WIDTH: usize = 2200;
const HEIGHT: usize = 1300;
const EPS: f64 = 1e-5;

pub struct RayResults {
    pub final_position: Vec3,
    pub distance: f64,
}

pub struct RayInfo {
    pub start_position: Vec3,
    pub cast_direction: Vec3,
    pub min_depth: f64,
    pub max_depth: f64,
}

pub struct Light {
    pub position: Vec3,
    pub color: Vec3,
    pub intensity: f64,
}

// Schlick’s approximation of Fresnel
fn fresnel_schlick(cos_theta: f64, f0: Vec3) -> Vec3 {
    f0 + (Vec3::new(1.0,1.0,1.0) - f0) * (1.0 - cos_theta).powf(5.0)
}

// GGX / Trowbridge‐Reitz normal distribution
fn d_ggx(n_dot_h: f64, alpha: f64) -> f64 {
    let a2 = alpha * alpha;
    let denom = n_dot_h * n_dot_h * (a2 - 1.0) + 1.0;
    a2 / (PI * denom * denom)
}

// Schlick‐GGX Geometry term, single direction
fn g1_schlick_ggx(n_dot_v: f64, k: f64) -> f64 {
    n_dot_v / (n_dot_v * (1.0 - k) + k)
}

// Smith’s joint masking‐shadowing with Schlick‐GGX
fn g_smith(n_dot_v: f64, n_dot_l: f64, alpha: f64) -> f64 {
    // k = (alpha+1)^2 / 8 is a common choice
    let k = (alpha + 1.0).powi(2) / 8.0;
    g1_schlick_ggx(n_dot_v, k) * g1_schlick_ggx(n_dot_l, k)
}

fn cast_ray(scene: &impl Sdf, ray: &RayInfo, start_total_distance: f64, inside: bool) -> RayResults {
    let sign = match inside {
        false => 1.0,
        true => -1.0,
    };
    let mut total_dist = start_total_distance;
    let mut dist = scene.distance_to(ray.start_position) * sign ;
    let mut pos = ray.start_position;
    while (dist > ray.min_depth) && (total_dist < ray.max_depth) {
        total_dist += dist;
        pos = pos + ray.cast_direction * dist;
        dist = scene.distance_to(pos);
    }
    RayResults {
        final_position: pos,
        distance: total_dist,
    }
}

fn shade_pbr(
    scene: &impl Sdf,
    hit_pos: Vec3,
    normal: Vec3,
    view_dir: Vec3,       // = -ray.direction
    material: &Material,
    light: Light,
    safe_pos: Vec3,
) -> Vec3
{
    // 1) build L, H, the various cosines
    let to_light = (light.position - hit_pos);       // if directional light, this is infinite.
    let light_dist2 = to_light.magnitude2();
    let L = to_light.normalize();
    let H = (view_dir + L).normalize();

    let n_dot_l = normal.dot(L).max(0.0);
    let n_dot_v = normal.dot(view_dir).max(0.0);
    let n_dot_h = normal.dot(H).max(0.0);
    let v_dot_h = view_dir.dot(H).max(0.0);
    if n_dot_l <= 0.0 || n_dot_v <= 0.0 {
        return Vec3::new(0.0, 0.0, 0.0);
    }

    // 2) shadow‐ray
    let shadow_ray = RayInfo {
        start_position: safe_pos,
        cast_direction: to_light,
        min_depth: EPS,
        max_depth: to_light.magnitude(),
    };
    let shadow_cast = cast_ray(scene, &shadow_ray, 0.0, false);
    if shadow_cast.distance <= shadow_ray.max_depth {
        // in shadow: no direct lighting
        return Vec3::new(0.0, 0.0, 0.0);
    }

    // 3) fetch BRDF parameters
    let albedo     = material.albedo;
    let f0         = material.f0;
    // remap artist roughness²→alpha
    let alpha      = material.roughness * material.roughness;

    // 4) compute D, G, F
    let D = d_ggx(n_dot_h, alpha);
    let G = g_smith(n_dot_v, n_dot_l, alpha);
    let F = fresnel_schlick(v_dot_h, f0);

    // 5) Specular term (Cook‐Torrance)
    //    note the +EPS in denom to avoid NaNs on a perfect mirror
    let spec_numer = F *D * G;
    let spec_denom = 4.0 * n_dot_v * n_dot_l + EPS;
    let specular = spec_numer / spec_denom;

    // 6) Diffuse term: energy‐conserving Lambert
    //    kd = 1 – ks  (we do it per‐channel on Vec3!)
    let kd = Vec3::new(1.0, 1.0, 1.0) - F;
    let diffuse = kd * albedo / PI;

    // 7) light radiance & final
    //    if it’s a point light: falloff = 1 / dist²
    //    if it’s directional: you can just use 1.0
    let radiance = light.color * light.intensity / light_dist2;
    let base = (diffuse + specular) * radiance * n_dot_l;
    base
}

fn get_ray_colour(
    scene: &impl Sdf,
    ray: &RayInfo,
    light_pos: Vec3,
    start_total_distance: f64,
    bounces: usize,
) -> Vec3 {
    let light_colour = Vec3::new(1.0, 1.0, 1.0);
    let sky_colour = Vec3::new(1.0, 1.0, 1.0);
    let scene_ray = cast_ray(scene, ray, start_total_distance, false);
    if scene_ray.distance >= ray.max_depth {
        // Sky box
        return sky_colour
    }

    let eps = 1e-2;
    let dx = scene.distance_to(scene_ray.final_position + Vec3::new(eps, 0.0, 0.0))
        - scene.distance_to(scene_ray.final_position - Vec3::new(eps, 0.0, 0.0));
    let dy = scene.distance_to(scene_ray.final_position + Vec3::new(0.0, eps, 0.0))
        - scene.distance_to(scene_ray.final_position - Vec3::new(0.0, eps, 0.0));
    let dz = scene.distance_to(scene_ray.final_position + Vec3::new(0.0, 0.0, eps))
        - scene.distance_to(scene_ray.final_position - Vec3::new(0.0, 0.0, eps));

    let normal = Vec3 {
        x: dx,
        y: dy,
        z: dz,
    }
        .normalize();

    let mat = scene.get_material(scene_ray.final_position);
    let light = Light {
        position : light_pos,
        color : light_colour,
        intensity : 500000.0,
    };

    let safe_pos = scene_ray.final_position + (-ray.cast_direction * ray.min_depth * 10.0);

    let base = shade_pbr( scene, scene_ray.final_position, normal, -ray.cast_direction, &mat, light, safe_pos );
    if mat.roughness == 0.0 && bounces < 3 {
        let reflection_ray = RayInfo {
            start_position: safe_pos,
            cast_direction: ray.cast_direction.reflect(normal).normalize(),
            min_depth: ray.min_depth,
            max_depth: ray.max_depth,
        };

        let mirror_col = get_ray_colour(
            scene,
            &reflection_ray,
            light_pos,
            scene_ray.distance,
            bounces + 1,
        );
        return mirror_col;
    }
    base
}

fn linear_to_srgb_channel(c: f64) -> f64 {
    let c = c.clamp(0.0, 1.0);
    if c <= 0.0031308 {
        12.92 * c
    } else {
        1.055 * c.powf(1.0/2.4) - 0.055
    }
}
fn to_srgb(linear: Vec3) -> Vec3 {
    Vec3::new(
        linear_to_srgb_channel(linear.x),
        linear_to_srgb_channel(linear.y),
        linear_to_srgb_channel(linear.z),
    )
}

fn get_pixel_colour(
    x: usize,
    y: usize,
    width: usize,
    height: usize,
    scene: &impl Sdf,
    cam_pos: Vec3,
    screen_x: Vec3,
    screen_y: Vec3,
    screen_pos: Vec3,
    screen_width: f64,
    light_pos: Vec3,
) -> u32 {
    let min_d: f64 = 0.001;
    let max_d: f64 = 1e3;

    let aspect_ratio  = width as f64 / height as f64;
    let screen_height = screen_width / aspect_ratio;

    // Get point on screen.
    let screen_x_pos = ((x as f64 - width as f64 / 2.0) / width as f64) * screen_width;
    let screen_y_pos = ((y as f64 - height as f64 / 2.0) / height as f64) * screen_height;

    let pixel_pos = screen_pos + screen_x * screen_x_pos - screen_y * screen_y_pos;
    let cast_dir = (pixel_pos - cam_pos).normalize();

    let ray = RayInfo {
        start_position: cam_pos,
        cast_direction: cast_dir,
        min_depth: min_d,
        max_depth: max_d,
    };

    let result = to_srgb(get_ray_colour(scene, &ray, light_pos, 0.0, 0));


    let r = (result.x * 255.0).round() as u32;
    let g = (result.y * 255.0).round() as u32;
    let b = (result.z * 255.0).round() as u32;
    (255 << 24) | (r << 16) | (g << 8) | b
}

fn main() {
    // ThreadPoolBuilder::new()
    //     .num_threads(6)
    //     .build_global()
    //     .expect("Failed to build global Rayon thread pool");

    let mut buffer: Vec<u32> = vec![0; WIDTH * HEIGHT];

    let mut window = Window::new(
        "My Rust Framebuffer",
        WIDTH,
        HEIGHT,
        WindowOptions::default(),
    )
    .unwrap();

    let mut screen_multiplier: usize = 4;

    let mut last_instant = Instant::now();
    let mut frame_count = 0u32;

    let mut cam_pos = Vec3 {
        x: 0.0,
        y: 0.0,
        z: -10.0,
    };
    let mut look_pos = Vec3 {
        x: 0.0,
        y: 0.0,
        z: 0.0,
    };
    let mut screen_distance = 0.5;
    let mut screen_width = 2.0;

    let mut light_pos;

    let up = Vec3 {
        x: 0.0,
        y: 1.0,
        z: 0.0,
    };

    let white_mat = Material::new(Vec3::new(1.0, 1.0, 1.0), Vec3::new(0.04, 0.04, 0.04), 0.8);
    let red_mirror_mat = Material::new(Vec3::new(0.9, 0.7, 0.4), Vec3::new(1.0, 1.0, 1.0), 0.0);
    let black_mat = Material::new(Vec3::new(0.1, 0.1, 0.1), Vec3::new(0.04, 0.04, 0.04), 0.8);
    let green_mat = Material::new(Vec3::new(0.2, 0.8, 0.2), Vec3::new(0.04, 0.04, 0.04), 0.8);


    let dark_brown = Material::new(Vec3::new(0.5, 0.4, 0.2), Vec3::new(0.04, 0.04, 0.04), 0.8);
    let light_brown = Material::new(Vec3::new(0.9, 0.7, 0.4), Vec3::new(0.04, 0.04, 0.04), 0.8);

    let shapes: Vec<Box<dyn Sdf>> = vec![
        Box::new(Sphere::new(
            8.0,
            Vec3::new(-5.0, -12.0, 0.0),
            red_mirror_mat,
        )),
        Box::new(Sphere::new(
            8.0,
            Vec3::new(25.0, -12.0, 10.0),
            red_mirror_mat,
        )),
        Box::new(SmoothUnion::new(
            Sphere::new(8.0, Vec3::new(0.0, 11.0, 20.0), white_mat),
            SmoothUnion::new(
                Sphere::new(12.0, Vec3::new(0.0, -12.0, 20.0), white_mat),
                Sphere::new(10.0, Vec3::new(0.0, 0.0, 20.0), white_mat),
                0.3,
            ),
            0.3,
        )),
        Box::new(Sphere::new(1.0, Vec3::new(-3.0, 12.5, 12.7), black_mat)),
        Box::new(Sphere::new(1.0, Vec3::new(3.0, 12.5, 12.7), black_mat)),
        Box::new(Cuboid::new(
            Vec3::new(1.0, 1.0, 1.0),
            Vec3::new(0.0, 0.0, 5.0),
            green_mat,
        )),
        Box::new(Sphere::new(3.0, Vec3::new(10.0, 5.0, 20.0), green_mat)),
        Box::new(Checker::new(
            Cuboid::new(
                Vec3::new(50.0, 1.0, 50.0),
                Vec3::new(0.0, -20.0, 0.0),
                dark_brown,
            ),
            0.1,
            light_brown,
        )),
    ];

    // let scene = SpeedField::new(UnionN::new(shapes));
    let scene = UnionN::new(shapes);

    // let scene = CheckerFloor { height: 10.0, scale: 10.0, radius: 1.0, mat1: green_mat, mat2: red_mat};
    // let scene = Mandelbulb { power: 8, iterations: 12, scale: 1.0, pos: Vec3 {x: 0.0, y: 0.0, z: 0.0}};
    let start = Instant::now();
    while window.is_open() && !window.is_key_down(Key::Escape) {
        let render_height = HEIGHT / screen_multiplier;
        let render_width = WIDTH / screen_multiplier;

        frame_count += 1;
        let now = Instant::now();
        let elapsed = now.duration_since(last_instant);
        let dt = elapsed.as_secs_f64().max(1.0);

        let t = now.duration_since(start).as_secs_f64();

        let speed = 1.0;
        let radius = 200.0;

        let angle = t * speed;
        // let angle = 4.1 as f64;
        light_pos = Vec3 {
            x: angle.cos() * radius,
            y: 500.0,
            z: angle.sin() * radius,
        };

        if elapsed >= Duration::from_secs(1) {
            let fps = frame_count as f64 / dt;
            window.set_title(&format!("My Rust Framebuffer — {:.2} FPS", fps));

            frame_count = 0;
            last_instant = now;
        }

        window.update_with_buffer(&buffer, WIDTH, HEIGHT).unwrap();

        let look_dir = (look_pos - cam_pos).normalize();
        let screen_pos = cam_pos + look_dir * screen_distance;
        let screen_x = look_dir.cross(up).normalize();
        let screen_y = screen_x.cross(look_dir).normalize();


        let chunk_size: usize = 16;
        let x_chunks = render_width / chunk_size;
        let y_chunks = render_height / chunk_size;
        let no_chunks = x_chunks * y_chunks;

        let chunk_buffers: Vec<Vec<u32>> = (0..no_chunks)
            .into_par_iter()
            .map(|chunk_idx| {
                let mut chunk_buffer = vec![0u32; chunk_size * chunk_size];
                let chunk_x = (chunk_idx % x_chunks) * chunk_size;
                let chunk_y = (chunk_idx / x_chunks) * chunk_size;
                for y in 0..chunk_size {
                    for x in 0..chunk_size {
                        let px = get_pixel_colour(
                            x + chunk_x,
                            y + chunk_y,
                            render_width,
                            render_height,
                            &scene,
                            cam_pos,
                            screen_x,
                            screen_y,
                            screen_pos,
                            screen_width,
                            light_pos,
                        );
                        chunk_buffer[y * chunk_size + x] = px;
                    }
                }
                chunk_buffer
            })
            .collect();

        for (chunk_idx, chunk_buffer) in chunk_buffers.iter().enumerate() {
            let chunk_x = (chunk_idx % x_chunks) * chunk_size * screen_multiplier;
            let chunk_y = (chunk_idx / x_chunks) * chunk_size * screen_multiplier;
            for (pixel_idx, pixel) in chunk_buffer.iter().enumerate() {
                let pixel_x = (pixel_idx % chunk_size) * screen_multiplier;
                let pixel_y = (pixel_idx / chunk_size) * screen_multiplier;
                for dy in 0..screen_multiplier {
                    for dx in 0..screen_multiplier {
                        let screen_x = chunk_x + pixel_x + dx;
                        let screen_y = chunk_y + pixel_y + dy;
                        let idx = screen_y * WIDTH + screen_x;
                        buffer[idx] = *pixel;
                    }
                }
            }
        }

        let speed = 1.0; // world units per second
        if window.is_key_down(Key::W) {
            cam_pos = cam_pos + look_dir * speed * dt;
            look_pos = look_pos + look_dir * speed * dt;
        }
        if window.is_key_down(Key::S) {
            cam_pos = cam_pos - look_dir * speed * dt;
            look_pos = look_pos - look_dir * speed * dt;
        }
        if window.is_key_down(Key::D) {
            cam_pos = cam_pos + screen_x * speed * dt;
            look_pos = look_pos + screen_x * speed * dt;
        }
        if window.is_key_down(Key::A) {
            cam_pos = cam_pos - screen_x * speed * dt;
            look_pos = look_pos - screen_x * speed * dt;
        }
        if window.is_key_down(Key::Space) {
            cam_pos = cam_pos + screen_y * speed * dt;
            look_pos = look_pos + screen_y * speed * dt;
        }
        if window.is_key_down(Key::LeftShift) {
            cam_pos = cam_pos - screen_y * speed * dt;
            look_pos = look_pos - screen_y * speed * dt;
        }

        if window.is_key_down(Key::NumPadPlus) {
            screen_distance -= 0.1 * dt;
        }
        if window.is_key_down(Key::NumPadMinus) {
            screen_distance += 0.1 * dt;
        }

        if window.is_key_down(Key::NumPadAsterisk) {
            screen_width -= 0.1 * dt;
        }
        if window.is_key_down(Key::NumPadSlash) {
            screen_width += 0.1 * dt;
        }

        if window.is_key_down(Key::Key1) {
            screen_multiplier = 1;
        }
        if window.is_key_down(Key::Key2) {
            screen_multiplier = 2;
        }
        if window.is_key_down(Key::Key3) {
            screen_multiplier = 4;
        }
        if window.is_key_down(Key::Key4) {
            screen_multiplier = 8;
        }

        let look_speed = 0.1;
        let up_key = window.is_key_down(Key::Up);
        let down_key = window.is_key_down(Key::Down);
        let left_key = window.is_key_down(Key::Left);
        let right_key = window.is_key_down(Key::Right);
        if up_key || down_key || left_key || right_key {
            let look_vector = look_pos - cam_pos;
            let look_r = look_vector.magnitude();

            let mut phi = look_vector.z.atan2(look_vector.x);
            let mut theta = (look_vector.y / look_r).acos();

            if right_key {
                phi += look_speed * dt;
            }
            if left_key {
                phi -= look_speed * dt;
            }
            if up_key {
                theta -= look_speed * dt;
            }
            if down_key {
                theta += look_speed * dt;
            }

            let eps = 0.001;
            theta = theta.clamp(eps, PI - eps);

            let sin_t = theta.sin();
            let new_v = Vec3 {
                x: look_r * sin_t * phi.cos(),
                y: look_r * theta.cos(),
                z: look_r * sin_t * phi.sin(),
            }
            .normalize();

            look_pos = cam_pos + new_v;
        }
    }
}
