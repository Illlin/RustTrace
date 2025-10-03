use std::f32::consts::PI;
use minifb::{Key, Window, WindowOptions};
use std::time::{Duration, Instant};
use vec3::Vec3;
use shapes::{Sphere, Sdf, Union, Box, SmoothUnion, Material};
use crate::shapes::{Checker, Mandelbulb};
use rayon::prelude::*;

mod vec3;
mod shapes;

const WIDTH: usize = 1280;
const HEIGHT: usize = 1280;
const SCREEN_MULTIPLIER: usize = 4;

pub fn depth_to_gamma(
    depth:     f32,
    min_depth: f32,
    max_depth: f32,
    gamma:     f32,
) -> f32 {
    let t = (depth - min_depth) / (max_depth - min_depth);
    let t = t.clamp(0.0, 1.0);
    t.powf(1.0 / gamma)
}

pub fn depth_to_u32(
    depth:     f32,
    min_depth: f32,
    max_depth: f32,
    gamma:     f32,
) -> u32 {
    let v = depth_to_gamma(depth, min_depth, max_depth, gamma);
    (v).round().clamp(0.0, 255.0) as u32
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
    screen_width: f32,
    look_dir: Vec3,
    light_pos: Vec3,
) -> u32 {
    let min_d:f32 = 1e-3;
    let max_d:f32 = 1e2;

    // Get point on screen.
    let screen_x_pos = ((x as f32 - width as f32/2.0) / width as f32) * screen_width;
    let screen_y_pos = ((y as f32 - height as f32/2.0) / height as f32) * screen_width;
    let pixel_pos = screen_pos + screen_x * screen_x_pos - screen_y * screen_y_pos;
    let cast_dir = (pixel_pos-cam_pos).normalize();

    let mut total_dist = 0.0;
    let mut dist = scene.distance_to(cam_pos);
    let mut pos = cam_pos.clone();
    while (dist > min_d) && (dist < max_d) {
        total_dist += dist;
        pos = pos + cast_dir * dist;
        dist = scene.distance_to(pos);
    }

    let result: Vec3;

    if dist < min_d {
        let light_vec = (light_pos - pos).normalize();

        let eps = 1e-2;
        let dx = scene.distance_to(pos + Vec3::new(eps, 0.0, 0.0)) - scene.distance_to(pos - Vec3::new(eps, 0.0, 0.0));
        let dy = scene.distance_to(pos + Vec3::new(0.0, eps, 0.0)) - scene.distance_to(pos - Vec3::new(0.0, eps, 0.0));
        let dz = scene.distance_to(pos + Vec3::new(0.0, 0.0, eps)) - scene.distance_to(pos - Vec3::new(0.0, 0.0, eps));

        let normal = Vec3 {x: dx, y: dy, z: dz}.normalize();

        let mat = scene.get_material(pos);

        let diffuse = normal.dot(light_vec).max(0.0);

        let reflection = (light_vec).reflect(normal);
        let specular = look_dir.dot(reflection).max(0.0).powi(32) * mat.specular;

        let lightness = (specular + diffuse + 0.1)/2.1;

        let gamma_lightness = depth_to_gamma(lightness, 0.0, 1.0, 2.2 );

        result = mat.colour * gamma_lightness;
    }
    else {
        result = Vec3 { x: 0.0, y: 0.0, z: 0.0};
    }


    let r = (result.x * 255.0).round() as u32;
    let g = (result.y * 255.0).round() as u32;
    let b = (result.z * 255.0).round() as u32;
    (255 << 24) | (r << 16) | (g << 8) | b
}


fn main() {
    let mut buffer: Vec<u32> = vec![0; WIDTH * HEIGHT];

    let mut window = Window::new(
        "My Rust Framebuffer",
        WIDTH,
        HEIGHT,
        WindowOptions::default(),
    ).unwrap();

    let render_height = HEIGHT/SCREEN_MULTIPLIER;
    let render_width = WIDTH/SCREEN_MULTIPLIER;

    let mut last_instant = Instant::now();
    let mut frame_count = 0u32;

    let mut cam_pos = Vec3 {x:0.0, y: 0.0, z:-10.0};
    let mut look_pos = Vec3 {x:0.0, y:0.0, z:0.0};
    let mut screen_distance = 0.5;
    let mut screen_width = 2.0;

    let mut light_pos = Vec3 {x:0.0, y:20.0, z:-20.0};

    let up = Vec3 {x: 0.0, y: 1.0, z: 0.0};

    let red_mat = Material { colour: Vec3 {x: 0.8, y: 0.2, z: 0.2}, specular: 0.8 };
    let white_mat = Material { colour: Vec3 {x: 1.0, y: 1.0, z: 1.0}, specular: 0.1 };
    let black_mat = Material { colour: Vec3 {x: 0.1, y: 0.1, z: 0.1}, specular: 1.0 };
    let green_mat = Material { colour: Vec3 {x: 0.2, y: 0.8, z: 0.2}, specular: 0.8 };

    let dark_brown = Material { colour: Vec3 {x: 0.5, y: 0.4, z: 0.2}, specular: 0.2 };
    let light_brown = Material { colour: Vec3 {x: 0.9, y: 0.7, z: 0.4}, specular: 0.2 };

    let scene = Union {
        a: Union {
            a: SmoothUnion {
                a: Sphere { radius: 8.0, pos: Vec3 { x: 0.0, y: 11.0, z: 20.0 }, material: white_mat },
                b: SmoothUnion {
                    a: Sphere { radius: 12.0, pos: Vec3 { x: 0.0, y: -12.0, z: 20.0 }, material: white_mat },
                    b: Sphere { radius: 10.0, pos: Vec3 { x: 0.0, y: 0.0, z: 20.0 }, material: white_mat },
                    smooth: 0.3
                },
                smooth: 0.3
            },
            b: Union {
                a: Sphere { radius: 1.0, pos: Vec3 { x: -3.0, y: 12.5, z: 12.7 }, material: black_mat },
                b: Sphere { radius: 1.0, pos: Vec3 { x: 3.0, y: 12.5, z: 12.7 }, material: black_mat },
            }
        },
        b: Union {
            a: Box { sides: Vec3 { x: 1.0, y: 1.0, z: 1.0}, pos: Vec3 { x: 0.0, y: 0.0, z: 5.0 }, material: green_mat },
            b: Checker {
                a: Box { sides: Vec3 { x: 100.0, y: 1.0, z: 100.0}, pos: Vec3 { x: 0.0, y: -22.0, z: 0.0 }, material: dark_brown },
                scale: 0.1,
                material: light_brown
            }
        }
    };

    // let scene = CheckerFloor { height: 10.0, scale: 10.0, radius: 1.0, mat1: green_mat, mat2: red_mat};
    // let scene = Mandelbulb { power: 8, iterations: 12, scale: 1.0, pos: Vec3 {x: 0.0, y: 0.0, z: 0.0}};
    let start =  Instant::now();
    while window.is_open() && !window.is_key_down(Key::Escape) {
        frame_count += 1;
        let now = Instant::now();
        let elapsed = now.duration_since(last_instant);
        let dt = elapsed.as_secs_f32().max(1.0);

        let t = now.duration_since(start).as_secs_f32();

        let speed = 1.0;
        let radius = 200.0;

        let angle = t * speed;
        light_pos = Vec3 {
            x: angle.cos() * radius,
            y: 10.0,
            z: angle.sin() * radius,
        };

        if elapsed >= Duration::from_secs(1) {
            let fps = frame_count as f32 / dt;
            window.set_title(&format!("My Rust Framebuffer — {:.2} FPS", fps));

            frame_count = 0;
            last_instant = now;
        }

        window
            .update_with_buffer(&buffer, WIDTH, HEIGHT)
            .unwrap();

        let look_dir = (look_pos - cam_pos).normalize();
        let screen_pos = cam_pos+look_dir*screen_distance;
        let screen_y = up - look_dir * look_dir.dot(up);
        let screen_x = look_dir.cross(up);

        for y in 0..render_height {
            for x in 0..render_width {
                let px = get_pixel_colour(
                    x,
                    y,
                    render_width,
                    render_height,
                    &scene,
                    cam_pos,
                    screen_x,
                    screen_y,
                    screen_pos,
                    screen_width,
                    look_dir,
                    light_pos
                );

                for dy in 0..SCREEN_MULTIPLIER {
                    for dx in 0..SCREEN_MULTIPLIER {
                        let idx = (y*SCREEN_MULTIPLIER+dy) * WIDTH + x*SCREEN_MULTIPLIER+dx;
                        buffer[idx] = px;
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
            screen_distance = screen_distance - 0.1 * dt;
        }
        if window.is_key_down(Key::NumPadMinus) {
            screen_distance = screen_distance + 0.1 * dt;
        }

        if window.is_key_down(Key::NumPadAsterisk) {
            screen_width = screen_width - 0.1 * dt;
        }
        if window.is_key_down(Key::NumPadSlash) {
            screen_width = screen_width + 0.1 * dt;
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

            if right_key { phi   += look_speed * dt; }
            if left_key { phi   -= look_speed * dt; }
            if up_key { theta -= look_speed * dt; }
            if down_key { theta += look_speed * dt; }

            let eps = 0.001;
            theta = theta.clamp(eps, PI - eps);

            let sin_t = theta.sin();
            let new_v = Vec3 {
                x: look_r * sin_t * phi.cos(),
                y: look_r * theta.cos(),
                z: look_r * sin_t * phi.sin(),
            }.normalize();

            look_pos = cam_pos + new_v;
        }
    }
}