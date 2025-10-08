use std::f32::consts::PI;
use minifb::{Key, Window, WindowOptions};
use std::time::{Duration, Instant};
use vec3::Vec3;
use shapes::{Sphere, Sdf, Union, Box, SmoothUnion, Material};
use crate::shapes::{Checker, SpeedField};

mod vec3;
mod shapes;

const WIDTH: usize = 1280;
const HEIGHT: usize = 1280;

pub struct RayResults {
    pub final_position: Vec3,
    pub distance: f32,
}

pub struct RayInfo {
    pub start_position: Vec3,
    pub cast_direction: Vec3,
    pub min_depth: f32,
    pub max_depth: f32,
}

pub fn depth_to_gamma(
    depth: f32,
    min_depth: f32,
    max_depth: f32,
    gamma: f32,
) -> f32 {
    let t = (depth - min_depth) / (max_depth - min_depth);
    let t = t.clamp(0.0, 1.0);
    t.powf(1.0 / gamma)
}

pub fn depth_to_u32(
    depth: f32,
    min_depth: f32,
    max_depth: f32,
    gamma: f32,
) -> u32 {
    let v = depth_to_gamma(depth, min_depth, max_depth, gamma);
    v.round().clamp(0.0, 255.0) as u32
}
#[inline]
fn cast_ray(
    scene: &impl Sdf,
    ray: &RayInfo,
    start_total_distance: f32,
) -> RayResults {
    let mut total_dist = start_total_distance;
    let mut dist = scene.distance_to(ray.start_position);
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

fn get_ray_colour(
    scene: &impl Sdf,
    ray: &RayInfo,
    light_pos: Vec3,
    start_total_distance: f32,
    bounces: usize,
) -> Vec3 {
    let scene_ray = cast_ray(
        scene,
        ray,
        start_total_distance,
    );
    if scene_ray.distance >= ray.max_depth {
        // Sky box
        return Vec3 { x: 0.9, y: 0.9, z: 0.9 };
    }
    let mat = scene.get_material(scene_ray.final_position);

    let eps = 1e-2;
    let dx = scene.distance_to(scene_ray.final_position + Vec3::new(eps, 0.0, 0.0)) - scene.distance_to(scene_ray.final_position - Vec3::new(eps, 0.0, 0.0));
    let dy = scene.distance_to(scene_ray.final_position + Vec3::new(0.0, eps, 0.0)) - scene.distance_to(scene_ray.final_position - Vec3::new(0.0, eps, 0.0));
    let dz = scene.distance_to(scene_ray.final_position + Vec3::new(0.0, 0.0, eps)) - scene.distance_to(scene_ray.final_position - Vec3::new(0.0, 0.0, eps));

    let normal = Vec3 { x: dx, y: dy, z: dz }.normalize();

    let safe_pos = scene_ray.final_position + (normal * ray.min_depth * 1.1);

    // Phong Lighting
    let light_vec = (light_pos - scene_ray.final_position).normalize();

    // Shadow
    let light_ray = RayInfo {
        start_position: safe_pos,
        cast_direction: light_vec,
        min_depth: ray.min_depth,
        max_depth: ray.max_depth
    };

    let light_check = cast_ray(
        scene,
        &light_ray,
        start_total_distance,
    );

    let gamma_lightness = if light_check.distance >= light_ray.max_depth {
        let diffuse = normal.dot(light_vec).max(0.0);
        let reflection = light_vec.reflect(normal);
        let specular = light_ray.cast_direction.dot(reflection).max(0.0).powi(32) * mat.specular;
        let lightness = (specular + diffuse + 0.1) / 2.1;
        depth_to_gamma(lightness, 0.0, 1.0, 2.2)
    } else {
        0.1
    };

    let result = mat.colour * gamma_lightness;

    if mat.mirror && (bounces < 3) {
        let reflection_ray = RayInfo {
            start_position: safe_pos,
            cast_direction: ray.cast_direction.reflect(normal),
            min_depth: ray.min_depth,
            max_depth: ray.max_depth
        };
        return get_ray_colour(
            scene,
            &reflection_ray,
            light_pos,
            scene_ray.distance,
            bounces + 1,
        ) * (1.0 - mat.mirror_mix) + (result * mat.mirror_mix);
    }

    result
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
    light_pos: Vec3,
) -> u32 {
    let min_d: f32 = 0.05;
    let max_d: f32 = 1e3;

    // Get point on screen.
    let screen_x_pos = ((x as f32 - width as f32 / 2.0) / width as f32) * screen_width;
    let screen_y_pos = ((y as f32 - height as f32 / 2.0) / height as f32) * screen_width;
    let pixel_pos = screen_pos + screen_x * screen_x_pos - screen_y * screen_y_pos;
    let cast_dir = (pixel_pos - cam_pos).normalize();

    let ray = RayInfo {
        start_position: cam_pos,
        cast_direction: cast_dir,
        min_depth: min_d,
        max_depth: max_d
    };

    let result = get_ray_colour(
        scene,
        &ray,
        light_pos,
        0.0,
        0,
    );

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

    const SCREEN_MULTIPLIER: usize = 4;

    let render_height = HEIGHT / SCREEN_MULTIPLIER;
    let render_width = WIDTH / SCREEN_MULTIPLIER;

    let mut last_instant = Instant::now();
    let mut frame_count = 0u32;

    let mut cam_pos = Vec3 { x: 0.0, y: 0.0, z: -10.0 };
    let mut look_pos = Vec3 { x: 0.0, y: 0.0, z: 0.0 };
    let mut screen_distance = 0.5;
    let mut screen_width = 2.0;

    let mut light_pos;

    let up = Vec3 { x: 0.0, y: 1.0, z: 0.0 };

    let red_mat = Material::new(Vec3::new(0.8, 0.2, 0.2 ), 0.8,  false, 0.0);
    let white_mat = Material::new(Vec3::new(1.0, 1.0, 1.0 ), 0.1,  false, 0.0);
    let red_mirror_mat = Material::new(Vec3::new(0.8, 0.2, 0.2 ), 1.0,  true, 0.5);
    let blue_mirror_mat = Material::new(Vec3::new(0.2, 0.2, 0.8 ), 1.0,  true, 0.5);
    let black_mat = Material::new(Vec3::new(0.1, 0.1, 0.1 ), 1.0,  false, 0.0);
    let green_mat = Material::new(Vec3::new(0.2, 0.8, 0.2 ), 0.8,  false, 0.0);

    let dark_brown = Material::new(Vec3::new(0.5, 0.4, 0.2 ), 0.2,  true, 0.96);
    let light_brown = Material::new(Vec3::new(0.9, 0.7, 0.4 ), 0.2,  true, 0.96);

    let scene = SpeedField::new(
        Union::new(
            Union::new(
                Sphere::new(8.0, Vec3::new(-5.0, -12.0, 0.0 ), red_mirror_mat),
                Sphere::new(8.0, Vec3::new(25.0, -12.0, 10.0 ), blue_mirror_mat ),
            ),
            Union::new(
                Union::new(
                    SmoothUnion::new(
                        Sphere::new( 8.0, Vec3::new(0.0, 11.0, 20.0 ), white_mat ),
                        SmoothUnion::new(
                            Sphere::new( 12.0, Vec3::new(0.0, -12.0, 20.0 ), white_mat ),
                            Sphere::new( 10.0, Vec3::new(0.0, 0.0, 20.0 ), white_mat ),
                            0.3,
                        ),
                        0.3,
                    ),
                    Union::new(
                        Sphere::new( 1.0, Vec3::new(-3.0, 12.5, 12.7 ), black_mat ),
                        Sphere::new( 1.0, Vec3::new(3.0, 12.5, 12.7 ), black_mat ),
                    ),
                ),
                Union::new(
                    Union::new(
                        Box::new(Vec3::new(1.0, 1.0, 1.0), Vec3::new(0.0, 0.0, 5.0), green_mat),
                        Sphere::new( 3.0, Vec3::new(10.0, 5.0, 20.0 ), red_mat ),
                    ),
                    Checker::new(
                        Box::new(Vec3::new(50.0, 1.0, 50.0), Vec3::new(0.0, -20.0, 0.0), dark_brown),
                        0.1,
                        light_brown,
                    ),
                ),
            ),
        )
    );


    // let scene = CheckerFloor { height: 10.0, scale: 10.0, radius: 1.0, mat1: green_mat, mat2: red_mat};
    // let scene = Mandelbulb { power: 8, iterations: 12, scale: 1.0, pos: Vec3 {x: 0.0, y: 0.0, z: 0.0}};
    let start = Instant::now();
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
            y: 500.0,
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
        let screen_pos = cam_pos + look_dir * screen_distance;
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
                    light_pos,
                );

                for dy in 0..SCREEN_MULTIPLIER {
                    for dx in 0..SCREEN_MULTIPLIER {
                        let idx = (y * SCREEN_MULTIPLIER + dy) * WIDTH + x * SCREEN_MULTIPLIER + dx;
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

            if right_key { phi += look_speed * dt; }
            if left_key { phi -= look_speed * dt; }
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