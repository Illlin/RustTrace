use minifb::{Key, Window, WindowOptions};
use std::time::{Duration, Instant};
use vec3::Vec3;

mod vec3;

const WIDTH: usize = 640;
const HEIGHT: usize = 640;

struct Sphere {
    radius: f32,
    pos: Vec3,
}

impl Sphere {
    fn new(radius: f32, pos: Vec3) -> Self {
        Self { radius, pos }
    }

    fn distance_to(&self, p: Vec3) -> f32 {
        let a = p - self.pos;
        a.magnitude() - self.radius
    }
}

pub fn depth_to_gamma(depth: f32, min_depth: f32, max_depth: f32, gamma: f32) -> f32 {
    // 1. normalize
    let t = (depth - min_depth) / (max_depth - min_depth);
    // 2. clamp
    let t = t.clamp(0.0, 1.0);
    // 3. gamma‐correct (use 1.0 / gamma so that gamma>1 lightens midtones)
    t.powf(1.0 / gamma)
}

/// If you’d like an 8‐bit value instead of f32, you can do:
pub fn depth_to_u32(depth: f32, min_depth: f32, max_depth: f32, gamma: f32) -> u32 {
    let v = depth_to_gamma(depth, min_depth, max_depth, gamma);
    // scale to 0..255, round to nearest and cast
    (v * 255.0).round().clamp(0.0, 255.0) as u32
}

fn main() {
    // A flat buffer of WIDTH * HEIGHT pixels (u32 = 0xAARRGGBB)
    let mut buffer: Vec<u32> = vec![0; WIDTH * HEIGHT];
    let mut window = Window::new(
        "My Rust Framebuffer",
        WIDTH,
        HEIGHT,
        WindowOptions::default(),
    )
    .unwrap();

    // ─── CAMERA STATE ───────────────────────────────────────────────
    let mut cam_pos = Vec3::new(0.0, 0.0, 0.0);
    let mut yaw = 0.0f32; // left/right
    let mut pitch = 0.0f32; // up/down
    let world_up = Vec3::new(0.0, 1.0, 0.0);

    // For frame‐rate independent movement:
    let mut last_frame = Instant::now();

    // ─── FPS COUNTER STATE ───────────────────────────────────────────
    let mut last_fps_time = Instant::now();
    let mut frame_count = 0u32;

    while window.is_open() && !window.is_key_down(Key::Escape) {
        // ——— 1) compute dt ————————————————————————————————
        let now = Instant::now();
        let dt = (now - last_frame).as_secs_f32();
        last_frame = now;

        // ——— 2) build camera basis vectors ————————————————————
        // forward vector from yaw & pitch (spherical coords → cartesian)
        let forward = Vec3::new(
            pitch.cos() * yaw.cos(),
            pitch.sin(),
            pitch.cos() * yaw.sin(),
        )
        .normalize();

        // right and recalculated up
        let right = forward.cross(world_up).normalize();
        let cam_up = right.cross(forward).normalize();

        // ——— 3) handle movement keys ————————————————————————
        let speed = 5.0; // world units per second
        if window.is_key_down(Key::W) {
            cam_pos = cam_pos + forward * speed * dt;
        }
        if window.is_key_down(Key::S) {
            cam_pos = cam_pos - forward * speed * dt;
        }
        if window.is_key_down(Key::D) {
            cam_pos = cam_pos + right * speed * dt;
        }
        if window.is_key_down(Key::A) {
            cam_pos = cam_pos - right * speed * dt;
        }
        if window.is_key_down(Key::Space) {
            cam_pos = cam_pos + world_up * speed * dt;
        }
        if window.is_key_down(Key::LeftShift) {
            cam_pos = cam_pos - world_up * speed * dt;
        }

        // ——— 4) handle look (yaw/pitch) keys ————————————————————
        let rot_speed = 1.2; // radians per second
        if window.is_key_down(Key::Left) {
            yaw -= rot_speed * dt;
        }
        if window.is_key_down(Key::Right) {
            yaw += rot_speed * dt;
        }
        if window.is_key_down(Key::Up) {
            pitch = (pitch + rot_speed * dt).clamp(
                -std::f32::consts::FRAC_PI_2 + 0.01,
                std::f32::consts::FRAC_PI_2 - 0.01,
            );
        }
        if window.is_key_down(Key::Down) {
            pitch = (pitch - rot_speed * dt).clamp(
                -std::f32::consts::FRAC_PI_2 + 0.01,
                std::f32::consts::FRAC_PI_2 - 0.01,
            );
        }

        // ─── 5) ray‐march loop ────────────────────────────────────────
        let sphere = Sphere::new(10.0, Vec3::new(0.0, 0.0, 20.0));
        let screen_dist = 1.0;
        let screen_width = 5.0;
        let min_d: f32 = 1e-3;
        let max_d: f32 = 1e3;

        for y in 0..HEIGHT {
            for x in 0..WIDTH {
                // build point on screen plane in world‐space
                let screen_center = cam_pos + forward * screen_dist;

                let sx = ((x as f32 / WIDTH as f32) - 0.5) * screen_width;
                let sy = ((y as f32 / HEIGHT as f32) - 0.5) * screen_width;
                let pixel_pos = screen_center + right * sx + cam_up * sy;

                let cast_dir = (pixel_pos - cam_pos).normalize();

                // ray‐march
                let mut total_dist = 0.0;
                let mut dist = sphere.distance_to(cam_pos);
                let mut pos = cam_pos;
                while dist > min_d && dist < max_d {
                    total_dist += dist;
                    pos = pos + cast_dir * dist;
                    dist = sphere.distance_to(pos);
                }

                let r = depth_to_u32(total_dist, 0.0, 30.0, 2.2);
                let idx = y * WIDTH + x;
                buffer[idx] = (255 << 24)  // alpha
                    | (r   << 16)  // red
                    | (r   <<  8)  // green
                    |  r; // blue
            }
        }

        // ─── 6) FPS COUNTER & TITLE UPDATE ──────────────────────────
        frame_count += 1;
        let fps_elapsed = now.duration_since(last_fps_time);
        if fps_elapsed >= Duration::from_secs(1) {
            let fps = frame_count as f64 / fps_elapsed.as_secs_f64();
            window.set_title(&format!("My Rust Framebuffer — {:.2} FPS", fps));
            frame_count = 0;
            last_fps_time = now;
        }

        // ─── 7) BLIT TO SCREEN ───────────────────────────────────────
        window.update_with_buffer(&buffer, WIDTH, HEIGHT).unwrap();
    }
}
