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

pub fn depth_to_gamma(
    depth:     f32,
    min_depth: f32,
    max_depth: f32,
    gamma:     f32,
) -> f32 {
    // 1. normalize
    let t = (depth - min_depth) / (max_depth - min_depth);
    // 2. clamp
    let t = t.clamp(0.0, 1.0);
    // 3. gamma‐correct (use 1.0 / gamma so that gamma>1 lightens midtones)
    t.powf(1.0 / gamma)
}

/// If you’d like an 8‐bit value instead of f32, you can do:
pub fn depth_to_u32(
    depth:     f32,
    min_depth: f32,
    max_depth: f32,
    gamma:     f32,
) -> u32 {
    let v = depth_to_gamma(depth, min_depth, max_depth, gamma);
    // scale to 0..255, round to nearest and cast
    (v * 255.0).round().clamp(0.0, 255.0) as u32
}


fn main() {
    let mut buffer: Vec<u32> = vec![0; WIDTH * HEIGHT];

    let mut window = Window::new(
        "My Rust Framebuffer",
        WIDTH,
        HEIGHT,
        WindowOptions::default(),
    ).unwrap();

    let mut last_instant = Instant::now();
    let mut frame_count = 0u32;

    while window.is_open() && !window.is_key_down(Key::Escape) {
        let screen_distance = 1.0;
        let screen_width = 5.0;
        let sphere = Sphere { radius: 10.0, pos: Vec3 {x: 0.0, y:0.0, z:20.0 }};
        let mut cam_pos = Vec3 {x:0.0, y: 0.0, z:0.0};
        let look_pos = Vec3 {x:0.0, y:0.0, z:1.0};
        let up = Vec3 {x: 0.0, y: 1.0, z:0.0};
        let min_d:f32 = 1e-3;
        let max_d:f32 = 1e3;

        // Fill the buffer with some pattern
        for y in 0..HEIGHT {
            for x in 0..WIDTH {
                // Get point on screen.
                let look_dir = (look_pos - cam_pos).normalize();
                let screen_pos = cam_pos+look_dir*screen_distance;
                let screen_y = up;
                let screen_x = look_dir.cross(up);
                let screen_x_pos = ((x as f32 - WIDTH as f32/2.0) / WIDTH as f32) * screen_width;
                let screen_y_pos = ((y as f32 - HEIGHT as f32/2.0) / HEIGHT as f32) * screen_width;
                let pixel_pos = screen_pos + screen_x * screen_x_pos + screen_y * screen_y_pos;
                let cast_dir = (pixel_pos-cam_pos).normalize();

                let mut total_dist = 0.0;
                let mut dist = sphere.distance_to(cam_pos);
                let mut pos = cam_pos.clone();
                while (dist > min_d) && (dist < max_d) {
                    total_dist += dist;
                    pos = pos + cast_dir * dist;
                    dist = sphere.distance_to(pos);
                }

                let r = depth_to_u32(total_dist, 0.0, 30.0, 2.2 );
                let (g, b) = (r, r);
                let idx = y * WIDTH + x;
                // let r = ((x + y) % 256) as u32;
                // let g = ((2 * x + y) % 256) as u32;
                // let b = ((x + 2 * y) % 256) as u32;
                buffer[idx] = (255 << 24) | (r << 16) | (g << 8) | b;
            }
        }

        // ===== FPS COUNT & TITLE UPDATE =====
        frame_count += 1;
        let now = Instant::now();
        let elapsed = now.duration_since(last_instant);

        // update once per second
        if elapsed >= Duration::from_secs(1) {
            let fps = frame_count as f64 / elapsed.as_secs_f64();
            // update window title
            window.set_title(&format!("My Rust Framebuffer — {:.2} FPS", fps));
            // or just println!("FPS: {:.2}", fps);

            // reset counters
            frame_count = 0;
            last_instant = now;
        }

        // Feed it to the window.  Window handles vsync internally.
        window
            .update_with_buffer(&buffer, WIDTH, HEIGHT)
            .unwrap();

        // You can also query key events:
        if window.is_key_pressed(Key::Space, minifb::KeyRepeat::Yes) {
            println!("Space was pressed!");
            cam_pos = Vec3 {y: cam_pos.y + 1.0, ..cam_pos}
        }
        if window.is_key_pressed(Key::LeftShift, minifb::KeyRepeat::Yes) {
            cam_pos = Vec3 {y: cam_pos.y - 1.0, ..cam_pos}
        }
    }
}