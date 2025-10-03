use minifb::{Key, Window, WindowOptions};
use std::time::{Duration, Instant};
use vec3::Vec3;
use shapes::{Sphere, Sdf, Union, Box};

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
        let screen_width = 2.0;
        let scene = Union {
            a: Union {
                a: Sphere { radius: 8.0, pos: Vec3 { x: 0.0, y: 11.0, z: 20.0 } },
                b: Sphere { radius: 10.0, pos: Vec3 { x: 0.0, y: 0.0, z: 20.0 } }
            },
            b: Sphere { radius: 12.0, pos: Vec3 { x: 0.0, y: -12.0, z: 20.0 } }
        };
        // let scene = Box { sides: Vec3 { x: 10.0, y: 10.0, z: 10.0 }, pos: Vec3 { x: 0.0, y: 0.0, z: 20.0 } };

        let mut cam_pos = Vec3 {x:0.0, y: 0.0, z:-10.0};

        let look_pos = Vec3 {x:0.0, y:0.0, z:1.0};
        let up = Vec3 {x: 0.0, y: -1.0, z:0.0};
        let min_d:f32 = 1e-3;
        let max_d:f32 = 1e4;

        frame_count += 1;
        let now = Instant::now();
        let elapsed = now.duration_since(last_instant);

        if elapsed >= Duration::from_secs(1) {
            let fps = frame_count as f64 / elapsed.as_secs_f64();
            window.set_title(&format!("My Rust Framebuffer — {:.2} FPS", fps));

            frame_count = 0;
            last_instant = now;
        }

        window
            .update_with_buffer(&buffer, WIDTH, HEIGHT)
            .unwrap();

        let render_height = HEIGHT/SCREEN_MULTIPLIER;
        let render_width = WIDTH/SCREEN_MULTIPLIER;

        for y in 0..render_height {
            for x in 0..render_width {
                // Get point on screen.
                let look_dir = (look_pos - cam_pos).normalize();
                let screen_pos = cam_pos+look_dir*screen_distance;
                let screen_y = up;
                let screen_x = look_dir.cross(up);
                let screen_x_pos = ((x as f32 - render_width as f32/2.0) / render_width as f32) * screen_width;
                let screen_y_pos = ((y as f32 - render_height as f32/2.0) / render_height as f32) * screen_width;
                let pixel_pos = screen_pos + screen_x * screen_x_pos + screen_y * screen_y_pos;
                let cast_dir = (pixel_pos-cam_pos).normalize();

                let mut total_dist = 0.0;
                let mut dist = scene.distance_to(cam_pos);
                let mut pos = cam_pos.clone();
                while (dist > min_d) && (dist < max_d) {
                    total_dist += dist;
                    pos = pos + cast_dir * dist;
                    dist = scene.distance_to(pos);
                }

                let r = depth_to_u32(total_dist, 15.0, 40.0, 2.2 );
                let (g, b) = (r, r);

                for dy in 0..SCREEN_MULTIPLIER {
                    for dx in 0..SCREEN_MULTIPLIER {
                        let idx = (y*SCREEN_MULTIPLIER+dy) * WIDTH + x*SCREEN_MULTIPLIER+dx;
                        buffer[idx] = (255 << 24) | (r << 16) | (g << 8) | b;
                    }
                }
            }
        }
    }
}