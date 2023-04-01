use glam::Vec3;
use pixels::{Error, Pixels, SurfaceTexture};
use std::time::Instant;
use winit::dpi::LogicalSize;
use winit::event::{Event, VirtualKeyCode};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::WindowBuilder;
use winit_input_helper::WinitInputHelper;

const WIDTH: usize = 320;
const HEIGHT: usize = 240;

const MAX_DIST: f32 = 50.0;
const EPSILON: f32 = 0.001;
const EPSILON1: f32 = 0.0011;

/// Representation of the application state. In this example, a box will bounce around the screen.
struct ImageData {
    pixel_data: [u8; WIDTH * HEIGHT * 4],
}

impl ImageData {
    fn new() -> Self {
        Self {
            pixel_data: [0; WIDTH * HEIGHT * 4],
        }
    }

    fn set(&mut self, x: usize, y: usize, c: (u8, u8, u8)) {
        let index = (x + y * WIDTH) * 4;
        self.pixel_data[index] = c.0;
        self.pixel_data[index + 1] = c.1;
        self.pixel_data[index + 2] = c.2;
        self.pixel_data[index + 3] = 255;
    }

    fn draw(&self, buffer: &mut [u8]) {
        buffer.copy_from_slice(&self.pixel_data as &[u8]);
    }
}

fn main() -> Result<(), Error> {
    let event_loop = EventLoop::new();
    let mut input = WinitInputHelper::new();
    let window = {
        let size = LogicalSize::new(WIDTH as f64, HEIGHT as f64);
        WindowBuilder::new()
            .with_title("SDF ray marching test")
            .with_inner_size(size)
            .with_min_inner_size(size)
            .build(&event_loop)
            .unwrap()
    };

    let mut pixels = {
        let window_size = window.inner_size();
        let surface_texture = SurfaceTexture::new(window_size.width, window_size.height, &window);
        Pixels::new(WIDTH as u32, HEIGHT as u32, surface_texture)?
    };

    let mut image = Box::new(ImageData::new());
    let mut x = 0;
    let mut y = 0;
    let mut samples = 1;

    event_loop.run(move |event, _, control_flow| {
        // Draw the current frame
        if let Event::RedrawRequested(_) = event {
            image.draw(pixels.get_frame_mut());
            if let Err(_) = pixels.render() {
                *control_flow = ControlFlow::Exit;
                return;
            }
        }

        // Handle input events
        if input.update(&event) {
            // Close events
            if input.key_pressed(VirtualKeyCode::Escape)
                || input.close_requested()
                || input.destroyed()
            {
                *control_flow = ControlFlow::Exit;
                return;
            }

            // Resize the window
            if let Some(size) = input.window_resized() {
                if let Err(_) = pixels.resize_surface(size.width, size.height) {
                    *control_flow = ControlFlow::Exit;
                    return;
                }
            }

            let start = Instant::now();
            while start.elapsed().as_millis() < 10 {
                let c = get_color_for_pixel(x, y, samples);
                image.set(x, y, c);
                x += 1;
                if x == WIDTH {
                    x = 0;
                    y += 1;
                }
                if y >= HEIGHT {
                    y = 0;
                    if samples < 1000 {
                        samples *= 4;
                    }
                    println!("Starting render with {} samples", samples);
                }
            }

            window.request_redraw();
        }
    });
}

// above here is all the stuff to get pixels onto the screen
// below here is all the stuff for actually making the image

enum Material {
    Sphere,
    Floor,
}

pub fn get_color_for_pixel(x: usize, y: usize, samples: u16) -> (u8, u8, u8) {
    let mut r = 0.0;
    let mut g = 0.0;
    let mut b = 0.0;
    for _ in 0..samples {
        let x1 = x as f32 + if samples == 1 { 0.0 } else { fastrand::f32() };
        let y1 = y as f32 + if samples == 1 { 0.0 } else { fastrand::f32() };
        let c = get_color_for_camera_space(
            ((x1 / WIDTH as f32) * 2.0 - 1.0) * (WIDTH as f32 / HEIGHT as f32),
            ((y1 / HEIGHT as f32) * 2.0 - 1.0) * -1.0,
        );
        r += c.0;
        g += c.1;
        b += c.2;
    }
    (
        ((r / samples as f32).powf(1.0 / 2.2) * 255.0) as u8,
        ((g / samples as f32).powf(1.0 / 2.2) * 255.0) as u8,
        ((b / samples as f32).powf(1.0 / 2.2) * 255.0) as u8,
    )
}

fn get_random_unit_vector() -> Vec3 {
    loop {
        let x = fastrand::f32() * 2.0 - 1.0;
        let y = fastrand::f32() * 2.0 - 1.0;
        let z = fastrand::f32() * 2.0 - 1.0;
        let v = Vec3::new(x, y, z);
        if v.length_squared() <= 1.0 {
            return v.normalize();
        }
    }
}

fn get_color_for_camera_space(x: f32, y: f32) -> (f32, f32, f32) {
    let cam_pos = Vec3::new(15.0, -9.0, 8.0);
    let look_vec = (-cam_pos).normalize();
    let up_vec = Vec3::new(0.0, 0.0, 1.0);
    let dx = look_vec.cross(up_vec).normalize();
    let dy = dx.cross(look_vec).normalize();
    let fov_factor = 0.2;
    get_color_for_ray(cam_pos, look_vec + (x * dx + y * dy) * fov_factor, 2)
}

fn get_collision_for_ray(start: Vec3, dir: Vec3) -> Option<(Vec3, Material)> {
    let dir_n = dir.normalize();
    let mut pos = start;
    let mut dist = 0.0;
    while dist < MAX_DIST {
        let (next, m) = sdf(pos);
        if next < EPSILON {
            return Some((pos, m));
        }
        pos += dir_n * next;
        dist += next;
    }
    return None;
}

fn get_color_for_ray(start: Vec3, dir: Vec3, bounces: u8) -> (f32, f32, f32) {
    let sun_pos = Vec3::new(20.0, 20.0, 20.0);
    let fill_pos = Vec3::new(1.0, 1.0, 50.0);
    match get_collision_for_ray(start, dir) {
        Some((pos, m)) => {
            let nx = sdf(pos + Vec3::new(EPSILON, 0.0, 0.0)).0
                - sdf(pos - Vec3::new(EPSILON, 0.0, 0.0)).0;
            let ny = sdf(pos + Vec3::new(0.0, EPSILON, 0.0)).0
                - sdf(pos - Vec3::new(0.0, EPSILON, 0.0)).0;
            let nz = sdf(pos + Vec3::new(0.0, 0.0, EPSILON)).0
                - sdf(pos - Vec3::new(0.0, 0.0, EPSILON)).0;
            let normal = Vec3::new(nx, ny, nz).normalize();
            let (diffuse_albedo, shiny_albedo) = match m {
                Material::Sphere => ((0.2, 0.1, 0.0), (0.0, 0.0, 0.0)),
                Material::Floor => {
                    let check = ((pos.x * 2.0).floor() + (pos.y * 2.0).floor()).rem_euclid(2.0);
                    ((0.0, check * 0.05, 0.05), (0.1, 0.1, 0.1))
                }
            };
            let shadow_ray_dir = (sun_pos - pos).normalize();
            let obscured = get_collision_for_ray(pos + normal * EPSILON1, shadow_ray_dir).is_some();
            let s = if obscured {
                0.0
            } else {
                (sun_pos - pos).normalize().dot(normal).max(0.0) * 0.8
            };
            let f = (fill_pos - pos).normalize().dot(normal).max(0.0) * 0.1;
            let diffuse = if bounces > 0 {
                let mut bd = (normal * 1.001 + get_random_unit_vector()).normalize();
                if bd.dot(normal) < 0.0 {
                    bd *= -1.0;
                }
                get_color_for_ray(pos + normal * EPSILON1, bd, bounces - 1)
            } else {
                (0.0, 0.0, 0.0)
            };
            let ex = if bounces > 0
                && (shiny_albedo.0 > 0.0 || shiny_albedo.1 > 0.0 || shiny_albedo.2 > 0.0)
            {
                let reflected_dir = dir + -2.0 * normal * dir.dot(normal);
                let c = get_color_for_ray(pos + normal * EPSILON1, reflected_dir, bounces - 1);
                (
                    c.0 * shiny_albedo.0,
                    c.1 * shiny_albedo.1,
                    c.2 * shiny_albedo.2,
                )
            } else {
                (0.0, 0.0, 0.0)
            };
            (
                diffuse_albedo.0 * (s + f + diffuse.0) + ex.0,
                diffuse_albedo.1 * (s + f + diffuse.1) + ex.1,
                diffuse_albedo.2 * (s + f + diffuse.2) + ex.2,
            )
        }
        None => (0.0, 0.0, 0.0),
    }
}

fn sdf(p: Vec3) -> (f32, Material) {
    let c = Vec3::new((p.x / 2.0).round() * 2.0, (p.y / 2.0).round() * 2.0, 0.0);
    let sd = (p - c).length() - 0.7;
    let fd = p.z + 0.7;
    if fd < sd {
        (fd, Material::Floor)
    } else {
        (sd, Material::Sphere)
    }
}
