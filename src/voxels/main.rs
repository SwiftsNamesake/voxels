// Voxels
// A world of voxels

// Main focus
// - Implement proper camera controls
// - 
// -
// -
// -

// - Nice debug tools, like coordinate systems with annotated offsets
// - Reloading at runtime

// Extra fun
// - Play with
// - Scripting

use std::env;
use std::error::Error;
use std::time::{Instant};
use std::io::Cursor;
use std::fs;

use glium::*;
use glium::{
    index::{ PrimitiveType },
};
use glutin::{
    event::{Event, KeyboardInput, VirtualKeyCode, WindowEvent, ElementState},
    event_loop::ControlFlow,
};

use cgmath::prelude::*;
use cgmath::{Matrix4, Vector3, Rad};

mod geometry;
mod voxel;

pub struct Camera {
    pub position: Vector3<f32>, 
    pub orientation: Vector3<f32>, // TODO: Use quaternion?
}

impl Camera {
    /// Construct the matrix for the current state of the camera
    fn transform(&self) -> Matrix4<f32> {
        Matrix4::from_value(1.0)
    }
}

fn render_frame(
    display: &glium::Display,
    primitive: glium::index::PrimitiveType,
    use_perspective: bool,
    distance: f32,
    textured_shader: &glium::Program,
    coloured_shader: &glium::Program,
    vertex_buffer: &glium::VertexBuffer<geometry::Vertex>,
    axes_buffer: &glium::VertexBuffer<geometry::Vertex>,
    grass_texture: &glium::texture::Texture2d,
    velocity: Vector3<f32>, position: &mut Vector3<f32>, rotation_y: f32, dt_seconds: f32
) {
    let chunk: voxel::Chunk = vec![
        vec![2, 2, 3, 3],
        vec![2, 2, 3, 3],
        vec![2, 2, 3, 3],
        vec![2, 2, 3, 3],
    ];

    // Animate
    *position += velocity * dt_seconds;

    // Render
    let mut target = display.draw();
    target.clear_color_and_depth((0.0, 0.0, 0.0, 1.0), 1.0);

    let (_screen_width, _screen_height) = {
        let (w, h) = display.get_framebuffer_dimensions();
        (w as f32, h as f32)
    };

    let projection_matrix = cgmath::perspective(
        Rad(45.0f32 * std::f32::consts::PI / 180.0),
        _screen_width/_screen_height,
        0.1f32,
        100.0f32
    );

    let orthographic = cgmath::ortho(-10.0, 10.0, -10.0, 10.0, 0.1f32, 100.0f32);

    let pov_matrix = if use_perspective { projection_matrix } else { orthographic };

    for (irow, row) in chunk.iter().enumerate() {
        for (icol, col) in row.iter().enumerate() {
            let y = *col as f32 - 3.0;
            let translation = *position + cgmath::vec3(irow as f32 * (1.0 + distance), y, icol as f32 * (1.0 + distance));

            let rotation = Matrix4::from_axis_angle(cgmath::vec3(0.0, 1.0, 0.0).normalize(), Rad(rotation_y));

            let model_matrix = rotation
                .concat(&Matrix4::from_translation(translation));
            let modelview: [[f32; 4]; 4] = pov_matrix.concat(&model_matrix).into();
            let uniforms = uniform! {
                texture: grass_texture,
                modelview: modelview
            };

            target.draw(
                vertex_buffer,
                glium::index::NoIndices(primitive),
                textured_shader,
                &uniforms,
                &glium::DrawParameters {
                    depth: glium::Depth {
                        test: glium::draw_parameters::DepthTest::IfLess,
                        write: true,
                        .. Default::default()
                    },
                    blend: glium::Blend::alpha_blending(),
                    multisampling: true,
                    ..Default::default()
                },
            ).unwrap();
        }
    }

    let rotation = Matrix4::from_axis_angle(cgmath::vec3(0.0, 1.0, 0.0).normalize(), Rad(rotation_y));

    let model_matrix = rotation
        .concat(&Matrix4::from_translation(*position));
    let modelview: [[f32; 4]; 4] = pov_matrix.concat(&model_matrix).into();
    let axes_uniforms = uniform! {
        modelview: modelview
    };

    target.draw(
        axes_buffer,
        glium::index::NoIndices(glium::index::PrimitiveType::LinesList),
        coloured_shader,
        &axes_uniforms,
        &glium::DrawParameters {
            depth: glium::Depth {
                test: glium::draw_parameters::DepthTest::IfLess,
                write: true,
                .. Default::default()
            },
            blend: glium::Blend::alpha_blending(),
            multisampling: true,
            line_width: Some(6.0),
            ..Default::default()
        },
    ).unwrap();

    target.finish().unwrap();
}

// TODO: error handling
//pub fn load_texture(display: &glium::Display, data: &'static [u8; N]) -> glium::texture::Texture2d {
//    let image = image::load(
//        Cursor::new(&include_bytes!(path)[..]),
//        image::ImageFormat::Png
//    ).unwrap().to_rgba();
//    let image_dimensions = image.dimensions();
//    let image = glium::texture::RawImage2d::from_raw_rgba_reversed(&image.into_raw(), image_dimensions);
//    return glium::texture::Texture2d::new(&display, image).unwrap();
//}

pub fn run() -> Result<(), Box<dyn Error>> {
    if cfg!(target_os = "linux") && env::var("WINIT_UNIX_BACKEND").is_err() {
        env::set_var("WINIT_UNIX_BACKEND", "x11");
    }

    let width = 64*10;
    let height = 64*10;

    let window = glium::glutin::window::WindowBuilder::new()
        .with_inner_size(glium::glutin::dpi::PhysicalSize::new(width, height))
        .with_title("voxels");

    let context = glium::glutin::ContextBuilder::new().with_vsync(true).with_depth_buffer(24).with_multisampling(4);
    let event_loop = glium::glutin::event_loop::EventLoop::new();
    let display = glium::Display::new(window, context, &event_loop)?;

    let image = image::load(Cursor::new(&include_bytes!("/Users/jonatan/kuliga kodprojekt/oxide/src/voxels/assets/textures/grass_cube_atlas.png")[..]),
        image::ImageFormat::Png).unwrap().to_rgba();
    let image_dimensions = image.dimensions();
    let image = glium::texture::RawImage2d::from_raw_rgba_reversed(&image.into_raw(), image_dimensions);
    let mut grass_texture = glium::texture::Texture2d::new(&display, image).unwrap();

    let mut rotation_y: f32 = 0.0;
    let mut rotation_x: f32 = 0.0;
    let mut position = cgmath::vec3(0.0, 0.0, -0.8);
    let mut velocity = cgmath::vec3(0.0, 0.0, 0.0);

    let mut current_primitive_iter = [PrimitiveType::TrianglesList, PrimitiveType::LinesList, PrimitiveType::Points].iter().cycle();
    let mut primitive = current_primitive_iter.next().unwrap();

    let mut use_perspective = true;

    let mut distance: f32 = 0.0;

    let textured_shader = program!(
        &display,
        140 => {
                vertex: "
                    #version 140
                    in vec3 position;
                    in vec2 tex_coords;
                    in vec4 colour;
                    out vec2 v_tex_coords;
                    out vec4 v_colour;
                    uniform mat4 modelview;
                    void main() {
                        gl_Position = modelview * vec4(position, 1.0);
                        v_tex_coords = tex_coords;
                        v_colour = colour;
                    }
                ",
    
                fragment: "
                    #version 140
                    uniform sampler2D tex;
                    in vec2 v_tex_coords;
                    // in vec4 gl_FragCoord;
                    in vec4 v_colour;
                    out vec4 f_colour;
                    void main() {
                        //f_colour = v_colour;
                        f_colour = texture(tex, v_tex_coords);
                    }
                "
        }
    )?;

    let coloured_shader = program!(
        &display,
        140 => {
                vertex: "
                    #version 140
                    in vec3 position;
                    //in vec2 tex_coords;
                    in vec4 colour;
                    //out vec2 v_tex_coords;
                    out vec4 v_colour;
                    uniform mat4 modelview;
                    void main() {
                        gl_Position = modelview * vec4(position, 1.0);
                        //v_tex_coords = tex_coords;
                        v_colour = colour;
                    }
                ",
    
                fragment: "
                    #version 140
                    //uniform sampler2D tex;
                    in vec2 v_tex_coords;
                    // in vec4 gl_FragCoord;
                    in vec4 v_colour;
                    out vec4 f_colour;
                    void main() {
                        f_colour = v_colour;
                    }
                "
        }
    )?;

    let _scale = display.gl_window().window().scale_factor();

    let vertex_buffer = glium::VertexBuffer::new(&display, &geometry::cube_vertices(1.0)).unwrap();

    let axes_buffer = glium::VertexBuffer::new(&display, &vec![
        geometry::Vertex {
            position: [-100.0, 0.0, 0.0],
            tex_coords: [0.0, 0.0],
            colour: [0.5, 0.0, 0.0, 1.0],
        },
        geometry::Vertex {
            position: [100.0, 0.0, 0.0],
            tex_coords: [0.0, 0.0],
            colour: [1.0, 0.0, 0.0, 1.0],
        },
        geometry::Vertex {
            position: [0.0, -100.0, 0.0],
            tex_coords: [0.0, 0.0],
            colour: [0.0, 0.2, 0.0, 1.0],
        },
        geometry::Vertex {
            position: [0.0, 100.0, 0.0],
            tex_coords: [0.0, 0.0],
            colour: [0.0, 1.0, 0.0, 1.0],
        },
        geometry::Vertex {
            position: [0.0, 0.0, -100.0],
            tex_coords: [0.0, 0.0],
            colour: [0.0, 0.0, 0.5, 1.0],
        },
        geometry::Vertex {
            position: [0.0, 0.0, 100.0],
            tex_coords: [0.0, 0.0],
            colour: [0.0, 0.0, 1.0, 1.0],
        }
    ]).unwrap();

    let mut time_of_last_frame = Instant::now();

    event_loop.run(move |event, _, control_flow| {
        let now = Instant::now();
        let dt_seconds: f32 = now.duration_since(time_of_last_frame).as_secs_f32();
        let should_render = dt_seconds > 1.0/30.0;

        match event {
            Event::WindowEvent { event, .. } => match event {
                WindowEvent::CloseRequested => { *control_flow = ControlFlow::Exit },
                WindowEvent::KeyboardInput { input: KeyboardInput { virtual_keycode: Some(key), state, .. }, .. } => {
                    let speed = 10.0;
                    match (key, state) {
                        (VirtualKeyCode::Escape, _) => { *control_flow = ControlFlow::Exit },
                        (VirtualKeyCode::W, ElementState::Pressed) => { velocity = cgmath::vec3(0.0, 0.0, 1.0) * speed; },
                        (VirtualKeyCode::W, ElementState::Released) => { velocity = cgmath::vec3(0.0, 0.0, 0.0); },

                        (VirtualKeyCode::S, ElementState::Pressed) => { velocity = cgmath::vec3(0.0, 0.0, -1.0) * speed; },
                        (VirtualKeyCode::S, ElementState::Released) => { velocity = cgmath::vec3(0.0, 0.0, 0.0); },

                        (VirtualKeyCode::A, ElementState::Pressed) => { velocity = cgmath::vec3(-1.0, 0.0, 0.0) * speed; },
                        (VirtualKeyCode::A, ElementState::Released) => { velocity = cgmath::vec3(0.0, 0.0, 0.0); },

                        (VirtualKeyCode::D, ElementState::Pressed) => { velocity = cgmath::vec3(1.0, 0.0, 0.0) * speed; },
                        (VirtualKeyCode::D, ElementState::Released) => { velocity = cgmath::vec3(0.0, 0.0, 0.0); },

                        (VirtualKeyCode::Left, ElementState::Pressed) => { distance -= 0.05 },

                        (VirtualKeyCode::Right, ElementState::Pressed) => { distance += 0.05 },

                        (VirtualKeyCode::Space, ElementState::Released) => { primitive = current_primitive_iter.next().unwrap(); },

                        (VirtualKeyCode::P, ElementState::Pressed) => { use_perspective = !use_perspective; },
                        (VirtualKeyCode::R, ElementState::Pressed) => {
                            let data = fs::read("/Users/jonatan/kuliga kodprojekt/oxide/src/voxels/assets/textures/grass_cube_atlas.png").unwrap();
                            let image = image::load(Cursor::new(&data[..]),
                                image::ImageFormat::Png).unwrap().to_rgba();
                            let image_dimensions = image.dimensions();
                            let image = glium::texture::RawImage2d::from_raw_rgba_reversed(&image.into_raw(), image_dimensions);
                            grass_texture = glium::texture::Texture2d::new(&display, image).unwrap();
                        },
                        _ => {}
                    }
                },
                WindowEvent::CursorMoved { position, .. } => {
                    rotation_y = 360.0 * std::f32::consts::PI/180.0 * ((position.x as f32)/(width as f32) - 0.5);
                },
                WindowEvent::MouseInput { state, button, .. } => {
                   match (button, state) {
                       _ => {}
                   }
                },
                _ => (),
            },
            Event::RedrawRequested(_) => {
                
            },
            _ => (),
        }

        // let next_frame_time = Instant::now() + Duration::from_millis(1000/60);

        if should_render {
            render_frame(&display, *primitive, use_perspective, distance, &textured_shader, &coloured_shader, &vertex_buffer, &axes_buffer, &grass_texture, velocity, &mut position, rotation_y, dt_seconds);
            time_of_last_frame = now;
        }
        
    
        // if !(*control_flow == glutin::event_loop::ControlFlow::Exit) {
        //     *control_flow = glutin::event_loop::ControlFlow::WaitUntil(next_frame_time);
        // }
    })
}

pub fn main() -> Result<(), Box<dyn Error>> {
    run()
}