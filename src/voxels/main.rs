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

use std::path::Path;
use std::ops::{Mul};
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
use cgmath::{Matrix3, Matrix4, Vector3, Point3, Rad, Deg};

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

pub struct Orientation {
    pub rotation_y: Rad<f32>, pub rotation_z: Rad<f32>
}

impl Orientation {
    fn forwards(&self) -> Vector3<f32> {
        // let f = Matrix3::from_axis_angle(cgmath::vec3(0.0, 1.0, 0.0), self.rotation_y) * cgmath::vec3(1.0, 0.0, 0.0);
        // println!("{:?}", f);
        // return f;
        Vector3::new(
            self.rotation_y.cos(),
            0.0, // We only want to walk along the x and z axes
            self.rotation_y.sin()
        ).normalize()
        // cgmath::vec3(self.x.cos(), 0.0, self.x.sin())
        // let forwards = Vector3 { y: 0.0, ..player_orientation };
        // let rotation_angle = forwards.angle(cgmath::vec3(1.0, 0.0, 0.0));
    }

    // fn backwards(&self) -> Vector3<f32>
    // fn left(&self) -> Vector3<f32>
    // fn right(&self) -> Vector3<f32>
}

static PI: f32 = std::f32::consts::PI;

fn render_frame(
    display: &glium::Display,
    primitive: glium::index::PrimitiveType,
    use_perspective: bool,
    textured_shader: &glium::Program,
    coloured_shader: &glium::Program,
    vertex_buffer: &glium::VertexBuffer<geometry::Vertex>,
    axes_buffer: &glium::VertexBuffer<geometry::Vertex>,
    grass_texture: &glium::texture::Texture2d,
    velocity: Vector3<f32>, player_position: &mut Vector3<f32>, player_orientation: &Orientation, dt_seconds: f32
) {
    let chunk: voxel::Chunk = vec![
        vec![2, 2, 3, 3],
        vec![2, 2, 3, 3],
        vec![2, 2, 3, 3],
        vec![2, 2, 3, 3],
    ];

    // Animate
    *player_position += velocity * dt_seconds;

    // Some settings
    let sky_colour = (0.0/255.0, 206.0/255.0, 237.0/255.0, 1.0);

    // Render
    let mut target = display.draw();
    target.clear_color_and_depth(sky_colour, 1.0);

    let (_screen_width, _screen_height) = {
        let (w, h) = display.get_framebuffer_dimensions();
        (w as f32, h as f32)
    };

    let perspective_matrix = cgmath::perspective(
        Rad(45.0f32 * PI / 180.0),
        _screen_width/_screen_height,
        0.1f32,
        100.0f32
    );

    let orthographic = cgmath::ortho(-10.0, 10.0, -10.0, 10.0, 0.1f32, 100.0f32);

    let projection_matrix = if use_perspective { perspective_matrix } else { orthographic };

    let render_distance_x = 16;
    let render_distance_z = 16;

    // let orientation_xz = player_orientation.forwards().normalize();
    // let rotation_angle_y_axis = cgmath::vec3(1.0, 0.0, 0.0).angle(orientation_xz);
    // println!("{:?}", rotation_angle_y_axis);
    let camera_rotation = Matrix4::from_axis_angle(cgmath::vec3(0.0, 1.0, 0.0).normalize(), player_orientation.rotation_y);
 
    println!("{:?}", player_position);
    // let camera_translation = Matrix4::from_translation(*player_position);
    //let view_matrix = camera_rotation
    //    .concat(&Matrix4::from_translation(-*player_position));
    // let camera_matrix = camera_rotation * camera_translation;
    // let view_matrix = camera_matrix.inverse_transform().unwrap();
    let view_matrix = Matrix4::look_to_rh(
        Point3 { x: player_position.x, y: player_position.y, z: player_position.z },
        Vector3::new(
            player_orientation.rotation_y.cos(),
            player_orientation.rotation_z.sin(),
            player_orientation.rotation_y.sin()
        ).normalize(),
        Vector3::unit_y(),
    );

    for x in -render_distance_x .. render_distance_x {
        for z in -render_distance_z .. render_distance_z {
            let translation = cgmath::vec3(x as f32, 0.0, z as f32);

            let model_matrix = Matrix4::from_translation(translation);
            let modelview: [[f32; 4]; 4] = (projection_matrix * view_matrix * model_matrix).into();
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

    let model_matrix = Matrix4::from_translation(cgmath::vec3(0.0, 0.0, 0.0));
    let modelview: [[f32; 4]; 4] = projection_matrix.concat(&view_matrix).concat(&model_matrix).into();
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

/// 
// TODO: error handling
pub fn load_texture(display: &glium::Display, path: std::string::String) -> glium::texture::Texture2d {
   let data = fs::read(path).unwrap();
    let image = image::load(Cursor::new(&data[..]),
    image::ImageFormat::Png).unwrap().to_rgba();
    let image_dimensions = image.dimensions();
    let image = glium::texture::RawImage2d::from_raw_rgba_reversed(&image.into_raw(), image_dimensions);
    return glium::texture::Texture2d::new(display, image).unwrap();
}

pub fn run() -> Result<(), Box<dyn Error>> {
    if cfg!(target_os = "linux") && env::var("WINIT_UNIX_BACKEND").is_err() {
        env::set_var("WINIT_UNIX_BACKEND", "x11");
    }

    let width = 720*2;
    let height = 480*2;

    let window = glium::glutin::window::WindowBuilder::new()
        .with_inner_size(glium::glutin::dpi::PhysicalSize::new(width, height))
        .with_title("voxels");

    let context = glium::glutin::ContextBuilder::new()
        .with_vsync(true)
        .with_depth_buffer(24)
        .with_multisampling(4);
    let event_loop = glium::glutin::event_loop::EventLoop::new();
    let display = glium::Display::new(window, context, &event_loop)?;

    let mut grass_texture = load_texture(&display, "/Users/jonatan/kuliga kodprojekt/oxide/src/voxels/assets/textures/grass_cube_atlas.png".to_string());

    let mut player_orientation = Orientation { rotation_y: Rad(0.0), rotation_z: Rad(0.0) }; // (0.0, 0.0); // cgmath::vec3(0.0, 0.0, -1.0).normalize();
    let player_speed: f32 = 10.0;
    let mut position = cgmath::vec3(0.0, 2.0, 0.0);
    let mut velocity = cgmath::vec3(0.0, 0.0, 0.0);

    let mut current_primitive_iter = [PrimitiveType::TrianglesList, PrimitiveType::LinesList, PrimitiveType::Points].iter().cycle();
    let mut primitive = current_primitive_iter.next().unwrap();

    let mut use_perspective = true;

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
                    match (key, state) {
                        (VirtualKeyCode::Escape, ElementState::Pressed) => { *control_flow = ControlFlow::Exit },

                        (VirtualKeyCode::W, ElementState::Pressed) => { velocity = player_orientation.forwards() * player_speed; },
                        (VirtualKeyCode::W, ElementState::Released) => { velocity = cgmath::vec3(0.0, 0.0, 0.0); },

                        (VirtualKeyCode::S, ElementState::Pressed) => { velocity = -1.0 * player_orientation.forwards() * player_speed; },
                        (VirtualKeyCode::S, ElementState::Released) => { velocity = cgmath::vec3(0.0, 0.0, 0.0); },

                        // (VirtualKeyCode::A, ElementState::Pressed) => { velocity = cgmath::vec3(-1.0, 0.0, 0.0) * player_speed; },
                        // (VirtualKeyCode::A, ElementState::Released) => { velocity = cgmath::vec3(0.0, 0.0, 0.0); },

                        // (VirtualKeyCode::D, ElementState::Pressed) => { velocity = cgmath::vec3(1.0, 0.0, 0.0) * player_speed; },
                        // (VirtualKeyCode::D, ElementState::Released) => { velocity = cgmath::vec3(0.0, 0.0, 0.0); },

                        (VirtualKeyCode::Space, ElementState::Released) => { primitive = current_primitive_iter.next().unwrap(); },

                        (VirtualKeyCode::P, ElementState::Pressed) => { use_perspective = !use_perspective; },

                        (VirtualKeyCode::R, ElementState::Pressed) => {
                            grass_texture = load_texture(&display, "/Users/jonatan/kuliga kodprojekt/oxide/src/voxels/assets/textures/grass_cube_atlas.png".to_string());
                        },
                        _ => {}
                    }
                },
                WindowEvent::CursorMoved { position, .. } => {
                    // println!("{:?}", position.y);
                    let (_screen_width, _screen_height) = {
                        let (w, h) = display.get_framebuffer_dimensions();
                        (w as f32, h as f32)
                    };
                    // let rotation_y_matrix = Matrix4::from_axis_angle(cgmath::vec3(0.0, 1.0, 0.0), Deg(0.0));
                    // player_orientation = rotation_y_matrix * cgmath::Vector4::<f32> { x: 0.0, y: 0.0, z: 0.0, w: 0.0 };
                    let normalised_x_along_screen = 1.0 - (position.x as f32)/(_screen_width);
                    player_orientation.rotation_y = Rad(2.0 * PI * normalised_x_along_screen);

                    let normalised_y_along_screen = 1.0 - (position.y as f32)/(_screen_width);
                    player_orientation.rotation_z = Rad(2.0 * PI * normalised_y_along_screen);
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
            render_frame(&display, *primitive, use_perspective, &textured_shader, &coloured_shader, &vertex_buffer, &axes_buffer, &grass_texture, velocity, &mut position, &player_orientation, dt_seconds);
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