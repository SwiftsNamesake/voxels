// Voxels
// A world of voxels

// Main focus
// - Implement proper camera controls
// - Dynamic lighting (eg. holding a torch that lights up the surrounding area and maybe flickers a bit)

// Architecture
// - Investigate some sort of entity-component system

// Professional Mode
// - Proper error handling
// - Proper logging

// - Nice debug tools, like coordinate systems with annotated offsets, chunk borders, bounding boxes, lines of sight, etc.
// - Reloading at runtime
// - Multiple rendering backends (Metal would be first in line)

// Extra fun
// - Mods, well-integrated into the game itself (with an API and so forth) 
// - Scripting, with an editor

// Lova's ideas
// - tiny blocks that you can use to make vehicles

// use std::path::Path;
use std::env;
use std::error::Error;
use std::time::{Instant};
use std::io::Cursor;
use std::fs;
use std::collections::{HashSet};
use std::iter::Iterator;

use glium::*;
use glium::{
    index::{ PrimitiveType },
};
use glutin::{
    event::{Event, KeyboardInput, VirtualKeyCode, WindowEvent, ElementState, DeviceEvent, MouseButton},
    event_loop::ControlFlow,
};

use cgmath::prelude::*;
use cgmath::{Matrix3, Matrix4, Vector3, Point3, Rad};

mod geometry;
mod voxel;
mod world;

pub struct Camera {
    pub position: Vector3<f32>, 
    pub orientation: Orientation, // TODO: Use quaternion?
}

impl Camera {
    /// Construct the matrix for the current state of the camera
    fn transform(&self) -> Matrix4<f32> {
        Matrix4::from_value(1.0)
    }
}

pub struct Inventory {}

pub struct Orientation {
    pub rotation_y: Rad<f32>, pub rotation_z: Rad<f32>
}

impl Orientation {
    fn forwards(&self) -> Vector3<f32> {
        Vector3::new(
            self.rotation_y.cos(),
            0.0, // We only want to walk along the x and z axes
            self.rotation_y.sin()
        ).normalize()
    }

    fn backwards(&self) -> Vector3<f32> { -self.forwards() }

    fn left(&self) -> Vector3<f32> { self.forwards().rotate(Rad(PI * 0.5)) }

    fn right(&self) -> Vector3<f32> { self.forwards().rotate(Rad(-PI * 0.5)) }

    fn looking_at(&self) -> Vector3<f32> {
        Vector3::new(
            self.rotation_y.cos(),
            self.rotation_z.cos(), // TODO: Make sure this is correct
            self.rotation_y.sin()
        ).normalize()
    }
}

// TODO: This is probably included in cgmath via a trait somehow, so we really should be using that
trait Rot {
    fn rotate<T: Into<Rad<f32>>>(&self, angle: T) -> Self;
}

impl Rot for Vector3<f32> {
    fn rotate<T: Into<Rad<f32>>>(&self, angle: T) -> Self {
        Matrix3::from_angle_y(angle) * self
    }
}

pub struct Actor {
    // Extrinsic properties
    pub position: Vector3<f32>,
    pub velocity: Vector3<f32>, // Walk velocity
    pub jump_velocity: f32,
    pub orientation: Orientation,

    // Intrinsic properties
    pub speed: f32,
}

pub struct AssetCatalogue {
    pub sounds: (),
    pub textures: (),
}

pub struct GraphicsCardResources {
    pub textured_shader: glium::Program,
    pub coloured_shader: glium::Program,
    pub cube_vertex_buffer: glium::VertexBuffer<geometry::Vertex>,
    pub axes_vertex_buffer: glium::VertexBuffer<geometry::Vertex>,
    pub texture_atlas: glium::texture::SrgbTexture2d,
}

pub struct Settings {
    primitive: glium::index::PrimitiveType,
    use_perspective: bool, // TODO: Replace with something cleverer (enum, function, etc)
}

#[derive(Copy, Clone, Debug)]
struct CubeInstanceAttributes {
    world_position: (f32, f32, f32),
    texture_offset: (f32, f32),
}
implement_vertex!(CubeInstanceAttributes, world_position, texture_offset);

static PI: f32 = std::f32::consts::PI;

fn create_world() -> voxel::World {
    use voxel::{Block, Chunk};

    return (0 .. 10)
        .map(|ichunk| {
            let mut chunk: Chunk = vec![];

            for layer in 0 .. 32 {
                let block = match layer {
                    0 => Block::AIR,
                    1 => Block::AIR,
                    2 => Block::AIR,
                    3 => Block::AIR,
                    4 => Block::AIR,
                    5 => Block::AIR,
                    6 => Block::AIR,
                    7 => Block::STONE,
                    8 => Block::AIR,
                    9 => Block::AIR,
                    10 => Block::AIR,
                    11 => Block::GRASS,
                    12 => Block::DIRT,
                    13 => Block::DIRT,
                    14 => Block::DIRT,
                    15 => Block::DIRT,
                    _ => Block::STONE,
                };

                chunk.push(
                    [
                        [block, block, block, block, block, block, block, block, block, block, block, block, block, block, block, block],
                        [block, block, block, block, block, block, block, block, block, block, block, block, block, block, block, block],
                        [block, block, block, block, block, block, block, block, block, block, block, block, block, block, block, block],
                        [block, block, block, block, block, block, block, block, block, block, block, block, block, block, block, block],
                        [block, block, block, block, block, block, block, block, block, block, block, block, block, block, block, block],
                        [block, block, block, block, block, block, block, block, block, block, block, block, block, block, block, block],
                        [block, block, block, block, block, block, block, block, block, block, block, block, block, block, block, block],
                        [block, block, block, block, block, block, block, block, block, block, block, block, block, block, block, block],
                        [block, block, block, block, block, block, block, block, block, block, block, block, block, block, block, block],
                        [block, block, block, block, block, block, block, block, block, block, block, block, block, block, block, block],
                        [block, block, block, block, block, block, block, block, block, block, block, block, block, block, block, block],
                        [block, block, block, block, block, block, block, block, block, block, block, block, block, block, block, block],
                        [block, block, block, block, block, block, block, block, block, block, block, block, block, block, block, block],
                        [block, block, block, block, block, block, block, block, block, block, block, block, block, block, block, block],
                        [block, block, block, block, block, block, block, block, block, block, block, block, block, block, block, block],
                        [block, block, block, block, block, block, block, block, block, block, block, block, block, block, block, block]
                    ]
                );
            }

            chunk
        })
        .collect();
}

fn texture_offset_for_block(block: voxel::Block) -> (f32, f32) {
    match block {
        voxel::Block::GRASS => (0.0, 5.0 * 1.0 / 6.0),
        voxel::Block::STONE => (0.0, 4.0 * 1.0 / 6.0),
        voxel::Block::DIRT  => (0.0, 3.0 * 1.0 / 6.0),
        _ => (0.0, 0.0)
    }
}

fn tick<'a>(keyboard: &'a HashSet::<VirtualKeyCode>, actor: &'a mut Actor, world: &'a mut voxel::World, dt_seconds: f32) {
    // use VirtualKeyCode::*;
    // if keyboard.contains(&W) {
    //     actor.velocity = actor.orientation.forwards() * actor.speed + cgmath::vec3(0.0, actor.velocity.y, 0.0);
    // }

    // Animate
    // TODO: Define actor bounding-box properly
    let ichunk: usize = (actor.position.x / 16.0).floor() as usize;
    let chunk_x = (actor.position.x as usize) % 16;
    let chunk_z = (actor.position.z as usize) % 16;
    if ichunk < world.len() {
        println!("{} {} {}", ichunk, chunk_x, chunk_z);
        let y_of_lowest_solid = world[ichunk]
            .iter()
            .enumerate()
            .filter_map(|(i, layer)|
                if (-11.0 + i as f32) < actor.position.y && layer[chunk_x][chunk_z] != voxel::Block::AIR {
                    Some(-11.0 + i as f32)
                } else {
                    None
                })
            .last()
            .unwrap_or(-11.0);

        println!("{:?}", y_of_lowest_solid);
        println!("{:?}", actor.position);

        if (actor.position.y - 2.0) > y_of_lowest_solid {
            let gravity = cgmath::vec3(0.0, -9.82, 0.0);
            actor.velocity += gravity * dt_seconds;
            actor.position += actor.velocity * dt_seconds;
            if (actor.position.y - 2.0) <= (y_of_lowest_solid) {
                actor.velocity.y = 0.0;
                actor.position.y = y_of_lowest_solid + 2.0;
            }
        } else {
            actor.position += actor.velocity * dt_seconds;
        }
    }
}

///
fn make_cube_instance(kind: voxel::Block, world_position: (f32, f32, f32)) -> Option<CubeInstanceAttributes> {
    if kind == voxel::Block::AIR {
        Option::None
    } else {
        Some(CubeInstanceAttributes {
            world_position: world_position,
            texture_offset: texture_offset_for_block(kind)
        })
    }
}

///
fn make_chunk_instance_attributes<'a>(ichunk: usize, chunk: &'a voxel::Chunk) -> impl Iterator<Item = CubeInstanceAttributes> + 'a {
    let chunk_origin = Vector3::new((ichunk * 16) as f32, 0.0, 0.0);

    chunk
        .iter()
        .enumerate()
        .flat_map(move |(ilayer, layer)| {
            return layer
                .iter()
                .enumerate()
                .flat_map(move |(x, row)| {
                    row
                        .iter()
                        .enumerate()
                        .filter_map(move |(z, block)| {
                            let cube_offset_in_chunk = Vector3::new(x as f32 + 0.5, 11.0 - ilayer as f32 - 0.5, z as f32 + 0.5);
                            make_cube_instance(*block, (chunk_origin + cube_offset_in_chunk).into())
                        })
                })
        })
}

fn render_frame<'a>(
    display: &'a Display,
    resources: &'a GraphicsCardResources,
    settings: &'a Settings,
    actor: &'a Actor,
    world: &'a Vec<voxel::Chunk>,
) {
    let bla = match 5 {
        0 => "hello",
        1 => "world",
        2 => "hola",
        _ => "no idea"
    };

    // Some settings
    let sky_colour = (0.0/255.0, 206.0/255.0, 237.0/255.0, 1.0);

    // Render
    let mut target = display.draw();
    target.clear_color_and_depth(sky_colour, 1.0);

    let (screen_width, screen_height) = {
        let (w, h) = display.get_framebuffer_dimensions();
        (w as f32, h as f32)
    };

    let field_of_view = Rad(45.0f32 * PI / 180.0);
    let aspect_ratio = screen_width/screen_height;
    let perspective_matrix = cgmath::perspective(field_of_view, aspect_ratio, 0.1f32, 100.0f32);

    let orthographic = cgmath::ortho(-10.0, 10.0, -10.0, 10.0, 0.1f32, 100.0f32);

    let projection_matrix = if settings.use_perspective { perspective_matrix } else { orthographic };

    let view_matrix = Matrix4::look_to_rh(
        Point3 { x: actor.position.x, y: actor.position.y, z: actor.position.z },
        Vector3::new(
            actor.orientation.rotation_y.cos(),
            actor.orientation.rotation_z.sin(),
            actor.orientation.rotation_y.sin()
        ).normalize(),
        Vector3::unit_y(),
    );

    let render_distance_x: i32 = 32;
    let render_distance_z: i32 = 32;

    // TODO: We'll need to optimise the hell out of this eventually...
    // TODO: Factor out coordinate logic to make it more readable and re-usable.
    // TODO: Caching? Is there a way of caching per-chunk and then concatenating the results?
    let per_instance = {
        let data: Vec<CubeInstanceAttributes> = world
            .iter()
            .enumerate()
            .flat_map(|(ichunk, chunk)| { make_chunk_instance_attributes(ichunk, chunk) })
            .collect();
        glium::VertexBuffer::dynamic(display, &data).unwrap()
    };

    let light_position: [f32; 3] = [0.0, 8.0, 0.0];

    let uniforms = uniform! {
        texture: resources.texture_atlas.sampled().magnify_filter(glium::uniforms::MagnifySamplerFilter::Nearest),
        projection: Into::<[[f32; 4]; 4]>::into(projection_matrix),
        view: Into::<[[f32; 4]; 4]>::into(view_matrix),
        light_position: Into::<[f32; 3]>::into(actor.position),
    };

    target.draw(
        (&resources.cube_vertex_buffer, per_instance.per_instance().unwrap()),
        glium::index::NoIndices(settings.primitive),
        &resources.textured_shader,
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

    let model_matrix = Matrix4::from_translation(cgmath::vec3(0.0, 0.0, 0.0));
    let modelview: [[f32; 4]; 4] = projection_matrix.concat(&view_matrix).concat(&model_matrix).into();
    let axes_uniforms = uniform! {
        modelview: modelview
    };

    target.draw(
        &resources.axes_vertex_buffer,
        glium::index::NoIndices(glium::index::PrimitiveType::LinesList),
        &resources.coloured_shader,
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

    // let held_block_uniforms = uniform! {
    //     texture: resources.texture_atlas.sampled().magnify_filter(glium::uniforms::MagnifySamplerFilter::Nearest),
    //     projection: Into::<[[f32; 4]; 4]>::into(projection_matrix),
    //     view: Into::<[[f32; 4]; 4]>::into(view_matrix),
    //     light_position: Into::<[f32; 3]>::into(actor.position),
    // };

    // let held_block_data = vec![
    //     CubeInstanceAttributes {
    //         world_position: (actor.position + actor.orientation.forwards() * 0.2).into(),
    //         texture_offset: (0.0, 5.0 * 1.0 / 6.0)
    //     }
    // ];

    // let held_block_attr_buffer = glium::VertexBuffer::dynamic(
    //     display,
    //     &held_block_data
    // );

    // target.draw(
    //     (&resources.cube_vertex_buffer, held_block_attr_buffer.unwrap().per_instance().unwrap()),
    //     glium::index::NoIndices(glium::index::PrimitiveType::TrianglesList),
    //     &resources.textured_shader,
    //     &held_block_uniforms,
    //     &glium::DrawParameters {
    //         depth: glium::Depth {
    //             test: glium::draw_parameters::DepthTest::IfLess,
    //             write: true,
    //             .. Default::default()
    //         },
    //         blend: glium::Blend::alpha_blending(),
    //         multisampling: true,
    //         ..Default::default()
    //     },
    // ).unwrap();

    target.finish().unwrap();
}

/// 
// TODO: error handling
pub fn load_texture(display: &glium::Display, path: std::string::String) -> glium::texture::SrgbTexture2d {
    let data = fs::read(path).unwrap();
    let image = image::load(Cursor::new(&data[..]), image::ImageFormat::Png).unwrap().to_rgba();
    let image_dimensions = image.dimensions();
    let image = glium::texture::RawImage2d::from_raw_rgba_reversed(&image.into_raw(), image_dimensions);
    return glium::texture::SrgbTexture2d::new(display, image).unwrap();
}

pub fn run() -> Result<(), Box<dyn Error>> {
    if cfg!(target_os = "linux") && env::var("WINIT_UNIX_BACKEND").is_err() {
        env::set_var("WINIT_UNIX_BACKEND", "x11");
    }

    let width = 720*3;
    let height = 480*3;

    let window = glium::glutin::window::WindowBuilder::new()
        .with_inner_size(glium::glutin::dpi::PhysicalSize::new(width, height))
        .with_title("Voxels");

    let context = glium::glutin::ContextBuilder::new()
        .with_vsync(true)
        .with_depth_buffer(24)
        .with_multisampling(4);

    let event_loop = glium::glutin::event_loop::EventLoop::new();
    let display = glium::Display::new(window, context, &event_loop)?;

    let mut player = Actor {
          position: cgmath::vec3(2.0, 2.0, 0.0)
        , velocity: cgmath::vec3(0.0, 0.0, 0.0)
        , jump_velocity: 4.3
        , orientation: Orientation { rotation_y: Rad(0.0), rotation_z: Rad(0.0) }

        , speed: 6.0
    };

    let mut keyboard = HashSet::<VirtualKeyCode>::default();

    let mut current_primitive_iter = [PrimitiveType::TrianglesList, PrimitiveType::LinesList, PrimitiveType::Points].iter().cycle();
    let mut primitive = current_primitive_iter.next().unwrap();

    let textured_shader = program!(
        &display,
        140 => {
            vertex: "
                #version 140

                in vec3 position;
                in vec2 tex_coords;
                in vec4 colour;

                // Per instance
                in vec3 world_position;
                in vec2 texture_offset;

                uniform mat4 projection;
                uniform mat4 view;

                out vec2 v_tex_coords;
                out vec4 v_colour;
                out vec3 pos;

                uniform vec3 light_position;

                vec2 atlas_size = vec2(6.0, 6.0);
                vec2 cube_face_size = vec2(192.0/atlas_size.x, 192.0/atlas_size.y);

                void main() {
                    gl_Position = projection * view * vec4(position + world_position, 1.0);
                    // world_position = vec4(position + world_position, 1.0);
                    pos = position + world_position;
                    v_tex_coords = vec2(tex_coords.x, tex_coords.y / atlas_size.y) + texture_offset;
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

                uniform vec3 light_position;
                // in vec4 world_position;
                in vec3 pos;

                // Converts a color from sRGB gamma to linear light gamma
                vec4 toLinear(vec4 sRGB) {
                    bvec4 cutoff = lessThan(sRGB, vec4(0.04045));
                    vec4 higher = pow((sRGB + vec4(0.055))/vec4(1.055), vec4(2.4));
                    vec4 lower = sRGB/vec4(12.92);

                    return mix(higher, lower, cutoff);
                }

                void main() {
                    //f_colour = v_colour;
                    float d = distance(pos, light_position);
                    // float light_intensity = clamp(1.0/(d*d*0.1*0.1), 0.0, 1.0);
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
    
    let mut time_of_last_frame = Instant::now();

    let maximum_frames_per_second: f32 = 30.0;

    let cube_vertex_buffer = glium::VertexBuffer::new(&display, &geometry::cube_vertices(1.0)).unwrap();
    let axes_vertex_buffer = glium::VertexBuffer::new(&display, &geometry::perpendicular_axes_xyz(200.0, 200.0, 200.0)).unwrap();

    let texture_atlas = load_texture(&display, "/Users/jonatan/kuliga kodprojekt/oxide/src/voxels/assets/textures/texture_atlas_layers.png".to_string());

    let mut graphics_resources = GraphicsCardResources {
        textured_shader: textured_shader
      , coloured_shader: coloured_shader
      , cube_vertex_buffer: cube_vertex_buffer
      , axes_vertex_buffer: axes_vertex_buffer
      , texture_atlas: texture_atlas
    };

    let mut world = create_world();

    event_loop.run(move |event, _, control_flow| {
        let now = Instant::now();
        let dt_seconds: f32 = now.duration_since(time_of_last_frame).as_secs_f32();
        let should_render = dt_seconds > 1.0/maximum_frames_per_second;

        use VirtualKeyCode::*;
        use ElementState::*;

        match event {
            Event::WindowEvent { event, .. } => match event {
                WindowEvent::CloseRequested => { *control_flow = ControlFlow::Exit },
                WindowEvent::KeyboardInput { input: KeyboardInput { virtual_keycode: Some(key), state, .. }, .. } => {
                    match (key, state) {
                        (Escape, Pressed) => { *control_flow = ControlFlow::Exit },

                        (W, Pressed) => { player.velocity = player.orientation.forwards() * player.speed; },
                        (W, Released) => { player.velocity -= player.orientation.forwards() * player.speed; },

                        (S, Pressed) => { player.velocity = player.orientation.backwards() * player.speed; },
                        (S, Released) => { player.velocity = cgmath::vec3(0.0, 0.0, 0.0); },

                        (A, Pressed) => { player.velocity = player.orientation.left() * player.speed; },
                        (A, Released) => { player.velocity = cgmath::vec3(0.0, 0.0, 0.0); },

                        (D, Pressed) => { player.velocity = player.orientation.right() * player.speed; },
                        (D, Released) => { player.velocity = cgmath::vec3(0.0, 0.0, 0.0); },

                        (Space, Released) => {
                            player.velocity = cgmath::vec3(0.0, player.jump_velocity, 0.0);
                        },

                        (Down, Released) => {
                            primitive = current_primitive_iter.next().unwrap();
                        },

                        // (P, Pressed) => { use_perspective = !use_perspective; },

                        (R, Pressed) => {
                            graphics_resources.texture_atlas = load_texture(&display, "/Users/jonatan/kuliga kodprojekt/oxide/src/voxels/assets/textures/texture_atlas_layers.png".to_string());
                        },
                        _ => {}
                    }

                    match state {
                        Pressed => { keyboard.insert(key); }
                        Released => { keyboard.remove(&key); }
                    }
                },
                WindowEvent::CursorMoved { .. } => {},

                WindowEvent::MouseInput { state, button, .. } => {
                   match (button, state) {
                        (MouseButton::Left, ElementState::Pressed) => {
                            // TODO: Factor out all coordinate calculations
                            let chunk_x: usize = (player.position.x as usize) / 16;
                            let chunk_z: usize = 0; // (player.position.x as usize) / 16;

                            // TODO: Ray intersection with world to find out what cube we're looking at.

                            // TODO: Factor out block placing logic
                            if chunk_z == 0 && chunk_x < world.len() {
                                world[chunk_x][0][(player.position.x as usize) % 16][(player.position.z as usize) % 16] = voxel::Block::DIRT;
                            }
                        },
                       _ => {}
                   }
                },
                _ => (),
            },
            Event::DeviceEvent { event, .. } => {
                match event {
                    DeviceEvent::MouseMotion { delta } => {
                        let (screen_width, screen_height) = {
                            let (w, h) = display.get_framebuffer_dimensions();
                            (w as f32, h as f32)
                        };

                        // TODO: Implement proper mouse sensitivity settings. There's no reason to tie it to the screen dimensions.

                        let normalised_x_delta = (delta.0 as f32)/(screen_width);
                        player.orientation.rotation_y += Rad(2.0 * PI * normalised_x_delta);
    
                        let normalised_y_delta = (delta.1 as f32) * 1.0 / 400.0;
                        player.orientation.rotation_z = Rad((player.orientation.rotation_z.0 + 1.0 * PI * normalised_y_delta).clamp(-PI/2.0, PI/2.0));
                    },
                    _ => {},
                }
            }
            _ => (),
        }

        if should_render {
            tick(&keyboard, &mut player, &mut world, dt_seconds);
            render_frame(
                &display,
                &graphics_resources,
                &Settings { use_perspective: true, primitive: *primitive },
                &player,
                &world,
            );
            time_of_last_frame = now;
        }
    })
}

pub fn main() -> Result<(), Box<dyn Error>> {
    run()
}
