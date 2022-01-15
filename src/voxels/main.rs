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
use std::borrow::Cow;

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

use rusttype::gpu_cache::Cache;
use rusttype::{point, vector, Font, PositionedGlyph, Rect, Scale, Point};

mod geometry;
mod voxel;
mod world;
mod ui;
mod text;

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
    pub position: Vector3<f32>, // Assumed to be the geometric centre.
    pub velocity: Vector3<f32>, // Walk velocity
    pub jump_velocity: f32,
    pub orientation: Orientation,

    // Intrinsic properties
    pub speed: f32, /// This is the speed at which the actor moves when walking.
    pub size: Vector3<f32>
}

pub struct AssetCatalogue {
    pub sounds: (),
    pub textures: (),
}

pub struct GraphicsCardResources<'a> {
    pub textured_shader: glium::Program,
    pub coloured_shader: glium::Program,
    pub monochrome_shader: glium::Program,
    pub textured_array_shader: glium::Program,

    pub cube_vertex_buffer: glium::VertexBuffer<geometry::Vertex>,
    pub axes_vertex_buffer: glium::VertexBuffer<geometry::Vertex>,

    pub texture_atlas: glium::texture::SrgbTexture2d,
    pub textures_array: glium::texture::SrgbTexture2dArray,

    pub shapes: Vec<((VertexBuffer<geometry::Vertex>, IndexBuffer<u16>), Vec<(Vector3<f32>, [f32; 4])>)>,

    pub font: Font<'a>,
    pub glyph_cache: Cache<'a>,
    pub glyph_cache_texture: glium::Texture2d,
    pub text_vertex_buffer: glium::VertexBuffer<text::Vertex>,
    pub text_quad_vertex_buffer: glium::VertexBuffer<geometry::Vertex>,
}

pub struct Settings {
    primitive: glium::index::PrimitiveType,
    use_perspective: bool, // TODO: Replace with something cleverer (enum, function, etc)
}

#[derive(Copy, Clone, Debug)]
struct CubeInstanceAttributes {
    world_position: (f32, f32, f32),
    texture_offset: (f32, f32),
    texture_index: f32,
}
implement_vertex!(CubeInstanceAttributes, world_position, texture_offset, texture_index);

static PI: f32 = std::f32::consts::PI;

fn create_world() -> voxel::World {
    use voxel::{Block};

    (0 .. 1)
        .map(|_ichunk| {
            (0 .. 32).map(|layer| {
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

                [[block; 16]; 16]
            }).collect()
        })
        .collect()
}

//fn texture_offset_for_block(block: voxel::Block) -> (f32, f32) {
//    match block {
//        voxel::Block::GRASS => (0.0, 5.0 * 1.0 / 6.0),
//        voxel::Block::STONE => (0.0, 4.0 * 1.0 / 6.0),
//        voxel::Block::DIRT  => (0.0, 3.0 * 1.0 / 6.0),
//        _ => (0.0, 0.0)
//    }
//}

fn texture_index_for_block(block: voxel::Block) -> f32 {
    match block {
        voxel::Block::DIRT  => 0.0,
        voxel::Block::GRASS => 1.0,
        voxel::Block::STONE => 2.0,
        _ => 0.0
    }
}

fn tick<'a>(keyboard: &'a HashSet::<VirtualKeyCode>, actor: &'a mut Actor, world: &'a mut voxel::World, dt_seconds: f32) {
    // use VirtualKeyCode::*;
    // if keyboard.contains(&W) {
    //     actor.velocity = actor.orientation.forwards() * actor.speed + cgmath::vec3(0.0, actor.velocity.y, 0.0);
    // }

    // Animate
    // TODO: Define actor bounding-box properly
    // TODO: Deal with collisions across chunk boundaries
    // TODO: Horizontal collisions
    let ichunk: usize = (actor.position.x / 16.0).floor() as usize;
    let chunk_x = (actor.position.x as usize) % 16;
    let chunk_z = (actor.position.z as usize) % 16;
    if ichunk < world.len() {
        println!("{} {} {}", ichunk, chunk_x, chunk_z);
        let current_chunk = &world[ichunk];
        let chunk_y = (current_chunk.len() as f32 - actor.position.y) as usize; // TODO: make sure this is correct

        let block_below_actor = current_chunk[chunk_y][chunk_x][chunk_z];

        let gravity = cgmath::vec3(0.0, -9.82, 0.0);

        if block_below_actor != voxel::Block::AIR {
            actor.velocity += gravity * dt_seconds;
            actor.position += actor.velocity * dt_seconds;
            if (actor.position.y - 2.0) <= (chunk_y as f32) {
                actor.velocity.y = 0.0;
                actor.position.y = (chunk_y as f32) + 2.0;
            }
        } else {
            actor.velocity += gravity * dt_seconds;
            actor.position += actor.velocity * dt_seconds;
        }
    } else {
        actor.position += actor.velocity * dt_seconds;
    }
}

/// Constructs `CubeInstanceAttributes` that can be used to render a voxel at a specific position in the world, without duplicating geometry data.
fn make_cube_instance(kind: voxel::Block, world_position: (f32, f32, f32)) -> Option<CubeInstanceAttributes> {
    if kind == voxel::Block::AIR {
        Option::None
    } else {
        Some(CubeInstanceAttributes {
            world_position: world_position,
            texture_offset: (0.0, 0.0), // texture_offset_for_block(kind),
            texture_index: texture_index_for_block(kind),
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
            layer
                .iter()
                .enumerate()
                .flat_map(move |(x, row)| {
                    row
                        .iter()
                        .enumerate()
                        .filter_map(move |(z, block)| {
                            // TODO: Factor out this mapping between 'world indices' to world coordinates. IMPORTANT!!!  n0
                            let cube_offset_in_chunk = Vector3::new(x as f32 + 0.5, chunk.len() as f32 - ilayer as f32 - 0.5, z as f32 + 0.5);
                            make_cube_instance(*block, (chunk_origin + cube_offset_in_chunk).into())
                        })
                })
        })
}

fn render_frame<'a>(
    display: &'a Display,
    resources: &'a mut GraphicsCardResources,
    settings: &'a Settings,
    actor: &'a Actor,
    world: &'a Vec<voxel::Chunk>,
    program_start_time: Instant
) {
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
        texture: resources.textures_array
            .sampled()
            .minify_filter(glium::uniforms::MinifySamplerFilter::Nearest)
            .magnify_filter(glium::uniforms::MagnifySamplerFilter::Nearest),
        projection: Into::<[[f32; 4]; 4]>::into(projection_matrix),
        view: Into::<[[f32; 4]; 4]>::into(view_matrix),
        light_position: Into::<[f32; 3]>::into(actor.position),
    };

    // Draw all the voxels
    target.draw(
        (&resources.cube_vertex_buffer, per_instance.per_instance().unwrap()),
        glium::index::NoIndices(settings.primitive),
        &resources.textured_array_shader,
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
            ..Default::default()
        },
    ).unwrap();

    let ui_projection = cgmath::ortho(-screen_width * 0.5, screen_width * 0.5, -screen_height * 0.5, screen_height * 0.5, -0.1f32, 100.0f32);

    // Render the u (just a rounded rectangle for now)
    for (shape, instances) in &resources.shapes {
        for (position, colour) in instances {
            let translation = Matrix4::from_translation(*position);
            let ui_uniforms = uniform! {
                modelview: Into::<[[f32; 4]; 4]>::into(ui_projection.concat(&translation)),
                colour: *colour
            };
    
            target.draw(
                &shape.0,
                &shape.1,
                &resources.monochrome_shader,
                &ui_uniforms,
                &Default::default(),
            ).unwrap();
        }
    }

    // Render text, just to see if it works (should be part of UI system later, and maybe for signs and things in-the-world too)
    if false {
        text::create_text_vertices(display, &resources.font, &mut resources.glyph_cache, 1.0, &resources.glyph_cache_texture, 20, &vec!["Once upon a time".to_string(), "in Mexico".to_string()]);
        let translation = Matrix4::from_translation(cgmath::vec3(0.0, 20.0, 0.0));
        let text_uniforms = uniform! {
            texture: resources.glyph_cache_texture
                .sampled()
                .magnify_filter(glium::uniforms::MagnifySamplerFilter::Nearest),
            projection: Into::<[[f32; 4]; 4]>::into(projection_matrix),
            view: Into::<[[f32; 4]; 4]>::into(view_matrix.concat(&translation)),
            light_position: Into::<[f32; 3]>::into(actor.position),
        };
        target.draw(
            &resources.text_quad_vertex_buffer,
            glium::index::NoIndices(glium::index::PrimitiveType::TrianglesList),
            &resources.textured_shader,
            &text_uniforms,
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

    let t = Instant::now().duration_since(program_start_time).as_secs_f32(); // Ugly hack just to try to animate colour

    let translation = Matrix4::from_translation(cgmath::vec3(0.0, -200.0, 0.0));
    let scale = Matrix4::from_scale(800.0);
    let modelview = translation.concat(&scale);
    text::render(&mut target, &resources.text_vertex_buffer, &resources.glyph_cache_texture, &resources.textured_shader, ui_projection, modelview);

    target.finish().unwrap();
}

pub fn create_shape_buffers(display: &glium::Display, shape: lyon::lyon_tessellation::VertexBuffers<lyon::math::Point, u16>) -> (VertexBuffer<geometry::Vertex>, IndexBuffer<u16>) {
    let vertices: Vec<geometry::Vertex> = shape.vertices
        .into_iter()
        .map(|p| geometry::Vertex { position: [p.x, p.y, 0.0], colour: [1.0, 0.0, 0.0, 1.0], normal: [0.0, 0.0, 1.0], tex_coords: [0.0, 0.0] })
        .collect();

    (
          glium::VertexBuffer::new(display, &vertices).unwrap()
        , IndexBuffer::new(display, glium::index::PrimitiveType::TriangleStrip, &shape.indices[..]).unwrap()
    )
}

/// 
// TODO: error handling
pub fn load_texture(display: &glium::Display, path: std::string::String) -> glium::texture::SrgbTexture2d {
    let image = load_raw_image_2d(path);
    glium::texture::SrgbTexture2d::new(display, image).unwrap()
}

pub fn load_raw_image_2d<'a>(path: std::string::String) -> glium::texture::RawImage2d<'a, u8> {
    let data = fs::read(path).unwrap();
    let image = image::load(Cursor::new(&data[..]), image::ImageFormat::Png).unwrap().to_rgba();
    let size = image.dimensions();
    glium::texture::RawImage2d::from_raw_rgba_reversed(&image.into_raw(), size)
}

pub fn load_array_texture(display: &glium::Display, paths: Vec<std::string::String>) -> glium::texture::SrgbTexture2dArray {
    glium::texture::SrgbTexture2dArray::new(
        display, paths.iter().map(|path| { load_raw_image_2d(path.to_string()) }).collect()
    ).unwrap()
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
          position: cgmath::vec3(2.0, 23.0, 0.0)
        , velocity: cgmath::vec3(0.0, 0.0, 0.0)
        , jump_velocity: 4.3
        , orientation: Orientation { rotation_y: Rad(0.0), rotation_z: Rad(0.0) }

        , speed: 4.0
        , size: cgmath::vec3(1.0, 2.0, 1.0)
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
                //in vec3 world_position;
                //in vec2 texture_offset;

                uniform mat4 projection;
                uniform mat4 view;

                out vec2 v_tex_coords;
                out vec4 v_colour;
                out vec3 pos;

                uniform vec3 light_position;

                vec2 atlas_size = vec2(6.0, 6.0);
                vec2 cube_face_size = vec2(192.0/atlas_size.x, 192.0/atlas_size.y);

                void main() {
                    gl_Position = projection * view * vec4(position, 1.0);
                    // world_position = vec4(position, 1.0);
                    pos = position;
                    v_tex_coords = vec2(tex_coords.x, tex_coords.y);
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
                    // f_colour = texture(tex, v_tex_coords);
                    f_colour = v_colour * vec4(1.0, 1.0, 1.0, texture(tex, v_tex_coords).r);
                }
            "
        }
    )?;

    let textured_array_shader = program!(
        &display,
        140 => {
            vertex: "
                #version 140

                in vec3 position;
                in vec2 tex_coords;
                in float texture_index;
                in vec4 colour;

                // Per instance
                in vec3 world_position;
                in vec2 texture_offset;

                uniform mat4 projection;
                uniform mat4 view;

                out vec2 v_tex_coords;
                flat out float v_tex_index;
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
                    v_tex_index = texture_index;
                }
            ",

            fragment: "
                #version 140
                uniform sampler2DArray textures;
                in vec2 v_tex_coords;
                flat in float v_tex_index;
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
                    f_colour = texture(textures, vec3(v_tex_coords.x, v_tex_coords.y * 6.0, v_tex_index));
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
                in vec4 colour;

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

    let monochrome_shader = program!(
        &display,
        140 => {
            vertex: "
                #version 140
                in vec3 position;
                uniform vec4 colour;
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
    
    let mut time_of_last_frame = Instant::now();

    let maximum_frames_per_second: f32 = 30.0;

    let cube_vertex_buffer = glium::VertexBuffer::new(&display, &geometry::cube_vertices(1.0)).unwrap();
    let axes_vertex_buffer = glium::VertexBuffer::new(&display, &geometry::perpendicular_axes_xyz(200.0, 200.0, 200.0)).unwrap();

    let texture_atlas = load_texture(&display, "/Users/jonatan/kuliga kodprojekt/oxide/src/voxels/assets/textures/texture_atlas_layers.png".to_string());
    let textures_array = load_array_texture(
        &display,
        vec![
              "/Users/jonatan/kuliga kodprojekt/oxide/src/voxels/assets/textures/blocks/dirt.png".to_string()
            , "/Users/jonatan/kuliga kodprojekt/oxide/src/voxels/assets/textures/blocks/grass.png".to_string()
            , "/Users/jonatan/kuliga kodprojekt/oxide/src/voxels/assets/textures/blocks/stone.png".to_string()
        ]
    );

    let o = cgmath::vec3(-(width as f32) / 2.0 + 20.0, -(height as f32) / 2.0, 0.0);

    let mut font_data = include_bytes!("./assets/fonts/Hasklig-Medium.otf");
    let mut font: Font = Font::try_from_bytes(font_data as &[u8]).unwrap();
    let scale = display.gl_window().window().scale_factor();

    let (cache_width, cache_height) = ((width as f64 * scale) as u32, (height as f64 * scale) as u32);
    println!("(cache_width, cache_height) {} {}", cache_width, cache_height);
    let mut glyph_cache = Cache::builder()
        .dimensions(cache_width, cache_height)
        .build();
    let glyph_cache_texture = glium::texture::Texture2d::with_format(
            &display,
            glium::texture::RawImage2d {
                data: Cow::Owned(vec![128u8; cache_width as usize * cache_height as usize]),
                width: cache_width,
                height: cache_height,
                format: glium::texture::ClientFormat::U8,
            },
            glium::texture::UncompressedFloatFormat::U8,
            glium::texture::MipmapsOption::NoMipmap,
        )?;
    let text_quad_vertex_buffer = glium::VertexBuffer::new(&display, &geometry::quad_vertices_xz(2.0, 2.0)).unwrap();
    let text_vertex_buffer = text::create_text_vertices(&display, &font, &mut glyph_cache, scale as f32, &glyph_cache_texture, cache_width, &vec!["Harry Potter".to_string(), "and the Fiddler of Razzmatazz".to_string()]);

    let mut graphics_resources = GraphicsCardResources {
        textured_shader: textured_shader
      , coloured_shader: coloured_shader
      , textured_array_shader: textured_array_shader
      , monochrome_shader: monochrome_shader

      , cube_vertex_buffer: cube_vertex_buffer
      , axes_vertex_buffer: axes_vertex_buffer

      , texture_atlas: texture_atlas
      , textures_array: textures_array

      , shapes: vec![
          (create_shape_buffers(&display, ui::rounded_rectangle_outlined(62.0, 62.0, 2.0, 8.0)),
          (0 .. 9)
            .map(|i| { (o + cgmath::vec3(i as f32 * 70.0, 62.0 * 0.5, 0.0), [0.0, 0.0, 0.85, 1.0]) })
            .collect()
          )
      , (
          create_shape_buffers(&display, ui::rounded_rectangle_filled(62.0, 62.0, 8.0)),
          (0 .. 9)
            .map(|i| { (o + cgmath::vec3(i as f32 * 70.0, 62.0 * 0.5, 0.0), [0.0, 1.0, 0.25, 1.0]) })
            .collect()
        )
      ]
      
      , font: font
      , glyph_cache: glyph_cache
      , glyph_cache_texture: glyph_cache_texture
      , text_vertex_buffer: text_vertex_buffer
      , text_quad_vertex_buffer: text_quad_vertex_buffer
    };

    let mut world = create_world();

    let program_start_time: Instant = Instant::now();

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
                &mut graphics_resources,
                &Settings { use_perspective: true, primitive: *primitive },
                &player,
                &world,
                program_start_time
            );
            time_of_last_frame = now;
        }
    })
}

pub fn main() -> Result<(), Box<dyn Error>> {
    run()
}
