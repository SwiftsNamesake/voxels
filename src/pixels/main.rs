/// Pixels
/// A pixel editor
/// 
/// - Plugins
/// - Constraint-based layout
/// - Customisable layout
/// - Scriptable commands (with repl? omgomgomg please yes)
/// - Integrate with terminal emulator

use glium::*;
use glutin::{
    event::{Event, KeyboardInput, VirtualKeyCode, WindowEvent, ElementState},
    event_loop::ControlFlow,
};
// use lyon::math::{rect, Point};
use std::io::Cursor;

// use std::convert::TryFrom;
use std::convert::TryInto;
use std::time::{Duration};

// use std::borrow::Cow;
use std::env;
use std::error::Error;
use std::cmp::{max,min};

use cgmath;
use cgmath::Transform;

extern crate libloading;

use libloading::{Library, Symbol};

extern crate notify;
use notify::{Watcher, RecursiveMode, watcher};
use std::sync::mpsc::channel;

type Palette = unsafe fn() -> Vec<(u8, u8, u8)>;

pub fn load_extensions() -> Vec<(u8, u8, u8)> {
    let library_path = "./src/pixels/extensions/compiled/libpixels_dlc.dylib";
    let lib = Library::new(library_path).unwrap();
    unsafe {
        let func: Symbol<Palette> = lib.get(b"palette").unwrap();
        return func();
    }
}

mod vector;

trait Tool {}

// struct Edit {}

#[derive(Copy, Clone)]
struct Vertex {
    position: [f32; 2],
    tex_coords: [f32; 2],
    colour: [f32; 4],
}

implement_vertex!(Vertex, position, tex_coords, colour);

fn observe_file_changes() {
    let (tx, rx) = channel();
    let mut watcher = watcher(tx, Duration::from_secs(10)).unwrap();
    watcher.watch("/Users/jonatan/kuliga kodprojekt/oxide/src/pixels/extensions", RecursiveMode::Recursive).unwrap();
    loop {
        match rx.recv() {
            Ok(event) => println!("{:?}", event),
            Err(e) => println!("watch error: {:?}", e),
        }
    }
}

// fn draw_user_interface(target: Frame, icon_pen_texture, vertex_buffers) {
//     let mv_pen: [[f32; 4]; 4] = cgmath::Matrix4::from_scale(0.3f32).concat(&cgmath::Matrix4::from_translation(
//         cgmath::Vector3 { x: 0.0, y: 0.0, z: 0.0 }
//     )).into();
//     let icon_uniforms = uniform! {
//         tex: icon_pen_texture.sampled().magnify_filter(glium::uniforms::MagnifySamplerFilter::Nearest),
//         modelview: mv_pen
//     };
//     target.draw(
//         &vertex_buffers.0,
//         glium::index::NoIndices(glium::index::PrimitiveType::TrianglesList),
//         &program,
//         &icon_uniforms,
//         &glium::DrawParameters {
//             blend: glium::Blend::alpha_blending(),
//             ..Default::default()
//         },
//     ).unwrap();
// }

// fn draw_shapes() {
//     let mv: [[f32; 4]; 4] = cgmath::Matrix4::from_scale(0.01f32).concat(&cgmath::Matrix4::from_translation(
//         cgmath::Vector3 { x: 0.0, y: 0.0, z: 0.0 }
//     )).into();
//     let shape_uniforms = uniform! {
//         tex: canvas.sampled().magnify_filter(glium::uniforms::MagnifySamplerFilter::Nearest),
//         modelview: mv
//     };

//     let _ = target.draw(
//         &vertex_buffers.1.0,
//         &vertex_buffers.1.1,
//         &program,
//         &shape_uniforms,
//         &glium::DrawParameters {
//             blend: glium::Blend::alpha_blending(),
//             ..Default::default()
//         }
//     );
// }

pub fn main() -> Result<(), Box<dyn Error>> {
    // observe_file_changes();

    if cfg!(target_os = "linux") && env::var("WINIT_UNIX_BACKEND").is_err() {
        env::set_var("WINIT_UNIX_BACKEND", "x11");
    }

    let width = 64*10;
    let height = 64*10;

    let window = glium::glutin::window::WindowBuilder::new()
        .with_inner_size(glium::glutin::dpi::PhysicalSize::new(width, height))
        .with_title("pixels");

    let context = glium::glutin::ContextBuilder::new().with_vsync(true);
    let event_loop = glium::glutin::event_loop::EventLoop::new();
    let display = glium::Display::new(window, context, &event_loop)?;

    let colour = (255u8, 255u8, 255u8);
    let canvas = glium::Texture2d::new(&display, vec![vec![colour; 64]; 64]).unwrap();

    let image = image::load(Cursor::new(&include_bytes!("./assets/icon.tool.pen.png")[..]),
    image::ImageFormat::Png).unwrap().to_rgba();
    let image_dimensions = image.dimensions();
    let image = glium::texture::RawImage2d::from_raw_rgba_reversed(&image.into_raw(), image_dimensions);
    let icon_pen_texture = glium::texture::Texture2d::new(&display, image).unwrap();

    // let mut cursor: glutin::dpi::PhysicalPosition<f64> = glutin::dpi::PhysicalPosition { x : 0.0, y : 0.0 };
    let mut is_drawing = false;
    let palette = load_extensions();
    let mut pen_colour = 0;

    let program = program!(
    &display,
    140 => {
            vertex: "
                #version 140
                in vec2 position;
                in vec2 tex_coords;
                in vec4 colour;
                out vec2 v_tex_coords;
                out vec4 v_colour;
                uniform mat4 modelview;
                void main() {
                    gl_Position = modelview * vec4(position, 0.0, 1.0);
                    v_tex_coords = tex_coords;
                    v_colour = colour;
                }
            ",

            fragment: "
                #version 140
                uniform sampler2D tex;
                in vec2 v_tex_coords;
                in vec4 v_colour;
                out vec4 f_colour;
                void main() {
                    f_colour = texture(tex, v_tex_coords);
                }
            "
    })?;

    event_loop.run(move |event, _, control_flow| {
        match event {
            Event::WindowEvent { event, .. } => match event {
                
                WindowEvent::KeyboardInput {
                    input:
                        KeyboardInput {
                            virtual_keycode: Some(VirtualKeyCode::Escape),
                            ..
                        },
                    ..
                }
                | WindowEvent::CloseRequested => *control_flow = ControlFlow::Exit,
                WindowEvent::DroppedFile(f) => {
                    println!("{}", f.display());
                    // font_data.clear();
                    // font_data.append(&mut fs::read(f).unwrap());
                    // font = Font::try_from_bytes(&font_data).unwrap();
                },
                WindowEvent::KeyboardInput {
                    input:
                        KeyboardInput {
                            virtual_keycode: Some(key),
                            state: ElementState::Pressed,
                            ..
                        },
                    ..
                } => {
                    match key {
                        VirtualKeyCode::Space => {
                            pen_colour = (pen_colour + 1) % palette.len();
                        },
                        _ => {}
                    }
                },
                WindowEvent::CursorMoved { position, .. } => {
                    // println!("{:?}", position);
                    if is_drawing {
                        let x = max(0, min((64.0 * position.x as f64 / (width as f64)) as u32, 63));
                        let y = max(0, min(64 - (64.0 * position.y as f64 / (height as f64)) as i32, 63)) as u32;

                        let w = 1; //rand::random::<u32>() % 64;
                        let h = 1; //rand::random::<u32>() % 64;

                        canvas.write(
                            glium::Rect {
                                width : w,
                                height : h,
                                left : x,
                                bottom : y
                            },
                            vec![
                                vec![palette[pen_colour]
                                    ;
                                    w.try_into().unwrap()
                                ]
                                ;
                                h.try_into().unwrap()
                            ]
                        );
                        // edits.push((x,y));
                        // canvas.write(glium::Rect { width : 1, height : 1, left : x, bottom : y }, vec![vec![colour]]);
                        display.gl_window().window().request_redraw();
                    }
                },
                WindowEvent::MouseInput { state, button, .. } => {
                   match (button, state) {
                       (glutin::event::MouseButton::Left, glutin::event::ElementState::Pressed) => { is_drawing = true; },
                       (glutin::event::MouseButton::Left, glutin::event::ElementState::Released) => { is_drawing = false },
                       _ => {}
                   }
                },
                WindowEvent::Moved(_position) => {
                    // println!("{:?}", position);
                },
                // WindowEvent::KeyboardInput {
                //     input:
                //         KeyboardInput {
                //             virtual_keycode: Some(VirtualKeyCode::Return),
                //             state: ElementState::Pressed,
                //             ..
                //         },
                //     ..
                // } => {

                // },
                WindowEvent::ReceivedCharacter(_c) => {
                    display.gl_window().window().request_redraw();
                }
                _ => (),
            }
            Event::RedrawRequested(_) => {
                let _scale = display.gl_window().window().scale_factor();
                // let (width, height): (u32, _) = display
                //     .gl_window()
                //     .window()
                //     .inner_size()
                //     .into();

                let vertex_buffers = {
                    let (_screen_width, _screen_height) = {
                        let (w, h) = display.get_framebuffer_dimensions();
                        (w as f32, h as f32)
                    };

                    let x_1 = -1.0;
                    let x_2 = 1.0;

                    let y_1 = -1.0;
                    let y_2 = 1.0;

                    let quad_vertices: Vec<Vertex> = vec![
                        Vertex {
                            position: [x_1, y_1],
                            tex_coords: [0.0, 0.0],
                            colour: [0.2, 0.5, 0.4, 1.0],
                        },
                        Vertex {
                            position: [x_2, y_1],
                            tex_coords: [1.0, 0.0],
                            colour: [0.2, 0.5, 0.4, 1.0],
                        },
                        Vertex {
                            position: [x_2, y_2],
                            tex_coords: [1.0, 1.0],
                            colour: [0.2, 0.5, 0.4, 1.0],
                        },
                        Vertex {
                            position: [x_1, y_1],
                            tex_coords: [0.0, 0.0],
                            colour: [0.2, 0.5, 0.4, 1.0],
                        },
                        Vertex {
                            position: [x_2, y_2],
                            tex_coords: [1.0, 1.0],
                            colour: [0.2, 0.5, 0.4, 1.0],
                        },
                        Vertex {
                            position: [x_1, y_2],
                            tex_coords: [0.0, 1.0],
                            colour: [0.2, 0.5, 0.4, 1.0],
                        },
                    ];

                    let rounded_rectangle = vector::rounded_rectangle();
                    let rounded_rectangle_vertices: Vec<Vertex> = rounded_rectangle.vertices
                        .into_iter()
                        .map(|p| Vertex { position: [p.x, p.y], tex_coords: [p.x * 0.01, p.y * 0.01], colour: [0.8, 0.2, 0.05, 1.0], })
                        .collect();

                    (
                        glium::VertexBuffer::new(&display, &quad_vertices).unwrap(),
                        (
                            glium::VertexBuffer::new(&display, &rounded_rectangle_vertices).unwrap(),
                            IndexBuffer::new(&display, glium::index::PrimitiveType::TriangleStrip, &rounded_rectangle.indices[..]).unwrap()
                        )
                    )
                };

                let canvas_modelview: [[f32; 4]; 4] = cgmath::Matrix4::one().into();
                let canvas_uniforms = uniform! {
                    tex: canvas.sampled().magnify_filter(glium::uniforms::MagnifySamplerFilter::Nearest),
                    modelview: canvas_modelview
                };

                let mut target = display.draw();
                target.clear_color(0.0, 0.0, 0.0, 0.0);
                target.draw(
                    &vertex_buffers.0,
                    glium::index::NoIndices(glium::index::PrimitiveType::TrianglesList),
                    &program,
                    &canvas_uniforms,
                    &glium::DrawParameters {
                        blend: glium::Blend::alpha_blending(),
                        ..Default::default()
                    },
                ).unwrap();

                // draw_user_interface(target, pen);

                target.finish().unwrap();
            }
            _ => (),
        }
    });
}
