
use cgmath::{Matrix3, Matrix4, Vector3, Point3, Rad};
use glium::*;
use rusttype::gpu_cache::Cache;
use rusttype::{point, vector, Font, PositionedGlyph, Rect, Scale, Point};
use std::borrow::Cow;
use std::fs;
use std::path::{ PathBuf };

//pub load_font() -> Font {
//    let mut font_data = include_bytes!("../assets/fonts/Hasklig-Medium.otf");
//    Font::try_from_bytes(font_data as &[u8]).unwrap()
//}

// fn load_font<'a>(s: PathBuf) -> Font<'a> {
//     let font_data = fs::read(s).unwrap();
//     return Font::try_from_bytes(&font_data).unwrap();
// }

///
pub fn layout_paragraph<'a>(
    font: &Font<'a>,
    scale: Scale,
    width: u32,
    lines: &Vec<String>,
) -> (Vec<PositionedGlyph<'a>>, Point<f32>) {
    let mut result = Vec::new();
    let v_metrics = font.v_metrics(scale);
    let advance_height = v_metrics.ascent - v_metrics.descent + v_metrics.line_gap;
    let mut caret = point(0.0, v_metrics.ascent);
    let mut caret_x = 0.0;
    let mut last_glyph_id = None;
    for line in lines {
        for c in line.chars() {
            if c.is_control() {
                match c {
                    '\r' => {
                        caret = point(0.0, caret.y + advance_height);
                    }
                    '\n' => {}
                    _ => {}
                }
                continue;
            }
            let base_glyph = font.glyph(c);
            if let Some(id) = last_glyph_id.take() {
                caret.x += font.pair_kerning(scale, id, base_glyph.id());
            }
            last_glyph_id = Some(base_glyph.id());
            let mut glyph = base_glyph.scaled(scale).positioned(caret);
            if let Some(bb) = glyph.pixel_bounding_box() {
                if bb.max.x > width as i32 {
                    caret = point(0.0, caret.y + advance_height);
                    glyph.set_position(caret);
                    last_glyph_id = None;
                }
            }
            caret.x += glyph.unpositioned().h_metrics().advance_width;
            result.push(glyph);
        }

        caret_x = caret.x;
        caret = point(0.0, caret.y + advance_height);
    }
    caret.x = caret_x;
    (result, caret)
}

#[derive(Copy, Clone)]
pub struct Vertex {
    position: [f32; 3],
    tex_coords: [f32; 2],
    colour: [f32; 4],
}
implement_vertex!(Vertex, position, tex_coords, colour);

pub fn create_text_vertices<'a>(
    display: &glium::Display,
    font: &Font<'a>,
    cache: &mut Cache<'a>,
    scale: f32,
    cache_tex: &glium::Texture2d,
    width: u32,
    lines: &Vec<String>
) -> glium::VertexBuffer<Vertex> {
    let (glyphs, caret) = layout_paragraph(&font, Scale::uniform(24.0 * 10.0 * scale), 720 * 6, &lines);
    for glyph in &glyphs {
        cache.queue_glyph(0, glyph.clone());
    }
    cache.cache_queued(|rect, data| {
        cache_tex.main_level().write(
            glium::Rect {
                left: rect.min.x,
                bottom: rect.min.y,
                width: rect.width(),
                height: rect.height(),
            },
            glium::texture::RawImage2d {
                data: Cow::Borrowed(data),
                width: rect.width(),
                height: rect.height(),
                format: glium::texture::ClientFormat::U8,
            },
        );
    }).unwrap();

    let colour = [1.0, 1.0, 1.0, 1.0];
    let (screen_width, screen_height) = {
        let (w, h) = display.get_framebuffer_dimensions();
        (w as f32, h as f32)
    };
    let origin = point(0.0, 0.0);
    let glyph_vertices: Vec<Vertex> = glyphs
        .iter()
        .filter_map(|g| cache.rect_for(0, g).ok().flatten())
        .flat_map(|(uv_rect, screen_rect)| {
            let s = 0.4;
            let gl_rect = Rect {
                min: origin
                    + (vector(
                        (screen_rect.min.x as f32 * s) / screen_width,
                        (1.0 - screen_rect.min.y as f32) * s / screen_height,
                    )) * 2.0,
                max: origin
                    + (vector(
                        (screen_rect.max.x as f32 * s) / screen_width,
                        (1.0 - screen_rect.max.y as f32) * s / screen_height,
                    )) * 2.0,
            };
            vec![
                Vertex {
                    position: [gl_rect.min.x, gl_rect.max.y, 0.0],
                    tex_coords: [uv_rect.min.x, uv_rect.max.y],
                    colour,
                },
                Vertex {
                    position: [gl_rect.min.x, gl_rect.min.y, 0.0],
                    tex_coords: [uv_rect.min.x, uv_rect.min.y],
                    colour,
                },
                Vertex {
                    position: [gl_rect.max.x, gl_rect.min.y, 0.0],
                    tex_coords: [uv_rect.max.x, uv_rect.min.y],
                    colour,
                },
                Vertex {
                    position: [gl_rect.max.x, gl_rect.min.y, 0.0],
                    tex_coords: [uv_rect.max.x, uv_rect.min.y],
                    colour,
                },
                Vertex {
                    position: [gl_rect.max.x, gl_rect.max.y, 0.0],
                    tex_coords: [uv_rect.max.x, uv_rect.max.y],
                    colour,
                },
                Vertex {
                    position: [gl_rect.min.x, gl_rect.max.y, 0.0],
                    tex_coords: [uv_rect.min.x, uv_rect.max.y],
                    colour,
                },
            ]
        })
        .collect();

    return glium::VertexBuffer::new(display, &glyph_vertices).unwrap();
}

pub fn render(target: &mut glium::Frame, text_vertex_buffer: &glium::VertexBuffer<Vertex>, cache_texture: &glium::Texture2d, program: &glium::Program, projection: Matrix4<f32>, modelview: Matrix4<f32>) {
    let uniforms = uniform! {
        tex: cache_texture.sampled().magnify_filter(glium::uniforms::MagnifySamplerFilter::Nearest),
        view: Into::<[[f32; 4]; 4]>::into(modelview),
        projection: Into::<[[f32; 4]; 4]>::into(projection),
        light_position: Into::<[f32; 3]>::into([0.0, 0.0, 0.0])
    };

    target.draw(
        text_vertex_buffer,
        glium::index::NoIndices(glium::index::PrimitiveType::TrianglesList),
        program,
        &uniforms,
        &glium::DrawParameters {
            blend: glium::Blend::alpha_blending(),
            ..Default::default()
        },
    ).unwrap();
}