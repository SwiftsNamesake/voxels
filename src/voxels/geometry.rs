use glium::*;

#[derive(Copy, Clone)]
pub struct Vertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub tex_coords: [f32; 2],
    pub colour: [f32; 4],
}

implement_vertex!(Vertex, position, tex_coords, colour);

const up: [f32; 3]       = [ 0.0,  1.0,  0.0];
const down: [f32; 3]     = [ 0.0, -1.0,  0.0];
const left: [f32; 3]     = [-1.0,  0.0,  0.0];
const right: [f32; 3]    = [ 1.0,  0.0,  0.0];
const inwards: [f32; 3]  = [ 0.0,  0.0, -1.0];
const outwards: [f32; 3] = [ 0.0,  0.0,  1.0];

pub fn perpendicular_axes_xyz(dx: f32, dy: f32, dz: f32) -> Vec<Vertex> {
    // Note that the normals don't matter for these vertices, they just represent lines
    vec![
        Vertex {
            position: [-dx*0.5, 0.0, 0.0],
            normal: up,
            tex_coords: [0.0, 0.0],
            colour: [0.5, 0.0, 0.0, 1.0],
        },
        Vertex {
            position: [dx*0.5, 0.0, 0.0],
            normal: up,
            tex_coords: [0.0, 0.0],
            colour: [1.0, 0.0, 0.0, 1.0],
        },
        Vertex {
            position: [0.0, -dy*0.5, 0.0],
            normal: up,
            tex_coords: [0.0, 0.0],
            colour: [0.0, 0.2, 0.0, 1.0],
        },
        Vertex {
            position: [0.0, dy*0.5, 0.0],
            normal: up,
            tex_coords: [0.0, 0.0],
            colour: [0.0, 1.0, 0.0, 1.0],
        },
        Vertex {
            position: [0.0, 0.0, -dz*0.5],
            normal: up,
            tex_coords: [0.0, 0.0],
            colour: [0.0, 0.0, 0.5, 1.0],
        },
        Vertex {
            position: [0.0, 0.0, dz*0.5],
            normal: up,
            tex_coords: [0.0, 0.0],
            colour: [0.0, 0.0, 1.0, 1.0],
        }
    ]
}

/// A quad in the XZ plane
pub fn quad_vertices_xz(dx: f32, dz: f32) -> Vec<Vertex> {
    vec![
        Vertex {
            position: [-dx*0.5, 0.0, dz*0.5],
            normal: up,
            tex_coords: [0.0, 0.0],
            colour: [0.0, 0.0, 1.0, 1.0],
        },
        Vertex {
            position: [dx*0.5, 0.0, dz*0.5],
            normal: up,
            tex_coords: [1.0, 0.0],
            colour: [0.0, 0.0, 1.0, 1.0],
        },
        Vertex {
            position: [dx*0.5, 0.0, -dz*0.5],
            normal: up,
            tex_coords: [1.0, 1.0],
            colour: [0.0, 0.0, 1.0, 1.0],
        },
        Vertex {
            position: [-dx*0.5, 0.0, dz*0.5],
            normal: up,
            tex_coords: [0.0, 0.0],
            colour: [0.0, 0.0, 1.0, 1.0],
        },
        Vertex {
            position: [dx*0.5, 0.0, -dz*0.5],
            normal: up,
            tex_coords: [1.0, 1.0],
            colour: [0.0, 0.0, 1.0, 1.0],
        },
        Vertex {
            position: [-dx*0.5, 0.0, -dz*0.5],
            normal: up,
            tex_coords: [0.0, 1.0],
            colour: [0.0, 0.0, 1.0, 1.0],
        },
    ]
}

/// Constructs the vertices for a cube
/// TODO: Separate abstract geometry from `Vertex` type
pub fn cube_vertices(size: f32) -> Vec<Vertex> {
    let pos = 1.0 * size * 0.5;
    let neg = -1.0 * size * 0.5;

    // Textures
    // Front, Right, Back, Left, Top, Bottom
    let front_offset: cgmath::Vector2<f32> = cgmath::vec2(1.0/6.0 * 0.0, 0.0);
    let right_offset: cgmath::Vector2<f32> = cgmath::vec2(1.0/6.0 * 1.0, 0.0);
    let back_offset: cgmath::Vector2<f32> = cgmath::vec2(1.0/6.0 * 2.0, 0.0);
    let left_offset: cgmath::Vector2<f32> = cgmath::vec2(1.0/6.0 * 3.0, 0.0);
    let top_offset: cgmath::Vector2<f32> = cgmath::vec2(1.0/6.0 * 4.0, 0.0);
    let bottom_offset: cgmath::Vector2<f32> = cgmath::vec2(1.0/6.0 * 5.0, 0.0);

    let dx = 1.0/6.0;

    return vec![
        // Front
        Vertex {
            position: [neg, neg, pos],
            normal: outwards,
            tex_coords: (cgmath::vec2(0.0, 0.0) + front_offset).into(),
            colour: [1.0, 0.0, 0.0, 1.0],
        },
        Vertex {
            position: [pos, neg, pos],
            normal: outwards,
            tex_coords: (cgmath::vec2(dx, 0.0) + front_offset).into(),
            colour: [1.0, 0.0, 0.0, 1.0],
        },
        Vertex {
            position: [pos, pos, pos],
            normal: outwards,
            tex_coords: (cgmath::vec2(dx, 1.0) + front_offset).into(),
            colour: [1.0, 0.0, 0.0, 1.0],
        },
        Vertex {
            position: [neg, neg, pos],
            normal: outwards,
            tex_coords: (cgmath::vec2(0.0, 0.0) + front_offset).into(),
            colour: [1.0, 0.0, 0.0, 1.0],
        },
        Vertex {
            position: [pos, pos, pos],
            normal: outwards,
            tex_coords: (cgmath::vec2(dx, 1.0) + front_offset).into(),
            colour: [1.0, 0.0, 0.0, 1.0],
        },
        Vertex {
            position: [neg, pos, pos],
            normal: outwards,
            tex_coords: (cgmath::vec2(0.0, 1.0) + front_offset).into(),
            colour: [1.0, 0.0, 0.0, 1.0],
        },

        // Back
        Vertex {
            position: [pos, neg, neg],
            normal: inwards,
            tex_coords: (cgmath::vec2(0.0, 0.0) + back_offset).into(),
            colour: [0.0, 1.0, 0.0, 1.0],
        },
        Vertex {
            position: [neg, neg, neg],
            normal: inwards,
            tex_coords: (cgmath::vec2(dx, 0.0) + back_offset).into(),
            colour: [0.0, 1.0, 0.0, 1.0],
        },
        Vertex {
            position: [neg, pos, neg],
            normal: inwards,
            tex_coords: (cgmath::vec2(dx, 1.0) + back_offset).into(),
            colour: [0.0, 1.0, 0.0, 1.0],
        },
        Vertex {
            position: [pos, neg, neg],
            normal: inwards,
            tex_coords: (cgmath::vec2(0.0, 0.0) + back_offset).into(),
            colour: [0.0, 1.0, 0.0, 1.0],
        },
        Vertex {
            position: [neg, pos, neg],
            normal: inwards,
            tex_coords: (cgmath::vec2(dx, 1.0) + back_offset).into(),
            colour: [0.0, 1.0, 0.0, 1.0],
        },
        Vertex {
            position: [pos, pos, neg],
            normal: inwards,
            tex_coords: (cgmath::vec2(0.0, 1.0) + back_offset).into(),
            colour: [0.0, 1.0, 0.0, 1.0],
        },

        // Left
        Vertex {
            position: [neg, neg, neg],
            normal: left,
            tex_coords: (cgmath::vec2(0.0, 0.0) + left_offset).into(),
            colour: [0.0, 1.0, 0.5, 1.0],
        },
        Vertex {
            position: [neg, pos, neg],
            normal: left,
            tex_coords: (cgmath::vec2(0.0, 1.0) + left_offset).into(),
            colour: [0.0, 1.0, 0.5, 1.0],
        },
        Vertex {
            position: [neg, pos, pos],
            normal: left,
            tex_coords: (cgmath::vec2(dx, 1.0) + left_offset).into(),
            colour: [0.0, 1.0, 0.5, 1.0],
        },
        Vertex {
            position: [neg, neg, neg],
            normal: left,
            tex_coords: (cgmath::vec2(0.0, 0.0) + left_offset).into(),
            colour: [0.0, 1.0, 0.5, 1.0],
        },
        Vertex {
            position: [neg, pos, pos],
            normal: left,
            tex_coords: (cgmath::vec2(dx, 1.0) + left_offset).into(),
            colour: [0.0, 1.0, 0.5, 1.0],
        },
        Vertex {
            position: [neg, neg, pos],
            normal: left,
            tex_coords: (cgmath::vec2(dx, 0.0) + left_offset).into(),
            colour: [0.0, 1.0, 0.5, 1.0],
        },

        // Right
        Vertex {
            position: [pos, neg, neg],
            normal: right,
            tex_coords: (cgmath::vec2(0.0, 0.0) + right_offset).into(),
            colour: [0.0, 0.0, 1.0, 1.0],
        },
        Vertex {
            position: [pos, neg, pos],
            normal: right,
            tex_coords: (cgmath::vec2(dx, 0.0) + right_offset).into(),
            colour: [0.0, 0.0, 1.0, 1.0],
        },
        Vertex {
            position: [pos, pos, pos],
            normal: right,
            tex_coords: (cgmath::vec2(dx, 1.0) + right_offset).into(),
            colour: [0.0, 0.0, 1.0, 1.0],
        },
        Vertex {
            position: [pos, neg, neg],
            normal: right,
            tex_coords: (cgmath::vec2(0.0, 0.0) + right_offset).into(),
            colour: [0.0, 0.0, 1.0, 1.0],
        },
        Vertex {
            position: [pos, pos, pos],
            normal: right,
            tex_coords: (cgmath::vec2(dx, 1.0) + right_offset).into(),
            colour: [0.0, 0.0, 1.0, 1.0],
        },
        Vertex {
            position: [pos, pos, neg],
            normal: right,
            tex_coords: (cgmath::vec2(0.0, 1.0) + right_offset).into(),
            colour: [0.0, 0.0, 1.0, 1.0],
        },

        // Top
        Vertex {
            position: [neg, pos, pos],
            normal: up,
            tex_coords: (cgmath::vec2(0.0, 0.0) + top_offset).into(),
            colour: [0.0, 0.0, 1.0, 1.0],
        },
        Vertex {
            position: [pos, pos, pos],
            normal: up,
            tex_coords: (cgmath::vec2(dx, 0.0) + top_offset).into(),
            colour: [0.0, 0.0, 1.0, 1.0],
        },
        Vertex {
            position: [pos, pos, neg],
            normal: up,
            tex_coords: (cgmath::vec2(dx, 1.0) + top_offset).into(),
            colour: [0.0, 0.0, 1.0, 1.0],
        },
        Vertex {
            position: [neg, pos, pos],
            normal: up,
            tex_coords: (cgmath::vec2(0.0, 0.0) + top_offset).into(),
            colour: [0.0, 0.0, 1.0, 1.0],
        },
        Vertex {
            position: [pos, pos, neg],
            normal: up,
            tex_coords: (cgmath::vec2(dx, 1.0) + top_offset).into(),
            colour: [0.0, 0.0, 1.0, 1.0],
        },
        Vertex {
            position: [neg, pos, neg],
            normal: up,
            tex_coords: (cgmath::vec2(0.0, 1.0) + top_offset).into(),
            colour: [0.0, 0.0, 1.0, 1.0],
        },

        // Bottom
        Vertex {
            position: [neg, neg, neg],
            normal: down,
            tex_coords: (cgmath::vec2(0.0, 0.0) + bottom_offset).into(),
            colour: [0.5, 0.0, 1.0, 1.0],
        },
        Vertex {
            position: [pos, neg, neg],
            normal: down,
            tex_coords: (cgmath::vec2(dx, 0.0) + bottom_offset).into(),
            colour: [0.5, 0.0, 1.0, 1.0],
        },
        Vertex {
            position: [pos, neg, pos],
            normal: down,
            tex_coords: (cgmath::vec2(dx, 1.0) + bottom_offset).into(),
            colour: [0.5, 0.0, 1.0, 1.0],
        },
        Vertex {
            position: [neg, neg, neg],
            normal: down,
            tex_coords: (cgmath::vec2(0.0, 0.0) + bottom_offset).into(),
            colour: [0.5, 0.0, 1.0, 1.0],
        },
        Vertex {
            position: [pos, neg, pos],
            normal: down,
            tex_coords: (cgmath::vec2(dx, 1.0) + bottom_offset).into(),
            colour: [0.5, 0.0, 1.0, 1.0],
        },
        Vertex {
            position: [neg, neg, pos],
            normal: down,
            tex_coords: (cgmath::vec2(0.0, 1.0) + bottom_offset).into(),
            colour: [0.5, 0.0, 1.0, 1.0],
        },
    ]
}
