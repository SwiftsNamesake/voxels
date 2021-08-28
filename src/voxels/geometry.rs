use glium::*;

#[derive(Copy, Clone)]
pub struct Vertex {
    pub position: [f32; 3],
    pub tex_coords: [f32; 2],
    pub colour: [f32; 4],
}

implement_vertex!(Vertex, position, tex_coords, colour);

/// A quad in the XZ plane
pub fn quad_vertices_xz(dx: f32, dz: f32) -> Vec<Vertex> {
    return vec![
        Vertex {
            position: [-dx*0.5, 0.0, dz*0.5],
            tex_coords: [0.0, 0.0],
            colour: [0.0, 0.0, 1.0, 1.0],
        },
        Vertex {
            position: [dx*0.5, 0.0, dz*0.5],
            tex_coords: [1.0, 0.0],
            colour: [0.0, 0.0, 1.0, 1.0],
        },
        Vertex {
            position: [dx*0.5, 0.0, -dz*0.5],
            tex_coords: [1.0, 1.0],
            colour: [0.0, 0.0, 1.0, 1.0],
        },
        Vertex {
            position: [-dx*0.5, 0.0, dz*0.5],
            tex_coords: [0.0, 0.0],
            colour: [0.0, 0.0, 1.0, 1.0],
        },
        Vertex {
            position: [dx*0.5, 0.0, -dz*0.5],
            tex_coords: [1.0, 1.0],
            colour: [0.0, 0.0, 1.0, 1.0],
        },
        Vertex {
            position: [-dx*0.5, 0.0, -dz*0.5],
            tex_coords: [0.0, 1.0],
            colour: [0.0, 0.0, 1.0, 1.0],
        },
    ];
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
            tex_coords: (cgmath::vec2(0.0, 0.0) + front_offset).into(),
            colour: [1.0, 0.0, 0.0, 1.0],
        },
        Vertex {
            position: [pos, neg, pos],
            tex_coords: (cgmath::vec2(dx, 0.0) + front_offset).into(),
            colour: [1.0, 0.0, 0.0, 1.0],
        },
        Vertex {
            position: [pos, pos, pos],
            tex_coords: (cgmath::vec2(dx, 1.0) + front_offset).into(),
            colour: [1.0, 0.0, 0.0, 1.0],
        },
        Vertex {
            position: [neg, neg, pos],
            tex_coords: (cgmath::vec2(0.0, 0.0) + front_offset).into(),
            colour: [1.0, 0.0, 0.0, 1.0],
        },
        Vertex {
            position: [pos, pos, pos],
            tex_coords: (cgmath::vec2(dx, 1.0) + front_offset).into(),
            colour: [1.0, 0.0, 0.0, 1.0],
        },
        Vertex {
            position: [neg, pos, pos],
            tex_coords: (cgmath::vec2(0.0, 1.0) + front_offset).into(),
            colour: [1.0, 0.0, 0.0, 1.0],
        },

        // Back
        Vertex {
            position: [pos, neg, neg],
            tex_coords: (cgmath::vec2(0.0, 0.0) + back_offset).into(),
            colour: [0.0, 1.0, 0.0, 1.0],
        },
        Vertex {
            position: [neg, neg, neg],
            tex_coords: (cgmath::vec2(dx, 0.0) + back_offset).into(),
            colour: [0.0, 1.0, 0.0, 1.0],
        },
        Vertex {
            position: [neg, pos, neg],
            tex_coords: (cgmath::vec2(dx, 1.0) + back_offset).into(),
            colour: [0.0, 1.0, 0.0, 1.0],
        },
        Vertex {
            position: [pos, neg, neg],
            tex_coords: (cgmath::vec2(0.0, 0.0) + back_offset).into(),
            colour: [0.0, 1.0, 0.0, 1.0],
        },
        Vertex {
            position: [neg, pos, neg],
            tex_coords: (cgmath::vec2(dx, 1.0) + back_offset).into(),
            colour: [0.0, 1.0, 0.0, 1.0],
        },
        Vertex {
            position: [pos, pos, neg],
            tex_coords: (cgmath::vec2(0.0, 1.0) + back_offset).into(),
            colour: [0.0, 1.0, 0.0, 1.0],
        },

        // Left
        Vertex {
            position: [neg, neg, neg],
            tex_coords: (cgmath::vec2(0.0, 0.0) + left_offset).into(),
            colour: [0.0, 1.0, 0.5, 1.0],
        },
        Vertex {
            position: [neg, pos, neg],
            tex_coords: (cgmath::vec2(0.0, 1.0) + left_offset).into(),
            colour: [0.0, 1.0, 0.5, 1.0],
        },
        Vertex {
            position: [neg, pos, pos],
            tex_coords: (cgmath::vec2(dx, 1.0) + left_offset).into(),
            colour: [0.0, 1.0, 0.5, 1.0],
        },
        Vertex {
            position: [neg, neg, neg],
            tex_coords: (cgmath::vec2(0.0, 0.0) + left_offset).into(),
            colour: [0.0, 1.0, 0.5, 1.0],
        },
        Vertex {
            position: [neg, pos, pos],
            tex_coords: (cgmath::vec2(dx, 1.0) + left_offset).into(),
            colour: [0.0, 1.0, 0.5, 1.0],
        },
        Vertex {
            position: [neg, neg, pos],
            tex_coords: (cgmath::vec2(dx, 0.0) + left_offset).into(),
            colour: [0.0, 1.0, 0.5, 1.0],
        },

        // Right
        Vertex {
            position: [pos, neg, neg],
            tex_coords: (cgmath::vec2(0.0, 0.0) + right_offset).into(),
            colour: [0.0, 0.0, 1.0, 1.0],
        },
        Vertex {
            position: [pos, neg, pos],
            tex_coords: (cgmath::vec2(dx, 0.0) + right_offset).into(),
            colour: [0.0, 0.0, 1.0, 1.0],
        },
        Vertex {
            position: [pos, pos, pos],
            tex_coords: (cgmath::vec2(dx, 1.0) + right_offset).into(),
            colour: [0.0, 0.0, 1.0, 1.0],
        },
        Vertex {
            position: [pos, neg, neg],
            tex_coords: (cgmath::vec2(0.0, 0.0) + right_offset).into(),
            colour: [0.0, 0.0, 1.0, 1.0],
        },
        Vertex {
            position: [pos, pos, pos],
            tex_coords: (cgmath::vec2(dx, 1.0) + right_offset).into(),
            colour: [0.0, 0.0, 1.0, 1.0],
        },
        Vertex {
            position: [pos, pos, neg],
            tex_coords: (cgmath::vec2(0.0, 1.0) + right_offset).into(),
            colour: [0.0, 0.0, 1.0, 1.0],
        },

        // Top
        Vertex {
            position: [neg, pos, pos],
            tex_coords: (cgmath::vec2(0.0, 0.0) + top_offset).into(),
            colour: [0.0, 0.0, 1.0, 1.0],
        },
        Vertex {
            position: [pos, pos, pos],
            tex_coords: (cgmath::vec2(dx, 0.0) + top_offset).into(),
            colour: [0.0, 0.0, 1.0, 1.0],
        },
        Vertex {
            position: [pos, pos, neg],
            tex_coords: (cgmath::vec2(dx, 1.0) + top_offset).into(),
            colour: [0.0, 0.0, 1.0, 1.0],
        },
        Vertex {
            position: [neg, pos, pos],
            tex_coords: (cgmath::vec2(0.0, 0.0) + top_offset).into(),
            colour: [0.0, 0.0, 1.0, 1.0],
        },
        Vertex {
            position: [pos, pos, neg],
            tex_coords: (cgmath::vec2(dx, 1.0) + top_offset).into(),
            colour: [0.0, 0.0, 1.0, 1.0],
        },
        Vertex {
            position: [neg, pos, neg],
            tex_coords: (cgmath::vec2(0.0, 1.0) + top_offset).into(),
            colour: [0.0, 0.0, 1.0, 1.0],
        },

        // Bottom
        Vertex {
            position: [neg, neg, neg],
            tex_coords: (cgmath::vec2(0.0, 0.0) + bottom_offset).into(),
            colour: [0.5, 0.0, 1.0, 1.0],
        },
        Vertex {
            position: [pos, neg, neg],
            tex_coords: (cgmath::vec2(dx, 0.0) + bottom_offset).into(),
            colour: [0.5, 0.0, 1.0, 1.0],
        },
        Vertex {
            position: [pos, neg, pos],
            tex_coords: (cgmath::vec2(dx, 1.0) + bottom_offset).into(),
            colour: [0.5, 0.0, 1.0, 1.0],
        },
        Vertex {
            position: [neg, neg, neg],
            tex_coords: (cgmath::vec2(0.0, 0.0) + bottom_offset).into(),
            colour: [0.5, 0.0, 1.0, 1.0],
        },
        Vertex {
            position: [pos, neg, pos],
            tex_coords: (cgmath::vec2(dx, 1.0) + bottom_offset).into(),
            colour: [0.5, 0.0, 1.0, 1.0],
        },
        Vertex {
            position: [neg, neg, pos],
            tex_coords: (cgmath::vec2(0.0, 1.0) + bottom_offset).into(),
            colour: [0.5, 0.0, 1.0, 1.0],
        },
    ]
}