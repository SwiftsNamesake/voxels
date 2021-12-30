
use cgmath::{Matrix3, Matrix4, Vector2, Vector3, Point3, Rad};
use cgmath::prelude::*;

// Converts a world coordinate to a chunk coordinate (ie. a coordinate whose origin is the origin of the chunk) {
fn position_in_chunk(position: Vector3<f32>) -> Vector3<f32> {
    Vector3::new(position.x % 16.0, position.y, position.z % 16.0)
}

fn chunk_of_position(position: Vector3<f32>) -> Vector2<i32> {
    Vector2::new((position.x / 16.0).floor() as i32, (position.z / 16.0).floor() as i32)
}
