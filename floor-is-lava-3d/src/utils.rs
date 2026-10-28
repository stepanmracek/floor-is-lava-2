use glam::{IVec2, Vec3, vec3};

pub fn ivec2_to_3d(position: IVec2) -> Vec3 {
    vec3(position.x as f32, position.y as f32, -position.y as f32)
}

pub fn y_to_3d(y: f32) -> Vec3 {
    vec3(0.0, y, -y)
}
