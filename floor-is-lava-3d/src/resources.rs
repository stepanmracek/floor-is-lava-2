use glam::{vec2, vec3};
use kiss3d::resource::{MeshManager3d, TextureManager};

#[rustfmt::skip]
const BLOCK_TEXTURES: &[(&str, &[u8])] = &[
    ("block-1-gray", include_bytes!("../assets/textures/block-1-gray.png")),
    ("block-1-blue", include_bytes!("../assets/textures/block-1-blue.png")),
    ("block-1-red",  include_bytes!("../assets/textures/block-1-red.png")),
    ("block-2-gray", include_bytes!("../assets/textures/block-2-gray.png")),
    ("block-2-blue", include_bytes!("../assets/textures/block-2-blue.png")),
    ("block-2-red",  include_bytes!("../assets/textures/block-2-red.png")),
    ("block-3-gray", include_bytes!("../assets/textures/block-3-gray.png")),
    ("block-3-blue", include_bytes!("../assets/textures/block-3-blue.png")),
    ("block-3-red",  include_bytes!("../assets/textures/block-3-red.png")),
    ("block-4-gray", include_bytes!("../assets/textures/block-4-gray.png")),
    ("block-4-blue", include_bytes!("../assets/textures/block-4-blue.png")),
    ("block-4-red",  include_bytes!("../assets/textures/block-4-red.png")),
    ("block-5-gray", include_bytes!("../assets/textures/block-5-gray.png")),
    ("block-5-blue", include_bytes!("../assets/textures/block-5-blue.png")),
    ("block-5-red",  include_bytes!("../assets/textures/block-5-red.png")),
    ("block-6-gray", include_bytes!("../assets/textures/block-6-gray.png")),
    ("block-6-blue", include_bytes!("../assets/textures/block-6-blue.png")),
    ("block-6-red",  include_bytes!("../assets/textures/block-6-red.png")),
    ("block-7-gray", include_bytes!("../assets/textures/block-7-gray.png")),
    ("block-7-blue", include_bytes!("../assets/textures/block-7-blue.png")),
    ("block-7-red",  include_bytes!("../assets/textures/block-7-red.png")),
    ("block-8-gray", include_bytes!("../assets/textures/block-8-gray.png")),
    ("block-8-blue", include_bytes!("../assets/textures/block-8-blue.png")),
    ("block-8-red",  include_bytes!("../assets/textures/block-8-red.png")),
    ("block-9-gray", include_bytes!("../assets/textures/block-9-gray.png")),
    ("block-9-blue", include_bytes!("../assets/textures/block-9-blue.png")),
    ("block-9-red",  include_bytes!("../assets/textures/block-9-red.png")),
];

fn generate_block_mesh() -> kiss3d::procedural::RenderMesh {
    let mut mesh = kiss3d::procedural::unit_cuboid();
    mesh.unify_index_buffer();

    let normals = mesh.normals.clone().unwrap();
    if let (coords, Some(uvs)) = (&mesh.coords.clone(), &mut mesh.uvs) {
        for (i, uv) in uvs.iter_mut().enumerate() {
            if normals[i].z > 0.5 {
                // Front face (-Z): map to top half [V: 0.0 .. 0.5]
                let u = if coords[i].x > 0.0 { 1.0 } else { 0.0 };
                let v = if coords[i].y > 0.0 { 0.0 } else { 0.5 };
                *uv = vec2(u, v);
            } else {
                // All other 5 faces: sample the plain solid color in the lower half
                *uv = vec2(0.5, 0.75);
            }
        }
    }

    mesh.coords.iter_mut().for_each(|i| {
        *i -= vec3(0.0, 0.5, 0.0);
    });

    mesh
}

pub fn init_resources() {
    MeshManager3d::get_global_manager(|mesh_mngr| {
        if mesh_mngr.get("block_cube").is_none() {
            mesh_mngr.add_render_mesh(generate_block_mesh(), false, "block_cube");
        }
    });

    TextureManager::get_global_manager(|texture_mngr| {
        for (name, bytes) in BLOCK_TEXTURES {
            texture_mngr.add_image_from_memory(bytes, name);
        }
    });
}
