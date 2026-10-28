use std::path::Path;

use floor_is_lava_lib::BlockColor;
use glam::{vec2, vec3};
use kiss3d::resource::{MeshManager3d, TextureManager};

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

    for p in 1..=9 {
        for color in [BlockColor::Neutral, BlockColor::Blue, BlockColor::Red] {
            let name = format!("block-{p}-{color}");
            let path = format!("assets/textures/{name}.png");
            TextureManager::get_global_manager(|texture_mngr| {
                texture_mngr.add(Path::new(&path), &name);
            });
        }
    }
}
