use kiss3d::resource::{MeshManager3d, TextureManager};

const STONE_CUBE_GLB: &[u8] = include_bytes!("../assets/models/stone_cube.glb");

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

pub fn init_resources() {
    if MeshManager3d::get_global_manager(|mm| mm.get("block_cube").is_none()) {
        let model = kiss3d::loader::gltf::load_from_slice(STONE_CUBE_GLB).expect("not valid glTF");
        let gltf_node = model.root.data().children()[0].clone();
        let object_node = gltf_node.data().children()[0].clone();
        let mesh = object_node
            .data()
            .object()
            .expect("object has has no mesh")
            .mesh()
            .clone();
        MeshManager3d::get_global_manager(|mm| mm.add(mesh.clone(), "block_cube"));
    }

    TextureManager::get_global_manager(|texture_mngr| {
        for (name, bytes) in BLOCK_TEXTURES {
            texture_mngr.add_image_from_memory(bytes, name);
        }
    });
}
