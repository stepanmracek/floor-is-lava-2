use std::{collections::HashMap, path::Path};

use floor_is_lava_lib::game::{BlockColor, Clock, Game, Player, PlayerColor, StepOutcome};
use glam::ivec2;
use kiss3d::{
    glamx::{IVec2, vec3},
    prelude::*,
};

fn camera(game: &Game) -> OrbitCamera3d {
    let mut camera = OrbitCamera3d::new(vec3(0.0, game.lava_height + 3.0, 10.0), Vec3::ZERO);
    //camera.rebind_rotate_button(None);
    camera.rebind_drag_button(None);
    camera.rebind_reset_key(None);
    camera.set_min_dist(camera.dist());
    camera.set_max_dist(camera.dist());
    camera
}

fn to_3d(position: IVec2) -> Vec3 {
    vec3(position.x as f32, position.y as f32, -position.y as f32)
}

fn player_3d_position(player: &Player) -> Vec3 {
    match player.state {
        floor_is_lava_lib::game::PlayerState::Idle(ref idle) => to_3d(idle.position),
        _ => todo!(),
    }
}

struct World {
    world_node: SceneNode3d,
    red_player_model: GltfModel,
    red_player_node: SceneNode3d,
    blue_player_model: GltfModel,
    blue_player_node: SceneNode3d,
    light: SceneNode3d,
    lava_node: SceneNode3d,
    camera: OrbitCamera3d,
    blocks: HashMap<usize, SceneNode3d>,
}

fn load_player_model(player: &Player, world_node: &mut SceneNode3d) -> (GltfModel, SceneNode3d) {
    let model_offset_x = match player.color {
        PlayerColor::Red => -0.15,
        PlayerColor::Blue => 0.15,
    };
    let model_texture = match player.color {
        PlayerColor::Red => "red",
        PlayerColor::Blue => "blue",
    };
    let mut scene_node = world_node
        .add_group()
        .set_position(player_3d_position(player));

    let mut model = scene_node.add_gltf(
        Path::new(&format!("assets/models/megarex/{model_texture}.gltf")),
        Vec3::ONE / 8.0,
    );
    model.root.set_position(vec3(model_offset_x, 0.0, 0.0));
    model.player.play_index(0);
    model.player.set_looping(true);

    (model, scene_node)
}

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
                *uv = Vec2::new(u, v);
            } else {
                // All other 5 faces: sample the plain solid color in the lower half
                *uv = Vec2::new(0.5, 0.75);
            }
        }
    }

    mesh.coords.iter_mut().for_each(|i| {
        *i = *i - vec3(0.0, 0.5, 0.0);
    });

    mesh
}

fn init_resources() {
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

impl World {
    fn from_game(game: &Game) -> Self {
        init_resources();
        let mut world_node = SceneNode3d::empty();
        let light = world_node
            .add_light(Light::point(100.0))
            .set_position(Vec3::new(5.0, 5.0, 10.0));
        let camera = camera(game);

        let (red_player_model, red_player_node) =
            load_player_model(&game.red_player, &mut world_node);

        let (blue_player_model, blue_player_node) =
            load_player_model(&game.blue_player, &mut world_node);

        let mut blocks = HashMap::new();
        for block in game.blocks.iter() {
            let texture_name = format!("block-{}-gray", block.points);
            let cube = world_node
                .add_geom_with_name("block_cube", Vec3::splat(0.8))
                .expect("block_cube mesh not registered")
                .set_position(to_3d(block.position))
                .set_texture_with_name(&texture_name);
            blocks.insert(block.id, cube);
        }

        let lava_node = world_node
            .add_cylinder_with_subdiv(12.0, 0.1, 32)
            .set_position(vec3(0.0, game.lava_height, 0.0))
            //.set_texture_from_file(Path::new("assets/textures/lava.png"), "lava")
            .set_color(DARK_RED.with_alpha(0.5));

        World {
            world_node,
            light,
            camera,
            red_player_model,
            red_player_node,
            blue_player_model,
            blue_player_node,
            blocks,
            lava_node,
        }
    }

    pub fn update(&mut self, step_outcome: StepOutcome, game: &Game) {
        self.lava_node
            .set_position(vec3(0.0, game.lava_height, 0.0));

        for removed_id in step_outcome.removed_block_ids {
            let node = self.blocks.remove(&removed_id);
            node.unwrap().remove();
        }

        for added_id in step_outcome.new_block_ids {
            todo!()
        }
    }
}

#[kiss3d::main]
async fn main() {
    let mut clock = Clock::realtime();
    let mut game = Game::new();
    Game::spawn_block(101, &mut game.blocks, ivec2(1, 1));
    Game::spawn_block(102, &mut game.blocks, ivec2(2, 2));
    Game::spawn_block(103, &mut game.blocks, ivec2(3, 3));

    let mut window = Window::new("Floor is lava!").await;
    window.set_shadows_enabled(false);
    //window.set_samples(NumSamples::One);
    let mut world = World::from_game(&game);

    while window
        .render_3d(&mut world.world_node, &mut world.camera)
        .await
    {
        clock.tick();
        let step_outcome = game.step(&clock, None, None);
        world.update(step_outcome, &game);
    }
}
