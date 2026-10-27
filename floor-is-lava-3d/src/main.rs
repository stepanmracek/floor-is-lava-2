mod controller;

use floor_is_lava_lib::{
    Block, BlockColor, BlockId, Clock, Game, GameEvent, Player, PlayerColor, PlayerState,
};
use glam::{IVec2, vec2, vec3};
use kiss3d::prelude::*;
use std::{collections::HashMap, path::Path};

use crate::controller::{Controller, KeyboardController};

const CAMERA_OFFSET: Vec3 = vec3(0.0, 3.0, 7.0);
const LIGHT_OFFSET: Vec3 = vec3(5.0, 7.0, 10.0);
const JUMP_CLIP_DURATION: f32 = 1.0;

fn camera(game: &Game) -> OrbitCamera3d {
    let mut camera = OrbitCamera3d::new(y_to_3d(game.lava_height) + CAMERA_OFFSET, Vec3::ZERO);
    camera.rebind_rotate_button(None);
    camera.rebind_drag_button(None);
    camera.rebind_reset_key(None);
    camera.set_min_dist(camera.dist());
    camera.set_max_dist(camera.dist());
    camera
}

fn ivec2_to_3d(position: IVec2) -> Vec3 {
    vec3(position.x as f32, position.y as f32, -position.y as f32)
}

fn y_to_3d(y: f32) -> Vec3 {
    vec3(0.0, y, -y)
}

struct PlayerView {
    model: GltfModel,
    node: SceneNode3d,
    current_animation: usize,
}

impl PlayerView {
    fn new(world_node: &mut SceneNode3d, player: &Player) -> Self {
        let model_offset_x = match player.color {
            PlayerColor::Red => -0.15,
            PlayerColor::Blue => 0.15,
        };
        let model_texture = match player.color {
            PlayerColor::Red => "red",
            PlayerColor::Blue => "blue",
        };

        let mut node = world_node.add_group();
        match &player.state {
            PlayerState::Idle(idle) => {
                node.set_position(ivec2_to_3d(idle.position));
            }
            _ => panic!(),
        }

        let mut model = node.add_gltf(
            Path::new(&format!("assets/models/megarex/{model_texture}.gltf")),
            Vec3::ONE / 8.0,
        );
        model.root.set_position(vec3(model_offset_x, 0.0, 0.0));
        let current_animation = PlayerView::state_to_animation(&player.state);
        model.player.play_index(current_animation);
        model.player.set_looping(true);

        Self {
            model,
            node,
            current_animation,
        }
    }

    fn state_to_animation(player_state: &PlayerState) -> usize {
        match player_state {
            PlayerState::Idle(_) | PlayerState::Falling(_) => 0,
            PlayerState::Dying(_) => 7,
            PlayerState::Jumping(_) => 8,
        }
    }

    fn position(player: &Player, clock: &Clock) -> Vec3 {
        match &player.state {
            PlayerState::Idle(idle) => ivec2_to_3d(idle.position),
            PlayerState::Jumping(jumping) => {
                let from = ivec2_to_3d(jumping.from);
                let to = ivec2_to_3d(jumping.to);
                let t = jumping.progress(clock.game_time);
                let u = ((t - 0.3) / (0.8 - 0.3)).clamp(0.0, 1.0); // remap from [0.3, 0.8] to [0.0, 1.0]
                let ease = u * u * (3.0 - 2.0 * u); // cubic ease
                from.lerp(to, ease)
            }
            PlayerState::Falling(falling) => {
                ivec2_to_3d(falling.started_falling_at).with_y(falling.vertical_pos)
            }
            PlayerState::Dying(dying) => {
                ivec2_to_3d(dying.started_falling_at).with_y(dying.vertical_pos)
            }
        }
    }

    fn update(&mut self, player: &Player, clock: &Clock) {
        let animation = PlayerView::state_to_animation(&player.state);
        if self.current_animation != animation {
            self.current_animation = animation;
            self.model.player.play_index(animation);
        }
        self.node.set_position(PlayerView::position(player, clock));

        if let PlayerState::Jumping(jumping) = &player.state {
            self.model
                .player
                .seek(jumping.progress(clock.game_time) * JUMP_CLIP_DURATION);
        } else {
            self.model.player.update(clock.dt);
        }
    }
}

struct WorldView {
    world_node: SceneNode3d,
    red: PlayerView,
    blue: PlayerView,
    light: SceneNode3d,
    lava_node: SceneNode3d,
    camera: OrbitCamera3d,
    blocks: HashMap<BlockId, SceneNode3d>,
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
        *i -= vec3(0.0, 0.5, 0.0);
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

impl WorldView {
    fn from_game(game: &Game) -> Self {
        init_resources();
        let mut world_node = SceneNode3d::empty();
        let light = world_node
            .add_light(Light::point(100.0))
            .set_position(y_to_3d(game.lava_height) + LIGHT_OFFSET);
        let camera = camera(game);

        let red = PlayerView::new(&mut world_node, &game.red_player);
        let blue = PlayerView::new(&mut world_node, &game.blue_player);

        let mut blocks = HashMap::new();
        for block in game.blocks.iter() {
            blocks.insert(block.id, WorldView::spawn_block(block, &mut world_node));
        }

        let lava_node = world_node
            .add_cylinder_with_subdiv(12.0, 0.1, 32)
            .set_position(vec3(0.0, game.lava_height, -game.lava_height))
            //.set_texture_from_file(Path::new("assets/textures/lava.png"), "lava")
            .set_color(DARK_RED.with_alpha(0.5));

        WorldView {
            world_node,
            light,
            camera,
            red,
            blue,
            blocks,
            lava_node,
        }
    }

    fn block_texture_name(block: &Block) -> String {
        format!("block-{}-{}", block.points, block.color)
    }

    pub fn update(&mut self, events: &[GameEvent], game: &Game, clock: &Clock) {
        for event in events.iter() {
            match event {
                GameEvent::BlockSpawned(id) => {
                    if let Some(block) = game.block(*id) {
                        self.blocks
                            .insert(*id, WorldView::spawn_block(block, &mut self.world_node));
                    }
                }
                GameEvent::BlockRemoved(id) => {
                    if let Some(mut node) = self.blocks.remove(id) {
                        node.remove();
                    }
                }
                GameEvent::BlockColored(id) => {
                    if let (Some(block_node), Some(block)) =
                        (self.blocks.get_mut(id), game.block(*id))
                    {
                        let texture_name = WorldView::block_texture_name(block);
                        block_node.set_texture_with_name(&texture_name);
                    }
                }
            }
        }

        self.red.update(&game.red_player, clock);
        self.blue.update(&game.blue_player, clock);

        let players = (self.blue.node.position() + self.red.node.position()) / 2.0;
        let at = self.camera.at().lerp(players, clock.dt);
        self.camera
            .look_at(y_to_3d(game.lava_height) + CAMERA_OFFSET, at);
        self.light
            .set_position(y_to_3d(game.lava_height) + LIGHT_OFFSET);
        self.lava_node
            .set_position(vec3(0.0, game.lava_height, -game.lava_height));
    }

    fn spawn_block(block: &Block, world_node: &mut SceneNode3d) -> SceneNode3d {
        let texture_name = WorldView::block_texture_name(block);
        world_node
            .add_geom_with_name("block_cube", Vec3::splat(0.8))
            .expect("block_cube mesh not registered")
            .set_position(ivec2_to_3d(block.position))
            .set_texture_with_name(&texture_name)
    }
}

#[kiss3d::main]
async fn main() {
    let mut clock = Clock::realtime();
    let mut game = Game::new();

    let mut window = Window::new("Floor is lava!").await;
    window.set_shadows_enabled(false);
    //window.set_samples(NumSamples::One);
    let font = Font::default();
    let mut world = WorldView::from_game(&game);

    let mut red_ctrl = KeyboardController::wasd();
    let mut blue_ctrl = KeyboardController::arrows();

    while window
        .render_3d(&mut world.world_node, &mut world.camera)
        .await
    {
        clock.tick();

        let events = game.step(
            &clock,
            red_ctrl.get_action(&window),
            blue_ctrl.get_action(&window),
        );

        world.update(&events, &game, &clock);

        window.draw_text(
            &format!("{}", game.red_player.score),
            vec2(20.0, 20.0),
            30.0,
            &font,
            RED,
        );
        window.draw_text(
            &format!("{}", game.blue_player.score),
            vec2(window.width() as f32 - 60.0, 20.0),
            30.0,
            &font,
            BLUE,
        );
    }
}
