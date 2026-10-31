use std::{collections::HashMap, path::Path};

use floor_is_lava_lib::{Block, BlockId, Clock, Game, GameEvent};
use glam::vec3;
use kiss3d::prelude::*;

use crate::{
    constants::{CAMERA_OFFSET, LIGHT_OFFSET},
    player::PlayerView,
    resources::init_resources,
    utils::{ivec2_to_3d, y_to_3d},
};

pub struct WorldView {
    pub world_node: SceneNode3d,
    pub camera: OrbitCamera3d,
    red: PlayerView,
    blue: PlayerView,
    light: SceneNode3d,
    lava_node: SceneNode3d,
    blocks: HashMap<BlockId, SceneNode3d>,
}

impl WorldView {
    pub fn from_game(game: &Game) -> Self {
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
            .add_cube(12.0, 0.1, 12.0)
            .set_position(y_to_3d(game.lava_height))
            .set_texture_from_file(Path::new("assets/textures/lava.png"), "lava");

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

fn camera(game: &Game) -> OrbitCamera3d {
    let mut camera = OrbitCamera3d::new(y_to_3d(game.lava_height) + CAMERA_OFFSET, Vec3::ZERO);
    camera.rebind_rotate_button(None);
    camera.rebind_drag_button(None);
    camera.rebind_reset_key(None);
    camera.set_min_dist(camera.dist());
    camera.set_max_dist(camera.dist());
    camera
}
