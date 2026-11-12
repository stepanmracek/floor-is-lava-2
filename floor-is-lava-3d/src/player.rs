use floor_is_lava_lib::{Clock, Player, PlayerColor, PlayerState};
use glam::vec3;
use kiss3d::prelude::*;

use crate::{
    constants::{DYING_CLIP_DURATION, JUMP_CLIP_DURATION},
    utils::ivec2_to_3d,
};

pub struct PlayerView {
    pub node: SceneNode3d,
    model: GltfModel,
    current_animation: usize,
}

const SCENE_GLB: &[u8] = include_bytes!("../assets/models/megarex/scene.glb");
const RED_TEXTURE: &[u8] = include_bytes!("../assets/models/megarex/red.png");
const BLUE_TEXTURE: &[u8] = include_bytes!("../assets/models/megarex/blue.png");

impl PlayerView {
    pub fn new(world_node: &mut SceneNode3d, player: &Player) -> Self {
        let model_offset_x = match player.color {
            PlayerColor::Red => -0.15,
            PlayerColor::Blue => 0.15,
        };

        let mut node = world_node.add_group();
        match &player.state {
            PlayerState::Idle(idle) => {
                node.set_position(ivec2_to_3d(idle.position));
            }
            _ => panic!(),
        }

        let mut model = node.add_gltf_from_memory(SCENE_GLB, Vec3::ONE / 8.0);
        model.root.set_texture_from_memory_recursive(
            match player.color {
                PlayerColor::Red => RED_TEXTURE,
                PlayerColor::Blue => BLUE_TEXTURE,
            },
            match player.color {
                PlayerColor::Red => "red_texture",
                PlayerColor::Blue => "blue_texture",
            },
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

    pub fn update(&mut self, player: &Player, clock: &Clock) {
        let animation = PlayerView::state_to_animation(&player.state);
        if self.current_animation != animation {
            self.current_animation = animation;
            self.model.player.play_index(animation);
        }
        self.node.set_position(PlayerView::position(player, clock));

        match &player.state {
            PlayerState::Jumping(jumping) => {
                self.model
                    .player
                    .seek(jumping.progress(clock.game_time) * JUMP_CLIP_DURATION);
            }
            PlayerState::Dying(dying) => {
                self.model
                    .player
                    .seek(dying.progress(clock.game_time) * DYING_CLIP_DURATION);
            }
            _ => {
                self.model.player.update(clock.dt);
            }
        }
    }
}
