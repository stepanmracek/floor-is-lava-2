use crate::{
    controller::PlayerAction,
    scene::{BlockColor, Clock, Scene},
};
use kiss3d::{
    glamx::{vec2, vec3},
    prelude::*,
};
use rand::seq::IteratorRandom;
use std::{f32::consts::PI, path::Path};

struct Jumping {
    start_time: f32,
    from: Vec2,
    to: Vec2,
}

struct Idle {
    xz: Vec2,
}

impl Idle {
    fn new(xz: Vec2) -> Self {
        Self { xz }
    }
}

struct Dying {
    start_time: f32,
}

enum PlayerState {
    Idle(Idle),
    Jumping(Jumping),
    Falling,
    Dying(Dying),
}

pub struct Player {
    state: PlayerState,
    pub scene_node: SceneNode3d,
    model: GltfModel,
    color: PlayerColor,
}

pub enum PlayerColor {
    Red,
    Blue,
}

const IDLE_CLIP: usize = 0;
const JUMP_CLIP: usize = 8;
const DEATH_CLIP: usize = 7;

impl Player {
    pub fn new(scene: &mut SceneNode3d, color: PlayerColor) -> Self {
        let start_x = match color {
            PlayerColor::Red => 4.0,
            PlayerColor::Blue => -4.0,
        };
        let model_offset_x = match color {
            PlayerColor::Red => 0.3,
            PlayerColor::Blue => -0.3,
        };
        let model_texture = match color {
            PlayerColor::Red => "red",
            PlayerColor::Blue => "blue",
        };
        let mut scene_node = scene.add_group().set_position(vec3(start_x, 0.0, 0.0));
        let mut model = scene_node.add_gltf(
            Path::new(&format!("assets/models/megarex/{model_texture}.gltf")),
            Vec3::ONE / 4.0,
        );
        model.root.set_position(vec3(model_offset_x, 0.0, 0.0));
        let mut player = Player {
            state: PlayerState::Idle(Idle::new(vec2(start_x, 0.0))),
            scene_node,
            model,
            color,
        };

        player
            .model
            .root
            .set_rotation(Quat::from_axis_angle(Vec3::Y, PI));
        player.model.player.play_index(0);
        player.model.player.set_looping(true);
        player
    }

    pub fn apply_action(&mut self, action: PlayerAction, clock: &Clock) {
        if let PlayerState::Idle(idle) = &self.state {
            let from = idle.xz;
            let d = match action {
                PlayerAction::Left => Vec2::X * 2.0,
                PlayerAction::Right => -Vec2::X * 2.0,
                PlayerAction::Up => Vec2::Y * 2.0,
                PlayerAction::Down => -Vec2::Y * 2.0,
            };

            let jumping = Jumping {
                start_time: clock.game_time,
                from,
                to: from + d,
            };
            self.state = PlayerState::Jumping(jumping);
            self.model.player.play_index(JUMP_CLIP);
        };
    }

    pub fn update(&mut self, clock: &Clock, scene: &mut Scene) {
        self.model.player.update(clock.dt);

        match &self.state {
            PlayerState::Jumping(jumping) => {
                let t = (clock.game_time - jumping.start_time).clamp(0.0, 1.0);
                let u = ((t - 0.3) / (0.8 - 0.3)).clamp(0.0, 1.0); // remap from [0.3, 0.8] to [0.0, 1.0]
                let ease = u * u * (3.0 - 2.0 * u); // cubic ease
                let mut pos = self.scene_node.position();
                let new_pos = jumping.from.lerp(jumping.to, ease);
                pos.x = new_pos.x;
                pos.z = new_pos.y;
                pos.y = new_pos.y;
                self.scene_node.set_position(pos);
                if t >= 1.0 {
                    if let Some(block) = scene.is_standing_on_block(jumping.to) {
                        block.set_color(match self.color {
                            PlayerColor::Red => BlockColor::Red,
                            PlayerColor::Blue => BlockColor::Blue,
                        });
                        self.model.player.play_index(IDLE_CLIP);
                        self.state = PlayerState::Idle(Idle::new(jumping.to));
                    } else {
                        // No block here! Transition to Falling
                        self.state = PlayerState::Falling;
                        self.model.player.play_index(3);
                    }
                }
            }
            PlayerState::Falling => {
                let mut pos = self.scene_node.position();
                pos.y -= 9.8 * clock.dt;
                self.scene_node.set_position(pos);

                if pos.y < scene.lava.position().y {
                    self.state = PlayerState::Dying(Dying {
                        start_time: clock.game_time,
                    });
                    self.model.player.play_index(DEATH_CLIP);
                    self.model.player.set_looping(false);
                }
            }
            PlayerState::Idle(_) => {
                let pos = self.scene_node.position();
                if pos.y < scene.lava.position().y {
                    self.state = PlayerState::Dying(Dying {
                        start_time: clock.game_time,
                    });
                    self.model.player.play_index(DEATH_CLIP);
                    self.model.player.set_looping(false);
                }
            }
            PlayerState::Dying(dying) => {
                if (clock.game_time - dying.start_time) > 1.0 {
                    self.respawn(scene);
                }
            }
        }
    }

    fn respawn(&mut self, scene: &Scene) {
        let lava_y = scene.lava.position().y;
        let mut rng = rand::rng();

        // Find some block that is safely above the lava (e.g. at least 1.0 unit above)
        if let Some(block) = scene
            .blocks
            .iter()
            .filter(|b| b.cube.position().y > lava_y + 1.0)
            .choose(&mut rng)
        {
            let bpos = block.cube.position();

            // Place player on top of the block
            self.scene_node
                .set_position(vec3(bpos.x, bpos.y + 0.75, bpos.z));

            // Reset animation
            self.model.player.play_index(IDLE_CLIP);
            self.model.player.set_looping(true);

            // Reset state
            self.state = PlayerState::Idle(Idle::new(vec2(bpos.x, bpos.z)));
        }
    }
}
