use crate::{
    game::STEP_SIZE,
    player::{Player, PlayerColor},
    scene::{Block, BlockColor, Scene},
};
use kiss3d::prelude::*;
use rand::seq::IteratorRandom;

#[derive(PartialEq, Clone, Copy)]
pub enum PlayerAction {
    Up,
    Down,
    Left,
    Right,
}

pub trait Controller {
    fn get_action(
        &mut self,
        window: &Window,
        scene: &Scene,
        this_player: &Player,
        other_player: &Player,
    ) -> Option<PlayerAction>;
}

pub struct KeyboardController {
    up: Key,
    down: Key,
    left: Key,
    right: Key,
}

impl KeyboardController {
    pub fn arrows() -> Self {
        Self {
            up: Key::Up,
            down: Key::Down,
            left: Key::Left,
            right: Key::Right,
        }
    }

    pub fn wasd() -> Self {
        Self {
            up: Key::W,
            down: Key::S,
            left: Key::A,
            right: Key::D,
        }
    }
}

impl Controller for KeyboardController {
    fn get_action(
        &mut self,
        window: &Window,
        _scene: &Scene,
        _this_player: &Player,
        _other_player: &Player,
    ) -> Option<PlayerAction> {
        if window.get_key(self.left) == Action::Press {
            Some(PlayerAction::Left)
        } else if window.get_key(self.right) == Action::Press {
            Some(PlayerAction::Right)
        } else if window.get_key(self.up) == Action::Press {
            Some(PlayerAction::Up)
        } else if window.get_key(self.down) == Action::Press {
            Some(PlayerAction::Down)
        } else {
            None
        }
    }
}

pub struct SimpleAiController;

struct CandidateMove<'a> {
    action: PlayerAction,
    block: &'a Block,
}

impl Controller for SimpleAiController {
    fn get_action(
        &mut self,
        _window: &Window,
        scene: &Scene,
        this_player: &Player,
        other_player: &Player,
    ) -> Option<PlayerAction> {
        let my_pos = this_player.scene_node.position();
        let my_x = (my_pos.x / STEP_SIZE).round() * STEP_SIZE;
        let my_z = (my_pos.z / STEP_SIZE).round() * STEP_SIZE;

        let lava_z = scene.lava.position().z;
        let opponent_block_color = match other_player.color {
            PlayerColor::Red => BlockColor::Red,
            PlayerColor::Blue => BlockColor::Blue,
        };

        // 4 possible jump offsets (dx, dz) matching Player::apply_action:
        // Left: +X, Right: -X, Up: +Z, Down: -Z
        const DIRECTIONS: [(PlayerAction, f32, f32); 4] = [
            (PlayerAction::Up, 0.0, STEP_SIZE),
            (PlayerAction::Left, STEP_SIZE, 0.0),
            (PlayerAction::Right, -STEP_SIZE, 0.0),
            (PlayerAction::Down, 0.0, -STEP_SIZE),
        ];

        // all adjacent blocks that exist and are safely above lava
        let safe_moves: Vec<CandidateMove> = DIRECTIONS
            .iter()
            .filter_map(|&(action, dx, dz)| {
                let target_x = my_x + dx;
                let target_z = my_z + dz;

                let block = scene.blocks.iter().find(|b| {
                    let pos = b.cube.position();
                    (pos.x - target_x).abs() < 0.1 && (pos.z - target_z).abs() < 0.1
                })?;

                // Only consider blocks safely ahead of lava
                if block.cube.position().z > lava_z + 0.5 {
                    Some(CandidateMove { action, block })
                } else {
                    None
                }
            })
            .collect();

        // 1. If near lava -> escape to the safest block (furthest ahead in Z)
        let distance_to_lava = my_z - lava_z;
        if distance_to_lava < 2.5 {
            return safe_moves
                .iter()
                .max_by_key(|m| (m.block.cube.position().z * 100.0) as i32)
                .map(|m| m.action);
        }

        // 2. Steal an opponent's block (prioritize highest points)
        let opponent_move = safe_moves
            .iter()
            .filter(|m| m.block.color == opponent_block_color)
            .max_by_key(|m| m.block.points);

        if let Some(m) = opponent_move {
            return Some(m.action);
        }

        // 3. Claim a neutral block (prioritize highest points)
        let neutral_move = safe_moves
            .iter()
            .filter(|m| m.block.color == BlockColor::Neutral)
            .max_by_key(|m| m.block.points);

        if let Some(m) = neutral_move {
            return Some(m.action);
        }

        // 4. Take any available safe move
        let mut rng = rand::rng();
        safe_moves.iter().choose(&mut rng).map(|m| m.action)
    }
}
