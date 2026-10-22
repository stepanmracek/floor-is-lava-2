use crate::{
    game::STEP_SIZE,
    player::{Player, PlayerColor},
    scene::{BlockColor, BlockInfo, Scene},
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

struct CandidateMove {
    action: PlayerAction,
    block: BlockInfo,
}

impl Controller for SimpleAiController {
    fn get_action(
        &mut self,
        _window: &Window,
        scene: &Scene,
        this_player: &Player,
        other_player: &Player,
    ) -> Option<PlayerAction> {
        let matrix = match this_player.color {
            PlayerColor::Red => scene.blocks_matrix(other_player, this_player),
            PlayerColor::Blue => scene.blocks_matrix(this_player, other_player),
        };

        // 1. Locate current position
        let (my_row, my_col, my_block) = matrix.iter().enumerate().find_map(|(r, row)| {
            row.iter().enumerate().find_map(|(c, cell)| {
                let block = cell.as_ref()?;
                if block.has_player(this_player.color) {
                    Some((r as isize, c as isize, block))
                } else {
                    None
                }
            })
        })?;

        let opponent_block_color = match other_player.color {
            PlayerColor::Red => BlockColor::Red,
            PlayerColor::Blue => BlockColor::Blue,
        };

        // 4 grid steps (d_row, d_col) matching Player::apply_action:
        // Up: +Z (row + 1), Left: +X (col + 1), Right: -X (col - 1), Down: -Z (row - 1)
        const DIRECTIONS: [(PlayerAction, isize, isize); 4] = [
            (PlayerAction::Up, 1, 0),
            (PlayerAction::Left, 0, 1),
            (PlayerAction::Right, 0, -1),
            (PlayerAction::Down, -1, 0),
        ];

        // Check each adjacent tile in the matrix
        let safe_moves: Vec<CandidateMove> = DIRECTIONS
            .iter()
            .filter_map(|&(action, d_row, d_col)| {
                let target_row = usize::try_from(my_row + d_row).ok()?;
                let target_col = usize::try_from(my_col + d_col).ok()?;

                let block = matrix.get(target_row)?.get(target_col)?.clone()?;

                if block.above_lava > 0.1 {
                    Some(CandidateMove { action, block })
                } else {
                    None
                }
            })
            .collect();

        // 1. If near lava -> escape to the safest block (furthest ahead of lava)
        if my_block.above_lava < STEP_SIZE / 2.0 {
            return safe_moves
                .iter()
                .max_by(|a, b| a.block.above_lava.total_cmp(&b.block.above_lava))
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
