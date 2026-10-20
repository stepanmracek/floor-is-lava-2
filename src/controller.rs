use crate::{player::Player, scene::Scene};
use kiss3d::prelude::*;

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
