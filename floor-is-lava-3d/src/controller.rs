use std::{cell::RefCell, collections::HashSet, rc::Rc};

use floor_is_lava_lib::{Cells, Controller, PlayerAction};
use kiss3d::prelude::*;

#[derive(Default)]
pub struct KeyboardState {
    watched: Vec<Key>,
    pressed: HashSet<Key>,
}

impl KeyboardState {
    pub fn update(&mut self, window: &Window) {
        self.pressed.clear();
        for &key in &self.watched {
            if window.get_key(key) == Action::Press {
                self.pressed.insert(key);
            }
        }
    }

    fn is_pressed(&self, key: Key) -> bool {
        self.pressed.contains(&key)
    }
}

pub struct KeyboardController {
    state: Rc<RefCell<KeyboardState>>,
    up: Key,
    down: Key,
    left: Key,
    right: Key,
}

impl KeyboardController {
    fn new(state: &Rc<RefCell<KeyboardState>>, up: Key, down: Key, left: Key, right: Key) -> Self {
        state.borrow_mut().watched.extend([up, down, left, right]);
        Self {
            state: state.clone(),
            up,
            down,
            left,
            right,
        }
    }

    pub fn arrows(state: &Rc<RefCell<KeyboardState>>) -> Self {
        KeyboardController::new(state, Key::Up, Key::Down, Key::Left, Key::Right)
    }

    pub fn wasd(state: &Rc<RefCell<KeyboardState>>) -> Self {
        KeyboardController::new(state, Key::W, Key::S, Key::A, Key::D)
    }
}

impl Controller for KeyboardController {
    fn get_action(&mut self, _cells: &Cells) -> Option<PlayerAction> {
        let state = self.state.borrow();
        if state.is_pressed(self.left) {
            Some(PlayerAction::Left)
        } else if state.is_pressed(self.right) {
            Some(PlayerAction::Right)
        } else if state.is_pressed(self.up) {
            Some(PlayerAction::Up)
        } else if state.is_pressed(self.down) {
            Some(PlayerAction::Down)
        } else {
            None
        }
    }
}
