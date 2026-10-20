mod controller;
mod game;
mod game_state;
mod player;
mod scene;

use crate::{
    game::Game,
    game_state::{GameState, StartState},
};
use kiss3d::prelude::*;

#[kiss3d::main]
async fn main() {
    let mut window = Window::new("Floor is lava!").await;
    window.set_shadows_enabled(false);
    window.set_samples(NumSamples::One);
    //window.set_vsync(false);

    let ai = std::env::args().nth(1).is_some_and(|arg| arg == "ai");
    let mut game = Game::new(ai);
    let mut state: Box<dyn GameState> = Box::new(StartState::default());

    while window
        .render_3d(&mut game.scene.world, &mut game.scene.camera)
        .await
    {
        if let Some(new_state) = state.step(&mut window, &mut game) {
            state = new_state;
        }
    }
}
