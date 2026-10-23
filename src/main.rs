mod controller;
mod game;
mod game_state;
mod player;
mod scene;

use crate::{
    game::Game,
    game_state::{GameState, StartState},
    scene::Clock,
};
use kiss3d::prelude::*;

fn headless() {
    let mut game = Game::new(true, true, true);
    let mut clock = Clock::fixed(1.0 / 60.0);

    for step in 0..60 * 60 * 60 {
        // one hour
        clock.tick();
        game.step_simulation(&clock);

        if step % 60 == 0 {
            println!(
                "time: {:.1}s: red: {}; blue: {}",
                clock.game_time, game.scene.red_score, game.scene.blue_score
            )
        }
    }
}

#[kiss3d::main]
async fn main() {
    let arg = std::env::args().nth(1);

    if arg.as_ref().is_some_and(|arg| arg == "headless") {
        headless();
        return;
    }

    let mut window = Window::new("Floor is lava!").await;
    window.set_shadows_enabled(false);
    window.set_samples(NumSamples::One);
    window.set_vsync(false);

    let ai = arg.is_some_and(|arg| arg == "ai");
    let mut game = Game::new(ai, false, false);
    let mut state: Box<dyn GameState> = Box::new(StartState::default());

    while window
        .render_3d(
            game.scene.world.as_mut().unwrap(),
            game.scene.camera.as_mut().unwrap(),
        )
        .await
    {
        if let Some(new_state) = state.step(&mut window, &mut game) {
            state = new_state;
        }
    }
}
