mod constants;
mod controller;
mod player;
mod resources;
mod utils;
mod world;

use floor_is_lava_lib::{Clock, Game};
use glam::vec2;
use kiss3d::prelude::*;

use crate::controller::{Controller, KeyboardController};
use crate::world::WorldView;

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
