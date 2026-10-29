mod constants;
mod controller;
mod player;
mod resources;
mod utils;
mod world;

use floor_is_lava_lib::{Clock, Controller, Game, PlayerColor, SimpleAiController};
use glam::vec2;
use kiss3d::prelude::*;

use crate::{
    controller::{KeyboardController, KeyboardState},
    world::WorldView,
};

#[kiss3d::main]
async fn main() {
    let mut clock = Clock::realtime();
    let mut game = Game::new();

    let mut window = Window::new("Floor is lava!").await;
    window.set_shadows_enabled(false);

    let font = Font::default();
    let mut world = WorldView::from_game(&game);
    let keyboard_state = Rc::new(RefCell::new(KeyboardState::default()));

    let arg = std::env::args().nth(1);
    let ai = arg.is_some_and(|arg| arg == "ai");

    let mut red_ctrl: Box<dyn Controller> = if ai {
        Box::new(SimpleAiController::new(PlayerColor::Red))
    } else {
        Box::new(KeyboardController::wasd(&keyboard_state))
    };

    let mut blue_ctrl = KeyboardController::arrows(&keyboard_state);

    while window
        .render_3d(&mut world.world_node, &mut world.camera)
        .await
    {
        clock.tick();
        keyboard_state.borrow_mut().update(&window);

        let events = game.step(
            &clock,
            red_ctrl.get_action(game.cells()),
            blue_ctrl.get_action(game.cells()),
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
