mod constants;
mod controller;
mod player;
mod resources;
mod utils;
mod world;

use floor_is_lava_lib::{Clock, Controller, Game, PlayerColor, SimpleAiController};
use floor_is_lava_policy::PolicyController;
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

    const HARD_AI_POLICY: &[u8] = include_bytes!(
        "../../floor-is-lava-learn/snapshots/actor-critic-ec-policy3-04000.safetensors"
    );

    let mut red_ctrl: Box<dyn Controller> = match arg.as_deref() {
        None => Box::new(KeyboardController::wasd(&keyboard_state)),
        Some("ai-easy") => Box::new(SimpleAiController::new(PlayerColor::Red, 42)),
        Some("ai-hard") => {
            Box::new(PolicyController::from_bytes(HARD_AI_POLICY, PlayerColor::Red, 42).unwrap())
        }
        Some(other) => panic!("Unknown controller: {other}"),
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

        let fps = 1.0 / clock.dt;
        window.draw_text(
            &format!("{fps:.0}FPS"),
            vec2(5.0, window.height() as f32 - 20.0),
            15.0,
            &font,
            WHITE,
        );

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
