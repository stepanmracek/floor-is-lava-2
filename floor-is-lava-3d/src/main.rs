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
    constants::GAME_DURATION,
    controller::{KeyboardController, KeyboardState},
    world::WorldView,
};

/// The red controller choice: the first command-line argument natively, the URL query
/// string (e.g. `index.html?ai-hard`) on the web.
fn controller_arg() -> Option<String> {
    #[cfg(not(target_arch = "wasm32"))]
    {
        std::env::args().nth(1)
    }
    #[cfg(target_arch = "wasm32")]
    {
        let search = web_sys::window()?.location().search().ok()?;
        let arg = search.trim_start_matches('?');
        (!arg.is_empty()).then(|| arg.to_string())
    }
}

#[kiss3d::main]
async fn main() {
    #[cfg(target_arch = "wasm32")]
    console_error_panic_hook::set_once();

    let mut clock = Clock::realtime();
    let mut game = Game::new();

    let mut window = Window::new("Floor is lava!").await;
    let font = Font::default();
    let mut world = WorldView::from_game(&game);
    let keyboard_state = Rc::new(RefCell::new(KeyboardState::default()));

    let arg = controller_arg();

    const HARD_AI_POLICY: &[u8] = include_bytes!(
        "../../floor-is-lava-learn/snapshots/flex/actor-critic-from-scratch-pool-policy-08000.safetensors"
    );

    let mut red_ctrl: Box<dyn Controller> = match arg.as_deref() {
        None => Box::new(KeyboardController::wasd(&keyboard_state)),
        Some("ai-easy") => Box::new(SimpleAiController::new(PlayerColor::Red, 42)),
        Some("ai-medium") => Box::new(
            SimpleAiController::new(PlayerColor::Red, 42)
                .with_urgency(0.3)
                .with_lookahead_decay(0.99),
        ),
        Some("ai-hard") => {
            Box::new(PolicyController::from_bytes(HARD_AI_POLICY, PlayerColor::Red, 42).unwrap())
        }
        Some(other) => panic!("Unknown controller: {other}"),
    };

    let mut blue_ctrl = KeyboardController::arrows(&keyboard_state);
    let mut started = false;

    while window
        .render_3d(&mut world.world_node, &mut world.camera)
        .await
    {
        if !started {
            let any_key_pressed = window
                .events()
                .iter()
                .any(|event| matches!(event.value, WindowEvent::Key(_, Action::Press, _)));
            if any_key_pressed {
                started = true;
                // Restart the clock
                clock = Clock::realtime();
            }
        }

        let time_left = (GAME_DURATION - clock.game_time).max(0.0);

        // Before the start and after the time runs out, the game is frozen.
        if started && time_left > 0.0 {
            clock.tick();
            keyboard_state.borrow_mut().update(&window);

            let events = game.step(
                &clock,
                red_ctrl.get_action(game.cells()),
                blue_ctrl.get_action(game.cells()),
            );

            world.update(&events, &game, &clock);
        }

        if started {
            // Remaining time: a thin white strip along the top edge
            let scale = window.scale_factor() as f32;
            let half_w = window.width() as f32 / scale / 2.0;
            let half_h = window.height() as f32 / scale / 2.0;
            let bar_half_w = half_w * time_left / GAME_DURATION;
            let bar_y = half_h - 4.0;
            if bar_half_w > 0.0 {
                let a = vec2(-bar_half_w, bar_y);
                let b = vec2(bar_half_w, bar_y);
                window.draw_line_2d(a, b, WHITE, 4.0);
            }

            let fps = 1.0 / clock.dt;
            let pos = vec2(5.0, window.height() as f32 - 20.0);
            window.draw_text(&format!("{fps:.0}FPS"), pos, 15.0, &font, WHITE);
        }

        let pos = vec2(20.0, 20.0);
        let score = &format!("{}", game.red_player.score);
        window.draw_text(score, pos, 30.0, &font, RED);

        let pos = vec2(window.width() as f32 - 60.0, 20.0);
        let score = &format!("{}", game.blue_player.score);
        window.draw_text(score, pos, 30.0, &font, BLUE);

        if !started {
            let pos = vec2(
                window.width() as f32 / 2.0 - 200.0,
                window.height() as f32 / 2.0 - 20.0,
            );
            window.draw_text("Press any key to start", pos, 40.0, &font, WHITE);
        }
    }
}
