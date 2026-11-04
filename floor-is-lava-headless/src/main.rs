use std::path::Path;

use clap::Parser;
use floor_is_lava_lib::{Clock, Controller, Game, PlayerColor, SimpleAiController};
use floor_is_lava_policy::PolicyController;

/// Policy-against-policy evaluation over 200 seeded games.
#[derive(Parser)]
struct Cli {
    /// Seed for the first game; game g uses seed + g.
    seed: u64,
    /// Red policy (.safetensors).
    red: String,
    /// Blue policy (.safetensors).
    blue: String,
}

fn load_controller(path: &str, color: PlayerColor, seed: u64) -> Box<dyn Controller> {
    match path {
        "simple" => Box::new(SimpleAiController::new(color, seed)),
        _ => Box::new(PolicyController::from_path(Path::new(path), color, seed).unwrap()),
    }
}

fn main() {
    let Cli { seed, red, blue } = Cli::parse();

    let mut red_ctrl = load_controller(&red, PlayerColor::Red, seed);
    let mut blue_ctrl = load_controller(&blue, PlayerColor::Blue, seed.wrapping_add(1 << 32));

    const GAMES: u64 = 200;
    let mut red_wins = 0;
    let mut blue_wins = 0;
    let mut total_margin = 0;

    for g in 0..GAMES {
        let mut game = Game::with_seed(seed + g);
        let mut clock = Clock::fixed(1.0 / 60.0);
        for i in 1..=60 * 60 {
            clock.tick();

            game.step(
                &clock,
                red_ctrl.get_action(game.cells()),
                blue_ctrl.get_action(game.cells()),
            );

            if i % (60 * 60) == 0 {
                println!(
                    "time: {:5.1}s; red: {}; blue: {}",
                    clock.game_time, game.red_player.score, game.blue_player.score
                );
            }
        }
        if game.red_player.score > game.blue_player.score {
            red_wins += 1;
        } else if game.red_player.score < game.blue_player.score {
            blue_wins += 1;
        }
        total_margin += game.red_player.score as i64 - game.blue_player.score as i64;
    }

    println!("Red wins: {red_wins}; Blue wins: {blue_wins}");
    println!(
        "Average margin in favour of red player: {:.1}",
        total_margin as f32 / GAMES as f32
    );
}
