use std::path::Path;

use floor_is_lava_lib::{Clock, Controller, Game, PlayerColor};
use floor_is_lava_policy::PolicyController;

fn main() {
    let args = std::env::args().collect::<Vec<_>>();
    let seed = args[1].parse::<u64>().unwrap();

    let mut red_ctrl =
        PolicyController::from_path(Path::new(&args[2]), PlayerColor::Red, seed).unwrap();
    let mut blue_ctrl = PolicyController::from_path(
        Path::new(&args[3]),
        PlayerColor::Blue,
        seed.wrapping_add(1 << 32),
    )
    .unwrap();

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
