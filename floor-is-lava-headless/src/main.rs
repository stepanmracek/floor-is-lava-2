use floor_is_lava_lib::{Clock, Controller, Game, PlayerColor, SimpleAiController};

fn main() {
    let seed = std::env::args().nth(1).unwrap().parse().unwrap();
    let mut game = Game::with_seed(seed);
    let mut clock = Clock::fixed(1.0 / 60.0);

    let mut red_ctrl = SimpleAiController::new(PlayerColor::Red);
    let mut blue_ctrl = SimpleAiController::new(PlayerColor::Blue);

    for i in 0..60 * 60 * 60 * 24 {
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
}
