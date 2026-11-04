use std::path::Path;

use floor_is_lava_lib::{Game, Player, PlayerColor};

pub fn opponent(my_color: PlayerColor) -> PlayerColor {
    match my_color {
        PlayerColor::Blue => PlayerColor::Red,
        PlayerColor::Red => PlayerColor::Blue,
    }
}

pub fn player(game: &Game, color: PlayerColor) -> &Player {
    match color {
        PlayerColor::Red => &game.red_player,
        PlayerColor::Blue => &game.blue_player,
    }
}

/// Path of the policy snapshot saved after `update`.
pub fn snapshot_path(prefix: &str, update: usize) -> String {
    format!("{prefix}policy-{update:05}.bpk")
}

/// Creates the directory the snapshots with this prefix go to.
pub fn create_snapshot_dir(prefix: &str) {
    let path = snapshot_path(prefix, 0);
    std::fs::create_dir_all(Path::new(&path).parent().unwrap()).unwrap();
}
