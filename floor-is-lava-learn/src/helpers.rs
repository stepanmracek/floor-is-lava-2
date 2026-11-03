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
