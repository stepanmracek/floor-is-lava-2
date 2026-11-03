use floor_is_lava_lib::{BlockColor, Cells, PlayerColor, VISIBLE_ROWS, find_player, get_cell};
use glam::ivec2;

const COLUMNS: usize = 7;
const FEATURES_PER_CELL: usize = 5;
pub const INPUTS: usize =
    ((VISIBLE_ROWS - 1) * 2 + 1) * ((COLUMNS - 1) * 2 + 1) * FEATURES_PER_CELL;

/// Observation centred on player `me`, `INPUTS` floats:
/// rows −3 … +3 (bottom to top) × columns −6 … +6 × 5 features.
/// Cells outside the board are encoded like a missing block.
/// `None` if `me` isn't on the board (dying, respawning).
pub fn observation(cells: &Cells, me: PlayerColor) -> Option<Vec<f32>> {
    let my_color = BlockColor::from(me);
    let (_, my_pos) = find_player(cells, me)?;

    let dy = VISIBLE_ROWS as i32 - 1;
    let dx = COLUMNS as i32 - 1;

    let mut data = Vec::with_capacity(INPUTS);
    for y in my_pos.y - dy..=my_pos.y + dy {
        for x in my_pos.x - dx..=my_pos.x + dx {
            if let Some(cell) = get_cell(cells, ivec2(x, y)) {
                let owner = match cell.color {
                    BlockColor::Neutral => 0.0,
                    color if color == my_color => 1.0,
                    _ => -1.0,
                };
                let (other_standing, other_landing) = match me {
                    PlayerColor::Red => (cell.blue_standing, cell.blue_landing),
                    PlayerColor::Blue => (cell.red_standing, cell.red_landing),
                };
                data.extend([
                    cell.points as f32 / 10.0,
                    owner,
                    cell.distance_from_lava,
                    other_standing,
                    other_landing,
                ]);
            } else {
                data.extend([-1.0, 0.0, 0.0, 0.0, 0.0])
            }
        }
    }

    Some(data)
}

#[cfg(test)]
mod tests {
    use super::*;
    use floor_is_lava_lib::Game;

    #[test]
    fn test_observation() {
        let game = Game::with_seed(42);
        for me in [PlayerColor::Red, PlayerColor::Blue] {
            let observation = observation(game.cells(), me).unwrap();
            assert_eq!(observation.len(), INPUTS);
        }
    }
}
