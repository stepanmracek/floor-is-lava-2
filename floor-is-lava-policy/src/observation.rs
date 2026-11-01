use floor_is_lava_lib::{BlockColor, Cells, PlayerColor, VISIBLE_ROWS};

const COLUMNS: usize = 7;
const FEATURES_PER_CELL: usize = 6;
pub const INPUTS: usize = VISIBLE_ROWS * COLUMNS * FEATURES_PER_CELL;

/// Observation of the game from the perspective of player `me`, `INPUTS` floats.
pub fn observation(cells: &Cells, me: PlayerColor) -> Vec<f32> {
    let my_color = BlockColor::from(me);

    let mut data = Vec::with_capacity(INPUTS);
    for cell in cells.iter().flatten() {
        match cell {
            Some(cell) => {
                let owner = match cell.color {
                    BlockColor::Neutral => 0.0,
                    color if color == my_color => 1.0,
                    _ => -1.0,
                };
                let (me_standing, other_standing, other_landing) = match me {
                    PlayerColor::Red => (cell.red_standing, cell.blue_standing, cell.blue_landing),
                    PlayerColor::Blue => (cell.blue_standing, cell.red_standing, cell.red_landing),
                };
                data.extend([
                    cell.points as f32 / 10.0,
                    owner,
                    cell.distance_from_lava,
                    me_standing,
                    other_standing,
                    other_landing,
                ]);
            }
            None => data.extend([-1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        }
    }
    data
}
