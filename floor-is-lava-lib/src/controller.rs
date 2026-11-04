use rand::{SeedableRng, rngs::StdRng, seq::IteratorRandom};

use glam::{IVec2, ivec2};

use crate::{BlockColor, Cell, Cells, JUMP_DURATION, LAVA_SPEED, PlayerAction, PlayerColor};

pub trait Controller {
    fn get_action(&mut self, cells: &Cells) -> Option<PlayerAction>;
}

pub struct SimpleAiController {
    color: PlayerColor,
    rng: StdRng,
}

impl SimpleAiController {
    pub fn new(color: PlayerColor, seed: u64) -> Self {
        Self {
            color,
            rng: StdRng::seed_from_u64(seed),
        }
    }

    fn value(color: PlayerColor, cell: &Cell) -> usize {
        if cell.color == color.into() {
            // my color -> no value
            0
        } else if cell.color == BlockColor::Neutral {
            // neutral color -> regular value
            cell.points
        } else {
            // opponent color -> twice as value
            cell.points * 2
        }
    }
}

pub fn find_player(cells: &Cells, player_color: PlayerColor) -> Option<(&Cell, IVec2)> {
    for (y, row) in cells.iter().enumerate() {
        for (x, cell) in row.iter().enumerate() {
            if let Some(cell) = cell
                && cell.has_player(player_color)
            {
                return Some((cell, ivec2(x as i32, y as i32)));
            }
        }
    }
    None
}

pub fn get_cell(cells: &Cells, pos: IVec2) -> Option<&Cell> {
    let row = usize::try_from(pos.y).ok()?;
    let col = usize::try_from(pos.x).ok()?;
    cells.get(row)?.get(col)?.as_ref()
}

impl Controller for SimpleAiController {
    fn get_action(&mut self, cells: &Cells) -> Option<PlayerAction> {
        let (_my_cell, my_pos) = find_player(cells, self.color)?;

        const DIRECTIONS: [(PlayerAction, IVec2); 4] = [
            (PlayerAction::Up, ivec2(0, 1)),
            (PlayerAction::Down, ivec2(0, -1)),
            (PlayerAction::Left, ivec2(-1, 0)),
            (PlayerAction::Right, ivec2(1, 0)),
        ];

        let plausible_moves = DIRECTIONS
            .iter()
            // consider only actions that lands on a block
            .filter_map(|(action, dir)| get_cell(cells, my_pos + dir).map(|cell| (action, cell)))
            // filter-out cell too close to the lava
            .filter(|(_action, cell)| cell.distance_from_lava > LAVA_SPEED * JUMP_DURATION)
            .collect::<Vec<_>>();

        let color = self.color;
        let best = plausible_moves
            .iter()
            .map(|(_, cell)| Self::value(color, cell))
            .max()?;
        plausible_moves
            .iter()
            .filter(|(_, cell)| Self::value(color, cell) == best)
            .choose(&mut self.rng)
            .map(|(action, _cell)| **action)
    }
}
