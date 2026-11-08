use rand::{SeedableRng, rngs::StdRng, seq::IteratorRandom};

use glam::{IVec2, ivec2};

use crate::{BlockColor, Cell, Cells, JUMP_DURATION, LAVA_SPEED, PlayerAction, PlayerColor};

pub trait Controller {
    fn get_action(&mut self, cells: &Cells) -> Option<PlayerAction>;
}

pub struct SimpleAiController {
    color: PlayerColor,
    rng: StdRng,
    urgency: f32,
    lookahead_decay: Option<f32>,
}

const DIRECTIONS: [(PlayerAction, IVec2); 4] = [
    (PlayerAction::Up, ivec2(0, 1)),
    (PlayerAction::Down, ivec2(0, -1)),
    (PlayerAction::Left, ivec2(-1, 0)),
    (PlayerAction::Right, ivec2(1, 0)),
];

impl SimpleAiController {
    pub fn new(color: PlayerColor, seed: u64) -> Self {
        Self {
            color,
            rng: StdRng::seed_from_u64(seed),
            urgency: 1.0,
            lookahead_decay: None,
        }
    }

    /// Looks one jump further: a move is worth the value of its target block plus
    /// `decay` times the best value reachable from there by the next jump.
    pub fn with_lookahead_decay(mut self, decay: f32) -> Self {
        self.lookahead_decay = Some(decay);
        self
    }

    /// Points bank only when a block sinks; the longer until then, the more time the
    /// opponent has to steal it back. A block's value is multiplied by
    /// `urgency ^ distance_from_lava`: 1.0 ignores the distance, smaller values prefer
    /// blocks close to the lava.
    pub fn with_urgency(mut self, urgency: f32) -> Self {
        self.urgency = urgency;
        self
    }

    fn value(&self, cell: &Cell) -> f32 {
        let points = if cell.color == self.color.into() {
            // my color -> no value
            0
        } else if cell.color == BlockColor::Neutral {
            // neutral color -> regular value
            cell.points
        } else {
            // opponent color -> twice as value
            cell.points * 2
        };
        points as f32 * self.urgency.powf(cell.distance_from_lava)
    }

    /// Value of a jump to `target`, or `None` if there is no block or it sinks before
    /// the `jumps`-th jump lands.
    fn jump_value(&self, cells: &Cells, target: IVec2, jumps: f32) -> Option<f32> {
        get_cell(cells, target)
            .filter(|cell| cell.distance_from_lava > LAVA_SPEED * JUMP_DURATION * jumps)
            .map(|cell| self.value(cell))
    }

    fn move_value(&self, cells: &Cells, my_pos: IVec2, target: IVec2) -> Option<f32> {
        let value = self.jump_value(cells, target, 1.0)?;
        let Some(decay) = self.lookahead_decay else {
            return Some(value);
        };
        let next = DIRECTIONS
            .iter()
            .map(|(_, dir)| target + dir)
            .filter(|&next| next != my_pos)
            .filter_map(|next| self.jump_value(cells, next, 2.0))
            .fold(0.0, f32::max);
        Some(value + decay * next)
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

        // consider only actions that land on a block that is not too close to the lava
        let plausible_moves = DIRECTIONS
            .iter()
            .filter_map(|&(action, dir)| {
                Some((action, self.move_value(cells, my_pos, my_pos + dir)?))
            })
            .collect::<Vec<_>>();

        let best = plausible_moves
            .iter()
            .map(|(_, value)| *value)
            .fold(f32::NEG_INFINITY, f32::max);
        // values are non-negative; relative tolerance keeps tiny values (low urgency) distinct
        plausible_moves
            .iter()
            .filter(|(_, value)| *value >= best * (1.0 - 1e-5))
            .choose(&mut self.rng)
            .map(|(action, _)| *action)
    }
}
