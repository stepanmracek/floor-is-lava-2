use glam::{IVec2, ivec2};
use rand::{RngExt, SeedableRng, rngs::StdRng, seq::IteratorRandom};
use std::time::Instant;

pub const LAVA_SPEED: f32 = 1.0 / 4.0;
pub const VISIBLE_ROWS: usize = 4;
pub const JUMP_DURATION: f32 = 1.0;
pub const DYING_DURATION: f32 = 3.0;
pub const FALLING_SPEED: f32 = 8.0;

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BlockId(usize);

impl BlockId {
    fn next(self) -> Self {
        BlockId(self.0 + 1)
    }
}

#[derive(Default, Clone, Copy, PartialEq)]
pub enum BlockColor {
    #[default]
    Neutral,
    Red,
    Blue,
}

impl std::fmt::Display for BlockColor {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            BlockColor::Neutral => write!(f, "gray"),
            BlockColor::Red => write!(f, "red"),
            BlockColor::Blue => write!(f, "blue"),
        }
    }
}

impl From<PlayerColor> for BlockColor {
    fn from(color: PlayerColor) -> Self {
        match color {
            PlayerColor::Red => BlockColor::Red,
            PlayerColor::Blue => BlockColor::Blue,
        }
    }
}

pub struct Block {
    pub id: BlockId,
    pub position: IVec2,
    pub points: usize,
    pub color: BlockColor,
}

#[derive(Clone, Copy)]
pub enum PlayerColor {
    Red,
    Blue,
}

pub struct Jumping {
    pub from: IVec2,
    pub to: IVec2,
    start_time: f32,
}

impl Jumping {
    pub fn progress(&self, game_time: f32) -> f32 {
        ((game_time - self.start_time) / JUMP_DURATION).clamp(0.0, 1.0)
    }
}

pub struct Idle {
    pub position: IVec2,
}

pub struct Falling {
    pub started_falling_at: IVec2,
    pub vertical_pos: f32,
}

pub struct Dying {
    pub started_falling_at: IVec2,
    pub vertical_pos: f32,
    start_time: f32,
}

impl Dying {
    pub fn progress(&self, game_time: f32) -> f32 {
        ((game_time - self.start_time) / DYING_DURATION).clamp(0.0, 1.0)
    }
}

pub enum PlayerState {
    Idle(Idle),
    Jumping(Jumping),
    Falling(Falling),
    Dying(Dying),
}

pub struct Player {
    pub score: usize,
    pub color: PlayerColor,
    pub state: PlayerState,
}

#[derive(Clone, Copy)]
pub enum PlayerAction {
    Up,
    Down,
    Left,
    Right,
}

impl PlayerAction {
    fn to_direction(self) -> IVec2 {
        match self {
            PlayerAction::Up => IVec2::Y,
            PlayerAction::Down => IVec2::NEG_Y,
            PlayerAction::Left => IVec2::NEG_X,
            PlayerAction::Right => IVec2::X,
        }
    }
}

#[derive(Default)]
pub struct Cell {
    pub points: usize,
    pub color: BlockColor,
    pub red_standing: f32,
    pub red_landing: f32,
    pub blue_standing: f32,
    pub blue_landing: f32,
    pub distance_from_lava: f32,
}

impl Cell {
    pub fn has_player(&self, color: PlayerColor) -> bool {
        match color {
            PlayerColor::Red => self.red_standing > 0.0,
            PlayerColor::Blue => self.blue_standing > 0.0,
        }
    }
}

pub type Cells = [[Option<Cell>; 7]; VISIBLE_ROWS];

pub struct Game {
    pub blocks: Vec<Block>,
    pub lava_height: f32,
    pub red_player: Player,
    pub blue_player: Player,
    next_block_id: BlockId,
    rng: StdRng,
    cells: Cells,
}

pub enum ClockMode {
    Realtime { last_time: Instant },
    Fixed { dt: f32 },
}

pub struct Clock {
    mode: ClockMode,
    pub dt: f32,
    pub game_time: f32,
}

impl Clock {
    pub fn realtime() -> Self {
        let now = Instant::now();
        Self {
            mode: ClockMode::Realtime { last_time: now },
            dt: 0.0,
            game_time: 0.0,
        }
    }

    pub fn fixed(dt: f32) -> Self {
        Self {
            mode: ClockMode::Fixed { dt },
            dt,
            game_time: 0.0,
        }
    }

    pub fn tick(&mut self) {
        match &mut self.mode {
            ClockMode::Realtime { last_time } => {
                let now = Instant::now();
                self.dt = now.duration_since(*last_time).as_secs_f32().min(0.05);
                self.game_time += self.dt;
                *last_time = now;
            }
            ClockMode::Fixed { dt } => {
                self.dt = *dt;
                self.game_time += *dt;
            }
        }
    }
}

pub enum GameEvent {
    BlockSpawned(BlockId),
    BlockRemoved(BlockId),
    BlockColored(BlockId),
}

impl Game {
    pub fn new() -> Self {
        Self::with_rng(StdRng::from_rng(&mut rand::rng()))
    }

    pub fn with_seed(seed: u64) -> Self {
        Self::with_rng(StdRng::seed_from_u64(seed))
    }

    fn with_rng(rng: StdRng) -> Self {
        let mut game = Self {
            blocks: vec![],
            lava_height: -2.0,
            blue_player: Player {
                score: 0,
                color: PlayerColor::Blue,
                state: PlayerState::Idle(Idle {
                    position: ivec2(2, 0),
                }),
            },
            red_player: Player {
                score: 0,
                color: PlayerColor::Red,
                state: PlayerState::Idle(Idle {
                    position: ivec2(-2, 0),
                }),
            },
            next_block_id: BlockId(0),
            cells: Default::default(),
            rng,
        };

        for x in -3..=3 {
            game.spawn_block(ivec2(x, 0));
        }

        game.update_cells(0.0);
        game
    }

    pub fn spawn_block(&mut self, position: IVec2) -> BlockId {
        let id = self.next_block_id;
        self.next_block_id = id.next();
        let points = self.rng.random_range(1..=9);
        self.blocks.push(Block {
            id,
            position,
            points,
            color: BlockColor::Neutral,
        });
        id
    }

    pub fn is_standing_on_block(blocks: &mut [Block], player_pos: IVec2) -> Option<&mut Block> {
        blocks.iter_mut().find(|block| block.position == player_pos)
    }

    pub fn block(&self, id: BlockId) -> Option<&Block> {
        self.blocks
            .binary_search_by_key(&id, |block| block.id)
            .ok()
            .map(|index| &self.blocks[index])
    }

    fn update_cells(&mut self, game_time: f32) {
        self.cells = Default::default();
        let min_height = self.blocks[0].position.y;

        for block in &self.blocks {
            let row = (block.position.y - min_height) as usize;
            let col = (block.position.x + 3) as usize;
            self.cells[row][col] = Some(Cell {
                points: block.points,
                color: block.color,
                distance_from_lava: block.position.y as f32 - self.lava_height,
                ..Default::default()
            });
        }

        for player in [&self.red_player, &self.blue_player] {
            let (from, to, progress) = match &player.state {
                PlayerState::Idle(idle) => (Some(idle.position), None, 0.0),
                PlayerState::Jumping(jumping) => (
                    Some(jumping.from),
                    Some(jumping.to),
                    jumping.progress(game_time),
                ),
                _ => (None, None, 0.0),
            };

            let pos_to_row_col = |pos: IVec2| {
                let row = usize::try_from(pos.y - min_height).ok()?;
                let col = usize::try_from(pos.x + 3).ok()?;
                Some((row, col))
            };

            if let Some(cell) = from
                .and_then(pos_to_row_col)
                .and_then(|(row, col)| self.cells.get_mut(row)?.get_mut(col)?.as_mut())
            {
                match player.color {
                    PlayerColor::Red => cell.red_standing = 1.0 - progress,
                    PlayerColor::Blue => cell.blue_standing = 1.0 - progress,
                }
            };

            if let Some(cell) = to
                .and_then(pos_to_row_col)
                .and_then(|(row, col)| self.cells.get_mut(row)?.get_mut(col)?.as_mut())
            {
                match player.color {
                    PlayerColor::Red => cell.red_landing = progress,
                    PlayerColor::Blue => cell.blue_landing = progress,
                }
            }
        }
    }

    pub fn step(
        &mut self,
        clock: &Clock,
        red_action: Option<PlayerAction>,
        blue_action: Option<PlayerAction>,
    ) -> Vec<GameEvent> {
        let mut events = vec![];

        if let PlayerState::Idle(idle) = &self.red_player.state
            && let Some(red_action) = red_action
        {
            self.red_player.state = PlayerState::Jumping(Jumping {
                start_time: clock.game_time,
                from: idle.position,
                to: idle.position + red_action.to_direction(),
            })
        }

        if let PlayerState::Idle(idle) = &self.blue_player.state
            && let Some(blue_action) = blue_action
        {
            self.blue_player.state = PlayerState::Jumping(Jumping {
                start_time: clock.game_time,
                from: idle.position,
                to: idle.position + blue_action.to_direction(),
            })
        }

        self.lava_height += clock.dt * LAVA_SPEED;

        let mut max_height = 0;
        self.blocks.retain(|block| {
            let block_height = block.position.y;
            if (block_height as f32) < self.lava_height {
                match block.color {
                    BlockColor::Red => self.red_player.score += block.points,
                    BlockColor::Blue => self.blue_player.score += block.points,
                    _ => {}
                }
                events.push(GameEvent::BlockRemoved(block.id));
                false
            } else {
                if block_height > max_height {
                    max_height = block_height;
                }
                true
            }
        });

        if (max_height as f32) - self.lava_height < (VISIBLE_ROWS - 1) as f32 {
            let y = max_height + 1;
            let guaranteed_x = (-3..=3).choose(&mut self.rng).unwrap();
            for x in -3..=3 {
                if x == guaranteed_x || self.rng.random::<f32>() < 0.7 {
                    let new_block_id = self.spawn_block(ivec2(x, y));
                    events.push(GameEvent::BlockSpawned(new_block_id));
                }
            }
        }

        let mut players: [&mut Player; 2] = [&mut self.blue_player, &mut self.red_player];
        if self.rng.random::<bool>() {
            players.swap(0, 1);
        }
        for player in players {
            match &mut player.state {
                PlayerState::Idle(idle) => {
                    if (idle.position.y as f32) < self.lava_height {
                        // in lava -> dying
                        player.state = PlayerState::Dying(Dying {
                            started_falling_at: idle.position,
                            vertical_pos: idle.position.y as f32,
                            start_time: clock.game_time,
                        })
                    }
                }
                PlayerState::Jumping(jumping) => {
                    // Jumped already to the target block?
                    if clock.game_time - jumping.start_time >= JUMP_DURATION {
                        if let Some(landing_block) =
                            Game::is_standing_on_block(&mut self.blocks, jumping.to)
                        {
                            // hit -> claim the block and become idle
                            landing_block.color = player.color.into();
                            events.push(GameEvent::BlockColored(landing_block.id));
                            player.state = PlayerState::Idle(Idle {
                                position: jumping.to,
                            })
                        } else {
                            // miss -> start falling
                            player.state = PlayerState::Falling(Falling {
                                started_falling_at: jumping.to,
                                vertical_pos: jumping.to.y as f32,
                            })
                        }
                    }
                }
                PlayerState::Falling(falling) => {
                    falling.vertical_pos -= FALLING_SPEED * clock.dt;
                    if falling.vertical_pos < self.lava_height {
                        // in lava -> dying
                        player.state = PlayerState::Dying(Dying {
                            started_falling_at: falling.started_falling_at,
                            vertical_pos: falling.vertical_pos,
                            start_time: clock.game_time,
                        })
                    }
                }
                PlayerState::Dying(dying) => {
                    if clock.game_time - dying.start_time > DYING_DURATION {
                        // find some safe block above to respawn
                        if let Some(block) = self
                            .blocks
                            .iter()
                            .filter(|b| (b.position.y as f32) > (self.lava_height + 1.0))
                            .choose(&mut self.rng)
                        {
                            player.state = PlayerState::Idle(Idle {
                                position: block.position,
                            })
                        }
                    }
                }
            }
        }

        self.update_cells(clock.game_time);
        events
    }

    pub fn cells(&self) -> &Cells {
        &self.cells
    }
}

impl Default for Game {
    fn default() -> Self {
        Self::new()
    }
}
