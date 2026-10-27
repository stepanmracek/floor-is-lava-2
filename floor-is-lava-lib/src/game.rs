use std::time::Instant;

use glam::{IVec2, ivec2};
use rand::{RngExt, seq::IteratorRandom};

const LAVA_SPEED: f32 = 1.0 / 4.0;
const JUMP_DURATION: f32 = 1.0;
const DYING_DURATION: f32 = 1.0;
const FALLING_SPEED: f32 = 9.8;

pub enum BlockColor {
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

pub struct Block {
    pub id: usize,
    pub position: IVec2,
    pub points: usize,
    pub color: BlockColor,
}

pub enum PlayerColor {
    Red,
    Blue,
}

pub struct Jumping {
    start_time: f32,
    from: IVec2,
    to: IVec2,
}

pub struct Idle {
    pub position: IVec2,
}

pub struct Falling {
    started_falling_at: IVec2,
    vertical_pos: f32,
}

pub struct Dying {
    started_falling_at: IVec2,
    vertical_pos: f32,
    start_time: f32,
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

pub enum PlayerAction {
    Up,
    Down,
    Left,
    Right,
}

impl PlayerAction {
    fn to_direction(&self) -> IVec2 {
        match self {
            PlayerAction::Up => IVec2::Y,
            PlayerAction::Down => IVec2::NEG_Y,
            PlayerAction::Left => IVec2::NEG_X,
            PlayerAction::Right => IVec2::X,
        }
    }
}

pub struct Game {
    pub blocks: Vec<Block>,
    pub lava_height: f32,
    pub red_player: Player,
    pub blue_player: Player,
}

pub enum ClockMode {
    Realtime { start: Instant, last_time: Instant },
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
            mode: ClockMode::Realtime {
                start: now,
                last_time: now,
            },
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
            ClockMode::Realtime { start, last_time } => {
                let now = Instant::now();
                self.game_time = now.duration_since(*start).as_secs_f32();
                self.dt = now.duration_since(*last_time).as_secs_f32().min(0.05);
                *last_time = now;
            }
            ClockMode::Fixed { dt } => {
                self.dt = *dt;
                self.game_time += *dt;
            }
        }
    }
}

pub struct StepOutcome {
    pub new_block_ids: Vec<usize>,
    pub removed_block_ids: Vec<usize>,
}

impl Game {
    pub fn new() -> Self {
        let mut blocks = vec![];
        for (id, x) in (-3..=3).enumerate() {
            Game::spawn_block(id, &mut blocks, ivec2(x, 0));
        }

        Self {
            blocks,
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
        }
    }

    pub fn spawn_block(id: usize, blocks: &mut Vec<Block>, position: IVec2) {
        let points = rand::rng().random_range(1..=9);
        let block = Block {
            id,
            position,
            points,
            color: BlockColor::Neutral,
        };
        blocks.push(block);
    }

    pub fn is_standing_on_block(blocks: &mut [Block], player_pos: IVec2) -> Option<&mut Block> {
        blocks.iter_mut().find(|block| block.position == player_pos)
    }

    pub fn step(
        &mut self,
        clock: &Clock,
        red_action: Option<PlayerAction>,
        blue_action: Option<PlayerAction>,
    ) -> StepOutcome {
        if let PlayerState::Idle(idle) = &self.red_player.state {
            if let Some(red_action) = red_action {
                self.red_player.state = PlayerState::Jumping(Jumping {
                    start_time: clock.game_time,
                    from: idle.position,
                    to: idle.position + red_action.to_direction(),
                })
            }
        }
        if let PlayerState::Idle(idle) = &self.blue_player.state {
            if let Some(blue_action) = blue_action {
                self.blue_player.state = PlayerState::Jumping(Jumping {
                    start_time: clock.game_time,
                    from: idle.position,
                    to: idle.position + blue_action.to_direction(),
                })
            }
        }

        self.lava_height += clock.dt * LAVA_SPEED;

        let mut max_height = 0;
        let max_id = self.blocks.last().unwrap().id;
        let mut removed_block_ids = vec![];
        self.blocks.retain(|block| {
            let block_height = block.position.y;
            if (block_height as f32) < self.lava_height {
                match block.color {
                    BlockColor::Red => self.red_player.score += block.points,
                    BlockColor::Blue => self.blue_player.score += block.points,
                    _ => {}
                }
                removed_block_ids.push(block.id);
                false
            } else {
                if block_height > max_height {
                    max_height = block_height;
                }
                true
            }
        });

        let mut new_block_ids = vec![];
        if (max_height as f32) - self.lava_height < 3.0 {
            let y = max_height + 1;
            let mut rng = rand::rng();
            let guaranteed_x = (-3..=3).choose(&mut rng).unwrap();
            for (id, x) in (-3..=3).enumerate() {
                if x == guaranteed_x || rng.random::<f32>() < 0.7 {
                    let new_block_id = id + 1 + max_id;
                    Game::spawn_block(new_block_id, &mut self.blocks, ivec2(x, y));
                    new_block_ids.push(new_block_id);
                }
            }
        }

        let mut players: [&mut Player; 2] = [&mut self.blue_player, &mut self.red_player];
        if rand::random::<bool>() {
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
                            landing_block.color = match player.color {
                                PlayerColor::Red => BlockColor::Red,
                                PlayerColor::Blue => BlockColor::Blue,
                            };
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
                        let mut rng = rand::rng();
                        if let Some(block) = self
                            .blocks
                            .iter()
                            .filter(|b| (b.position.y as f32) > (self.lava_height + 1.0))
                            .choose(&mut rng)
                        {
                            player.state = PlayerState::Idle(Idle {
                                position: block.position,
                            })
                        }
                    }
                }
            }
        }

        StepOutcome {
            new_block_ids,
            removed_block_ids,
        }
    }
}
