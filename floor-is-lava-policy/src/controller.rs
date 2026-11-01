use std::{error::Error, path::Path};

use floor_is_lava_lib::{Cells, Controller, PlayerAction, PlayerColor};
use rand::{RngExt, SeedableRng, rngs::StdRng};

use crate::{Mlp, observation};

pub const ACTIONS: [Option<PlayerAction>; 5] = [
    Some(PlayerAction::Up),
    Some(PlayerAction::Down),
    Some(PlayerAction::Left),
    Some(PlayerAction::Right),
    None,
];

/// Plays with a policy trained in `floor-is-lava-learn` and exported to SafeTensors
/// (`floor-is-lava-learn export <policy.bpk> <policy.safetensors>`).
pub struct PolicyController {
    policy: Mlp,
    color: PlayerColor,
    rng: StdRng,
}

impl PolicyController {
    pub fn from_path(path: &Path, color: PlayerColor, seed: u64) -> Result<Self, Box<dyn Error>> {
        Self::new(Mlp::load(path)?, color, seed)
    }

    pub fn from_bytes(bytes: &[u8], color: PlayerColor, seed: u64) -> Result<Self, Box<dyn Error>> {
        Self::new(Mlp::from_bytes(bytes)?, color, seed)
    }

    fn new(policy: Mlp, color: PlayerColor, seed: u64) -> Result<Self, Box<dyn Error>> {
        let outputs = policy.output.bias.len();
        if outputs != ACTIONS.len() {
            return Err(format!("policy has {outputs} outputs, expected {}", ACTIONS.len()).into());
        }
        Ok(Self {
            policy,
            color,
            rng: StdRng::seed_from_u64(seed),
        })
    }
}

impl Controller for PolicyController {
    /// Samples from the policy instead of taking the most likely action: the policy was
    /// trained and evaluated that way, deciding on every tick while idle.
    fn get_action(&mut self, cells: &Cells) -> Option<PlayerAction> {
        let observation = observation(cells, self.color);
        ACTIONS[self.policy.sample_action(&observation, self.rng.random())]
    }
}
