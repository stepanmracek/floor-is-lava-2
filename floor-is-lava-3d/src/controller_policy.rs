use std::error::Error;

use floor_is_lava_lib::{Cells, Controller, PlayerAction, PlayerColor};
use floor_is_lava_policy::{ACTIONS, PlainPolicy, observation};

/// Plays with a policy trained in `floor-is-lava-learn` and exported to SafeTensors
/// (`floor-is-lava-learn export <policy.bpk> <policy.safetensors>`).
pub struct PolicyController {
    policy: PlainPolicy,
    color: PlayerColor,
}

impl PolicyController {
    pub fn from_bytes(bytes: &[u8], color: PlayerColor) -> Result<Self, Box<dyn Error>> {
        Ok(Self {
            policy: PlainPolicy::from_bytes(bytes)?,
            color,
        })
    }
}

impl Controller for PolicyController {
    /// Samples from the policy instead of taking the most likely action: the policy was
    /// trained and evaluated that way, deciding on every tick while idle.
    fn get_action(&mut self, cells: &Cells) -> Option<PlayerAction> {
        let observation = observation(cells, self.color);
        ACTIONS[self.policy.sample_action(&observation, rand::random())]
    }
}
