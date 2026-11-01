//! Learned policy without any ML framework: the observation encoding and a plain-Rust
//! forward pass of the policy network. Shared by the training (fast rollouts) and by
//! the game front-ends.
mod controller;
mod mlp;
mod observation;

pub use controller::*;
pub use mlp::*;
pub use observation::*;
