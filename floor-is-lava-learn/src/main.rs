mod actor_critic;
mod helpers;
mod policy;
mod reinforce;

use clap::{Parser, Subcommand};

/// RL training for floor-is-lava.
#[derive(Parser)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Train a policy from scratch with REINFORCE.
    Reinforce {
        seed: u64,
        /// Number of policy updates.
        #[arg(long, default_value_t = 10_000)]
        updates: usize,
        /// Print a log line every this many updates; 0 = never.
        #[arg(long, default_value_t = 20)]
        report_every: usize,
        /// Save a policy snapshot every this many updates; 0 = never.
        #[arg(long, default_value_t = 20)]
        snapshot_every: usize,
        /// Snapshots are saved as `<prefix>policy-<update>.bpk`.
        #[arg(long, default_value = "snapshots/reinforce-")]
        snapshot_prefix: String,
    },
    /// Train with actor-critic, optionally fine-tuning an existing policy.
    ActorCritic {
        seed: u64,
        /// Policy snapshot (.bpk) to start from; trains from scratch if omitted.
        policy: Option<String>,
        /// Number of updates, including the critic-only warm-up when fine-tuning.
        #[arg(long, default_value_t = 50_000)]
        updates: usize,
        /// Print a log line every this many updates; 0 = never.
        #[arg(long, default_value_t = 20)]
        report_every: usize,
        /// Save a policy snapshot every this many updates; 0 = never. Once the policy
        /// beats the simple AI, every snapshot also joins the opponent pool, so 0 means
        /// training against the simple AI only.
        #[arg(long, default_value_t = 500)]
        snapshot_every: usize,
        /// Snapshots are saved as `<prefix>policy-<update>.bpk`.
        #[arg(long, default_value = "snapshots/actor-critic-")]
        snapshot_prefix: String,
    },
    /// Export a burn snapshot to SafeTensors.
    Export {
        /// Input snapshot (.bpk).
        snapshot: String,
        /// Output file (.safetensors).
        output: String,
    },
}

fn main() {
    match Cli::parse().command {
        Command::Reinforce {
            seed,
            updates,
            report_every,
            snapshot_every,
            snapshot_prefix,
        } => reinforce::train(
            updates,
            seed,
            report_every,
            snapshot_every,
            &snapshot_prefix,
        ),
        Command::ActorCritic {
            seed,
            policy,
            updates,
            report_every,
            snapshot_every,
            snapshot_prefix,
        } => actor_critic::train(
            policy.as_deref(),
            updates,
            seed,
            report_every,
            snapshot_every,
            &snapshot_prefix,
        ),
        Command::Export { snapshot, output } => policy::export(&snapshot, &output),
    }
}
