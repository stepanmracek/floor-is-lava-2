use burn::optim::{AdamConfig, GradientsParams};
use burn::prelude::*;
use burn::tensor::activation::log_softmax;
use floor_is_lava_lib::{Clock, Controller, Game, PlayerColor, PlayerState, SimpleAiController};
use floor_is_lava_policy::{PlainPolicy, observation};
use rand::{RngExt, SeedableRng, rngs::StdRng};
use rayon::prelude::*;

use crate::policy::{ACTIONS, Policy};

const EPISODE_SECONDS: f32 = 60.0;
const DT: f32 = 1.0 / 60.0;
const GAMMA_PER_SECOND: f32 = 0.99;
const ENTROPY_COEF: f32 = 0.01;
const EPISODES_PER_UPDATE: usize = 32;
/// Width of the time buckets used for the time-dependent baseline.
const BASELINE_BUCKET_SECONDS: f32 = 1.0;
const LEARNING_RATE: f64 = 1e-3;

struct Episode {
    /// One observation per decision, i.e. per tick when the agent was idle.
    /// Flattened, stacking many small tensors at once is very slow.
    observations: Vec<f32>,
    actions: Vec<i64>,
    /// Tick of the episode at which each decision was made.
    decision_ticks: Vec<usize>,
    /// Discounted return for each decision.
    returns: Vec<f32>,
    my_score: usize,
    opponent_score: usize,
    deaths: usize,
}

fn discounted_returns(rewards: &[f32]) -> Vec<f32> {
    let gamma = GAMMA_PER_SECOND.powf(DT);
    let mut returns = vec![0.0; rewards.len()];
    let mut ret = 0.0;
    for t in (0..rewards.len()).rev() {
        ret = rewards[t] + gamma * ret;
        returns[t] = ret;
    }
    returns
}

/// Plays one episode as red against the simple AI.
/// `game_seed` seeds the game, `rng_seed` the sampling of actions.
fn play_episode(policy: &PlainPolicy, game_seed: u64, rng_seed: u64) -> Episode {
    let mut game = Game::with_seed(game_seed);
    let mut rng = StdRng::seed_from_u64(rng_seed);
    let mut clock = Clock::fixed(DT);
    let mut opponent = SimpleAiController::new(PlayerColor::Blue);

    let mut observations = vec![];
    let mut actions = vec![];
    let mut decision_ticks = vec![];
    let mut rewards = vec![];
    let mut deaths: usize = 0;
    let mut last_dying = false;

    let ticks = (EPISODE_SECONDS / DT) as usize;
    for tick in 0..ticks {
        let my_action = if matches!(game.red_player.state, PlayerState::Idle(_)) {
            let observation = observation(game.cells(), PlayerColor::Red);
            let action = policy.sample_action(&observation, rng.random());
            observations.extend(observation);
            actions.push(action as i64);
            decision_ticks.push(tick);
            ACTIONS[action]
        } else {
            None
        };

        let my_score_before = game.red_player.score;
        let opponent_score_before = game.blue_player.score;
        clock.tick();
        let opponent_action = opponent.get_action(game.cells());
        game.step(&clock, my_action, opponent_action);
        let dying = matches!(game.red_player.state, PlayerState::Dying(_));
        if !last_dying && dying {
            deaths += 1;
        }
        last_dying = dying;
        let my_gain = (game.red_player.score - my_score_before) as f32;
        let opponent_gain = (game.blue_player.score - opponent_score_before) as f32;
        rewards.push(my_gain - opponent_gain);
    }

    let tick_returns = discounted_returns(&rewards);
    Episode {
        observations,
        actions,
        returns: decision_ticks.iter().map(|&t| tick_returns[t]).collect(),
        decision_ticks,
        my_score: game.red_player.score,
        opponent_score: game.blue_player.score,
        deaths,
    }
}

/// Advantage = return minus the mean return of all decisions made in the same time bucket
/// of the episode, normalized to unit variance.
///
/// The return mostly depends on how much of the episode is left, so comparing against
/// decisions made at a similar time isolates the effect of the action itself.
fn advantages(returns: &[f32], ticks: &[usize]) -> Vec<f32> {
    let bucket_ticks = (BASELINE_BUCKET_SECONDS / DT).round() as usize;
    let buckets = ticks
        .iter()
        .max()
        .map_or(0, |&tick| tick / bucket_ticks + 1);
    let mut sums = vec![0.0; buckets];
    let mut counts = vec![0usize; buckets];
    for (&ret, &tick) in returns.iter().zip(ticks) {
        sums[tick / bucket_ticks] += ret;
        counts[tick / bucket_ticks] += 1;
    }

    let advantages: Vec<f32> = returns
        .iter()
        .zip(ticks)
        .map(|(&ret, &tick)| {
            let bucket = tick / bucket_ticks;
            ret - sums[bucket] / counts[bucket] as f32
        })
        .collect();

    let n = advantages.len() as f32;
    let std = (advantages.iter().map(|a| a * a).sum::<f32>() / n).sqrt();
    advantages.iter().map(|a| a / (std + 1e-8)).collect()
}

/// REINFORCE loss: -mean(log π(a|s) * advantage) - ENTROPY_COEF * mean(entropy)
///
/// Returns the loss and the mean policy entropy (for logging).
fn loss(policy: &Policy, episodes: Vec<Episode>, device: &Device) -> (Tensor<1>, f32) {
    let mut observations = vec![];
    let mut actions = vec![];
    let mut returns = vec![];
    let mut ticks = vec![];
    for episode in episodes {
        observations.extend(episode.observations);
        actions.extend(episode.actions);
        returns.extend(episode.returns);
        ticks.extend(episode.decision_ticks);
    }

    let advantages = advantages(&returns, &ticks);

    let shape = [actions.len(), observations.len() / actions.len()];
    let observations = Tensor::<2>::from_data(TensorData::new(observations, shape), device);
    let actions = Tensor::<1, Int>::from_data(actions.as_slice(), device);
    let advantages = Tensor::<1>::from_data(advantages.as_slice(), device);

    let log_probs = log_softmax(policy.forward(observations), 1);
    let entropy = (log_probs.clone().exp() * log_probs.clone())
        .sum_dim(1)
        .neg()
        .mean();
    let chosen_log_probs = log_probs
        .gather(1, actions.unsqueeze_dim(1))
        .squeeze_dim::<1>(1);

    let policy_loss = (chosen_log_probs * advantages).mean().neg();
    let mean_entropy = entropy.clone().into_scalar::<f32>();
    (policy_loss - entropy * ENTROPY_COEF, mean_entropy)
}

pub fn train(updates: usize, seed: u64) {
    std::fs::create_dir_all("snapshots").unwrap();
    let device = Device::default().autodiff();
    let mut policy = Policy::new(&device);
    let mut optimizer = AdamConfig::new().init();
    let mut rng = StdRng::seed_from_u64(seed);

    for update in 1..=updates {
        // play with a plain-Rust copy of the policy, episodes in parallel
        let rollout_policy = policy.to_plain();
        let seeds: Vec<(u64, u64)> = (0..EPISODES_PER_UPDATE)
            .map(|_| (rng.random(), rng.random()))
            .collect();
        let episodes: Vec<Episode> = seeds
            .into_par_iter()
            .map(|(game_seed, rng_seed)| play_episode(&rollout_policy, game_seed, rng_seed))
            .collect();

        let decisions: usize = episodes.iter().map(|e| e.actions.len()).sum();
        let mut histogram = [0usize; ACTIONS.len()];
        for &action in episodes.iter().flat_map(|e| &e.actions) {
            histogram[action as usize] += 1;
        }
        let my_score: usize = episodes.iter().map(|e| e.my_score).sum();
        let opponent_score: usize = episodes.iter().map(|e| e.opponent_score).sum();
        let deaths: usize = episodes.iter().map(|e| e.deaths).sum();

        let (loss, entropy) = loss(&policy, episodes, &device);
        let grads = GradientsParams::from_grads(loss.backward(), &policy);
        policy = optimizer.step(LEARNING_RATE, policy, grads);

        if update % 20 != 0 {
            continue;
        }

        let [up, down, left, right, noop] =
            histogram.map(|count| count as f32 / EPISODES_PER_UPDATE as f32);
        println!(
            "update {update:5}: my score {:6.1}, opponent {:6.1}, decisions {:6.0}, deaths {:6.0}, \
             up {up:4.0}, down {down:4.0}, left {left:4.0}, right {right:4.0}, no-op {noop:4.0}, entropy {entropy:.2}",
            my_score as f32 / EPISODES_PER_UPDATE as f32,
            opponent_score as f32 / EPISODES_PER_UPDATE as f32,
            decisions as f32 / EPISODES_PER_UPDATE as f32,
            deaths as f32 / EPISODES_PER_UPDATE as f32,
        );

        policy
            .clone()
            .save_file(format!("snapshots/reinforce-policy-{update:05}.bpk"))
            .unwrap();
    }
}
