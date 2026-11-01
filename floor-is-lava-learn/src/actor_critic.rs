use burn::nn::{Linear, LinearConfig, Relu};
use burn::optim::{AdamConfig, GradientsParams, ModuleOptimizer};
use burn::prelude::*;
use burn::tensor::activation::log_softmax;
use floor_is_lava_lib::{Clock, Controller, Game, PlayerColor, PlayerState, SimpleAiController};
use floor_is_lava_policy::{INPUTS, Mlp, observation, sample};
use rand::{RngExt, SeedableRng, rngs::StdRng};
use rayon::prelude::*;

use crate::policy::{ACTIONS, HIDDEN, Policy, plain_layer};

const EPISODE_SECONDS: f32 = 60.0;
const DT: f32 = 1.0 / 60.0;
const GAMMA_PER_SECOND: f32 = 0.99;
const ENTROPY_COEF: f32 = 0.01;
const VALUE_COEF: f32 = 0.5;
/// Rewards are divided by this so the critic's targets stay roughly within ±10.
const REWARD_SCALE: f32 = 10.0;
const EPISODES_PER_UPDATE: usize = 32;
const LEARNING_RATE: f64 = 1e-3;
/// Learning rate when fine-tuning a pre-trained policy.
const FINE_TUNE_LEARNING_RATE: f64 = 3e-4;
/// After warm-up, the fine-tuning learning rate rises linearly from 0 over this many
/// updates, so the fresh Adam state doesn't make full-size steps on a trained policy.
const LEARNING_RATE_RAMP_UPDATES: usize = 100;
/// Number of critic-only updates when starting from a pre-trained policy.
const WARMUP_UPDATES: usize = 200;
/// Extra critic-only gradient steps on each batch, after the main update.
const CRITIC_STEPS: usize = 10;

/// Value network with its own trunk: INPUTS → HIDDEN → ReLU → HIDDEN → ReLU → 1.
#[derive(Module, Debug)]
pub struct Critic {
    hidden1: Linear,
    hidden2: Linear,
    output: Linear,
    activation: Relu,
}

impl Critic {
    fn new(device: &Device) -> Self {
        Self {
            hidden1: LinearConfig::new(INPUTS, HIDDEN).init(device),
            hidden2: LinearConfig::new(HIDDEN, HIDDEN).init(device),
            output: LinearConfig::new(HIDDEN, 1).init(device),
            activation: Relu::new(),
        }
    }

    /// [batch, INPUTS] -> [batch] values
    fn forward(&self, observations: Tensor<2>) -> Tensor<1> {
        let x = self.activation.forward(self.hidden1.forward(observations));
        let x = self.activation.forward(self.hidden2.forward(x));
        self.output.forward(x).squeeze_dim::<1>(1)
    }

    /// Plain-Rust copy for the rollouts. It has the same shape as the policy network,
    /// only with a single output, so the value is `logits(observation)[0]`.
    fn to_plain(&self) -> Mlp {
        Mlp {
            hidden1: plain_layer(&self.hidden1),
            hidden2: plain_layer(&self.hidden2),
            output: plain_layer(&self.output),
        }
    }
}

/// Policy and a separate critic. They share no weights, so the policy gradient trains
/// only the policy and the value loss only the critic.
#[derive(Module, Debug)]
pub struct ActorCritic {
    policy: Policy,
    critic: Critic,
}

impl ActorCritic {
    fn new(policy: Policy, device: &Device) -> Self {
        Self {
            policy,
            critic: Critic::new(device),
        }
    }

    /// [batch, INPUTS] -> ([batch, ACTIONS] logits, [batch] values)
    fn forward(&self, observations: Tensor<2>) -> (Tensor<2>, Tensor<1>) {
        let logits = self.policy.forward(observations.clone());
        let values = self.critic.forward(observations);
        (logits, values)
    }
}

struct Episode {
    /// One observation per decision, flattened.
    observations: Vec<f32>,
    actions: Vec<i64>,
    /// Log-probability of each chosen action under the policy that played the episode.
    log_probs: Vec<f32>,
    /// Discounted return for each decision, bootstrapped at the episode cut.
    returns: Vec<f32>,
    my_score: usize,
    opponent_score: usize,
    deaths: usize,
}

/// Discounted per-tick returns. The game doesn't end at the episode cut, so instead of 0
/// the return after the last tick is the critic's estimate of the final state.
fn discounted_returns(rewards: &[f32], bootstrap: f32) -> Vec<f32> {
    let gamma = GAMMA_PER_SECOND.powf(DT);
    let mut returns = vec![0.0; rewards.len()];
    let mut ret = bootstrap;
    for t in (0..rewards.len()).rev() {
        ret = rewards[t] + gamma * ret;
        returns[t] = ret;
    }
    returns
}

/// Plays one episode as red against the simple AI.
/// `game_seed` seeds the game, `rng_seed` the sampling of actions.
fn play_episode(policy: &Mlp, critic: &Mlp, game_seed: u64, rng_seed: u64) -> Episode {
    let mut game = Game::with_seed(game_seed);
    let mut rng = StdRng::seed_from_u64(rng_seed);
    let mut clock = Clock::fixed(DT);
    let mut opponent = SimpleAiController::new(PlayerColor::Blue);

    let mut observations = vec![];
    let mut actions = vec![];
    let mut log_probs = vec![];
    let mut decision_ticks = vec![];
    let mut rewards = vec![];
    let mut deaths: usize = 0;
    let mut last_dying = false;

    let ticks = (EPISODE_SECONDS / DT) as usize;
    for tick in 0..ticks {
        let my_action = if matches!(game.red_player.state, PlayerState::Idle(_)) {
            let observation = observation(game.cells(), PlayerColor::Red);
            let probabilities = policy.probabilities(&observation);
            let action = sample(&probabilities, rng.random());
            observations.extend(observation);
            actions.push(action as i64);
            log_probs.push(probabilities[action].ln());
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
        rewards.push((my_gain - opponent_gain) / REWARD_SCALE);
    }

    let final_observation = observation(game.cells(), PlayerColor::Red);
    let bootstrap = critic.logits(&final_observation)[0];
    let tick_returns = discounted_returns(&rewards, bootstrap);
    Episode {
        observations,
        actions,
        log_probs,
        returns: decision_ticks.iter().map(|&t| tick_returns[t]).collect(),
        my_score: game.red_player.score,
        opponent_score: game.blue_player.score,
        deaths,
    }
}

struct LossInfo {
    entropy: f32,
    value_loss: f32,
    /// 1 - Var(return - V) / Var(return): 0 = no better than predicting the mean return,
    /// 1 = perfect, negative = worse than the mean.
    explained_variance: f32,
}

/// Actor-critic loss:
/// -mean(log π(a|s) * advantage) + VALUE_COEF * mse(V(s), return) - ENTROPY_COEF * entropy
/// where advantage = return - V(s), normalized to unit variance.
///
/// During warm-up only the critic is trained: the policy terms are left out, so the
/// policy stays untouched.
fn loss(
    model: &ActorCritic,
    episodes: &[Episode],
    warmup: bool,
    device: &Device,
) -> (Tensor<1>, LossInfo) {
    let mut observations = vec![];
    let mut actions = vec![];
    let mut returns = vec![];
    for episode in episodes {
        observations.extend_from_slice(&episode.observations);
        actions.extend_from_slice(&episode.actions);
        returns.extend_from_slice(&episode.returns);
    }

    let shape = [actions.len(), observations.len() / actions.len()];
    let observations = Tensor::<2>::from_data(TensorData::new(observations, shape), device);
    let actions = Tensor::<1, Int>::from_data(actions.as_slice(), device);
    let returns_tensor = Tensor::<1>::from_data(returns.as_slice(), device);

    let (logits, values) = model.forward(observations);

    let value_errors = values.clone() - returns_tensor;
    let value_loss = (value_errors.clone() * value_errors).mean();

    let log_probs = log_softmax(logits, 1);
    let entropy = (log_probs.clone().exp() * log_probs.clone())
        .sum_dim(1)
        .neg()
        .mean();

    let values = values.try_into_vec_as::<f32>().unwrap();
    let advantages: Vec<f32> = returns.iter().zip(&values).map(|(r, v)| r - v).collect();
    let n = advantages.len() as f32;
    let mean = advantages.iter().sum::<f32>() / n;
    let variance = advantages.iter().map(|a| (a - mean).powi(2)).sum::<f32>() / n;
    let mean_return = returns.iter().sum::<f32>() / n;
    let return_variance = returns
        .iter()
        .map(|r| (r - mean_return).powi(2))
        .sum::<f32>()
        / n;

    let info = LossInfo {
        entropy: entropy.clone().into_scalar::<f32>(),
        value_loss: value_loss.clone().into_scalar::<f32>(),
        explained_variance: 1.0 - variance / (return_variance + 1e-8),
    };

    if warmup {
        return (value_loss, info);
    }

    let std = variance.sqrt();
    let advantages: Vec<f32> = advantages
        .iter()
        .map(|a| (a - mean) / (std + 1e-8))
        .collect();
    let advantages = Tensor::<1>::from_data(advantages.as_slice(), device);

    let chosen_log_probs = log_probs
        .gather(1, actions.unsqueeze_dim(1))
        .squeeze_dim::<1>(1);
    let policy_loss = (chosen_log_probs * advantages).mean().neg();

    (
        policy_loss + value_loss * VALUE_COEF - entropy * ENTROPY_COEF,
        info,
    )
}

/// Extra critic-only steps on the batch, done after the main update so the advantages were
/// computed with a critic that hadn't seen these returns yet.
///
/// Only the critic is trained. It always uses the full `LEARNING_RATE`, so the critic
/// keeps up with the policy even while the policy is fine-tuned with a lower learning rate.
fn train_critic(
    mut model: ActorCritic,
    optimizer: &mut ModuleOptimizer,
    episodes: &[Episode],
    device: &Device,
) -> ActorCritic {
    let observations: Vec<f32> = episodes
        .iter()
        .flat_map(|e| e.observations.iter().copied())
        .collect();
    let returns: Vec<f32> = episodes
        .iter()
        .flat_map(|e| e.returns.iter().copied())
        .collect();
    let shape = [returns.len(), INPUTS];
    let observations = Tensor::<2>::from_data(TensorData::new(observations, shape), device);
    let returns = Tensor::<1>::from_data(returns.as_slice(), device);

    for _ in 0..CRITIC_STEPS {
        let values = model.critic.forward(observations.clone());
        let errors = values - returns.clone();
        let loss = (errors.clone() * errors).mean();
        let grads = GradientsParams::from_grads(loss.backward(), &model);
        model = optimizer.step(LEARNING_RATE, model, grads);
    }
    model
}

/// Approximate KL divergence between the policy that played `episodes` and `policy`,
/// averaged over the episodes' decisions. Uses the low-variance estimator
/// `(r - 1) - ln r` with `r = π_new(a|s) / π_old(a|s)`, which is always ≥ 0.
///
/// Measures how far one update moved the policy; values above ~0.02 mean it moves fast.
fn approx_kl(policy: &Mlp, episodes: &[Episode]) -> f32 {
    let (sum, count) = episodes
        .par_iter()
        .map(|episode| {
            let decisions = episode
                .observations
                .as_chunks::<INPUTS>()
                .0
                .iter()
                .zip(&episode.actions)
                .zip(&episode.log_probs);
            let sum: f32 = decisions
                .map(|((observation, &action), &old_log_prob)| {
                    let new_prob = policy.probabilities(observation)[action as usize];
                    let log_ratio = new_prob.max(1e-30).ln() - old_log_prob;
                    log_ratio.exp() - 1.0 - log_ratio
                })
                .sum();
            (sum, episode.actions.len())
        })
        .reduce(|| (0.0, 0), |a, b| (a.0 + b.0, a.1 + b.1));
    sum / count as f32
}

/// Trains an actor-critic agent. Starts from the policy snapshot at `policy_path`
/// (warming up the critic first) or from a fresh random policy.
pub fn train(policy_path: Option<&str>, updates: usize, seed: u64) {
    std::fs::create_dir_all("snapshots").unwrap();
    let device = Device::default().autodiff();
    let (policy, warmup_updates) = match policy_path {
        Some(path) => (Policy::new(&device).load_file(path), WARMUP_UPDATES),
        None => (Policy::new(&device), 0),
    };
    let mut model = ActorCritic::new(policy, &device);
    let mut optimizer = AdamConfig::new().init();
    let mut rng = StdRng::seed_from_u64(seed);

    for update in 1..=updates {
        let warmup = update <= warmup_updates;
        let learning_rate = if warmup || policy_path.is_none() {
            LEARNING_RATE
        } else {
            let ramp = (update - warmup_updates) as f64 / LEARNING_RATE_RAMP_UPDATES as f64;
            FINE_TUNE_LEARNING_RATE * ramp.min(1.0)
        };

        // play with a plain-Rust copy of the model, episodes in parallel
        let rollout_policy = model.policy.to_plain();
        let rollout_critic = model.critic.to_plain();
        let seeds: Vec<(u64, u64)> = (0..EPISODES_PER_UPDATE)
            .map(|_| (rng.random(), rng.random()))
            .collect();
        let episodes: Vec<Episode> = seeds
            .into_par_iter()
            .map(|(game_seed, rng_seed)| {
                play_episode(&rollout_policy, &rollout_critic, game_seed, rng_seed)
            })
            .collect();

        let decisions: usize = episodes.iter().map(|e| e.actions.len()).sum();
        let mut histogram = [0usize; ACTIONS.len()];
        for &action in episodes.iter().flat_map(|e| &e.actions) {
            histogram[action as usize] += 1;
        }
        let my_score: usize = episodes.iter().map(|e| e.my_score).sum();
        let opponent_score: usize = episodes.iter().map(|e| e.opponent_score).sum();
        let deaths: usize = episodes.iter().map(|e| e.deaths).sum();

        let (loss, info) = loss(&model, &episodes, warmup, &device);
        let grads = GradientsParams::from_grads(loss.backward(), &model);
        model = optimizer.step(learning_rate, model, grads);
        model = train_critic(model, &mut optimizer, &episodes, &device);

        if update % 20 != 0 {
            continue;
        }

        let kl = approx_kl(&model.policy.to_plain(), &episodes);
        let [up, down, left, right, noop] =
            histogram.map(|count| count as f32 / EPISODES_PER_UPDATE as f32);
        println!(
            "update {update:5}{}: my score {:6.1}, opponent {:6.1}, decisions {:6.0}, deaths {:6.0}, \
             up {up:4.0}, down {down:4.0}, left {left:4.0}, right {right:4.0}, no-op {noop:4.0}, \
             entropy {:.2}, value loss {:.3}, explained var {:.2}, kl {kl:.4}, lr {learning_rate:.0e}",
            if warmup { " (warm-up)" } else { "" },
            my_score as f32 / EPISODES_PER_UPDATE as f32,
            opponent_score as f32 / EPISODES_PER_UPDATE as f32,
            decisions as f32 / EPISODES_PER_UPDATE as f32,
            deaths as f32 / EPISODES_PER_UPDATE as f32,
            info.entropy,
            info.value_loss,
            info.explained_variance,
        );

        model
            .policy
            .clone()
            .save_file(format!("snapshots/actor-critic-policy-{update:05}.bpk"))
            .unwrap();
    }
}
