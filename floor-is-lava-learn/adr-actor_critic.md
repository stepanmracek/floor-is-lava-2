# ADR: Actor-critic agent for Floor is Lava

Implementation: `src/actor_critic.rs`.

Builds on the REINFORCE agent; see `adr-reinforce.md` for the decisions that are
shared and unchanged:

- observation (168 floats from `Cells`);
- 5 actions including "stay";
- a decision on every idle tick;
- reward = my gain − opponent's gain;
- discounting per second (γ = 0.99/s);
- entropy bonus 0.01;
- 32 episodes of 60 s per update;
- plain-Rust parallel rollouts.

## Why actor-critic

REINFORCE plateaued at a margin of +22 against the simple AI. Its time-dependent
baseline knows _when_ a decision was made but not _where_. A learned critic `V(s)` can
tell that a position is dangerous or that the opponent is about to steal a block. Those
are the situations where stealing and defending have to be learned, and where a better
baseline reduces gradient noise the most.

## Network

**Decision: a separate critic network, `ActorCritic { policy, critic }`, sharing no
weights with the policy.**

```
observation (168) → Policy: 128 (ReLU) → 128 (ReLU) → 5 logits
observation (168) → Critic: 128 (ReLU) → 128 (ReLU) → 1 value (no activation)
```

- The policy loss trains only the policy and the value loss only the critic. Nothing
  needs to be detached and neither network can disturb the other.
- **The existing `Policy` is reused unchanged.**
  - REINFORCE snapshots load directly with `Policy::load_file`.
  - Snapshots save only `model.policy`, so policy files stay interchangeable between
    the two algorithms and with whatever plays the game.
- `Critic::to_plain()` gives a plain-Rust copy for the rollouts, used for the
  bootstrap value only. It reuses `Mlp`, which has the same shape with a single
  output, so the value is `logits(observation)[0]`.

**History: a shared trunk with a linear value head was tried first and abandoned.**

| critic | explained variance after warm-up |
|---|---|
| linear head on the policy's trunk, 1 step per batch | ≈ 0.10, sometimes negative |
| the same + 10 critic steps per batch | ≈ 0.18 |
| **separate network + 10 critic steps per batch** | **≈ 0.32, stable** |

- **With the critic's gradient flowing into the shared trunk, the policy collapsed**
  within 20 updates after warm-up: the margin went from +22 to −160 and the agent made
  1783 no-ops out of 1794 decisions per episode. See *Fine-tuning learning rate* for
  the other half of the cause.
- **Detaching the trunk for the critic prevented the collapse**, but the critic was then
  limited to a linear map of features that had been shaped for choosing actions, not
  for predicting value. More steps barely helped: even with frozen features, explained
  variance topped out at ~0.3–0.38. Once the policy trained, the features also shifted
  under the critic, costing another ~0.1.
- A separate network learns its own features and holds its explained variance while the
  policy trains.

## Starting from a REINFORCE snapshot

**Decision: `train(policy_path, …)` loads a policy snapshot if given, otherwise
starts from a random policy.** The critic always starts random.

## Critic warm-up

**Decision: when starting from a snapshot, the first `WARMUP_UPDATES` = 200 updates
train only the critic.**

- A random critic predicts nonsense, so its advantages would be pure noise and would
  push the trained policy in random directions.
- During warm-up the loss is the value MSE only, with no policy or entropy terms.
  - Verified: the policy snapshot after warm-up is **byte-for-byte identical** to the
    REINFORCE snapshot, and the logged KL is exactly 0.
- **No warm-up for a fresh policy.** There's nothing learned to protect, so the critic
  and policy learn together from the first update.
- Explained variance reaches ~0.3 by the end of warm-up.

## Fine-tuning learning rate

**Decision: when starting from a snapshot, the policy's learning rate after warm-up is
`FINE_TUNE_LEARNING_RATE` = 3e-4, rising linearly from 0 over
`LEARNING_RATE_RAMP_UPDATES` = 100.** A fresh policy uses 1e-3 without a ramp.

- During warm-up only the critic had gradients, so the policy's Adam state was empty
  when it started training. Adam's first steps are effectively `lr · sign(gradient)`:
  every weight moves by ~lr, however small its gradient. At 1e-3, 20 steps changed every
  weight of the trained network by ~0.02, against weights of only ~0.08 in the first
  layer (init range √(1/168)). That was the other half of the collapse.
- The ramp lets Adam build sensible per-parameter statistics before the steps reach full
  size. With it, the margin didn't dip at all during the ramp.
- KL stayed at 0.0001–0.0007 at 3e-4, far below the ~0.02 danger zone, so 3e-4 is
  conservative and raising it is a candidate.

## Extra critic steps

**Decision: after the main update, the critic gets `CRITIC_STEPS` = 10 extra gradient
steps on the same batch, always at the full `LEARNING_RATE` = 1e-3.**

- The steps come *after* the update. Fitting the critic to the batch's returns first
  would pull `V(s)` towards those same returns and shrink the advantages towards zero.
- The full learning rate lets the critic keep up with the changing returns even while
  the policy is fine-tuned at 3e-4. The extra steps only touch the critic, so this needs
  no parameter groups in the optimizer.
- Cost: ~0.45 s per update with the separate critic (against ~0.25 s without the
  extra steps).

## Bootstrapping at the episode cut

**Decision: each episode's return is bootstrapped with the critic's estimate of the final
state, `γ^dt · V(cells after the last tick)`, instead of 0.**

- The game is endless; the 60 s episode length is an artificial cut. Without
  bootstrapping the return mostly measures how much of the episode is left. That's the
  problem REINFORCE solved with its time-bucket baseline.
- The observation contains no time, so a critic `V(cells)` could not predict the time
  effect and would be a poor baseline. With bootstrapping, the return estimates the
  value of an endless game, which doesn't depend on time and _is_ a function of the
  cells.
- This replaces the time-bucket baseline completely.
- Alternative considered: giving the critic the remaining time as an extra input
  (critic only; the policy stays cells-only). Bootstrapping was preferred as the
  standard, principled fix for truncated episodes.

## Loss

```
loss = −mean(log π(a|s) · Â) + VALUE_COEF · mean((V(s) − G)²) − ENTROPY_COEF · entropy
Â   = normalize(G − V(s))        // V treated as a constant, zero mean and unit variance per batch
```

- `G` is the Monte Carlo return from the decision tick, bootstrapped at the cut.
  GAE / TD(λ) was not used: the critic serves as a baseline and to remove the
  truncation bias, and Monte Carlo returns keep the code close to REINFORCE.
- **Rewards are divided by `REWARD_SCALE` = 10,** so the critic's regression targets
  stay roughly within ±10 instead of tens to hundreds of points.
- `VALUE_COEF` = 0.5 is the common default. With separate networks it no longer trades
  off against the policy loss. It only scales the critic's gradient, which Adam largely
  normalizes away.

**Rejected: normalizing the loss per episode instead of per decision.** This was meant
to stop no-op-heavy batches from diluting the gradient of jump decisions. But within a
batch both versions weight every decision equally; they differ only by a factor of
decisions per episode, which Adam's per-parameter scaling removes. The dilution comes
from per-tick decisions themselves, and weighting decisions by duration would no
longer estimate the true policy gradient.

## Logged diagnostics

The same as REINFORCE, plus:

- `value loss`: the MSE in scaled units. Hard to interpret on its own: a critic that
  always predicts the mean return scores exactly the variance of the returns.
- `explained var`: `1 − Var(G − V) / Var(G)`, measured on the batch *before* any
  training on it, so it's an out-of-sample number.
  - ≈ 0: no better than predicting the mean return;
  - 0.2–0.5: a useful baseline;
  - < 0: worse than the mean.
  - Returns include the opponent's points and the random block layout, which the critic
    can't foresee, so 1 is out of reach.
- `kl`: approximate KL divergence between the policy before and after the update.
  - Each decision stores the log-probability of its chosen action under the rollout
    policy. The updated policy re-evaluates the same decisions in plain Rust, using the
    low-variance estimator `(r − 1) − ln r` with `r = π_new / π_old`.
  - Values above ~0.02 per update mean the policy is moving too fast.
  - It would have flagged the collapse at its first update.
- `lr`: the policy's current learning rate.
- `(warm-up)` on lines from warm-up updates.

Snapshots: `snapshots/actor-critic-policy-{update:05}.bpk` every 20 updates (policy only;
the kept ones are now in `snapshots/cpu/`).
Each run's update counter starts at 1, so the kept final snapshots were renamed by hand:
`actor-critic-policy.bpk` (first run) and `actor-critic-policy2.bpk` (second run).

## Results

### First run

10 000 updates from `reinforce-policy-10000.bpk`, with the separate critic.
Parameters are saved in `snapshots/cpu/actor-critic-params.txt` and the log in
`snapshots/cpu/actor-critic-log.txt`. Final policy: `snapshots/cpu/actor-critic-policy.bpk`.

| updates | margin (my − opp) | deaths | Up | no-ops | entropy | explained var | KL max |
|---|---|---|---|---|---|---|---|
| REINFORCE end (for reference) | +22 | 4.40 | 7.9 | ~100 | 0.26 | – | – |
| 1–500 (incl. warm-up) | +26 | 4.28 | 8.1 | 84 | 0.29 | 0.29 | 0.0006 |
| 1001–1500 | +42 | 3.92 | 9.0 | 12 | 0.44 | 0.26 | 0.0006 |
| 3001–3500 | +44 | 3.80 | 9.2 | 14 | 0.46 | 0.26 | 0.0007 |
| 5001–5500 | +52 | 3.44 | 9.7 | 12 | 0.47 | 0.25 | 0.0006 |
| 7501–8000 | +55 | 3.40 | 9.8 | 12 | 0.47 | 0.24 | 0.0009 |
| 9501–10000 | **+57** | **3.16** | 9.9 | 12 | 0.45 | 0.25 | 0.0007 |

- **The margin went from REINFORCE's +22 to +57**, finishing at ≈ 144 : 87 per minute.
  Compared with the end of REINFORCE the agent both scores more (+19) and holds the
  opponent to less (−16), so it improved at stealing and defending, not just at
  scoring.
- **It hadn't plateaued.** The margin was still rising by ~+1–2 per 500 updates and
  deaths were still falling. REINFORCE had been flat for its last 2000 updates.
- **No-ops dropped from ~140 to ~12 per episode within the first ~500 joint updates,
  and entropy rose from ~0.2 to ~0.45.** The critic's advantages show that most 1-tick
  hesitations are worth nothing, and the policy explores more rather than less.
- **Stable throughout:** no collapse, steady entropy, explained variance ~0.25.
- **The learning rate was too cautious.** KL averaged 0.0002 with a maximum of 0.0014
  over the whole run, ~15× below the ~0.02 danger zone.

### Second run

20 000 more updates from `actor-critic-policy.bpk` with the same parameters,
including `FINE_TUNE_LEARNING_RATE` = 3e-4. The critic isn't saved, so it warmed up again.
Parameters in `snapshots/cpu/actor-critic-params2.txt`, log in `snapshots/cpu/actor-critic-log2.txt`.
Final policy: `snapshots/cpu/actor-critic-policy2.bpk`.

| updates | margin (my − opp) | deaths | Up | explained var | KL max |
|---|---|---|---|---|---|
| 1–1000 (incl. warm-up) | +57.4 | 3.18 | 9.8 | 0.20 | 0.0009 |
| 4001–5000 | +63.6 | 2.98 | 10.4 | 0.24 | 0.0014 |
| 9001–10000 | +68.8 | 2.72 | 10.9 | 0.25 | 0.0015 |
| 14001–15000 | +73.4 | 2.65 | 11.0 | 0.23 | 0.0015 |
| 15001–16000 | +74.6 | 2.44 | 11.1 | 0.25 | 0.0008 |
| 16001–17000 | +77.6 | 2.34 | 11.2 | 0.26 | 0.0009 |
| 17001–18000 | +76.2 | 2.52 | 11.2 | 0.26 | 0.0008 |
| 18001–19000 | +75.6 | 2.50 | 11.3 | 0.26 | 0.0009 |
| 19001–20000 | **+76.8** | 2.56 | 11.3 | 0.26 | 0.0009 |

- **The margin went from +57 to about +76,** ≈ 153 : 77 per minute.
- **Plateau from ~16 000 updates on:** the gains slowed from ~+1.3 to ~+0.9 per 1000
  updates and then stopped, while deaths levelled off at ~2.5 per minute. The margin
  per update varies by about ±8, so a 1000-update average is good to about ±1.2.
- **Decision: no further hyperparameter tuning against the simple AI.** Further gains
  would mostly exploit that one opponent's weaknesses; the next step is other opponents
  (see `todo.md`). A higher fine-tuning learning rate (KL max stayed at 0.0025) was not
  tried.
- `snapshots/cpu/actor-critic-policy2.bpk` was the best policy at the time. Its export,
  `actor-critic-policy2.safetensors`, was embedded in the 3D client as `ai-hard` (for
  the current one see `adr-policy-deployment.md`).
- **No longer loadable with the current code.** Both runs used the absolute observation
  (168 inputs), which was replaced by the egocentric one (455 inputs) and removed.
  `actor-critic-policy.bpk`, `actor-critic-policy2.bpk` and their `.safetensors`
  exports fail to load with a shape error; commit `78f73b0` is the last one that loads
  them. `ai-hard` no longer embeds `actor-critic-policy2`. The logs and params files
  stay as the record of these runs.

## Usage

```
floor-is-lava-learn actor-critic <seed> [path/to/policy.bpk] [--updates N] [--report-every N] [--snapshot-every N] [--snapshot-prefix P]
```

Defaults and details: `--help` and `practicalities.md`.

## Open issues / next steps

- **Next: opponents other than the simple AI.** The roadmap and its order are in
  `todo.md`:
  - check how a red-trained snapshot plays as blue;
  - randomize the agent's colour;
  - save the critic with the snapshots;
  - evaluation against fixed benchmarks;
  - an opponent pool of frozen snapshots plus the simple AI;
  - PPO if KL spikes or the evaluation shows forgetting.
- **Duplicated code:**
  - the rollout loop is duplicated between `reinforce.rs` and `actor_critic.rs`, a
    candidate for a shared function if both are kept;
  - the critic reuses the policy's plain-Rust MLP for its copy. Resolved: it was
    renamed from `PlainPolicy` to the neutral `Mlp`, which takes its number of outputs
    from the file.
