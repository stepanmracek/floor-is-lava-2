# Roadmap: learned agent

Decisions and results behind the done items are in `adr-reinforce.md`,
`adr-actor_critic.md` and `adr-policy-deployment.md`.

Order: **1. egocentric observation → 2. training against older snapshots → 3. CNN.**

## Done

- [x] Observation encoding from `Cells` (standing / landing channels)
- [x] REINFORCE with time-bucket baseline, entropy bonus, γ = 0.99/s, relative reward
  - beats the simple AI from ~5500 updates, plateau at margin +22
- [x] Plain-Rust parallel rollouts (~50–90× faster)
- [x] Actor-critic from the REINFORCE snapshot
  - separate critic, critic warm-up, fine-tuning learning rate with ramp, extra critic
    steps, bootstrapping at the episode cut
  - margin +57 after 10 000 updates, still rising
- [x] Diagnostics: entropy, explained variance, approximate KL
- [x] Export to SafeTensors, `PolicyController` in the 3D client (`ai-hard`)
- [x] Second actor-critic run against the simple AI: 20 000 updates from
  `actor-critic-policy.bpk` (`FINE_TUNE_LEARNING_RATE` = 3e-4); params in
  `snapshots/actor-critic-params2.txt`, log in `snapshots/actor-critic-log2.txt`,
  final policy `actor-critic-policy2.bpk` (exported, used by `ai-hard`).
  - Margin +57 → **+76**, ≈ 153 : 77 per minute; deaths 3.2 → 2.5; KL max 0.0025.
  - Gains slowed from ~+1.3 per 1000 updates to a **plateau at about +76 from ~16 000
    updates on** (1000-update windows: +74.6, +77.6, +76.2, +75.6, +76.8).
  - **Decision: no further hyperparameter tuning against the simple AI.** Further gains
    would mostly exploit that one opponent's weaknesses; move on to other opponents.

## Observations from playing against `actor-critic-policy2`

- Inhuman reaction time in the chicken game (expected: it decides on every tick).
- Very aggressive play, especially close to the lava (expected).
- Sometimes "panics": very close to the lava, with blocks to the left or right but none
  above, it jumps into the void. A dumb move, but human-like.
- **Plays badly in the upper-left and upper-right corners, usually jumping into the
  void.** Likely because these situations were rare in training, and with absolute
  columns the network has to learn each corner as a separate input pattern.

## 1. Egocentric observation

Do this first: every snapshot is tied to its observation encoding, so changing the
encoding after self-play would throw that training away.

**Expected benefits:**

- Translation invariance: "void above me, a block to my left" looks the same anywhere
  on the board, so experience from the middle transfers to the rare corner situations.
- It solves the colour problem: without absolute columns a policy plays blue the way it
  plays red, so colour randomization becomes easy.

**Tasks:**

- [ ] **New encoding in `floor_is_lava_policy::observation`,** centred on the player:
  - the window covers the whole board from any position: columns −6 … +6 (13),
    rows −3 … +3 (7);
  - cells outside the board are encoded like a missing block (points −1, everything else
    0): both are void, and jumping there kills;
  - drop the "my standing" channel: I'm always in the centre, and I decide only while
    idle, so it would always be 1 there at decision time. That leaves 5 features per
    cell: 13 × 7 × 5 = 455 inputs (168 today);
  - height stays visible through `distance_from_lava` per cell;
  - keep the layout `[rows][columns][features]` in a known order, so the same vector
    can be reshaped to `[features, rows, columns]` for step 3 (CNN).
- [ ] **Randomize the agent's colour per episode** from the start, so the new snapshots
  can play both sides.
- [ ] **Save the critic alongside the policy snapshots,** and load it on restart. Each
  restart currently costs the 200-update warm-up (explained variance starts at ~0.1
  instead of ~0.25).
- [ ] **Run name in the snapshot path,** e.g. `snapshots/<run>/policy-{update:05}.bpk`.
  Each run's counter starts at 1, so a continued run overwrites the previous run's
  snapshots; final snapshots are currently renamed by hand.
- [ ] **Retrain from scratch:** the existing snapshots can't be converted. Roughly
  REINFORCE 10 000 updates (~30 min) plus actor-critic ~30 000 (~4 h); possibly
  actor-critic directly from a random policy.
- [ ] **Re-export `ai-hard`;** `PolicyController` follows the new observation
  automatically through the shared crate.

**Evaluation harness** (needed from here on; all numbers so far come from training
rollouts: a sampling policy, always red, against the simple AI):

- [ ] Evaluation in `floor-is-lava-headless`: fixed seeds, many games, win rate and
  margin, as both colours
- [ ] **Death classification:** position on the board, and the situation:
  - jumped into the void while all reachable blocks were already its own (deliberate
    island escape, a good move);
  - jumped into the void with unclaimed or opponent blocks reachable (a mistake, e.g. the
    corner and "panic" deaths);
  - sank while standing still.
- [ ] **Compare** the egocentric agent with `actor-critic-policy2.bpk` against the simple
  AI. Expect fewer corner deaths, and see whether the overall margin improves.

## 2. Training against older snapshots

**Order: opponent pool on the current actor-critic first, PPO only if needed.** The
existing safeguards cover most of what PPO would provide at first:

- KL logging shows immediately if updates get too large;
- the learning rate is cautious (KL max 0.0025);
- the simple AI stays in the pool against forgetting.

Snapshots with the old (absolute) encoding can't be opponents unless the encoding is
versioned. Simpler: the pool starts with the simple AI and the new egocentric snapshots
only.

**Tasks:**

- [ ] **Benchmark opponents for the evaluation:** the simple AI plus frozen egocentric
  snapshots. Evaluate every ~100–200 updates. Progress means rising numbers against all
  benchmarks; falling numbers against old ones mean forgetting, so roll back.
- [ ] **Opponent pool, one opponent per episode:**
  - ~50% recent snapshots, ~30% older snapshots, ~20% simple AI;
  - snapshot opponents sample their actions (deterministic opponents are exploitable
    in ways that don't generalize).
  - Forgetting comes mainly from the opponent mix changing, not from the learning rate:
    without old opponents in the batches nothing reminds the policy how to beat them.
  - Implementation notes:
    - each parallel episode gets its own opponent as `Box<dyn Controller>`;
    - a policy opponent is a `PlainPolicy` with its own rng, seeded from the episode's
      seeds for reproducibility;
    - snapshots are loaded through burn into `PlainPolicy` once, when the pool changes,
      not per episode.
- [ ] **PPO** (clipped probability ratio, several epochs per batch), if KL spikes or the
  evaluation shows forgetting.
  - Limits how far one update moves the policy, the real fix for "too aggressive and it
    forgets".
  - Builds on what exists: critic, stored rollout log-probabilities, KL logging.
- [ ] **On every restart from a snapshot:**
  - ramp the learning rate up again (the fresh Adam state makes sign-sized steps), or
    save and restore the optimizer state;
  - keep the entropy bonus.

## 3. Convolutional layers

A separate experiment after steps 1 and 2, compared against their result. Changing the
encoding and the architecture at once would make it impossible to tell which helped.

- [ ] **Conv layers on the egocentric grid** (`[features, rows, columns]`), then dense
  layers reading the result relative to the centre.
  - Convolution shares weights across positions, so local patterns ("void next to a
    block near the lava") are recognized the same way everywhere.
  - The board is tiny: two 3×3 layers already see a 5×5 neighbourhood, so the gain over
    an egocentric MLP may be modest. Measure it.
- [ ] **Plain-Rust convolution in `floor-is-lava-policy`,** matching burn's `Conv2d`
  exactly: weight layout `[out_channels, in_channels, kh, kw]`, padding, stride.
  Extend the burn-vs-plain tests and the SafeTensors export (`conv1.weight`, …).
- [ ] If the architecture keeps growing, reconsider ONNX with `tract` instead of
  hand-written inference (see `adr-policy-deployment.md`).

## Smaller items

- [ ] Throttle `PolicyController` to 60 decisions per second if the frame rate visibly
  changes the agent's behaviour (needs the game time in `Controller::get_action`)
- [ ] Difficulty levels for the 3D client: per-decision mixing with the simple AI (drawn
  when the player becomes idle, not per tick), and/or reaction delay. Measure the win
  rate curve over the setting.
- [ ] Rename `PlainPolicy` to a neutral `PlainMlp`; the critic reuses it with one output
- [ ] Share the rollout loop between `reinforce.rs` and `actor_critic.rs` if both stay
- [ ] Maybe: merge `floor-is-lava-policy` into the lib behind a `policy` feature
- [ ] Maybe: GAE / TD(λ) instead of Monte Carlo returns, if the critic gets good enough
- [ ] Untried hyperparameter options, should training against a fixed opponent plateau
  again:
  - `FINE_TUNE_LEARNING_RATE` = 1e-3 (KL shows the current 3e-4 is very cautious; step
    back to 5e-4 if KL climbs above ~0.01);
  - plateau with low entropy: raise `ENTROPY_COEF` to 0.02–0.03;
  - plateau while the action mix keeps moving: 64 episodes per update.
