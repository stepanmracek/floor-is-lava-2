# Roadmap: learned agent

Decisions and results behind the done items are in `adr-reinforce.md`,
`adr-actor_critic.md` and `adr-policy-deployment.md`.

Order: **0. mirror the board for blue → 1. egocentric observation (if corner deaths
justify it) → 2. training against older snapshots → 3. CNN → 4. procedural level
generation.**

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
- [x] Export to SafeTensors, `PolicyController` in `floor-is-lava-policy` (seeded rng),
  used by the 3D client (`ai-hard`) and the headless evaluation
- [x] Second actor-critic run against the simple AI: 20 000 updates from
  `actor-critic-policy.bpk` (`FINE_TUNE_LEARNING_RATE` = 3e-4); params in
  `snapshots/actor-critic-params2.txt`, log in `snapshots/actor-critic-log2.txt`,
  final policy `actor-critic-policy2.bpk` (exported, used by `ai-hard`).
  - Margin +57 → **+76**, ≈ 153 : 77 per minute; deaths 3.2 → 2.5; KL max 0.0025.
  - Gains slowed from ~+1.3 per 1000 updates to a **plateau at about +76 from ~16 000
    updates on** (1000-update windows: +74.6, +77.6, +76.2, +75.6, +76.8).
  - **Decision: no further hyperparameter tuning against the simple AI.** Further gains
    would mostly exploit that one opponent's weaknesses; move on to other opponents.
- [x] Headless self-play of `actor-critic-policy2` against itself (`PolicyController` on
  both sides, now in `floor-is-lava-policy`):
  - **One 24-hour game:** red 143 158, blue 144 731, i.e. 102.5 vs 103.6 per minute.
    The difference is within noise (±~1.2 per minute), so over a long game the
    colours are equal: once both players have moved around, the start position
    doesn't matter.
  - **200 games of 60 s, seeds 42 … 241:** red wins 147, blue 51 (74% for red, ~6.8
    standard errors from 50%), average margin **+26.0 for red**. Varying seeds rule out a
    single favourable board.
    - Reproducible: each `PolicyController` samples from its own seeded rng (red from
      the seed, blue from the seed + 2³²), so the run repeats exactly. A first run with
      unseeded sampling gave 77% / +28.9.
  - **Conclusion: the red-trained policy plays the opening badly as blue.** It learned an
    opening for starting at x = −2; as blue it starts at x = 2. Training episodes are
    60 s, so the opening is a large part of each one, and blue snapshot opponents would
    be handicapped exactly there.

## Observations from playing against `actor-critic-policy2`

- Inhuman reaction time in the chicken game (expected: it decides on every tick).
- Very aggressive play, especially close to the lava (expected).
- Sometimes "panics": very close to the lava, with blocks to the left or right but none
  above, it jumps into the void. A dumb move, but human-like.
- **Plays badly in the upper-left and upper-right corners, usually jumping into the
  void.** Likely because these situations were rare in training, and with absolute
  columns the network has to learn each corner as a separate input pattern.

## 0. Mirror the board for blue

The game is left-right symmetric: start positions −2 / +2, uniform block spawning,
random player order. So blue can play as if it were red in a mirrored world, which
gives existing snapshots full strength as blue **without retraining**. It's exact, not
an approximation, because the rules themselves are symmetric.

- [ ] In `floor_is_lava_policy`, when `me` is blue:
  - `observation` reads the columns in reverse order (x → −x);
  - the action is mirrored: Left ↔ Right (Up, Down, None unchanged).

  Both belong to the shared crate, so `PolicyController`, training rollouts and future
  snapshot opponents all get them.
- [ ] **Verify with the headless test** (same snapshot on both sides, 200 seeds of 60 s):
  red's win rate should drop from the 74% baseline to ~50%, and the margin from +26.0
  to ~0. Same seed (42), so the comparison is direct.
- [ ] Re-export nothing: the snapshots stay the same, only the encoding for blue changes.
  Update `adr-policy-deployment.md` (the agent can now play either colour).

Consequences:

- **Colour randomization in training becomes unnecessary:** the agent can keep
  training as red, and blue players use the mirrored view.
- **Existing snapshots can be opponents at full strength,** so step 2 no longer depends
  on step 1.

## 1. Egocentric observation

With mirroring solving the colour problem, **the remaining reason is the corner
weakness.** Egocentric means retraining from scratch, so measure first.

- [ ] **Measure corner deaths first:** the death classification from the evaluation
  harness below, run on `actor-critic-policy2` in the headless setup.
  - Corner deaths a big share of all deaths: egocentric is justified; do it before
    step 2, because every snapshot is tied to its observation encoding and changing it
    after self-play would throw that training away.
  - Corner deaths rare: go straight to step 2 with the existing snapshots, and keep
    egocentric for later.

**Expected benefit:** translation invariance. "Void above me, a block to my left" looks
the same anywhere on the board, so experience from the middle transfers to the rare
corner situations.

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
    can be reshaped to `[features, rows, columns]` for step 3 (CNN);
  - for blue, mirror the window the same way as in step 0.
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

**Evaluation harness** (needed from here on; apart from the headless colour test, all
numbers so far come from training rollouts: a sampling policy, always red, against the
simple AI):

- [ ] Evaluation in `floor-is-lava-headless`: fixed seeds, many games, win rate and
  margin, as both colours. Started: the 200-game colour test (policy against policy)
  is the first version; it still needs other opponents such as the simple AI.
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

Which snapshots can be opponents depends on step 1:

- **Egocentric done:** snapshots with the old (absolute) encoding can't be opponents
  unless the encoding is versioned. Simpler: the pool starts with the simple AI and the
  new egocentric snapshots only.
- **Egocentric skipped:** the existing snapshots, playing blue through the mirrored view
  from step 0.

**Tasks:**

- [ ] **Benchmark opponents for the evaluation:** the simple AI plus frozen snapshots
  (egocentric ones if step 1 was done). Evaluate every ~100–200 updates. Progress means rising numbers against all
  benchmarks; falling numbers against old ones mean forgetting, so roll back.
- [ ] **Opponent pool, one opponent per episode:**
  - ~50% recent snapshots, ~30% older snapshots, ~20% simple AI;
  - snapshot opponents sample their actions (deterministic opponents are exploitable
    in ways that don't generalize).
  - Forgetting comes mainly from the opponent mix changing, not from the learning rate:
    without old opponents in the batches nothing reminds the policy how to beat them.
  - Implementation notes:
    - each parallel episode gets its own opponent as `Box<dyn Controller>`;
    - a policy opponent is a `PolicyController` (from `floor-is-lava-policy`, already
      used by the 3D client and headless), seeded from the episode's seeds with an offset
      from the agent's sampling seed, so the two players don't draw correlated random
      numbers;
    - snapshots are loaded into `PlainPolicy` once, when the pool changes, not per
      episode. `PolicyController` owns its `PlainPolicy`, so sharing one loaded snapshot
      between parallel episodes needs a small change, e.g. holding it in an `Arc`.
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

A separate experiment after steps 1 and 2, compared against their result. It needs the
egocentric grid from step 1. Changing the
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

## 4. Procedural level generation

New rows are random today (each column 0.7, one guaranteed block), so some blocks are
dead ends: no way up, and the player dies through no fault of their own. Simulated
per new row: 17.9% contain a segment with no block above it, 0.28% can't be reached
from the row below at all.

Holes between rows are fine as long as every block can still reach the top: columns
hanging from a row above, U-turns through lower rows. But lower rows sink, so a U-turn
is only safe if the player can finish it before the lava gets there. **The invariant is
escapability, not just connectivity.**

- [ ] **Row-by-row generation like Eller's maze algorithm, with timing:**
  - generate a candidate row `y+1` randomly;
  - for every block `b`, find the top-row cells reachable in time (BFS over jump
    count; the lava at landing after k jumps is `lava + k · JUMP_DURATION ·
    LAVA_SPEED`);
  - repair: every block that can reach the top row in time needs a block above at
    least one of its reachable top-row cells; otherwise add one at a random column
    among them.
  - A block that can still escape never becomes a dead end; a player dies only by
    wasting time or jumping into a hole.
  - Keep the repair choices uniformly random so the board stays left-right symmetric in
    distribution (step 0 relies on that).
  - The 0.7 density becomes a difficulty knob: sparser, more maze-like boards stay fair.
- [ ] **Separate `level_rng`** for rows and points; the existing `rng` keeps player order
  and respawns. A seed is then a level, independent of how the game goes, so headless
  evaluations compare policies on the same towers.
- [ ] **Re-measure** the existing policies headlessly before and after: the observation
  doesn't change, but the boards do (and existing seeds give different towers), so
  numbers before and after aren't comparable seed by seed. Fine-tune if needed.

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
