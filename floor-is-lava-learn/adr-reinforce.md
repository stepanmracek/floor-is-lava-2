# ADR: REINFORCE agent for Floor is Lava

Implementation:

- `src/reinforce.rs`;
- `src/policy.rs`;
- the observation encoding and the plain-Rust policy in the `floor-is-lava-policy` crate.

## Goal

Find out whether plain policy-gradient RL can learn to play the game against the
simple AI (`SimpleAiController`) **without hints** about strategy. The agent is never
told to steal blocks, defend its own, avoid holes or play the "chicken game";
it should discover that from the score alone.

## Observation

**Decision: the observation is a pure function of `Cells` (plus which colour "I" am).**
No access to `Game`, `Player` or the clock. Anything the policy needs is put into `Cell`
by `Game::update_cells` in the lib.

The encoding lives in `floor_is_lava_policy::observation(cells, me) -> Vec<f32>`. That
makes it the single source of truth for training and for any front-end that runs a
trained policy. A mismatch there would silently break a trained agent.

The grid is 7 columns × 4 visible rows. Each cell becomes 6 floats, so the input is a
flat vector of 168 values, in row-major order.

| # | feature | encoding |
|---|---|---|
| 0 | points | `points / 10` (0.1 … 0.9); **-1.0 if the cell is missing** |
| 1 | owner | 1.0 mine, 0.0 neutral, -1.0 opponent's |
| 2 | `distance_from_lava` | raw (≈ 0 … 4) |
| 3 | my standing | 1.0 idle; `1 - p` while jumping away (p = jump progress) |
| 4 | opponent standing | same as above |
| 5 | opponent landing | `p` on the cell the opponent is jumping to |

Justification:

- **Points divided by 10** instead of mapping 1..9 onto 0.01..1.0: simpler, and the
  range doesn't matter to the network.
- **Per-cell player channels instead of player coordinates.** Coordinates were
  considered and rejected:
  - Multiplying a coordinate by the jump progress is ambiguous (x=1 at 25% and
    x=3 at 75% encode the same), and x = 0 always encodes as 0.
  - Absolute `y` grows without bound during a game.
  - An MLP has to learn to match a coordinate to a cell index, which is
    sample-hungry. With per-cell channels, "am I next to a valuable block" is a
    local feature.
- **Separate standing and landing channels.** A single channel with
  `1-p` on `from` and `p` on `to` can't tell the direction of a jump
  ((1,2)→(2,2) at 25% looks like (2,2)→(1,2) at 75%). Landing on a separate
  channel fixes that.
- **Standing fades as `1 - p`** rather than staying 1.0 while jumping. A jump off the
  board (outside the grid, so no landing cell exists) then shows as fading presence
  with no landing anywhere, which can only mean a fatal jump.
- **Only the opponent's landing is included.** The agent decides only while idle, so
  its own landing channel would always be 0 at decision time.
- Falling or dying players are not encoded. "Opponent is nowhere" is enough.
- Known limitation: a jump into a hole (`None` cell) has no landing cell to mark
  either, so it looks the same as a jump off the board. Both are fatal, so this is
  acceptable.

## Network

`Policy`: MLP 168 → 128 → ReLU → 128 → ReLU → 5 (~38k parameters).

- **The output is logits.** `log_softmax` is used for the loss (numerically stable),
  softmax only for sampling.
- **5 actions: Up, Down, Left, Right, None.** Staying still is a valid move and has to
  be a choice the policy can make.
- `Policy::to_plain()` copies the weights into a `PlainPolicy`, a plain-Rust MLP from
  `floor-is-lava-policy`, for fast inference (see *Performance*).
  - burn's `Linear` stores the weight row-major as `[d_input, d_output]`, so the plain
    forward pass computes `input · weight + bias`.
  - A test (`plain_policy_matches_burn`) checks the plain logits against burn's
    `forward` to within 1e-4.

## Decision timing

**Decision: the agent decides on every tick in which it is `Idle`.**

`Game::step` ignores actions in any other state, so recording decisions on other ticks
would only add noise to the gradient.

Alternatives considered and rejected:

- **Holding `None` for a fixed time (e.g. 15 ticks)**, plus early exits when the
  opponent jumps and when the agent's own block is about to sink. This made waiting
  cheap to learn, but it was "too much wiring". It also gives a 0.25 s reaction delay,
  which loses the chicken game (both players on a sinking block, whoever jumps first
  gets their block stolen) against a player who waits until the last moment.
- **Per-tick decisions were chosen for simplicity.** They also give frame-accurate
  timing for free. The cost is that a long deliberate wait needs many consecutive
  `None` decisions; this is accepted as part of the experiment.

Practical consequences:

- One `None` is one tick (1/60 s). 100 no-ops per episode is only ~1.7 s of waiting,
  so the no-op count in the log overstates how much the agent waits.
- The number of decisions, and with it the per-update cost, grows as the policy learns
  to hesitate (from ~35 to ~120–200 per episode). This is what made fast inference
  necessary (see *Performance*).

## Reward

**Decision: reward per tick = my score gain − opponent's score gain.**

- First version: own score gain only. This tested whether stealing and defending
  emerge on their own, since a stolen block is lost future reward anyway.
- Changed to the difference because **the game is "my score is higher than yours",
  not "I score X"**. The difference rewards stealing and defending directly. It adds
  some variance (most of the opponent's points come from its own blocks sinking), but
  in a 3000-update comparison both rewards reached the same margin (≈ −27), with a
  lower opponent score for the relative reward.
- **No death penalty.** Dying already costs ~4 s of play (falling + 3 s dying). A
  penalty is a hint, and if larger than that natural cost it would distort risky but
  correct play such as the chicken game. The agent learned to die less without it
  (10 → 4.4 deaths per minute).
- Scoring is delayed: points are banked only when a block sinks. That is the core of
  the game design and was kept.

## Returns and discounting

**Decision: discount by game time, not by number of decisions:** γ = 0.99 **per second**,
applied per tick as `γ^dt`.

- Actions take different amounts of time (a jump takes 60 ticks, a no-op 1 tick), so
  discounting per decision would make waiting look much cheaper than jumping.
- Returns are computed for every tick, then read at the decision ticks. This is the same
  as discounting each decision by `γ^Δt` without having to track durations.
- **γ was raised from 0.95 to 0.99.** With 0.95 a block sinking 16 s after it was
  claimed kept only ~44% of its value. The agent stopped going Up: it grabbed
  low blocks for quick points, died, and got respawned higher for free.
  With 0.99 it keeps ~85%.

## Baseline: time-dependent

**Decision: advantage = return − mean return of all decisions made in the same second
of the episode (over the whole batch), normalized to unit variance.**

- With γ = 0.99/s the horizon (~100 s) is longer than the 60 s episode, so the return
  mostly measures how much of the episode is left. A single batch-mean baseline makes
  early actions look good and late actions bad regardless of what they were.
  This made the policy wander between actions without improving.
- Bucketing by time compares an action against others taken at the same point of the
  game. Bucket width: `BASELINE_BUCKET_SECONDS` = 1 s.
- Bucket boundaries use `round()`: `1.0 / DT` truncates to 59 ticks and crashed with an
  out-of-bounds bucket.

## Exploration: entropy bonus

**Decision: loss = −mean(log π(a|s) · advantage) − 0.01 · mean entropy.**

- Without it the policy collapsed early into a single habit. Red starts on the left,
  so it learned "always jump Right" and fell off the board like a lemming. The
  dominant action then kept rotating between updates without the score improving.
- The entropy bonus doesn't encode any game knowledge. It only stops the policy
  from becoming deterministic too early.
- **Entropy is logged** (0 … ln 5 ≈ 1.61) to tune the coefficient:
  - falling below ~0.5 while one action dominates → coefficient too weak;
  - stuck near 1.6 with no progress → too strong.

## Batch and optimizer

- **32 episodes of 60 s per update** (raised from 8), to give each 1 s baseline
  bucket enough samples and reduce gradient noise.
- **Adam, learning rate 1e-3.**
- **64 episodes per update was considered and postponed.** With fixed training time it
  halves the number of updates for only ~√2 less noise. Revisit when the margin
  plateaus for ~1000 updates while the action mix keeps moving.

## Performance

**Decision: rollouts run on a plain-Rust copy of the policy, and the episodes of a batch
are played in parallel (`rayon`). burn is used only for the batched loss and backward
pass.**

- Running the policy through burn cost **~1 ms per decision**, almost all of it
  framework overhead: a tiny forward pass, a synchronising read-back, and a tensor
  round trip for the observation. With a trained policy making ~180 decisions per
  episode, that was ~6.8 s per update, and it kept growing as the agent improved.
- The plain-Rust forward pass costs microseconds, and the 32 episodes are independent,
  so they parallelize across cores.
- **Result: ~0.1–0.2 s per update**, flat regardless of how many no-ops the agent makes.
- Each episode gets its own game seed and sampling seed from the main rng, so a run
  stays reproducible for a given seed despite the parallelism.
- **Observations are stored as a flat `Vec<f32>`** and turned into one `[n, 168]`
  tensor for the loss. `Tensor::stack` over ~1100 small tensors took ~19 s per update.

## Logged diagnostics

Logged every 20 updates, averaged per episode:

- my score and the opponent's score (the margin is the number that matters);
- decisions;
- deaths;
- action histogram (up/down/left/right/no-op);
- entropy.

The histogram and the death count were what revealed the policy collapse and the
"never go Up" local optimum.

Snapshots: `snapshots/reinforce-policy-{update:05}.bpk` (burnpack) every 20 updates.
Opponent: `SimpleAiController` playing blue. The agent always plays red.

## Results

10 000 updates against the simple AI, relative reward. Parameters are saved in
`snapshots/reinforce-params.txt` and the log in `snapshots/reinforce-log.txt`.

| updates | margin (my − opp) | deaths | Up | entropy |
|---|---|---|---|---|
| 1–500 | −118 | 9.0 | 1.7 | 1.05 |
| 2501–3000 | −25 | 6.4 | 5.6 | 0.43 |
| 5001–5500 | −2 | 5.6 | 6.7 | 0.34 |
| 8001–8500 | +21 | 4.4 | 8.0 | 0.27 |
| 9501–10000 | +22 | 4.4 | 7.9 | 0.26 |

- **The agent beats the simple AI** from about update 5500, finishing at ≈ 125 : 103
  per minute. It learned this from the score alone: deaths halved, Up came back from
  almost nothing, and the opponent lost ~65 points per episode compared with the start.
- **It plateaued** at +21–22 for the last ~2000 updates, while entropy kept falling.
  The policy kept becoming more deterministic without getting better.
- `snapshots/reinforce-policy-10000.bpk` is the starting point of the actor-critic.

## Open issues / next steps

- The agent always plays red, so "right" is partly a starting-position habit.
  Randomizing the colour is required before self-play.
- Next: a learned value baseline, see `adr-actor_critic.md`. Then PPO before
  self-play against a pool of snapshots, to avoid forgetting.
