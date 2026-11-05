# Roadmap: learned agent

Decisions and results behind the done items are in `adr-reinforce.md`,
`adr-actor_critic.md` and `adr-policy-deployment.md`.

Order: **0. mirror the board for blue (tried, reverted) → 1. egocentric observation,
training as both colours → 2. training against older snapshots → 3. CNN → 4.
procedural level generation.**

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
  `snapshots/cpu/actor-critic-params2.txt`, log in `snapshots/cpu/actor-critic-log2.txt`,
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

- [x] In `floor_is_lava_policy`, when `me` is blue:
  - `observation` reads the columns in reverse order (x → −x);
  - the action is mirrored: Left ↔ Right (Up, Down, None unchanged).

  Both belong to the shared crate, so `PolicyController`, training rollouts and future
  snapshot opponents all get them.
- [x] **Verify with the headless test** (`actor-critic-policy2` on both sides, 200 seeds
  of 60 s, seed 42):
  - red wins 104, blue 93, 3 draws (**52% for red**, was 74%);
  - average margin **+0.6 for red** (was +26.0).

  As predicted: mirrored, the red-trained snapshot plays blue at full strength.
- [x] **Reverted, not kept.** Mirroring ties the encoding to the current start positions
  (−2 / +2) and their symmetry, which shouldn't be wired into the policy. The experiment
  only confirms that the colour gap is the opening learned for x = −2, nothing else.

**Decision:** solve the colour problem in step 1 instead: egocentric observation, and
training runs for both colours (red and blue), so the policy learns both openings
itself. Until then, existing snapshots stay weak as blue.

## 1. Egocentric observation

Chosen over mirroring (step 0), together with training as both colours. Do it before
step 2, because every snapshot is tied to its observation encoding and changing it
after self-play would throw that training away.

- [x] **Dropped: measure corner deaths** on `actor-critic-policy2` in the headless setup
  (with a death classification, since dropped as well), as the baseline for the
  comparison.
  - Not measured. Deaths overall fell from 4.3 to ~1.5 per minute with the egocentric
    observation (REINFORCE plateaus), and in the 3D client the egocentric policy shows
    no visible weakness in the corners. Corner deaths specifically weren't counted.

**Expected benefit:** translation invariance. "Void above me, a block to my left" looks
the same anywhere on the board, so experience from the middle transfers to the rare
corner situations.

**Tasks:**

- [x] **New encoding in `floor_is_lava_policy::observation`,** centred on the player:
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
  - no mirroring for blue: the colour only decides which channels are "mine".

  Done as `observation_egocentric` (455 inputs, `None` while the player is off the
  board). Void stays −1 in `points`; all zeros, with or without a "block present" flag,
  was considered and not taken.
- [x] **Train as both colours:** pick the agent's colour per episode (half red, half
  blue), so it learns the opening from both start positions. Check with the headless
  colour test: ~50% / margin ~0 with the same snapshot on both sides.
  - Done in `reinforce.rs` and `actor_critic.rs`. Same plateau as blue only (+80.1 vs
    +80.8); against the simple AI equal from both sides (+76.2 as red, +77.2 as blue;
    blue only: +82.7 / +65.8).
  - Self-play gap +19.9 (blue only, 400 games) → **+6.3, 55.7% for red** (2000 games).
    Not ~0 yet; likely cause is the simple AI's tie-break (next item).
- [x] **Random tie-break in `SimpleAiController`:** among the plausible moves with the
  highest value, choose one at random, from a seeded rng (`SimpleAiController::new(color,
  seed)`).
  - Today `max_by_key` returns the last maximum, so ties go Right > Left > Down > Up.
    Red at x = −2 drifts towards the centre and the opponent, blue at x = 2 towards its
    edge: simple AI against itself gives red 220 : 178 (55%), margin **+9.5** over 400
    games, while a uniformly random player against itself gives +0.3 over 2000, so the
    game itself is symmetric.
  - A fixed tie-break order is also a pattern the agent can learn to exploit. Likely
    cause of the residual self-play gap of the egocentric policy trained as both colours
    (55.7% / +6.3 for red over 2000 games): it learned to counter an opponent that
    drifts right.
  - Seeds: training from the episode's seeds with an offset from the agent's sampling
    seed; headless from the game seed; the 3D client (`ai-easy`) any.
  - Check: simple AI against itself, 2000 games, ~50% / margin ~0. Then retrain (or
    continue actor-critic) against it and re-run the self-play colour test.
  - The opponent changes, so margins against the simple AI aren't comparable with older
    logs; note it in the params files.

  Done. Training seeds it from `rng_seed + 2³²`, the 3D client with 42.
  - Simple AI against itself, 2000 games: red 968 : 1023 (48.6%), margin **−1.7**, was
    +9.5. Symmetric within noise.
  - The existing policies against it, 400 games per colour: `reinforce-ec-policy2-10000`
    +77.4 as red / +79.9 as blue (old AI: +76.2 / +77.2); `actor-critic-ec-policy3-04000`
    +94.2 / +95.3, 786 of 799 games won. No colour difference, and about the same
    margins as against the old AI, so the policies didn't depend on its tie-break.
  - **Not retrained against it.** The simple AI is nearly beaten, and the opponent pool
    (step 2) includes it anyway. Whether the old tie-break caused the self-play gap
    (+6.3) stays open; check the gap again after the first pool run.
- [x] **Dropped: save the critic alongside the policy snapshots,** and load it on
  restart. Each restart currently costs the 200-update warm-up (explained variance
  starts at ~0.1 instead of ~0.25).
  - Not worth it: the warm-up costs ~1.5 min (200 × ~0.45 s), and the critic doesn't
    keep improving anyway. In the egocentric actor-critic run explained variance fell
    from ~0.31 (warm-up) to ~0.20 and stayed there, so a saved critic isn't much better
    than a freshly warmed-up one.
- [x] **Run name in the snapshot path,** e.g. `snapshots/<run>/policy-{update:05}.bpk`.
  Each run's counter starts at 1, so a continued run overwrites the previous run's
  snapshots; final snapshots are currently renamed by hand.
  - Done: `--snapshot-prefix snapshots/<run>/` (the directory is created). Also
    `--updates`, `--report-every` and `--snapshot-every`; see `practicalities.md`.
- [x] **Retrain from scratch:** the existing snapshots can't be converted. Roughly
  REINFORCE 10 000 updates (~30 min) plus actor-critic ~30 000 (~4 h); possibly
  actor-critic directly from a random policy.
  - REINFORCE, both colours, 10 000 updates (`reinforce-ec-params2.txt`): margin 0 at
    update ~360 (absolute: ~5500), plateau **+80** (absolute: +21), deaths 1.5 (4.3).
  - Actor-critic from it, 4120 updates (`actor-critic-params3.txt`), final
    `actor-critic-ec-policy3-04000.bpk`: training margin +77 → ~+92 by update ~1000,
    then flat. Head-to-head against the REINFORCE policy, both colour arrangements, 200
    games each: **72.5% wins, margin +25.2** (snapshot 1780: 63.4% / +13.3), so it kept
    improving after the training margin went flat.
  - Both runs are in `snapshots/cpu/`.
- [x] **Retrain on the `flex` backend** (`snapshots/flex/`, seed 42, same constants).
  The simple AI now breaks ties randomly and the same for both colours (`231ac99`).
  - REINFORCE, 10 000 updates (~1 min): margin 0 at update ~400, plateau **+78**, deaths
    1.5, the same as on `cpu`. Head-to-head against `cpu/reinforce-ec-policy2-10000`,
    both colour arrangements, 200 games each: **52.8% wins, margin +3.7**, a draw within
    noise. So the backend doesn't change what is learned.
  - Actor-critic from it, stopped after 4000 updates (~6.5 min), final
    `actor-critic-policy-04000.bpk`. Training margin +91 → **+98**, still rising slowly;
    KL max < 0.001, explained variance ~0.2.
  - Head-to-head against its REINFORCE start: 68.8% / +24.2. The `cpu` actor-critic
    against the same: 75.0% / +30.3.
  - Head-to-head against `cpu/actor-critic-ec-policy3-04000`, 5 seeds × 2 colour
    arrangements × 200 games: **38.1% wins, margin −15.5**. It loses clearly despite
    its higher training margin.
    - Likely cause: each was trained against a different simple AI (fixed vs random
      tie-breaks). A margin against one opponent is a poor strength measure, which
      supports the opponent pool and the benchmark opponents (step 2).
  - `ai-hard` then embedded the `flex` actor-critic, trained against the current simple
    AI. Both actor-critics are candidates for the opponent pool.
- [x] **Actor-critic from scratch on `flex`** (no REINFORCE start, constant lr 1e-3),
  22 300 updates, final `flex/actor-critic-from-scratch-policy-20000.bpk`; details in
  `adr-actor_critic.md`, *From scratch*.
  - Training margin +65 by update 4000, **+95–100** at the end, still creeping up
    (~+3 per 2000 updates); deaths 1.0.
  - Head-to-head, seed 42, both colour arrangements, 200 games each: against
    `flex/actor-critic-policy-04000` **62% / +15.8**, against
    `cpu/actor-critic-ec-policy3-04000` **53% / +1.3** (a draw).
  - So REINFORCE pretraining isn't needed. `ai-hard` embedded this policy until the
    first pool run (step 2).
  - Untried: continuing from the 20 000 snapshot at lr 3e-4.
- [x] **Re-export `ai-hard`;** `PolicyController` follows the new observation
  automatically through the shared crate.
  - Embedded `reinforce-ec-policy2-10000.safetensors` first, then
    `cpu/actor-critic-ec-policy3-04000.safetensors`, then
    `flex/actor-critic-policy-04000.safetensors`, then
    `flex/actor-critic-from-scratch-policy-20000.safetensors`, now
    `flex/actor-critic-from-scratch-pool-policy-08000.safetensors` (step 2).
    `PolicyController` picks the encoding from the model's input size.

## 2. Training against older snapshots

**Order: opponent pool on the current actor-critic first, PPO only if needed.** The plan
was that the existing safeguards cover most of what PPO would provide at first:

- KL logging shows immediately if updates get too large;
- the learning rate is cautious (KL max < 0.001 in the fine-tuned `flex` actor-critic
  run);
- the simple AI stays in the pool against forgetting.

The first pool run showed the limits: it trained from scratch at a constant lr 1e-3
(logged KL up to ~0.01), and the update that collapsed it fell between two report lines,
so KL logging didn't show it. See *Guard against the policy collapse* below.

The pool was planned to start with the simple AI and the egocentric `cpu/` and `flex/`
snapshots; the first run used only the simple AI and its own snapshots (see the done
item below). Snapshots with the old (absolute) encoding no longer load.

**Tasks:**

- [ ] **Benchmark opponents for the evaluation:** the simple AI plus frozen snapshots
  (egocentric). Evaluate every ~100–200 updates. Progress means rising numbers against all
  benchmarks; falling numbers against old ones mean forgetting, so roll back.
- [x] **Opponent pool, one opponent per episode:**
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
    - snapshots are loaded into `Mlp` once, when the pool changes, not per
      episode. `PolicyController` owns its `Mlp`, so sharing one loaded snapshot
      between parallel episodes needs a small change, e.g. holding it in an `Arc`.

  Done in `actor_critic.rs` only (`e498921`), differently from the plan:
  - the pool holds only the simple AI and the run's own snapshots, not the older
    `cpu/` and `flex/` ones;
  - shares: empty pool → simple AI; one snapshot → 50% it, 50% simple AI; otherwise 50%
    latest, 10% simple AI, 40% uniform over the older snapshots;
  - snapshots join only once the margin against the simple AI over the last 10 updates is
    positive (a one-time gate, like the critic warm-up); after that every
    `--snapshot-every` snapshot joins;
  - snapshot opponents borrow the shared `Mlp` and sample like `PolicyController`, so no
    `Box<dyn Controller>` or `Arc` was needed.

  First run, from scratch on `flex` (`snapshots/flex/actor-critic-from-scratch-pool-*`):
  - Pool open at update 614. Margin against the simple AI flat at ~+55 (no-pool run:
    +60–75 at the same updates), but much stronger head-to-head: `pool-policy-08000`
    wins 83–90% (margin +37 to +47) against all earlier final snapshots, including the
    no-pool `actor-critic-from-scratch-policy-20000` (83.0%, +39.5).
  - Round robin of the pool snapshots 1000–8000: strictly ordered by age, every later
    one beats every earlier one, so no cycling. Much of 8000's edge is against its own
    earlier snapshots (+68–70 against 5000/6000 vs. +40 against outside ones).
  - From ~4000 on, waiting standoffs: two copies on the same block wait for each other,
    in sync. No-ops grew from ~20 to 400–800 per episode, entropy fell from ~0.6 to
    0.07–0.10, explained variance to ~0.05.
  - **Collapse at update ~9000:** the policy saturated on always Down (12 deaths per
    episode, entropy 0.00, KL 0.0000, no recovery). Likely a single large update at the
    constant lr 1e-3 on a near-saturated policy. Collapsed snapshots joined the pool too.
    Log and snapshots are kept up to 8000.
  - `pool-policy-08000` is now `ai-hard`. Details in `adr-actor_critic.md`, *Opponent
    pool*.
- [ ] **Guard against the policy collapse** seen in the first pool run (update ~9000,
  saturated on always Down, no recovery):
  - [ ] **KL guard on every update:** compute `approx_kl` after each update (plain-Rust
    forward pass, cheap next to the rollouts), and discard the update if it is above a
    threshold (e.g. 0.02), keeping the model from before the step. A basic trust region
    without PPO.
  - [ ] **Log the maximum KL over the report window,** not only the KL of the reported
    update: the update that broke the policy fell between two report lines.
  - [ ] **Check each snapshot before it joins the pool:** only if its margin against the
    simple AI since the last report is still positive, so a broken policy doesn't take
    over the 50% "latest" share.
  - [ ] **Count no-ops per opponent group** in the log. If waiting is a standoff between
    near-copies, no-ops against the simple AI stay low; if they are high there too, it's
    a habit that doesn't depend on the opponent.
  - [ ] **Watch entropy;** if it falls below ~0.2 again, raise `ENTROPY_COEF` to 0.02
    (pushes towards the mixed strategy the chicken game calls for; not a strategy hint).
  - [ ] **Restart from `flex/actor-critic-from-scratch-pool-policy-08000.bpk`,** the
    last snapshot before the collapse: critic warm-up and `FINE_TUNE_LEARNING_RATE` 3e-4
    with the ramp instead of the constant 1e-3.
    - Open: reload the run's own snapshots 1000–8000 into the pool, or start with an
      empty pool.
- [ ] **PPO** (clipped probability ratio, several epochs per batch), if KL spikes or the
  evaluation shows forgetting.
  - The first pool run collapsed; try the KL guard above first, PPO if it isn't
    enough.
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
    distribution (the colours stay equal).
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
- [x] Rename `PlainPolicy` to a neutral `Mlp`; the critic reuses it with one output.
  The policy crate is split into `observation.rs`, `mlp.rs` and `controller.rs`.
- [x] **Slow updates with new batch sizes.** In the egocentric actor-critic run, some
  blocks of 20 updates took ~1 min on a single core, others a few seconds.
  - Hypothesis: burn's `cpu` backend compiles kernels per tensor shape, and the batch
    shape `[decisions, INPUTS]` changes every update (~3800 ± a few hundred rows).
  - **Rejected.** Timing every REINFORCE update (100 updates, ~1100 decisions each):
    stalls of ~59 s at updates 2, 3 and 22, while 85 of 88 first-seen batch sizes were
    fast.
    - Timing each phase put the whole stall in the next `to_plain()`, the first
      read-back after the lazily queued backward pass and Adam step. The rollout,
      loss, backward and step themselves took ≤ 0.3 s.
    - Only one core was busy during a stall. That fits cubecl compiling merged kernels
      with LLVM.
  - **Fixed by switching burn to the `flex` backend:** no stalls, and ~5.5 ms instead of
    ~0.2 s per REINFORCE update. Padding batches isn't needed. See `practicalities.md`
    (*burn 0.22*, *Speed*).
- [ ] Share the rollout loop between `reinforce.rs` and `actor_critic.rs` if both stay
- [ ] Maybe: merge `floor-is-lava-policy` into the lib behind a `policy` feature
- [ ] Maybe: GAE / TD(λ) instead of Monte Carlo returns, if the critic gets good enough
- [ ] Untried hyperparameter options, should training against a fixed opponent plateau
  again:
  - `FINE_TUNE_LEARNING_RATE` = 1e-3 (KL shows the current 3e-4 is very cautious; step
    back to 5e-4 if KL climbs above ~0.01);
  - plateau with low entropy: raise `ENTROPY_COEF` to 0.02–0.03;
  - plateau while the action mix keeps moving: 64 episodes per update.
