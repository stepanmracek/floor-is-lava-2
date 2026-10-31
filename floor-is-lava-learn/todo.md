# Roadmap: learned agent

Decisions and results behind the done items are in `adr-reinforce.md`,
`adr-actor_critic.md` and `adr-policy-deployment.md`.

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

## Before self-play

- [ ] **Check how a red-trained snapshot plays as blue.** Headless, against the simple
  AI, a few dozen games.
  - Every snapshot so far only played red. The observation is relative to "me" for
    ownership and players, but the columns are absolute and the board isn't mirrored
    (red starts at x = −2, blue at x = 2).
  - Plays fine: existing snapshots can be opponents right away.
  - Plays badly: use as opponents only snapshots trained with randomized colours,
    otherwise the agent learns to beat crippled opponents.
- [ ] **Randomize the agent's colour per episode,** so new snapshots can play both sides.
- [ ] **Save the critic alongside the policy snapshots,** and load it on restart.
  Restarts become frequent with changing opponents, and each currently costs the
  200-update warm-up (explained variance starts at ~0.1 instead of ~0.25).

## Evaluation harness

All numbers so far come from training rollouts: a sampling policy, always red, against
the simple AI. Against a mixed opponent pool the training margin depends on which
opponents were drawn, so it stops being comparable over time.

- [ ] Evaluation in `floor-is-lava-headless`: fixed seeds, many games, win rate and
  margin
- [ ] Benchmark opponents: the simple AI plus the frozen snapshots kept in `snapshots/`:
  `reinforce-policy-10000.bpk` (margin +22), `actor-critic-policy.bpk` (+57) and
  `actor-critic-policy2.bpk` (+76)
- [ ] Evaluate as both colours
- [ ] During self-play, evaluate every ~100–200 updates. Progress means rising numbers
  against all benchmarks; falling numbers against old ones mean forgetting, so roll back.

## Self-play

**Order: opponent pool on the current actor-critic first, PPO only if needed.** The
existing safeguards cover most of what PPO would provide at first:

- KL logging shows immediately if updates get too large;
- the learning rate is cautious (KL max 0.0025);
- the simple AI stays in the pool against forgetting.

Tasks:

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

## Smaller items

- [ ] Run name in the snapshot path, e.g. `snapshots/<run>/policy-{update:05}.bpk`.
  Each run's counter starts at 1, so a continued run overwrites the previous run's
  snapshots with the same names; final snapshots are currently renamed by hand.
- [ ] Throttle `PolicyController` to 60 decisions per second if the frame rate visibly
  changes the agent's behaviour (needs the game time in `Controller::get_action`)
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
