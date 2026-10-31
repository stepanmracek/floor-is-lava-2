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

## Next: squeeze more out of actor-critic

- [ ] Continue from `actor-critic-policy-10000.bpk` with `FINE_TUNE_LEARNING_RATE` = 1e-3
  - watch KL; step back to 5e-4 if it climbs above ~0.01
  - the margin was still rising, so this is the cheapest way to find the plateau
- [ ] If the margin plateaus with low entropy: raise `ENTROPY_COEF` to 0.02–0.03
- [ ] If the margin plateaus while the action mix keeps moving: 64 episodes per update

## Evaluation harness (before trusting any number further)

All numbers so far come from training rollouts: a sampling policy, always red, against
the simple AI.

- [ ] Evaluation in `floor-is-lava-headless`: fixed seeds, many games, win rate and
  margin
- [ ] Benchmark opponents: simple AI plus a few frozen snapshots (e.g. REINFORCE 10 000,
  actor-critic 10 000)
- [ ] Evaluate as both colours

## Self-play

Order matters: PPO first, then opponents other than the simple AI.

- [ ] **Randomize the agent's colour per episode.** Required: snapshot opponents play
  blue through `observation(cells, Blue)`, so the network must have seen both sides.
- [ ] **PPO** (clipped probability ratio, several epochs per batch)
  - limits how far one update moves the policy, which is the real fix for "too
    aggressive and it forgets"
  - builds on what exists: critic, stored rollout log-probabilities, KL logging
- [ ] **Opponent pool, one opponent per episode:**
  - ~50% recent snapshots, ~30% older snapshots, ~20% simple AI;
  - snapshot opponents sample their actions (deterministic opponents are exploitable
    in ways that don't generalize).
  - Forgetting comes mainly from the opponent mix changing, not from the learning rate:
    without old opponents in the batches nothing reminds the policy how to beat them.
- [ ] **Track the win rate against fixed benchmark opponents** every ~100 updates and roll
  back if it falls against old ones
- [ ] **On every restart from a snapshot:**
  - ramp the learning rate up again (the fresh Adam state makes sign-sized steps), or
    save and restore the optimizer state;
  - keep the entropy bonus.

## Smaller items

- [ ] Throttle `PolicyController` to 60 decisions per second if the frame rate visibly
  changes the agent's behaviour (needs the game time in `Controller::get_action`)
- [ ] Rename `PlainPolicy` to a neutral `PlainMlp`; the critic reuses it with one output
- [ ] Share the rollout loop between `reinforce.rs` and `actor_critic.rs` if both stay
- [ ] Maybe: merge `floor-is-lava-policy` into the lib behind a `policy` feature
- [ ] Maybe: GAE / TD(λ) instead of Monte Carlo returns, if the critic gets good enough
