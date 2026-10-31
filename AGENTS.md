# Agents guide

## Crates

| crate | purpose |
|---|---|
| `floor-is-lava-lib` | the game: rules, `Game::step`, `Cells`, `Controller` trait, `SimpleAiController` |
| `floor-is-lava-3d` | playable 3D client (kiss3d); red: keyboard / `ai-easy` (simple AI) / `ai-hard` (learned policy, embedded) |
| `floor-is-lava-headless` | runs games without graphics; currently policy-against-policy evaluation over 200 seeded games |
| `floor-is-lava-learn` | RL training with burn: REINFORCE, actor-critic, export |
| `floor-is-lava-policy` | burn-free observation encoding, plain-Rust policy, SafeTensors I/O, `PolicyController` (seeded) |

## Documentation

- `readme.md`, *Game Mechanics*: the rules, including timing details within a step that
  matter for the learned agent.
- `floor-is-lava-learn/adr-reinforce.md`: observation, decision timing, reward,
  discounting, baseline, entropy, performance, REINFORCE results.
- `floor-is-lava-learn/adr-actor_critic.md`: separate critic, warm-up, fine-tuning
  learning rate, bootstrapping, diagnostics, actor-critic results.
- `floor-is-lava-learn/adr-policy-deployment.md`: the shared policy crate, SafeTensors
  export, `PolicyController` (seeding, use in the 3D client and headless), playing blue.
- `floor-is-lava-learn/todo.md`: roadmap as a checklist (next experiments, evaluation,
  self-play).
- `floor-is-lava-learn/practicalities.md`: burn 0.22 and Adam pitfalls, training
  commands, snapshot and log conventions, how to read the training log.
- `floor-is-lava-learn/snapshots/*-params.txt` and `*-log.txt`: parameters and logs of
  finished runs.

## Working agreements

- **The user writes the code.** Propose changes as snippets and review the user's code;
  edit files only when explicitly asked to.
- **The policy's input is a function of `Cells` only** (plus the player's colour). Extra
  information the agent needs goes into `Cell` in the lib, not into the learn crate's
  access to `Game`.
- **The observation encoding is defined once,** in `floor_is_lava_policy::observation`.
  Changing it invalidates every trained snapshot and exported model.
- **Learning experiments run without strategy hints:** no reward shaping such as death
  penalties; the reward is my score gain minus the opponent's.
- **Add `#[derive]`s only when needed.**
- **Before claiming a training change works,** run it briefly from a scratch directory
  and look at the logged diagnostics.

## Commands

```
cargo build --release
cargo test -p floor-is-lava-learn            # includes burn-vs-plain policy checks
cargo run --release -p floor-is-lava-3d -- ai-hard
cargo run --release -p floor-is-lava-learn -- actor-critic 42 <policy.bpk>
cargo run --release -p floor-is-lava-learn -- export <policy.bpk> <policy.safetensors>
cargo run --release -p floor-is-lava-headless -- 42 <red.safetensors> <blue.safetensors>
```
