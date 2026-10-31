# ADR: Exporting a trained policy and running it in the game

Implementation:

- `floor-is-lava-policy`: observation, plain-Rust MLP, SafeTensors save/load, and
  `PolicyController` (`src/controller.rs`);
- `policy::export` in `floor-is-lava-learn`.

Users of `PolicyController`:

- the 3D client (`ai-hard`);
- the headless crate (policy-against-policy evaluation).

## Goal

Play against a trained agent in the 3D client, and run trained agents headless, without
linking burn into anything but the training.

## A burn-free shared crate: `floor-is-lava-policy`

**Decision: the observation encoding, the action table, a plain-Rust forward pass and the
`Controller` built on them live in their own crate. It depends only on the lib,
`safetensors` and `rand`.**

- **`observation(cells, me) -> Vec<f32>` is the single source of truth** for the
  encoding. Training and the game must use exactly the same encoding: any difference
  silently breaks a trained agent, and the features have changed several times.
- **`INPUTS`, `HIDDEN` and `ACTIONS`** (index → `Option<PlayerAction>`) live there too.
- **`PlainPolicy` / `Layer`** is a plain-Rust MLP: three dense layers with ReLU,
  softmax and sampling.
  - `sample_action(observation, uniform)` takes the random number as a parameter, so
    callers decide where randomness comes from: the training rollouts use their
    per-episode rng, `PolicyController` its own seeded rng.
  - A layer's weight is row-major `[inputs, outputs]`, the same layout as burn's
    `Linear`.
- The same code serves two purposes:
  - fast training rollouts (see `adr-reinforce.md`, *Performance*);
  - inference in the game and in headless evaluation.
- Correctness is guarded by tests in the learn crate that compare plain-Rust logits
  with burn's `forward`, both directly and after a full export/load round trip.

**`PolicyController` lives here, not in a front-end.** It was first written in the 3D
crate, then moved so the 3D client and the headless crate share one implementation, and
future snapshot opponents in training can use it too. Anything that changes how a policy
plays, such as mirroring the board for blue (see `todo.md`, step 0), then happens in one
place. The 3D crate no longer needs `rand`.

**Considered and postponed: merging the crate into the lib** as a `policy` cargo feature,
with `PolicyController` next to `SimpleAiController`.

- For:
  - the observation is a function of lib types;
  - one crate fewer.
- Against: the lib would gain `safetensors` and its dependencies, and become coupled to
  the training's observation layout.
- Moving `PolicyController` into this crate already gave the main benefit (every
  consumer shares it), so the merge is now mostly about having one crate fewer.

## File format: SafeTensors, written by us

**Decision: export to SafeTensors with our own tensor names, written from
`Policy::to_plain()` with the `safetensors` crate.**

- **Not ONNX:** burn 0.22 can't export ONNX (its ONNX support is import only), and an
  ONNX runtime in the game (`tract-onnx`, `ort`) is a lot of machinery for a 3-layer MLP.
- **Not burn's own SafeTensors adapter:** writing the file ourselves makes the tensor
  names and the weight layout match `PlainPolicy` exactly. It doesn't depend on how
  burn names tensors or whether it transposes `Linear` weights to PyTorch's layout.
- **Format,** all F32 little-endian:
  - `{layer}.weight` with shape `[inputs, outputs]` (row-major);
  - `{layer}.bias` with shape `[outputs]`;
  - layers `hidden1`, `hidden2`, `output`.
- **`PlainPolicy::from_bytes` / `load` validate** every tensor's dtype and shape against
  `INPUTS`, `HIDDEN` and `ACTIONS`. A file from an older network or observation layout
  fails with a clear error instead of playing garbage.
- **Trade-off:** the network's architecture is defined twice, once in burn's `Policy` and
  once in `PlainPolicy`. For a small MLP the duplication is a few lines and the
  round-trip test catches any mismatch. If the architecture grows (convolutions,
  attention), ONNX becomes worth reconsidering.

Usage:

```
floor-is-lava-learn export <policy.bpk> <policy.safetensors>
```

## `PolicyController`

**Decision: a `Controller` that builds the observation, samples an action from the policy
and maps it through `ACTIONS`.**

- **Construction:** `from_path(path, color, seed)` for files (headless),
  `from_bytes(bytes, color, seed)` for embedded models (3D client).
- **It samples, not argmax.** The policy was trained and evaluated by sampling, keeps
  entropy around 0.45, and decides on every idle tick. With argmax, a state where no-op
  is narrowly the top choice would make the agent wait until the observation changes
  enough.
- **Each controller samples from its own seeded rng** (`StdRng`).
  - The first version used `rand::random()` (thread-local, seeded by the OS), so
    headless games with a fixed game seed still differed from run to run.
  - With a seeded rng a headless run repeats exactly.
  - **Two controllers in the same game need different seeds.** Otherwise both draw the
    same random number on every tick, which correlates their decisions; the headless
    crate offsets blue's seed by 2³².
- **It returns an action every frame,** like the other controllers. `Game::step` ignores
  it while the player isn't idle, which matches training.

## In the 3D client

- **The model is embedded in the binary** with `include_bytes!`
  (`snapshots/actor-critic-policy2.safetensors`), so `ai-hard` can't fail to find its
  file and doesn't depend on the working directory. The original relative-path version
  broke when started from the workspace root.
- **The controller's seed is fixed (42).** The board (`Game::new()`) and the human's
  input differ every game, so games don't repeat, but the AI's random sequence starts
  the same every session. Seeding from the clock would vary it without adding `rand`
  back to the 3D crate.
- **The agent plays red,** the colour it was trained as.

## Playing blue

Every snapshot so far was trained as red only. Headless evaluation of
`actor-critic-policy2` against itself shows:

- **Over one 24-hour game the colours are equal** (102.5 vs 103.6 points per minute):
  once both players have moved around, the start position doesn't matter.
- **Over 200 games of 60 s, red wins 74% with a margin of +26.0** (seed 42,
  reproducible). The policy learned an opening for starting at x = −2 and plays it badly
  from x = 2.

Planned fix: mirror the board for blue (reverse the columns in the observation, swap
Left ↔ Right in the action). The rules are left-right symmetric, so this is exact and
needs no retraining. See `todo.md`, step 0.

Usage of the headless evaluation:

```
floor-is-lava-headless <seed> <red.safetensors> <blue.safetensors>
```

It plays 200 games of 60 s with game seeds `seed … seed + 199` and prints red's wins,
blue's wins and red's average margin.

## Known limitation: frame rate

- Training made exactly one decision per fixed 1/60 s tick. The 3D client uses a
  real-time clock and calls `get_action` once per rendered frame.
- At 60 FPS (vsync) this matches training. At higher frame rates the agent gets more
  chances per second to stop hesitating, so it waits less than it learned to; at lower
  frame rates it waits more.
- **Fix if it becomes visible:** throttle the controller to 60 decisions per second.
  That needs the game time, which `Controller::get_action` doesn't receive today.
