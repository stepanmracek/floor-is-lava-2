# ADR: Exporting a trained policy and running it in the game

Implementation:

- `floor-is-lava-policy` (observation, plain-Rust MLP, SafeTensors save/load);
- `policy::export` in `floor-is-lava-learn`;
- `PolicyController` in `floor-is-lava-3d/src/controller_policy.rs`.

## Goal

Play against a trained agent in the 3D client without linking burn into the front-end.

## A burn-free shared crate: `floor-is-lava-policy`

**Decision: the observation encoding, the action table and a plain-Rust forward pass
live in their own crate, which depends only on the lib and `safetensors`.**

- **`observation(cells, me) -> Vec<f32>` is the single source of truth** for the
  encoding. Training and the game must use exactly the same encoding: any difference
  silently breaks a trained agent, and the features have changed several times.
- **`INPUTS`, `HIDDEN` and `ACTIONS`** (index → `Option<PlayerAction>`) live there too.
- **`PlainPolicy` / `Layer`** is a plain-Rust MLP: three dense layers with ReLU,
  softmax and sampling.
  - `sample_action(observation, uniform)` takes the random number as a parameter, so the
    crate needs no `rand`.
  - A layer's weight is row-major `[inputs, outputs]`, the same layout as burn's
    `Linear`.
- The same code serves two purposes:
  - fast training rollouts (see `adr-reinforce.md`, *Performance*);
  - inference in the game.
- Correctness is guarded by tests in the learn crate that compare plain-Rust logits
  with burn's `forward`, both directly and after a full export/load round trip.

**Considered and postponed: merging the crate into the lib** as a `policy` cargo feature,
with `PolicyController` next to `SimpleAiController`.

- For:
  - the observation is a function of lib types;
  - the headless crate would get the controller for free, e.g. for evaluation;
  - one crate fewer.
- Against: the lib would gain `safetensors` and its dependencies, and become coupled to
  the training's observation layout.

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

## In the game: `PolicyController`

**Decision: a `Controller` that builds the observation, samples an action from the policy
and maps it through `ACTIONS`.**

- **It samples, not argmax.** The policy was trained and evaluated by sampling, keeps
  entropy around 0.45, and decides on every idle tick. With argmax, a state where no-op
  is narrowly the top choice would make the agent wait until the observation changes
  enough. Random numbers come from `rand::random()`.
- **It returns an action every frame,** like the other controllers. `Game::step` ignores
  it while the player isn't idle, which matches training.
- **The model is embedded in the binary** with `include_bytes!`, so `ai-hard` can't fail
  to find its file and doesn't depend on the working directory. The original
  relative-path version broke when started from the workspace root.
- **The agent plays red,** the colour it was trained as. The observation is relative to
  `me`, but the training never played blue, so blue play is untested.

## Known limitation: frame rate

- Training made exactly one decision per fixed 1/60 s tick. The 3D client uses a
  real-time clock and calls `get_action` once per rendered frame.
- At 60 FPS (vsync) this matches training. At higher frame rates the agent gets more
  chances per second to stop hesitating, so it waits less than it learned to; at lower
  frame rates it waits more.
- **Fix if it becomes visible:** throttle the controller to 60 decisions per second.
  That needs the game time, which `Controller::get_action` doesn't receive today.
