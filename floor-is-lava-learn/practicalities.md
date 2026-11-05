# Practicalities

Facts about the tools and the training runs that cost time to find out.

## burn 0.22

- **No backend type parameter.** Tensors are `Tensor<D>` (rank `D`, element kind
  defaults to `Float`, e.g. `Tensor<1, Int>`). The backend comes from the `Device`:
  - `Device::default()` for inference;
  - `Device::default().autodiff()` for training.
- **Default features include autodiff;** the backend is chosen by a feature. We use
  **`flex`**, a plain-Rust CPU backend that runs every operation right away.
  - The earlier `cpu` feature (cubecl) merges queued operations into kernels and
    compiles them with LLVM. For our small networks that cost ~0.2 s per REINFORCE update,
    plus occasional ~60 s stalls on a single core, not tied to new batch sizes. Those
    stalls appeared at the first read-back after a new combination of backward and Adam
    operations.
  - With several backend features enabled, `BURN_DEVICE` (`flex`, `cpu`, `ndarray`,
    `vulkan`, …) picks one at runtime. With only one enabled, it is the default.
  - Changing the backend changes floating-point results, so the same seed gives a
    different run. Record the backend in `*-params.txt`.
- **Modules:** `#[derive(Module, Debug)]` structs.
  - `Module::valid()` gives a copy without autodiff.
  - `Module::save_file(path)` / `load_file(path)` read and write **burnpack**. The path is
    used as given and no extension is added, so use `.bpk`.
  - `save_file` consumes the module, so call `.clone()` first.
- **`Linear` stores its weight as `[d_input, d_output]`** (default `LinearLayout::Row`)
  and computes `input · weight + bias`. That's the transpose of PyTorch's convention.
- **Training step:**
  ```rust
  let grads = GradientsParams::from_grads(loss.backward(), &model);
  model = optimizer.step(learning_rate, model, grads);
  ```
  `AdamConfig::new().init()` returns a `ModuleOptimizer`. Parameters without gradients
  in a step are left unchanged.
- **Reading tensors back:** `tensor.try_into_vec_as::<f32>()` (the older
  `into_data().to_vec()` is deprecated); `into_scalar::<f32>()` for single values.
- **`detach()`** stops gradients.
- **Building batches:**
  - `Tensor::from_data(TensorData::new(vec, [n, m]), &device)` from one flat `Vec<f32>`
    is fast.
  - `Tensor::stack` over ~1000 small tensors took ~19 s.
- **Per-call overhead was ~1 ms** (on `cpu`) for a tiny forward pass with a read-back. For many small
  inferences, as in rollouts, copy the weights out (`Policy::to_plain()`) and use plain
  Rust.
- **No ONNX export** (import only). SafeTensors through `burn-store` exists behind a
  feature but isn't used; we write SafeTensors ourselves (see
  `adr-policy-deployment.md`).

## Adam

- Adam's per-parameter normalization makes the overall scale of the loss mostly
  irrelevant. Multiplying the loss by a constant barely changes the steps. Rescaling the
  loss (e.g. per episode instead of per decision) isn't a lever.
- **A fresh Adam state on a trained network is dangerous.** The first steps are
  effectively `lr · sign(gradient)`, so every weight moves by ~lr regardless of its
  gradient. Ramp the learning rate up from 0 after (re)starting from a snapshot.

## Training runs

- **Commands:**
  - `cargo run --release -p floor-is-lava-learn -- reinforce <seed> [OPTIONS]`
  - `cargo run --release -p floor-is-lava-learn -- actor-critic <seed> [policy.bpk] [OPTIONS]`
  - options: `--updates N`, `--report-every N`, `--snapshot-every N`, `--snapshot-prefix P`
    (see `--help`)
  - `cargo run --release -p floor-is-lava-learn -- export <policy.bpk> <policy.safetensors>`

  Always use `--release`.
- **`snapshots/` is created relative to the working directory.** Existing runs are in
  `floor-is-lava-learn/snapshots/`, one subdirectory per burn backend: `cpu/` (all runs
  up to the egocentric actor-critic) and `flex/` (from the switch to `flex` on, e.g.
  `--snapshot-prefix snapshots/flex/reinforce-`; `actor-critic-from-scratch-policy-20000` is `ai-hard`).
- **Snapshots:**
  - `<prefix>policy-{update:05}.bpk`, saved every `--snapshot-every` updates (default 20;
    the log line is printed every `--report-every`, default 20; 0 = never, e.g.
    `--snapshot-every 0 --report-every 1` for debugging); the prefix defaults to
    `snapshots/reinforce-` / `snapshots/actor-critic-` (e.g. `--snapshot-prefix
    snapshots/flex/actor-critic-` for a run kept in `flex/`), and its directory is
    created if missing;
  - policy only: the critic isn't saved, so it has to warm up again on every restart.
- **Each run keeps `*-params.txt` (constants, seed, starting snapshot) and `*-log.txt`**
  next to its snapshots. Both finished runs used seed 42.
- **Reproducibility:** each episode gets its game seed and sampling seed from the main
  rng, so a run is reproducible for a given seed despite the parallel rollouts. The
  REINFORCE run was made before the parallel rollouts and won't reproduce exactly with
  the current code.
- **Speed** (`flex` backend, laptop with an Intel CPU):
  - REINFORCE ~5.5 ms per update (10 000 updates ≈ 1 min);
  - actor-critic with the separate critic and 10 critic steps ~0.1 s per update
    (4000 updates ≈ 6.5 min);
  - the rollout of 32 episodes takes only a few ms; the time goes into burn.
  - On the earlier `cpu` backend: REINFORCE ~0.2 s and actor-critic ~0.45 s per update,
    plus the ~60 s stalls.
  - For timing, `--report-every 1 --snapshot-every 0 | ts "%.T"` gives a timestamp per
    update without writing snapshots.
- **Smoke tests:** run from a scratch directory (e.g. `/tmp/…`) so test snapshots don't
  end up among the real ones.

## Reading the training log

- **All numbers are training-rollout averages per episode:** the policy samples, plays
  the simple AI in 60 s episodes, and gets a random colour per episode (the first
  REINFORCE and actor-critic runs always played red). For strength between policies,
  use the headless evaluation in both colour arrangements; the training margin against
  one opponent can mislead (see `todo.md`, *Retrain on the `flex` backend*).
- **The margin (my score − opponent) is the number that matters,** not the raw score.
- **`decisions` and `no-op` count ticks.** One no-op is 1/60 s, so 100 no-ops is ~1.7 s of
  waiting.
- **`explained var`:** ≈ 0 means the critic is useless, 0.2–0.5 useful, < 0 worse than
  the mean.
- **`kl`:** above ~0.02 per update the policy moves too fast; around 0.0002 it moves
  very cautiously.
- **`entropy`** ranges 0 … ln 5 ≈ 1.61. Watch it fall while one action dominates
  (collapse), or sit near 1.6 without progress (entropy bonus too strong).
- **The 500-update window summaries in the ADRs** came from `awk` over the log, matching
  fields by name. Warm-up lines have an extra `(warm-up)` column.
