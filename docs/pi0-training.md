# Fine-tune a graph adapter inside π0

Version 0.2 connects the graph encoder to the actual **openpi PyTorch π0 model**. The adapter is optimized through native flow matching on recorded demonstrations and used during every inference denoising step. This is adapter fine-tuning with a frozen pretrained base, not full-model fine-tuning or LoRA.

The implementation targets openpi revision `215abfb217dbac7d5f1273282331b9b1866c0479`, Transformers 4.53.2 with the official openpi replacements, and PyTorch 2.7.1. Source hashes are checked at startup. The training integration deliberately rejects π0.5 and compiled sampling: their different conditioning and execution paths need separate integration work. The existing remote π0.5 client remains usable with an external server.

You must provide real π0 weights and robot demonstrations. No pretrained weights, generated demonstration substitutes, or claimed robot success improvements are bundled. See [validation](validation.md) for what was actually executed.

## What is trained

The native π0 action/time embedding produces action tokens with width 1024. `GroundedPi0.embed_suffix` adds the learned graph residual to these tokens before the frozen action expert. The state token, attention masks, image/language prefix, KV cache, timestep convention, and output projection retain the upstream behavior.

\[
Z = E_\phi(G_t), \qquad H'_a = H_a + \tanh(\alpha)\,\mathrm{Attention}_\phi(\mathrm{LN}(H_a),Z,Z).
\]

Only the graph encoder, pooling queries, adapter cross-attention, normalization and scalar gate are trainable. Freezing the expert's parameters does **not** detach its activations: the flow loss still backpropagates through the expert into the adapter. The initial gate is 0.001. An all-invalid graph produces an exact identity residual.

The upstream model constructs `x_t = t * noise + (1 - t) * actions` and predicts velocity with target `noise - actions`. This trainer averages native elementwise squared error over valid timesteps and physical action dimensions. It does not substitute an independently trained regressor for π0.

Gradient accumulation weights each microbatch by its number of valid action elements. Graph dropout disables the entire graph for selected examples. The validation pass compares graph enabled and disabled using the **same noise and time samples** and preserves the training RNG state. Validation flow loss is a diagnostic, not a robot task success metric.

## 1. Build the training environment

The container targets Linux x86_64 and NVIDIA CUDA through the host NVIDIA Container Toolkit. It installs a CUDA 12.6 PyTorch wheel. The host needs a compatible driver; installing the host toolkit is outside the image. There is no measured VRAM requirement for this release: start with batch size 1 and increase accumulation if necessary. The full frozen base still occupies substantial memory, and gradients must cross the action expert.

From the repository root:

```bash
docker build -f Dockerfile.train -t grounded-vla:pi0 .
mkdir -p data/episodes models runs .cache
```

The container uses UID 1000. Ensure the mounted output/cache directories are writable by that user. It contains the pinned upstream source at `/opt/openpi`, an isolated virtual environment, our CLI, scripts, and example configs. Model weights and data are mounted separately.

A CPU image can be built for integration checks with `--build-arg TORCH_INDEX=https://download.pytorch.org/whl/cpu`. CPU execution also requires `device: "cpu"` and `precision: "float32"` in the config; it is not a practical recommendation for full-scale training.

Alternatively, use a dedicated local environment on Linux, Python 3.11 or 3.12:

```bash
python -m venv .training-venv
source .training-venv/bin/activate
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements-training.txt
git clone https://github.com/Physical-Intelligence/openpi.git ../openpi
git -C ../openpi checkout 215abfb217dbac7d5f1273282331b9b1866c0479
python -m pip install --no-deps ../openpi ../openpi/packages/openpi-client
python scripts/patch_openpi_transformers.py
python -m pip install -e '.[dev]'
python -c 'from grounded_vla.training.runtime import verify_runtime; verify_runtime()'
```

The patch script refuses to modify a global or inherited Transformers installation. Direct dependencies and upstream revision are pinned; the image tag and transitive dependencies are not an immutable environment lock. The optional environment is installed with `--no-deps` for openpi to avoid pulling in unrelated training stacks. A general `pip check` can therefore report upstream extras not used by this pipeline.

## 2. Obtain and convert the base checkpoint

Use a **π0** checkpoint compatible with your robot and action representation. A `.safetensors` file from the pinned native architecture can be used directly. A Hugging Face model with a different key layout is not automatically compatible. Loading checks complete pretrained coverage and refuses missing base weights.

For an official JAX/Orbax checkpoint, the wrapper uses the pinned openpi conversion functions, checks mapped parameter coverage including tied aliases, and saves the native PyTorch model:

```bash
# Local environment; replace the source with the π0 checkpoint you intend to use.
python scripts/convert_pi0_checkpoint.py \
  --openpi-dir ../openpi \
  --source gs://openpi-assets/checkpoints/pi0_base \
  --out models/pi0 \
  --precision float32
python scripts/fetch_tokenizer.py --out models/paligemma_tokenizer.model
```

The source is a checkpoint root containing `params/`, not the `params/` directory itself. Conversion needs substantial host RAM and disk, temporarily holding JAX arrays, mapped tensors and the model. It does not run automatically during Docker build. Use your licensed checkpoint and retain its upstream license/usage conditions when distributing derived weights.

To run the same preparation inside the image:

```bash
docker run --rm \
  -v "$PWD/models:/app/models" -v "$PWD/.cache:/cache" \
  grounded-vla:pi0 python scripts/convert_pi0_checkpoint.py \
  --openpi-dir /opt/openpi --source gs://openpi-assets/checkpoints/pi0_base \
  --out models/pi0 --precision float32

docker run --rm \
  -v "$PWD/models:/app/models" -v "$PWD/.cache:/cache" \
  grounded-vla:pi0 python scripts/fetch_tokenizer.py
```

The trainer builds mixed-precision modules according to the pinned implementation and loads these weights. `precision: "bfloat16"` uses openpi's selected parameter casting, not a new AMP/GradScaler path. The graph adapter remains float32. Full checkpoint conversion was not run in the packaged validation.

## 3. Record synchronized demonstrations and graph snapshots

Follow the complete [dataset contract](training-data.md). Each episode is an NPZ with images, state, expert actions, timestamps, prompts and the graph that was available **before** each action. Do not obtain graph features from the episode's eventual outcome.

`EpisodeRecorder` copies arrays when appending and pads variable graph sizes when saving. `encode_belief_graph` maps the existing evidence ledger to 26 features, retaining supported/refuted/unknown distinctions and ignoring expired, predicted and future evidence. Connect your existing KnowRob mirror/perception integration to this ledger; the recorder does not implement object detection or a new native KnowRob service.

```python
from grounded_vla.training.graph import SCHEMA_ID, encode_belief_graph
from grounded_vla.training.recording import EpisodeRecorder

recorder = EpisodeRecorder(SCHEMA_ID)
# Inside your real demonstration collection loop:
graph = encode_belief_graph(belief_store, tracked_objects, directed_relations, now=observation_time)
recorder.append(
    state=robot_state,
    images=rgb_images,
    action=expert_action,
    prompt=grounded_instruction,
    graph=graph,
    timestamp=observation_time,
    graph_timestamp=observation_time,
)
# At the actual episode boundary:
recorder.save("data/episodes/train-001.npz")
```

Here the variables come from your robot/simulator collector. `tracked_objects` contains `id`, `position` in a declared coordinate frame, and optional `visible`, `is_target`, `is_container` fields. `directed_relations` contains `(source_id, relation_name, target_id)` tuples. Images use the three canonical names listed in the data guide. These are integration inputs, not fabricated sensor values.

Split entire episodes, preferably also sessions/scenes/objects, before computing statistics. Copy `configs/dataset-manifest.example.json` to `data/manifest.json`, replace its episode paths and controller semantics with your actual data, then run:

```bash
grounded-vla-prepare data/manifest.json --out data/prepared.json
```

Or without a local environment:

```bash
docker run --rm -v "$PWD/data:/app/data" grounded-vla:pi0 \
  grounded-vla-prepare data/manifest.json --out data/prepared.json
```

This validates schemas, timestamps, camera availability, control period, finite targets and relation bounds; rejects duplicate episode IDs/paths/bytes; computes state/action statistics from training episodes only; and stores episode checksums. New output files are not silently overwritten.

## 4. Train

Edit `configs/pi0_adapter.json`. Its relative paths resolve against the config file's directory, consistently in Python and Docker. Defaults: π0 dimension 32, horizon 50, 26 graph features, eight relation types/context tokens/heads, batch size 2, accumulation 4, 10,000 optimizer steps, AdamW at 1e-4, warmup/cosine schedule, graph dropout 0.1 and gradient clipping 1.0.

```bash
docker run --rm --gpus all --shm-size=8g \
  -v "$PWD/data:/app/data:ro" \
  -v "$PWD/models:/app/models:ro" \
  -v "$PWD/configs:/app/configs:ro" \
  -v "$PWD/runs:/app/runs" \
  -v "$PWD/.cache:/cache" \
  grounded-vla:pi0
```

The local equivalent is `grounded-vla-train configs/pi0_adapter.json`. Start a short real-data check with `--stop-after 10` before committing to the full run. This preserves the planned learning-rate schedule. Inspect finite losses, gradient norm, adapter gate and the paired validation comparison in `runs/pi0-adapter/metrics.jsonl`.

The trainer is single-device and uses zero background data workers. It keeps at most two decompressed episodes cached; use reasonably sized episode files. DDP, sharded datasets and joint base-model fine-tuning are not implemented. Image augmentation is disabled because applying geometric transforms without updating graph grounding would create mismatched supervision.

## 5. Resume and understand artifacts

Each committed checkpoint directory contains:

- `adapter.safetensors`: only trainable graph-adapter weights.
- `metadata.json`: training configuration, graph schema, action semantics, normalization, source revision, base/tokenizer/dataset hashes, step and data cursor.
- `training_state.pt`: optimizer and Python/NumPy/PyTorch RNG states, loaded with `weights_only=True`.

Checkpoints are written into temporary directories then atomically renamed. The `latest` text file points to the latest committed directory. The pretrained model and tokenizer remain separate.

```bash
grounded-vla-train configs/pi0_adapter.json \
  --resume runs/pi0-adapter/step-00000500
```

Use the same Docker mounts and append this command after the image name to resume in Docker. Do not change the planned schedule, precision, batch size, graph settings or data when resuming an exact run. The stateless shuffle stream resumes from the saved sample cursor. An older checkpoint cannot overwrite a directory containing newer steps. Uncommitted metric rows after the resume step are removed.

Exact equality is tested on CPU in the included reduced native model. CUDA equality across drivers/hardware is not promised. For a different training plan, treat it as a new experiment rather than falsifying resume compatibility.

## 6. Run inference or serve to the existing executive

Export one real recorded observation, without its target action:

```bash
python examples/export_observation.py data/prepared.json \
  --episode val-001 --frame 0 --out runs/observation.npz
grounded-vla-predict runs/pi0-adapter/step-00010000 runs/observation.npz \
  --out runs/predicted-actions.npz
```

Predictions are denormalized and have shape `[horizon, physical_action_dim]`, not `[horizon,32]`. `--base-checkpoint` and `--tokenizer` can relocate dependencies; their content hashes must still match. `--disable-graph` runs the identity-adapter ablation. The same normalization and image transforms are used during training and serving.

```bash
grounded-vla-serve runs/pi0-adapter/step-00010000 --port 8000
python examples/openpi_probe.py runs/observation.npz \
  --embodiment libero --action-dim 7 --skill pick --object 'the clean cup'
```

The server defaults to localhost. For Docker serving, add `-p 127.0.0.1:8000:8000` to the training mounts and append `grounded-vla-serve ... --host 0.0.0.0`. The upstream websocket service does not provide authentication/TLS; keep it behind your access boundary.

The existing `OpenPiPolicy` checks server metadata and forwards `observation["graph"]` for a trained graph-conditioned server. Required fields are `schema_id`, `features`, `edges` and `valid`; the feature order and coordinate conventions must match training. Requests are serialized because graph context is per model invocation. The probe emits action proposals and never drives a robot. Real execution still needs the robot's matching action decoder, limits and observed-effect feedback; the symbolic toy runner is a separate demonstration.

## Evaluate the research claim

Compare unmodified π0, trained adapter with graph disabled, trained adapter with real graph, and controlled graph interventions using paired task/scene seeds and identical control budgets. A graph that changes actions is evidence of influence, not of improvement. Measure episode success, constraint violations, recovery rate, intervention consistency, inference latency and peak GPU memory. Report uncertainty across seeds and all started episodes.

Keep train/validation/test episodes and collection sessions separate. Hold out relevant object/task combinations and test stale, unknown and conflicting predicates. Vary only graph content in intervention pairs; do not also rewrite the image or prompt accidentally. Select hyperparameters on validation data and report final task metrics on the untouched test set.

The explicit decision journal explains symbolic preconditions, rejected proposals and revisions. Adapter attention is not a causal explanation. The graph residual adds a controllable information path but does not force the base model to obey the graph or turn the system into a complete concept bottleneck. Learned ranking/risk heads, native perception, π0.5 conditioning and robot benchmarks remain follow-up research.

## Primary implementation references

- [Official openpi repository](https://github.com/Physical-Intelligence/openpi).
- [Pinned π0 PyTorch implementation](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/models_pytorch/pi0_pytorch.py).
- [Pinned official checkpoint converter](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/examples/convert_jax_model_to_pytorch.py).
- [Official PyTorch 2.7.1 installation variants](https://pytorch.org/get-started/previous-versions/#v271).
