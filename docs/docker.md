# Docker: pi0, temporal ViT, and Isaac Sim

Build from the repository root. These are two environments communicating over
the existing openpi websocket protocol:

| Image | Dockerfile | Purpose |
| --- | --- | --- |
| `grounded-vla:pi0` | `Dockerfile.train` | π0 adapter training/serving and temporal ViT training |
| `grounded-vla:isaacsim` | `Dockerfile.isaacsim` | Isaac Sim cameras/joints, recordings, and online ViT monitoring |

The trainer pins torch 2.7.1 and torchvision 0.22.1 on CUDA 12.6. The simulator
uses `nvcr.io/nvidia/isaac-sim:6.0.1` and retains its bundled Python, NumPy,
torch 2.11.0 and torchvision 0.26.0. It installs only the transport dependencies
needed by this integration, rather than the trainer's JAX/Transformers stack.
Model weights, USD assets, and recorded data are mounted at runtime.

## Build

```bash
docker build -f Dockerfile.train -t grounded-vla:pi0 .
docker build -f Dockerfile.isaacsim -t grounded-vla:isaacsim .
```

The new `.dockerignore` excludes local virtual environments, Git history,
datasets, assets, model weights, runs, caches, and environment files from the
build context. It keeps the package's `src/grounded_vla/data` resources.

Both builds run `scripts/check_container_python.py`: imports/version checks,
an actual openpi MessagePack round-trip of RGB/graph arrays, and a small native
torchvision temporal-ViT forward pass. These checks run on CPU without starting
Kit or downloading model weights. The simulator check also verifies that the
Isaac Sim package is discoverable.

The upstream client is checked out at the same revision as the π0 trainer. In
the simulator it is installed with `--no-deps`: only its `msgpack_numpy` codec
is used. This avoids its NumPy `<2` metadata downgrading NVIDIA's bundled NumPy.
The codec check exercises the actual installed NumPy version. This is not a
claim that the client's unrelated image/policy utilities support NumPy 2.

You can use an immutable NVIDIA image digest via `--build-arg ISAACSIM_IMAGE=...`.
If changing the Isaac Sim release, update/revalidate the bundled version checks
and native bridge together; do not overwrite the simulator with training pins.
For CPU-only trainer checks, retain the existing build argument:

```bash
docker build -f Dockerfile.train -t grounded-vla:pi0-cpu \
  --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cpu .
```

GPU execution requires a compatible NVIDIA host and NVIDIA Container Toolkit.
Isaac Sim needs supported RTX rendering hardware; the CPU trainer variant does
not provide a CPU-only simulator.

## Train the temporal critic

Provide labelled episodes and edit `configs/critic-manifest.example.json` as
described in [isaacsim-vit.md](isaacsim-vit.md). Prepare `data`, `runs`, and `.cache`
on the host. Training uses UID 1000; mounted output/cache paths must be writable
by that UID.

```bash
docker run --rm --gpus all \
  -v "$PWD/data:/app/data:ro" \
  -v "$PWD/configs:/app/configs:ro" \
  -v "$PWD/runs:/app/runs:rw" \
  -v "$PWD/.cache:/cache:rw" \
  grounded-vla:pi0 grounded-vla-train-critic \
  configs/critic-manifest.example.json \
  --out runs/critic/critic.pt --device cuda --epochs 5 --window-size 8
```

`/cache/torch` retains the ImageNet initialization after the first download.
The π0 training commands in [pi0-training.md](pi0-training.md) still apply.

## Verify Isaac Sim cameras before policy execution

Set real USD and camera paths in `configs/isaacsim.example.json`. Host assets
mounted at `/app/assets` match the template's `../assets/scene.usd`. The simulator
runs as NVIDIA's UID 1234; its output and cache mounts must be writable by that
UID. Give each runtime its own writable subdirectory when they share a host.

The following commands require you to accept NVIDIA's license with
`ACCEPT_EULA=Y`. The Dockerfile does not embed that acceptance or enable optional
telemetry consent. Read the upstream terms before using the flag.

```bash
docker run --rm --gpus all -e ACCEPT_EULA=Y \
  -v "$PWD/configs:/app/configs:ro" \
  -v "$PWD/assets:/app/assets:ro" \
  -v "$PWD/runs/simulation:/app/runs:rw" \
  -v "$PWD/.cache/isaacsim:/cache:rw" \
  grounded-vla:isaacsim configs/isaacsim.example.json \
  --capture-only --out runs/camera-check
```

Create the mounted host directories with the correct ownership first. Choose a
new `--out` directory for each run; the CLI refuses to overwrite prior output.
The simulator image's default command is `--help`, so launching it without a
configuration does not execute a scene. All examples here are headless batch
jobs; they do not set up GUI or WebRTC streaming.

## Connect the simulator to a trained pi0 server

Create a Docker network once and start the server:

```bash
docker network create grounded-vla
docker run --rm --gpus all --name grounded-vla-policy --network grounded-vla \
  -v "$PWD/models:/app/models:ro" \
  -v "$PWD/runs:/app/runs:ro" \
  -v "$PWD/.cache/training:/cache:rw" \
  grounded-vla:pi0 grounded-vla-serve runs/pi0-adapter/step-00010000 \
  --host 0.0.0.0 --port 8000
```

Use a trained **canonical** checkpoint with matching joint order, absolute
position convention, and control period. Paths to its base checkpoint and
tokenizer must resolve inside the mounts; the serving CLI also accepts
`--base-checkpoint` and `--tokenizer` to relocate them with identity checks.

From another terminal, after the server is ready, check the connection with
explicit graph/critic ablations:

```bash
docker run --rm --gpus all --network grounded-vla -e ACCEPT_EULA=Y \
  -v "$PWD/configs:/app/configs:ro" \
  -v "$PWD/assets:/app/assets:ro" \
  -v "$PWD/runs/simulation:/app/runs:rw" \
  -v "$PWD/runs/critic:/app/critic:ro" \
  -v "$PWD/.cache/isaacsim:/cache:rw" \
  grounded-vla:isaacsim configs/isaacsim.example.json \
  --uri ws://grounded-vla-policy:8000 \
  --prompt 'pick the part and place it in the tray' \
  --empty-graph --no-critic --max-steps 20 --out runs/connection-check
```

For trained monitoring, replace `--no-critic` with
`--critic-checkpoint /app/critic/critic.pt --critic-device cpu`. The CPU choice
leaves GPU capacity for rendering and π0; it is not a real-time latency claim.
Replace `--empty-graph` with your real `--graph-provider module:function` and
mount that Python module/package under `/app` so the interpreter can import it.
The same applies to `--action-guard` and `--on-event` callbacks.

The server is reachable by its container name on this network; no host port is
published. Use a trusted network because the policy websocket has no built-in
authentication. Requests time out if the server is unavailable.

## What was checked

Docker is not available in the implementation environment, so neither image
has been built here and no native Isaac Sim/GPU rollout has been run. Local
checks exercise the CPU smoke script with the pinned upstream codec and the
available PyTorch/torchvision pair; the simulator's bundled pair is checked
by its Docker build. The existing source tests remain separate from native
container validation. Run both builds and the camera check on your GPU host
before relying on the container setup.

## Upstream references

- [NVIDIA container installation and runtime UID](https://docs.isaacsim.omniverse.nvidia.com/6.0.1/installation/install_container.html)
- [NVIDIA bundled Python launcher](https://docs.isaacsim.omniverse.nvidia.com/6.0.1/installation/install_python.html)
- [Isaac Sim 6.0.1 package versions](https://github.com/isaac-sim/IsaacSim/blob/v6.0.1/python_packages.toml)
- [Pinned openpi client](https://github.com/Physical-Intelligence/openpi/tree/215abfb217dbac7d5f1273282331b9b1866c0479/packages/openpi-client)
