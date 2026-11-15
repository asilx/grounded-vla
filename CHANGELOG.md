# Changelog

## Unreleased — grounded Isaac pick/place application

- Add typed Isaac scene/contact measurements, causal grasp/release evidence and
  task graphs, bounded pick/place/regrasp selection and measured inspection holds.
- Add observed-contact exemptions for pick geometry dependencies and journal the
  supporting evidence; keep ordinary dependencies and legacy rollouts compatible.
- Add native task CLI, scene/config examples, optional conditioned critic recording
  and raw-measurement scoring with all-attempt aggregation.
- Add CPU regression and composed task tests. Native Isaac GPU execution and
  pretrained-policy task performance remain unvalidated for this addition.

## Unreleased — isaac-trl-tests
- Add belief-aware native chunk execution with declared dependency checks,
  evidence-linked cancellation journals, observed-effect verification, bounded
  recovery and scene-preserving simulator resume.
- Add optional contract/state/graph conditioning to temporal ViT training and
  inference, versioned checkpoints, causal recording and task-bounded windows.
- Add a runnable CPU control-flow example, integration documentation and
  behavioral/gradient/data compatibility tests.
- Make three legacy experiment tests portable outside the original Runpod path.

## Unreleased — Isaac Sim and temporal ViT

- Update `Dockerfile.train` with paired torchvision, ViT checks, and a model cache.
- Add an official Isaac Sim 6.0.1 container, transport-only dependencies,
  `.dockerignore`, and two-container training/serving/rollout instructions.
- Add a CPU build check for real openpi MessagePack arrays and temporal ViT;
  actual Docker builds and native GPU execution still need a Docker/GPU host.

- Base: GitHub `pi0-integration` at `13f6d9129c05da8730700554f27eaf1377f51e39`.
- Add a lazy-import Isaac Sim RGB/articulation backend, strict canonical π0
  metadata handshake, bounded action-prefix execution, and causal NPZ recording.
- Add a torchvision ViT-B/16 encoder and causal temporal critic, supervised
  episode-split training, offline checkpoint loading, and pause-before-callback events.
- Add scene/critic configuration templates, CLI entry points, CPU tests, and CI.
  Native Isaac Sim execution remains unverified.
- Keep recording independent of PyTorch and expose state/graph dimensions in
  policy-server metadata. Existing LIBERO/DROID adapters and native π0 training remain available.

## 0.2.0 — 2026-09-18

- Integrate graph conditioning into actual openpi π0 action tokens in training and denoising.
- Add frozen-base flow-matching fine-tuning, causal episode preparation, train-only statistics, masked targets and graph ablations.
- Save adapter artifacts with identity checks and exact CPU optimizer/RNG/data-cursor resume.
- Restore native inference and serve graphs through the existing websocket client.
- Add checkpoint conversion, tokenizer setup, training Dockerfile, pinned dependencies and English training/data guides.
- Validate native algorithms with reduced constructor dimensions; full pretrained training and Docker build are not claimed tested.

