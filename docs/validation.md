# Packaged validation

## Version 0.2 — native π0 integration

Validation date: **18 September 2026**. The current source was exercised locally with Python 3.12.14, PyTorch 2.7.1+cpu, NumPy 1.26.4, Transformers 4.53.2 with official openpi replacements, and openpi revision 215abfb217dbac7d5f1273282331b9b1866c0479.

| Check | Result |
| --- | --- |
| Full automated suite | **92 passed**, including native integration; UserWarning treated as an error |
| Native flow matching | Finite loss, nonzero graph-encoder gradients, adapter optimizer update, frozen base unchanged |
| Native denoising | All-invalid graph exactly matches the unconditioned model; enabled graph can change actions |
| Training resume | Four uninterrupted CPU steps exactly match two steps plus checkpoint/resume, tensor for tensor |
| Artifact inference | Loaded adapter uses the same state/image transforms and action denormalization as training |
| Actual websocket transport | Official openpi server handler and repository client exchange metadata, graph and finite action arrays |
| Dataset controls | Train-only statistics, episode-boundary action masks, duplicate/changed data, future snapshots, invalid cameras/relations/targets rejected |
| Official converter | Pinned conversion module imports; wrapper command-line interface exercised |
| Existing showcase | Default disturbance succeeds; five matched intervention checks pass |
| Code checks | Ruff lint/format, JSON/TOML/YAML parsing and local documentation links pass |
| Clean archive installation | ZIP extracted, wheel built and installed with --no-index into a fresh environment; package resources/CLI entries verified; core demo and five interventions pass without torch |

Native integration tests reduce **constructor dimensions only**: the actual PaliGemma/SigLIP/Gemma modules, official π0 forward pass, expert attention, flow targets, gradient propagation, KV-cache and denoising methods execute. They use locally initialized, untrained small weights, a small test tokenizer and generated test episodes. They are not pretrained policy evaluations, and their losses are not robot-learning results. Production CLI configuration always builds the full π0 architecture; it cannot silently select the test fixture.

Full pretrained checkpoint conversion, full-size GPU training, measured VRAM/latency, real robot or simulator task evaluation, native KnowRob runtime and Docker build were **not** exercised here. A Dockerfile and GitHub Actions job are included, but neither a successful Docker build nor a remote CI run is claimed. π0.5 adapter training is outside this release.

Reproduce the complete suite after the [training environment setup](pi0-training.md):

```bash
OPENPI_SOURCE=../openpi python -m pytest -q -W error::UserWarning
python -m ruff check .
python -m ruff format --check .
```

OPENPI_SOURCE enables import validation of the official conversion script. The training tests skip when optional dependencies are absent, so the core-only test count is not the complete integration result.

## Historical v0.1 showcase validation

The following results were recorded for the original symbolic showcase and isolated adapter example. Retained HTML/JSON artifacts describe that experiment, not pretrained π0 performance.


Validation date: **11 September 2026**. These results come from local execution of this source release. A GitHub Actions workflow is included, but no remote CI run is claimed.

| Check | Result |
| --- | --- |
| Automated tests, including optional learning and observation adapters | 70 passed |
| Ruff lint and formatting | Passed |
| Matched belief/rule interventions | 5 of 5 passed |
| Synthetic event evaluation | 180 started episodes across six scenarios, ten scene seeds and three variants |
| Default disturbance demo | Constrained success; 10 decisions; one missing-effect recovery; 13 rejected candidates |
| Adapter optimization | 150 CPU steps; held-out MSE 0.027705 → 0.000455 on a synthetic regression target |
| Offline trace explorer | Desktop 1320×980 and mobile 390×844; navigation, expandable records and JSON export checked |
| External requests and browser errors | None during the local report checks |
| Clean archive installation | Source ZIP extracted; wheel built and installed in a fresh virtual environment; demo, five interventions and an 18-episode smoke evaluation passed |

The browser used Chromium 152.0.7977.0. Its test harness is a development-time tool and is not a runtime dependency or part of the archive.

### Historical runtime versions

| Component | Version |
| --- | --- |
| Python | 3.12.14 |
| PyTorch | 2.14.0+cpu |
| NumPy | 2.3.5 |
| pytest | 9.1.1 |
| Ruff | 0.16.7 |
| setuptools | 84.0.0 |

The core demo has no third-party runtime dependencies. PyTorch is optional; NumPy and the upstream openpi client are only needed for their respective examples. Declared version ranges allow other environments, but this table identifies the versions actually exercised locally.

The clean-install check ran outside the source tree with an isolated interpreter and imported the installed package from `site-packages`. Wheel installation used `--no-index`, and the report loaded its bundled HTML resource successfully. Building the wheel still requires setuptools; zero runtime dependencies does not mean zero build dependencies.

### Historical reproduction

```bash
python -m pip install -e ".[dev,openpi,learning]"
python -m pytest -q
python -m ruff check .
python -m ruff format --check .
grounded-vla demo
grounded-vla intervene
grounded-vla evaluate --seeds 10
python examples/train_adapter.py --steps 150
```

Generated artifacts include [the demo](demo.html), [evaluation episodes](evaluation-results.json), [intervention records](interventions.json), and [synthetic training history](synthetic-training.json). Regenerate them to inspect a changed implementation rather than treating these snapshots as its test results.

### Historical boundaries

Native KnowRob, π0.5 checkpoint inference, physical simulation, robot hardware, real perception, and a graph adapter inserted into a pretrained VLA were not exercised. KnowRob/openpi tests use explicit transport doubles and observation-contract checks. The local results establish prototype behavior and trainability of the isolated adapter, not VLA performance gains or physical safety.

See [integration steps](integrations.md), [evaluation scope](evaluation.md), and [the roadmap](roadmap.md) for the next implementation and research gates.
