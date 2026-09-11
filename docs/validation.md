# Packaged validation

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

## Runtime versions

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

## Reproduction

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

## Boundaries

Native KnowRob, π0.5 checkpoint inference, physical simulation, robot hardware, real perception, and a graph adapter inserted into a pretrained VLA were not exercised. KnowRob/openpi tests use explicit transport doubles and observation-contract checks. The local results establish prototype behavior and trainability of the isolated adapter, not VLA performance gains or physical safety.

See [integration steps](integrations.md), [evaluation scope](evaluation.md), and [the roadmap](roadmap.md) for the next implementation and research gates.
