# Contributing

Use Python 3.11 or newer and install `python -m pip install -e ".[dev,openpi]"`.
Install the `learning` extra for neural-interface work.

The complete CPU suite also needs `torchvision`, `safetensors`, and `sentencepiece`;
see [the CPU setup](docs/belief-aware-execution.md#tests). Pytest imports the
historical experiment modules directly from the tracked `experiments.zip` when
they are not unpacked. A clean checkout requires no manual extraction or
machine-specific paths. OpenPI-dependent tests need the separately pinned runtime.

Run `python -m pytest -q`, `python -m ruff check .`, and `python -m ruff format --check .` before proposing a change.

For a behavior change, add a test that distinguishes correct execution from a plausible failure: stale evidence, a missing effect, a changed rule, an uncertain backend response, or an invalid action. Keep geometry checks separate from task semantics and learned preferences.

When adding a backend, document the upstream revision, observation/action contract, timeout behavior, and validation actually performed. A transport double is useful for unit tests but must not be reported as a real integration run.

Keep all user-facing documentation, source comments, and example outputs in English. Do not commit credentials, model weights, private observations, or generated `runs/` directories.

Report research results with the executor, perception stack, knowledge available to every baseline, seeds, episode budgets, and all failed or abandoned episodes. Synthetic emulator results belong to a separate category from physics simulation and real hardware.
