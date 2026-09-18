# External runtime integrations

**v0.2:** native π0 graph-adapter training and serving are implemented. Start with [the training guide](pi0-training.md). The external π0.5 and KnowRob interfaces below retain their stated scope.

The offline demo and adapter tests run without KnowRob or model weights. Native KnowRob and an actual π0.5 server were not run during the packaged validation. The integrations below contain implementation code and explicit contracts; their remaining deployment steps are identified here.

## Reviewed upstream revisions

| Project | Revision inspected on 11 September 2026 |
| --- | --- |
| KnowRob `dev` | `d03b865927d8634cef30e2e4ef9b60e948777b35` |
| openpi `main` | `215abfb217dbac7d5f1273282331b9b1866c0479` |

Pin these revisions for the first integration attempt, or explicitly revalidate newer revisions. Dependency ranges in `pyproject.toml` are compatibility ranges; they are not a complete external-runtime lockfile.

## KnowRob

Build KnowRob using its [installation instructions](https://github.com/knowrob/knowrob/blob/d03b865927d8634cef30e2e4ef9b60e948777b35/README.md). Its Python extension must be importable from the interpreter running Grounded VLA. KnowRob is not installed by a `pip install grounded-vla` command.

```bash
python -c "import knowrob; print(knowrob.__file__)"
grounded-vla demo --knowrob-config knowledge/knowrob.json --backend-timeout 20
```

The implementation uses `KnowledgeBase`, `TripleCopy`, `insertOne`, `removeOne`, `QueryParser.parse`, and streamed query answers. The inspected [Python API tests](https://github.com/knowrob/knowrob/blob/d03b865927d8634cef30e2e4ef9b60e948777b35/tests/py/test_boost_python.py) demonstrate the relevant calls. Answer polarity and uncertainty are defined in the upstream [Answer interface](https://github.com/knowrob/knowrob/blob/d03b865927d8634cef30e2e4ef9b60e948777b35/include/knowrob/queries/Answer.h).

A worker process owns the native knowledge base. The parent bounds startup and every response wait, and terminates the worker after a timeout. Native errors are raised as backend errors; they never become `refuted` or silently fall back to the in-memory engine.

Each fresh ledger snapshot becomes explicit RDF truth-status edges. Superseded edges are removed before replacement. Use the supplied dedicated Prolog configuration; do not point this prototype at a shared production knowledge base. Both evidence polarities are queried explicitly, with a negative membership query never serving as positive evidence for its opposite.

The current mirror validates agreement with the local ledger. It is a verified interface boundary in unit tests, not a general importer of arbitrary KnowRob proofs. To add derived domain relations, import their supporting records into the decision journal and define their freshness/invalidation policy. The [Python integration documentation](https://github.com/knowrob/knowrob/blob/d03b865927d8634cef30e2e4ef9b60e948777b35/src/integration/python/README.md) is the reference for extensions.

The bundled native test doubles check synchronization, status mapping, failure propagation, and participation in task execution. They do not verify the C++ extension, Prolog setup, RDF loading, or native query completion. Run those checks in the target KnowRob environment before claiming integration success.

## π0.5 through openpi

Clone the upstream repository and use its own model-serving environment:

```bash
git clone https://github.com/Physical-Intelligence/openpi.git
cd openpi
git checkout 215abfb217dbac7d5f1273282331b9b1866c0479
```

Follow the repository's [setup and checkpoint instructions](https://github.com/Physical-Intelligence/openpi/tree/215abfb217dbac7d5f1273282331b9b1866c0479). For the named π0.5 LIBERO checkpoint, the intended explicit server command is:

```bash
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi05_libero \
  --policy.dir=gs://openpi-assets/checkpoints/pi05_libero
```

This example requires the upstream dependencies, model download, and suitable hardware. It has not been executed as part of the local showcase. The client/server separation follows the upstream [remote inference guide](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/docs/remote_inference.md).

In the Grounded VLA environment, install `openpi-client` from that checkout. For example, if both repositories are sibling directories:

```bash
python -m pip install -e ../openpi/packages/openpi-client
python -m pip install -e ".[openpi]"
```

`OpenPiTransport` uses the upstream MessagePack codec and a binary websocket connection. Startup, metadata reception, and response waits are bounded. An error closes the session so a late response cannot be mistaken for a later request. `OPENPI_API_KEY` is read by the probe example only when supplied in the environment.

### Observation contract

The current presets match the upstream [simple client examples](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/examples/simple_client/main.py):

| Preset | NPZ key | Required shape and dtype |
| --- | --- | --- |
| LIBERO | `observation/image` | `[224,224,3]`, RGB `uint8` |
| LIBERO | `observation/wrist_image` | `[224,224,3]`, RGB `uint8` |
| LIBERO | `observation/state` | `[8]`, finite numeric |
| DROID | `observation/exterior_image_1_left` | `[224,224,3]`, RGB `uint8` |
| DROID | `observation/wrist_image_left` | `[224,224,3]`, RGB `uint8` |
| DROID | `observation/joint_position` | `[7]`, finite numeric |
| DROID | `observation/gripper_position` | `[1]`, finite numeric |

Prepare a real captured observation using the camera ordering, image transforms, state conventions, and normalization expected by the selected checkpoint. The adapter checks shape and numeric validity; it cannot verify RGB semantics, units, synchronization, or physical grounding from the array alone.

Run `examples/openpi_probe.py` as shown in the README. The declared `--action-dim` is the expected physical output dimension after upstream output transforms. A mismatched 32-dimensional padded output is rejected rather than sliced. No exact flow-policy likelihood or candidate quality score is inferred: returned candidates have a neutral score and require an external selector.

### Token budget

The instruction compiler uses a conservative character cap, not a token-count guarantee. The reviewed π0.5 input pathway appends discretized state after task text. To enforce an exact budget, pass `serialized_input_guard` to `OpenPiPolicy`; it must reproduce the selected server tokenizer and validate the complete serialized input before inference. See the upstream [tokenizer source](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/models/tokenizer.py).

### Remaining robot integration

The supplied runner is an event emulator. Connecting real openpi actions requires replacing that environment with a robot-specific interface, providing fresh visual grounding, defining motion validation for the correct action space, and observing effects. A policy server connection by itself does not implement that execution loop.

ROS 2 integration, LIBERO/RoboCasa rollout adapters, hardware drivers, calibrated perception, contact-aware risk prediction, and direct π0.5 action-expert modification are not bundled. These are the next concrete implementation tasks rather than hidden dependencies of the offline demo.
