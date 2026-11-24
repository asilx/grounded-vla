# Grounded Isaac pick/place

This is a concrete, single-object application of the belief-aware executor.
It reads **Isaac ground-truth geometry and filtered contact forces**, creates
causal evidence and a policy graph, selects pick/place/inspection contracts,
and evaluates measured release at the goal. It is not RGB perception.

## Modules

| Module | Responsibility |
| --- | --- |
| `perception/isaac_scene.py` | Typed, validated measurements and native Isaac reader |
| `grounding/pick_place.py` | Timestamped evidence, contact/lift/co-motion checks, graph generation |
| `tasks/pick_place.py` | Bounded regrasp selection and measured-position inspection holds |
| `pick_place_cli.py` | Native scene/policy/critic/recording integration |
| `benchmarks/isaac_pick_place.py` | Raw-measurement evaluator and all-attempt result aggregation |

The benchmark lives under `benchmarks` because `grounded_vla.evaluation` already
names the existing symbolic evaluation module.

## Physical evidence and task phases

A grasp requires bilateral finger/object contact above the configured force
threshold, a nonempty measured aperture, proximity to the configured TCP,
lifting above the initial object height, and persistent co-motion in the
gripper frame. A closed-gripper command or a single contact frame is insufficient.
The initial scene must have the object resting on the support, not already held.
Both grounding and evaluation reject an initial sample without measured support
or with an existing bilateral grasp. Inconclusive finger contact and excessive
relative slip revoke placement eligibility, even when historical grasp diagnostics
remain true. A later release requires a new verified acquisition. After a long
sampling gap, the first new contact sample starts a fresh confirmation window.

In the pick phase the world position dependency detects an externally displaced
target. Once fresh measured bilateral contact is established, that dependency
can be exempted via `PositionDependency.ignore_when`. The exemption only accepts
an explicitly supported requirement, is rechecked every step, and is recorded
with evidence IDs. A predicted or unknown contact cannot enable it. This prevents
intentional lifting from repeatedly invalidating a pick before acquisition is
confirmed. Grasp confirmation is a separate, stricter test.

The place subtask requires verified `held` and monitors object/gripper relative
position and the world-frame goal. It completes on observed release or loss.
**That subtask completion is not whole-task success.** Release away from the goal
selects another bounded pick; inconclusive contact selects inspection. Inspection
sends one measured-position hold per observation through the normal joint/action
guard; it does not ask the model to invent an inspection trajectory.

After release near the goal, the controller holds while checking support contact,
low object and goal speed, target XY/Z tolerances, and a continuous settling
interval. Only then does `at_tray` become supported. The raw-measurement evaluator
also requires a verified grasp/lift and stable release, without consulting policy
outputs, critic predictions, or the belief ledger.

Samples must increase in time. Repeated paused queries reuse the same sample
without refreshing TTL or counting new confirmation time. Long sampling gaps
restart confirmation windows. Missing, nonfinite or wrong-shaped native
measurements raise errors and leave execution paused. Unknown cleanliness/fill
properties are not fabricated.

This application targets upright parallel-jaw manipulation of one rigid object.
The executor's relative position dependency uses world-frame displacement;
large wrist rotations are outside this initial task and can trigger replanning.

## Scene configuration

Copy and adjust:

- `configs/isaac-pick-place.example.json`: USD path, robot/cameras, joint ordering,
  joint limits, maximum measured joint step and control period.
- `configs/pick-place.example.json`: exact object/finger/support/goal prim paths,
  TCP offset, measured aperture and force thresholds, tolerances and budgets.

Units are metres, seconds, radians for arm joints, metres for prismatic finger
joints, and newtons for contact forces. The native reader passes physics dt to
the contact API to convert impulses to forces. The stage must be Z-up in metres.
The goal prim origin is the **desired object centre**, not the support surface.

Finger filters must resolve to the actual two contacting rigid finger bodies;
the support filter must resolve to the actual support collider. The reader
rejects an unexpected filter matrix shape. Contact sensing is registered before
the initial World reset using the new optional `scene_setup` hook. No scene reset
takes place during recovery. The default Franka TCP offset is an example that
must match your asset and grasp convention.

For a fresh scene, supply the matching Franka USD asset from your Isaac
installation or asset server:

~~~bash
# Run with Isaac Sim's Python; no policy is needed to author the scene.
python scripts/create_pick_place_scene.py \
  --robot-usd /absolute/path/to/franka.usd \
  --out assets/grounded-pick-place.usda
~~~

This creates a table, cargo, goal frame and top-down camera. It expects the usual
nine-joint Panda naming used by the example config. The robot asset is external;
the script deliberately takes its path instead of downloading vendor assets.
If using an existing scene, configure its exact paths and skip scene creation.

## Policy compatibility

Use the existing pinned OpenPI environment and serving instructions in
[pi0-training.md](pi0-training.md). The actual checkpoint must be trained for
these observations, joint convention, camera layout, control period, graph
features and pick/place prompts. There is no newly trained checkpoint in this
change. In particular, the historical V18 numeric baseline ignores language
and is not a demonstration of this task planner working with a VLA.

The default graph is `grounded-vla-belief-v1`: 26 features and eight relations.
Node order is cargo, goal, gripper; entity identities and observation timestamps
are retained on the execution side. Positions are oracle world coordinates.
`visible` defaults false because the reader does not perform camera visibility
estimation; `clean` and `filled` remain unknown. Collection and training must use
the same semantics. Do not assume matching tensor dimensions imply matching data.

Start your compatible canonical policy server separately:

~~~bash
grounded-vla-serve /absolute/path/to/checkpoint --host 127.0.0.1 --port 8000
~~~

Then run inside the Isaac environment, with the package installed. PyTorch and
NumPy are required for graph encoding; retain Isaac's compatible environment
instead of replacing its bundled GPU libraries with the CPU test wheels.

~~~bash
python examples/isaac_grounded_pick_place.py \
  configs/isaac-pick-place.example.json configs/pick-place.example.json \
  --uri ws://localhost:8000 --no-critic \
  --execute-steps 3 --max-steps 4000 \
  --out runs/pick-place-nominal-0
~~~

The equivalent installed entry point is `grounded-vla-pick-place`.
Add `--action-guard my_module:guard` to connect a task/collision validator.
The existing joint and measured-step checks always remain active.

For a controlled disturbance:

~~~bash
python examples/isaac_grounded_pick_place.py \
  configs/isaac-pick-place.example.json configs/pick-place.example.json \
  --uri ws://localhost:8000 --no-critic \
  --scenario target-shift --seed 0 --perturb-step 10 --shift-metres 0.08 \
  --execute-steps 3 --max-steps 4000 \
  --out runs/pick-place-shift-0
~~~

The disturbance changes the object's X position before the selected environment
step and clears its velocity. That action was inferred before the exogenous
change; the next observation must invalidate a stale suffix. This does not model
zero-latency disturbance detection. Choose a step while approaching the object
and a displacement that remains inside your authored support/workspace.
The result records whether the perturbation actually occurred; an early finish
before the scheduled disturbance is not counted as disturbance-task success.
`seed` chooses disturbance direction; it does not randomize nominal scenes.

## Critic and recordings

Replace `--no-critic` with `--critic-checkpoint path/to/trained.pt` for a compatible
monitor. Task-conditioned checkpoints work with the same canonical contract
context. `grasp_contact` is only an execution-side exemption predicate, not a new
critic input vocabulary. New task training data and labels are still required.

Add `--record-critic-context` to record causal context and terminal observations.
The run writes:

- `result.json`: configuration, source scope, policy metadata, termination,
  attempts/replans, perturbation status, measured score and any runtime error.
  New results include the measurement log's SHA-256 and sample count.
- `scene-measurements.jsonl`: immutable numeric oracle observations and derived
  predicate statuses; useful for independently inspecting the score.
- `execution-trace.jsonl`: selected contracts, dependency exemptions, discarded
  suffixes, evidence IDs and submitted actions.
- `episode.npz`: pre-action images/state/graph paired with submitted actions.
- `critic-observations.npz` when enabled: causal context and terminal frames.

Recordings are explicitly policy rollouts mixed with inspection holds, not
expert demonstrations. Critic labels are not fabricated.

Both simulator startup failures and execution failures write a failed
`result.json` once the run output directory has been created. Provide every
started run to the aggregator, including retries and failures:

~~~bash
python -m grounded_vla.benchmarks.isaac_pick_place \
  runs/pick-place-nominal-0/result.json runs/pick-place-shift-0/result.json \
  --verify-measurements \
  --out runs/pick-place-comparison.json
~~~

With `--verify-measurements`, the evaluator reconstructs typed samples from each
sibling `scene-measurements.jsonl`, using the saved `task_config`. It ignores
recorded predicates, critic scores and claimed physical outcomes. The replay needs
only NumPy (`pip install -e '.[simulation]'`), without PyTorch, Isaac or a policy
server. Nonfinite, malformed, out-of-order, repeated and mixed-source observations
are rejected, with a file and line number. Saved digests/counts are checked when
present. Older records without digests can still be replayed. Digests detect
changed bytes; they do not authenticate the sensor or attest to real physics.

Verified success requires the reported run to have succeeded without a runtime
error, independent raw-measurement success, and an actually applied disturbance
for `target-shift`. Missing or corrupt recordings remain failed attempts in the
denominator with a `verification_error`. Replaying valid measurements does not
turn a failed runtime into a successful run. Without the flag, `score_mode` is
`reported` and the aggregator trusts stored outcomes. Revised raw scoring is
explicitly identified as `grounded-vla-pick-place-measurements-v2`.

The report includes overall counts and separate groups by scenario and measurement
source. Startup failures with no observations have an unknown source; retain that
group and the overall denominator when reporting results. The aggregator cannot
discover attempts omitted by the caller or deduplicate copied result files under
new names. These are oracle-control metrics, not learned-perception or real-robot
generalization measurements. CPU fixture records retain their own source label.

## CPU checks and native validation boundary

~~~bash
# Separate CPU development environment (not Isaac's bundled environment).
python -m pip install -e '.[dev,learning,simulation]' safetensors sentencepiece
python -m pytest -q tests/test_pick_place.py tests/test_belief_executor.py \
  tests/test_isaacsim.py tests/test_conditioned_critic.py
~~~

Tests compose the real grounder, planner, executor, canonical policy validation,
inspection routing and independent evaluator with explicit measured-scene
fixtures. They cover closed-empty and unilateral grasps, lift/co-motion,
slip/drop, stable release, moving goals, stale/duplicate samples, bounded retries,
target-shift suffix cancellation, recording, and failure accounting.
Fixture trajectories are not physics or pretrained policy results.

Local validation on 2026-11-15: **213 passed, 4 skipped** in the complete CPU
suite (the four skips require the separate OpenPI runtime). This includes 31 new
pick/place tests. CLI help, missing-Isaac startup failure persistence and failed-run
aggregation were also exercised. Ruff checks cover the changed Python files.

The 2026-11-24 follow-up passed **237 tests, with four OpenPI-dependent skips**.
It adds contact-continuity, initial-state and provenance regressions; raw replay
and artifact-integrity checks; and successful/interrupted CLI runs using explicit
CPU environment/transport doubles. The CLI checks exercise actual artifact writes
and re-scoring, including the NumPy boolean serialization regression. See
[validation details](validation.md) for the remaining lint and runtime boundaries.

Native API choices were checked against the
[Isaac Sim 5.1 core prim API](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/py/source/extensions/isaacsim.core.prims/docs/index.html).
**The new native scene/contact path, scene-authoring script and trained-policy
closed loop have not been executed on an Isaac GPU in this change.** Isaac 6
compatibility also needs an actual runtime check. Contact thresholds, TCP offset,
scene collisions and policy performance require calibration and measurement.
Joint guards and these task checks are not a formal collision or safety proof.
