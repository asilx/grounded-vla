# Belief-aware execution and task-conditioned temporal monitoring

Implemented on the existing `isaac-trl-tests` branch, based on
`397d50e32ea942fce3f9244dca19bed931dba3b0`. No new remote branch or GitHub push
is required to use this source package.

## Scope

`execution.py` connects evidence/contract semantics to the native canonical
simulation-policy interface. It checks a chunk against its generation-time
dependencies instead of accepting a fresh graph timestamp as proof that an old
chunk is still valid. The existing `run_rollout` stays available unchanged.

`TemporalViTCritic` has an optional learned conditioning path: temporal image
features + contract/state projections + cross-attention to relation-aware graph
tokens. An explicit target mask binds the contract object; arbitrary entity IDs
are not learned embeddings. RGB-only models retain their original forward path.

No simulator or pretrained policy is bundled by this change. CPU behavior,
gradient, data-causality and serialization checks are distinct from Isaac GPU
execution and trained robot-task evaluation. The existing V18 artifacts retain
their original numeric-baseline scope.

## Executable CPU example

```bash
python -m pip install -e '.[dev,simulation]'
python examples/belief_aware_demo.py --out runs/belief-aware-demo
```

The deterministic example observes a displaced target, cancels the old suffix,
replans, and subsequently verifies a simulated contact report. It produces a
hash-chained `execution-trace.jsonl` and `summary.json`. Its policy and environment
are explicit control-flow doubles, not π0 or physical simulation.

## Application interfaces

Construct `BeliefAwareExecutor(view_provider, planner, journal=...)`.

* `view_provider(observation) -> ExecutionView(snapshot, belief)` ingests actual
  observations into a persistent `BeliefStore` and produces the contemporaneous
  policy graph. It may be called again at the same paused tick; ingestion must be
  idempotent for a sensor sample. Never refresh a fact's observation time merely
  because the executive asks for another view.
* `planner(view, observation, event) -> ExecutionPlan | None` runs while paused.
  It can select an `inspect` contract when evidence is missing, a new `pick` after
  a failed grasp, or the next subtask after verified effects. `None` means
  abstention (`reason="no_plan"`); it does not assert whole-task success.
* `on_event(event, observation)` is an optional paused notification callback.
  The planner receives the same cancellation event. Neither callback may advance
  physics or bypass the guarded action interface. Re-observation that needs
  robot motion must be expressed as an explicit inspection contract.

The application supplies sensor-to-predicate extraction, contracts and the task
planner. There is no built-in detector or automatic scene-specific planner.
An existing `Executive.choose()` can be wrapped, but its cups/cleanliness rules
must match the physical scene and its actual sensor evidence.

```python
from grounded_vla.belief import Atom, Truth
from grounded_vla.contracts import Contract, Requirement
from grounded_vla.execution import ExecutionPlan, PositionDependency, RelationDependency

pick_plan = ExecutionPlan(
    contract=Contract(
        "pick", "cargo", "the observed cargo",
        preconditions=(Requirement(Atom("held", "cargo"), Truth.REFUTED),),
        effects=(Requirement(Atom("held", "cargo"), Truth.SUPPORTED),),
    ),
    prompt="Pick up the observed cargo.",
    positions=(PositionDependency("cargo", tolerance=0.02, max_age_seconds=0.5),),
    relations=(RelationDependency("cargo", "goal"),),
    effect_timeout_steps=100,
    max_chunk_age_seconds=0.5,
)
```

The example plan requires both `cargo` and `goal` identities in its graph. Relation
dependencies compare the declared directed edge, including changes to `none`.

### Identity, time, and expected motion

`GraphSnapshot` accepts `entity_ids` and optional `entity_timestamps`, each aligned
with graph rows. Identity stays outside the neural policy wire payload. Node
permutations are resolved by identity and do not cause false cancellations.
Position dependencies interpret the first three features as xyz in the graph's
declared coordinate frame; choose tolerances in that frame's units. The canonical
belief graph uses metres.

When node observation times are supplied, a fresh snapshot cannot hide stale
geometry. Without them, the provider explicitly vouches for the snapshot time
as the observation time of every relevant node. The contract ledger independently
checks fact expiry, contradictions, retraction and confidence at every step.

Declare only the geometry that must remain invariant during the current skill.
For example, `PositionDependency("cargo", relative_to="gripper")` permits a held
object and gripper to move together while detecting relative slip. A placement
plan can additionally watch the destination in world coordinates. These are
declared execution conditions, not explanations of neural attention or proven
causal dependencies inside π0.

Every chunk is bound to its inference-time timestamp, graph digest, evidence IDs,
positions and edge values. Before another action is submitted, the executive checks:

1. Observed completion (before preconditions, because a successful pick changes
   `held=False` to `held=True`).
2. Current preconditions and the per-contract effect deadline.
3. Remaining-chunk age, relevant geometry and directed relations.
4. The optional temporal critic signal.

Joint limits, measured joint-step bounds and the external action guard still run
inside `IsaacSimEnvironment.step`. A critic alarm never inserts a belief fact or
certifies success. Formal collision/contact safety remains outside this module.

### Bounded recovery and scene-preserving resume

An invalidation pauses physics, records the discarded suffix and its reason,
clears the queue, and selects a new plan from freshly ingested evidence.
`max_steps`, `max_replans`, `max_plans`, `effect_timeout_steps`, and
`max_chunk_age_seconds` bound behavior, including planners that make no progress.
Inference errors, invalid inputs and callback failures leave the world paused.

`IsaacSimEnvironment.resume()` submits a measured-position hold through the
independent action guard before restarting the timeline. It neither resets the
scene nor intentionally advances a physics step. The new policy action is still
checked normally. This implementation targets a synchronous step-driven simulator;
it is not an asynchronous real-robot watchdog.

## Isaac CLI

Provide an application module, for example `my_robot.execution`, whose
`build(environment)` function returns the configured `BeliefAwareExecutor`.
Then run in the existing Isaac environment:

```bash
python -m grounded_vla.isaacsim_cli configs/runpod-franka-guarded.json \
  --executor-factory my_robot.execution:build \
  --uri ws://localhost:8000 --no-critic \
  --execute-steps 3 --max-steps 4000 --max-replans 5 --max-plans 50 \
  --record-critic-context --out runs/grounded-execution
```

`my_robot.execution:build` is an application integration point, not a supplied
scene detector. The scene/config and π0 server still need the matching camera,
joint, action, and timing contracts described in the existing Isaac guide.

The run writes:

* `execution-trace.jsonl`: hash-chained selections, proposals, cancellations,
  observed effects, submitted actions, resumption and termination.
* `episode.npz`: successfully executed actions with **pre-action** observations.
* `critic-observations.npz` when requested: observation-only frames, including the
  terminal effect, with causal task/graph context and **no fabricated actions**.
* `summary.json`: counters and termination reason. `completed_contracts` counts
  individually verified contracts, not arbitrary whole-task success.

Paused replanning does not duplicate or retimestamp camera frames. Recordings do
not contain automatically fabricated critic labels.

## Conditioned critic data and training

Use `configs/conditioned-critic-manifest.example.json` (format version 2).
Set state/graph dimensions, graph schema, and sampling period to the actual
recordings. The canonical contract context has **35 features**:

* 5 skill slots (`inspect`, `clean`, `pick`, `place`, `other`).
* For each of `clean`, `filled`, `held`, `at_tray`: supported/refuted/unknown
  preconditions, supported/refuted/unknown effects, and an observation-request bit.
* A tilt-enabled bit and normalized maximum tilt.

Predicates on other objects or outside this vocabulary are rejected by the
encoder; extend/version the schema deliberately for another task domain.
The graph has a separate boolean target mask and retains its relation types.
State is the measured physical state, with the same units as the recording;
this critic path applies no fitted state normalization.

Add independent frame labels to a **copy** of each observation recording:
`critic_labels: int64[T]`, with `progressing=0`, `stalled=1`, `failed=2`,
`completed=3`. Labels refer to the active contract. Preserve recorded context
instead of reconstructing it from eventual outcomes. Use complete, separate
train/validation/test episodes; select monitoring thresholds on validation data.

```bash
grounded-vla-train-critic configs/conditioned-critic-manifest.example.json \
  --out runs/conditioned-critic.pt --window-size 8 --epochs 5 --device cuda
```

The existing training command automatically selects conditioning from the
manifest. Windows never cross episode/split/contract boundaries. Variable graph
sizes are padded by `collate_critic_batch`; the endpoint's pre-observed context
conditions the causal video window. Future/stale graph timestamps and missing
target bindings are rejected. Training must include all four classes.

Online, replace `--no-critic` with `--critic-checkpoint runs/conditioned-critic.pt`
and use the same executor factory. A conditioned checkpoint requires a compatible
context schema, graph schema, camera and sampling period. Task changes reset the
monitor's temporal window and alarm latch. A learned `completed` score never
replaces observed contract-effect verification.

Version-1 RGB-only datasets and checkpoints remain readable. Version-2 conditioned
checkpoints are explicitly identified and cannot silently run without context.
No trained task critic or measured performance improvement is supplied.

## Tests

Pytest imports historical experiment sources directly from the tracked
`experiments.zip` if there is no unpacked `experiments/` directory. No manual
extraction or `/workspace/grounded-vla` path is required. The decoder parity test
uses the checked-in V18 metadata instead of an external training dataset.

```bash
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[dev,simulation]' safetensors sentencepiece
python -m pytest -q
```

Native OpenPI tests require the separately pinned OpenPI environment described
in `docs/pi0-training.md`. CPU checks cover target movement, irrelevant updates,
node permutations, relative-motion invariance, stale/conflicting/retracted facts,
relation changes, missing effects, bounded retries, error pausing, terminal
recording, context gradients, graph permutation invariance, temporal task
boundaries, training, and both checkpoint formats. They do not run Isaac physics
or a pretrained π0 model.
