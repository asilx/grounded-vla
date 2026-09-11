# Architecture and invariants

The prototype separates evidence, task decisions, action proposals, and observed outcomes. These boundaries make the system easy to inspect and give a real policy integration clear responsibilities.

## One execution cycle

1. Receive a timestamped observation and observed events.
2. Retract facts invalidated by events; ingest new measurements with provenance.
3. If enabled, synchronize the resolved snapshot with KnowRob and query through the native mirror.
4. Check an active contract's effects and preconditions. Cancel any remaining commands when a precondition is no longer established.
5. Choose a new contract when needed. Unknown required properties cause inspection.
6. Compile a short instruction and obtain action candidates.
7. Apply mandatory checks, then rank the admissible candidates.
8. Record the decision and its evidence before executing a prefix.
9. Obtain another observation. A missing terminal effect produces a recovery event and a new decision.

The runner bounds total logical ticks, including partial final prefixes. When no candidate is admissible it records abstention and ends the episode. This prototype does not pretend that generic zero joint velocities would be an appropriate physical fallback.

## Evidence semantics

`BeliefStore` keeps an archive of all assertions and an active set for each atom. Evidence has a source, timestamp, optional expiry, confidence, and kind. The acceptance threshold is a configurable gate; the demo's confidence values are not calibrated sensor probabilities.

The validity interval is half-open: evidence at time `t` with TTL `d` can be used when `t <= now < t + d`. Future-dated measurements, expired evidence, predictions, and low-confidence reports do not establish preconditions.

| Active usable evidence | Query answer |
| --- | --- |
| Positive only | `supported` |
| Negative only | `refuted` |
| Both polarities | `unknown`, with both evidence sets retained |
| None | `unknown`, with ignored evidence IDs where relevant |

Explicit replacement means a trusted application update supersedes earlier active evidence. Adding another source without replacement preserves a conflict. The initial conflict scenario deliberately uses this second path. The ledger does not implement a general probabilistic fusion algorithm or dependency-aware confidence calibration.

## Contracts and action spaces

A contract contains a skill, grounded object description, preconditions, expected effects, observation requirements, orientation limits, rule IDs, and the query results used to select it. Fill state remains a monitored precondition, even when it initially permits an unconstrained orientation.

The scripted policy emits `toy_pose` chunks, each `[H,4]` in normalized `x,y,z` and degrees of tilt. `ToyValidator` checks numeric validity, workspace bounds, orientation bounds, contract validity, and the task-specific tilt limit. It does not claim collision checking, inverse kinematics, contact prediction, or physical safety.

The openpi adapter returns an embodiment-specific action space and the declared physical action dimension. `ToyValidator` rejects these actions. A real integration needs its own controller interface, units, transform chain, calibrated perception, kinematic/collision checks, and outcome monitor.

## Knowledge backend

The default ledger is an in-memory reference component, not KnowRob. With `--knowrob-config`, `KnowRobView` synchronizes a snapshot, resolves queried statuses through the native backend, and checks agreement with the ledger's evidence. The executive's actual query path uses that view.

KnowRob stores explicit `hasTruth` edges to `supported`, `refuted`, or `unknown`. It never infers refutation from absence of a supporting edge. The example vocabulary is project-specific. Richer SOMA roles, temporal reasoning, external inference proofs, and NEEM export are extensions rather than completed capabilities.

## Explanation boundary

Explanations are rendered from the recorded decision payload. They describe the controller's object/skill choice, active rules, and candidate rejection. They do not expose hidden model chain-of-thought or explain every coordinate produced by an action expert.

JSON serialization freezes each record before it is appended. A hash chain detects modification, reordering, or removal inside a trace. It is not an authenticity signature: replacing the entire chain or truncating an unanchored suffix can evade that check. Publish or otherwise retain the final hash if an externally anchored audit is required.
