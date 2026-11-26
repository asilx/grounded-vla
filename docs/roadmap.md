# Roadmap

Version 0.2 implements the native π0 adapter insertion, data/training pipeline,
checkpoints and serving. Items concerning pretrained integration refer to
validation at full scale and π0.5 extensions, not the absence of a π0 code path.

## Priority review — 26 November 2026

The latest main branch already connects oracle Isaac observations to grounded
pick/place execution. Before collecting new performance numbers, the immediate
priority is correct evidence, durable outputs, and reproducible scoring.

| Priority | Gap and consequence | Status / next acceptance gate |
| --- | --- | --- |
| P0 | Inconclusive contact or slip could retain old placement credit; NumPy booleans could crash JSON saving | Fixed and covered by regressions, including successful and interrupted CLI artifact writes |
| P1 | Aggregation trusted stored success flags; saved raw samples had no replay command or integrity check | Implemented optional raw replay, protocol version, digests/counts and scenario/source groups; failed records remain in totals |
| P1 | Clean-checkout tests could not import experiment sources stored only in ZIP | Pytest imports the archived sources. The CPU suite passes 243 tests and skips four that require OpenPI |
| P1 — next | Native contact APIs, scene configuration and compatible trained policy are still unvalidated together | Use the [policy-free scene check](isaac-pick-place.md#check-the-initial-scene) to inspect the initial TCP and contact readings. Calibrate on Isaac GPU, run nominal and displaced-target episodes, and verify the raw records for every attempt |
| P2 | No matched evidence of benefit from the graph/critic on real demonstrations | Train compatible adapters/critic and compare direct VLA, graph-disabled and grounded variants with equal observations, episode budgets and seeds |
| P2 | No current GitHub Actions workflow; existing repo-wide lint debt | Restore automated CPU checks and resolve the pre-existing 85 lint findings / 17 formatting failures without changing archived experiment semantics |

The CPU checks establish control-flow and scoring behavior, not native physics
or pretrained policy performance. Full OpenPI validation, learned perception and
native KnowRob remain separate research gates.

## Research milestones

The offline showcase is complete within its documented scope. The following work would turn it into the full system proposed in the research document.

| Milestone | Implementation | Acceptance criterion |
| --- | --- | --- |
| Native KnowRob validation | Run the mirror with the pinned build, inspect native query streams, add integration tests to that environment | Positive, explicit negative, unknown, expired, and retracted facts agree with the evidence ledger |
| Real VLA simulation | Implement a LIBERO rollout adapter and connect the verified openpi checkpoint | Reproduce a declared baseline with correct camera/state/action transforms and all started episodes counted |
| Grounded perception | Object tracking, ROI/entity bindings, timestamped predicate observations, calibrated confidence | Identity switches and delayed observations are measured; unknown states remain explicit |
| Rich task knowledge | SOMA-aligned roles, task-dependent rules, inference provenance, and domain-specific validity policies | Derived facts have traceable support and are withdrawn when their support becomes invalid |
| Learned graph integration | Train and validate the implemented native π0 adapter on robot demonstrations | Full-size gradient connectivity, knowledge-disabled regression, graph ablations, and held-out task performance are measured |
| Contact-aware validation | Robot kinematics, collision checking, contact/effect models, and hardware-specific fallback | Validation assumptions and false acceptance/rejection rates are documented |
| Evaluation and publication | Matched hierarchical/memory baselines, intervention experiments, multiple training seeds, real-robot trials | Improvements and explanation faithfulness survive the stronger comparisons in the proposal |

The native KnowRob mirror and the openpi probe are separate integration surfaces in this release. A complete closed-loop system must connect both to the same observation and robot-execution stack. Transport tests or a successful model request alone do not establish that milestone.
