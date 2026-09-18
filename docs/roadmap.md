# Roadmap

Version 0.2 implements the native π0 adapter insertion, data/training pipeline, checkpoints and serving. The immediate next milestone is training on real demonstrations and measuring paired task success and graph interventions. Items below that concern pretrained integration now refer to validation at full scale and π0.5 extensions, not the absence of a π0 code path. from showcase to research system

The offline showcase is complete within its documented scope. The following work would turn it into the full system proposed in the research document.

| Milestone | Implementation | Acceptance criterion |
| --- | --- | --- |
| Native KnowRob validation | Run the mirror with the pinned build, inspect native query streams, add integration tests to that environment | Positive, explicit negative, unknown, expired, and retracted facts agree with the evidence ledger |
| Real VLA simulation | Implement a LIBERO rollout adapter and connect the verified openpi checkpoint | Reproduce a declared baseline with correct camera/state/action transforms and all started episodes counted |
| Grounded perception | Object tracking, ROI/entity bindings, timestamped predicate observations, calibrated confidence | Identity switches and delayed observations are measured; unknown states remain explicit |
| Rich task knowledge | SOMA-aligned roles, task-dependent rules, inference provenance, and domain-specific validity policies | Derived facts have traceable support and are withdrawn when their support becomes invalid |
| Learned graph integration | Insert the PyTorch module into a concrete action-expert forward path and train with robot demonstrations | Gradient connectivity, knowledge-disabled regression, graph ablations, and held-out task performance are measured |
| Contact-aware validation | Robot kinematics, collision checking, contact/effect models, and hardware-specific fallback | Validation assumptions and false acceptance/rejection rates are documented |
| Evaluation and publication | Matched hierarchical/memory baselines, intervention experiments, multiple training seeds, real-robot trials | Improvements and explanation faithfulness survive the stronger comparisons in the proposal |

The native KnowRob mirror and the openpi probe are separate integration surfaces in this release. A complete closed-loop system must connect both to the same observation and robot-execution stack. Transport tests or a successful model request alone do not establish that milestone.
