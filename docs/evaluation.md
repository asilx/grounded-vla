# Evaluation scope and reproduction

The included evaluation asks whether particular controller mechanisms behave as intended in a controlled event model. It does not measure VLA generalization, grasp quality, physics fidelity, native KnowRob query performance, or hardware safety.

```bash
grounded-vla evaluate --seeds 10 --out runs/evaluation
grounded-vla intervene --out runs/interventions.json
```

The first command runs six scenarios × ten scene seeds × three controller variants = **180 started episodes**. Seeds alter initial object placement and the second cup's cleanliness. There are no training seeds or learned policy stochasticity in these runs. The action candidate set, observations, task rules, and maximum logical tick budget are shared across variants.

The packaged run is available as [evaluation-results.json](evaluation-results.json), including every episode, and [interventions.json](interventions.json). These are generated outputs from the included implementation. The intentionally diagnostic fixtures make the full controller succeed across all 60 of its episodes; this is not an estimate of real robot success.

## Metrics

Constrained success requires both the final goal and no recorded violation of the active task rules. Task completion alone can therefore be true after a spill while constrained success remains false. Every started episode counts; reaching the budget or abandoning the task is not excluded.

Elapsed logical ticks are reported as simulator steps. They are not wall-clock control latency or robot cycle time. Failure recovery counts a completed chunk whose required effect was not observed. Queue cancellation counts a contract whose precondition changed, expired, or became uncertain before its remaining commands were executed.

The environment's hidden state is consulted only for observations, state transitions, and final scoring. The executive receives the evidence ledger, grounded object descriptions, and task rules. Oracle task success never feeds back into planning.

## Interventions

The five controller-level checks use cloned initial beliefs and the actual executive/validator implementation. They remove clean evidence, refute cleanliness, add an irrelevant fact, remove a cleanliness rule, or change fill state. Each check has a predefined expected contract or selected candidate. Merely changing an arbitrary decision is not counted as success.

These are belief/rule interventions under deterministic candidate generation. They do not establish a physical causal effect, explain the internals of π0.5, or measure human trust. Real VLA experiments need matched random seeds or action-sampling distributions, counterbalanced interventions, and a declared set of acceptable responses.

## Interpreting the ablations

The toy candidate generator intentionally includes a high-score invalid path and a tilted path. Disabling its semantic guard is therefore designed to reveal a task violation. The dirty-contact scenario is designed to expose the difference between stale and retracted knowledge. A high success rate on these fixtures establishes implementation behavior, not a state-of-the-art robotics result.

`no_retraction` can still benefit from later inspections and expiry. It is not a globally memoryless system. `no_semantic_guard` keeps precondition and workspace validation; it removes only the upright-orientation filter. Neither is a reproduction of a published baseline.

## Next research gate

Connect an actual action policy and physics environment, then compare direct VLA execution, a strong hierarchical planner, text/event memory, a simple task graph, and the proposed knowledge layer using the same perceptions and information. Report all started trials, resource budgets, interventions, and uncertainty. The broader study is described in [the proposal](research-proposal.md).
