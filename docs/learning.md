# Learnable knowledge interface

`GatedKnowledgeAdapter` implements a small trainable interface between a task-relevant graph and action hidden states:

\[
Z = E(G), \qquad H' = H + \tanh(\alpha)\,\operatorname{Attention}(\operatorname{LN}(H), Z, Z).
\]

The default gate is initialized at `1e-3`, keeping the initial residual small while allowing gradients to reach the encoder immediately. A gate initialized at exactly zero would give an exact identity at initialization, but would initially block encoder gradients through that multiplicative path.

## Graph representation

Input node features have shape `[B,N,F]`. They should contain grounded object features, task-role features, and explicitly encoded predicate/temporal information. The prototype does not generate these from images. Relation type indices have shape `[B,N,N]`; index `0` can represent no relation or padding, while the vocabulary is defined by the application. `valid` is a boolean node mask.

A relation embedding supplies a per-head bias to node self-attention. Fixed learned queries pool the variable-size graph into a fixed number of context tokens. There are no object-ID or node-position embeddings. Jointly permuting features, edges, and validity preserves the pooled context, while changing relation types can change it.

Invalid features are removed before projection. All-invalid graph rows and zero-node graphs yield an exact identity adapter output, including after attention biases have changed. Padded edge indices must still be valid vocabulary indices; use zero for them.

The design is a graph-conditioned residual module. It is not a full concept bottleneck: the base visuomotor path can still use raw observations. It is also not an implementation of the complete Knowledge Insulation training recipe.

## Training utilities

`masked_predicate_loss` uses labels `1` for supported, `0` for refuted, and `-1` for unknown. Unknown targets contribute no loss or gradient. A fully unknown batch produces finite zero loss.

`flow_matching_loss` uses target `noise - actions`, matching the reviewed openpi decreasing-time convention. A boolean `[B,H]` mask excludes padding; padded values do not leak into the loss. A real training loop must pair this convention with the matching interpolation and sampler. The helper does not itself sample timesteps or implement an action-expert trainer.

Expected terminal effects should be supervised at the subtask's terminal horizon. Intermediate approach chunks are not failed grasps merely because `held(object)` is not yet true. An effect/risk head, ranking loss, and replay/distillation objective remain research extensions.

## Synthetic optimization example

`examples/train_adapter.py` creates 256 training and 64 held-out samples with a known regression target derived from node features. It trains only the adapter and saves a weights-only state dictionary plus loss history. This is an optimization and serialization example, not a robot-learning result.

The packaged validation ran 150 optimization steps on CPU. The recorded held-out MSE was approximately **0.027705 before training and 0.000455 afterward**. These values concern this synthetic target only. Full local output is in [synthetic-training.json](synthetic-training.json), with the runtime version and training history.

Tests also verify that gradients pass through a frozen downstream linear layer to the graph encoder and gate, while the frozen layer's parameters remain without gradients. Freezing parameters must not be confused with detaching their input activations.

## Native π0 integration in v0.2

The implementation now subclasses the pinned official PI0Pytorch model. It conditions only action tokens returned by embed_suffix; the same insertion participates in native flow matching and all denoising steps. The base remains frozen while gradients reach the adapter through its action expert. State tokens and upstream attention masks retain their original behavior.

The [π0 training guide](pi0-training.md) covers real demonstration preparation, artifact identity, masked physical targets, exact CPU resume, inference and Docker. The [data contract](training-data.md) defines graph features and causal snapshots. π0.5, learned risk heads and joint base-model fine-tuning remain separate work. No trained robot checkpoint is bundled.
