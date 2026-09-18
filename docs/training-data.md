# Demonstration and graph data contract

The pipeline consumes **real recorded episodes**, one NPZ per episode, plus a JSON manifest. It deliberately uses a small explicit interchange format instead of assuming your LeRobot/RLDS data already contains synchronized symbolic state. Export your source dataset into this contract while preserving controller semantics. No LeRobot/RLDS importer or simulator collector is bundled.

## Episode arrays

NPZ loading uses `allow_pickle=False`. `T` is episode length, `S` physical state dimension, `A` physical action dimension, `N` the largest graph in the episode and `F` graph feature width.

| Key | Shape / type | Meaning |
| --- | --- | --- |
| `state` | `[T,S]`, finite numeric | State measured before the corresponding expert action |
| `actions` | `[T,A]`, finite numeric | One expert controller target per frame, not a prebuilt action chunk |
| `timestamps` | `[T]`, increasing seconds | Observation clock; fixed control period within 5% tolerance |
| `graph_timestamps` | `[T]`, finite nonnegative seconds | Snapshot availability time; at most the observation timestamp |
| `prompt` | `[T]`, Unicode strings | Grounded instruction available at that frame |
| `images/base_0_rgb` | `[T,224,224,3]`, uint8 | RGB scene camera |
| `images/left_wrist_0_rgb` | same, optional | RGB wrist camera |
| `images/right_wrist_0_rgb` | same, optional | Optional third RGB camera |
| `image_masks/<camera>` | `[T]`, boolean, optional | Availability; defaults true only for a present camera array |
| `graph_features` | `[T,N,F]`, finite float | Ordered node attributes; zero-pad unused nodes |
| `graph_edges` | `[T,N,N]`, int64 | Directed relation indices; zero for padding/no relation |
| `graph_valid` | `[T,N]`, boolean | True only for present nodes |

At least one camera must be available in every frame. Missing cameras become zero images with false masks. Image conversion is shared between training and inference: uint8 HWC RGB becomes float32 CHW in `[-1,1]`. Resize/crop and RGB conversion belong in your collector and must be consistent online. The native model's training augmentation is disabled to avoid changing visual geometry without updating graph grounding.

Construct state and action vectors explicitly from your controller. A dimension match alone does not establish semantic compatibility. Document units, joint order, rotation representation, absolute/delta commands, gripper sign/range and control period. The pipeline does not guess these conventions or perform implicit delta-to-absolute conversion. The `libero` and `droid` names below specify input key layouts only.

## Manifest and preparation

Copy [the example manifest](../configs/dataset-manifest.example.json) into `data/manifest.json`. Episode paths resolve relative to that file. Declare:

- `format_version: 1`, `state_dim` and `action_dim` in `[1,32]`.
- `input_preset`: `canonical`, `libero` or `droid`.
- `action_convention`: a precise description of your controller's outputs.
- `control_period_seconds`: the fixed demonstration sampling interval.
- `graph_schema`: ID, coordinate frame/normalization, ordered feature names and relation names.
- `episodes`: unique ID, file path and `train`, `val` or `test` split for each episode.

Training and validation episodes are required. Split by episode and preferably collection session, scene and object instance; adjacent frames of a single trajectory are not independent examples. Duplicate file bytes and path aliases are rejected, but the tool cannot detect every semantic near-duplicate. Keep the held-out test set untouched during model selection.

Run `grounded-vla-prepare data/manifest.json --out data/prepared.json`. The prepared manifest stores frame counts and SHA256 checksums. Checksums are verified before training, so replacing an NPZ requires explicit re-preparation and a new experiment. Statistics use training frames only, computed in float64 with merged variance and standard deviation floored at `1e-6`. Validation/test frames cannot alter them.

States and actions use `(x - mean) / std` on physical dimensions, then pad to π0's width 32. For frame `t`, targets are actions `t:t+horizon` within the same episode. The tail is padded to the configured horizon. A boolean `[horizon,32]` mask excludes both padded timesteps and unused motor dimensions from the loss. Inference applies the inverse action transform and returns only the physical dimensions. This is a declared new normalization for adapter training; it does not silently mix a pretrained checkpoint's asset statistics into a different dataset transform.

## Default evidence graph

`grounded-vla-belief-v1` contains 26 features:

1. `x`, `y`, `z`, `visible`, `is_target`, `is_container`.
2. For each predicate `clean`, `filled`, `held`, `at_tray`: `supported`, `refuted`, `unknown`, `confidence`, `age`.

Object IDs are used for joining evidence to tracked entities; IDs are not learned embeddings. Bind graph nodes to actual observed entities with a consistent coordinate frame. Positions are supplied by your tracker, not inferred by this package. Position scaling must be specified in `coordinate_frame` and applied identically in the collector and online encoder. Task role flags must be derived from the current instruction.

The default encoder queries the evidence ledger at observation time. Predicted, future, expired and insufficiently confident facts cannot create supported/refuted indicators. Conflict is unknown. Unknown predicates receive confidence 0 and age 1; otherwise age is the age of the newest usable evidence divided by `max_age` (default 60 seconds), clipped at 1. Both positive and negative observations retain provenance in the separate ledger/journal.

Relation indices are `0 none`, `1 self`, `2 on`, `3 in`, `4 near`, `5 held_by`, `6 target_of`, `7 supports`. An entry `[i,j]` means the declared directed relation from node i to node j. If multiple simultaneous relation types per ordered pair matter, extend the encoder rather than silently overwriting one. Padded relation entries remain zero.

The timestamp check can reject an explicitly future snapshot. It cannot prove that your perception/labeling process avoided future information. Collect graph snapshots causally; do not annotate `held` from the outcome of an action that has not happened yet. If a fact expires before observation, re-query the ledger at observation time. A historical feature matrix is not automatically refreshed just by changing its timestamp.

You can define another schema with a new ID and feature/relation lists, then match `adapter.feature_dim` and `relation_types`. Merely changing the ID string does not transform features. The live server checks the ID and tensor shapes/types; producers are responsible for semantics, units, timestamps and feature order.

## Live payloads

Every trained-server request includes `prompt` and `graph` with `schema_id`, `features: [N,F]`, `edges: int64 [N,N]`, `valid: bool [N]`.

| Preset | State keys | Image keys |
| --- | --- | --- |
| `canonical` | `state: [S]` | `images` dictionary with canonical camera names; optional `image_masks` |
| `libero` | `observation/state: [8]` | `observation/image`, `observation/wrist_image` |
| `droid` | `observation/joint_position: [7]`, `observation/gripper_position: [1]` | `observation/exterior_image_1_left`, `observation/wrist_image_left` |

LIBERO/DROID presets require eight state and seven physical action dimensions and both listed cameras during live inference. They map the two images to base and left-wrist cameras. `canonical` supports other dimensions and available camera combinations. All live images are uint8 HWC RGB at 224×224.

Tokenization uses the local PaliGemma SentencePiece model, official π0 whitespace/underscore cleanup, BOS and separately encoded newline. Prompts over 48 tokens raise an error instead of silently truncating constraints. Make grounded instructions concise upstream.

The recorder is an in-memory episode collector. The dataset loader caches at most two entire decompressed NPZ episodes and indexes their frames. This is useful for a first real adapter experiment; larger corpora should migrate to a sharded, indexed store while retaining this causal contract and normalization identity.
