"""Capture the observation and evidence snapshot before executing each expert action."""

from pathlib import Path

import numpy as np

from grounded_vla.observation import CAMERAS


class EpisodeRecorder:
    def __init__(self, schema_id):
        self.schema_id = schema_id
        self.frames = []

    def append(
        self,
        *,
        state,
        images,
        action,
        prompt,
        graph,
        timestamp,
        graph_timestamp,
        critic_context=None,
    ):
        self._append_observation(
            state=state,
            images=images,
            graph=graph,
            timestamp=timestamp,
            graph_timestamp=graph_timestamp,
            critic_context=critic_context,
            extra={"actions": np.asarray(action, np.float32).copy(), "prompt": str(prompt)},
        )

    def _append_observation(
        self, *, state, images, graph, timestamp, graph_timestamp, critic_context, extra
    ):
        if graph["schema_id"] != self.schema_id or graph_timestamp > timestamp:
            raise ValueError("Graph schema mismatch or future graph snapshot")
        if self.frames and timestamp <= self.frames[-1]["timestamps"]:
            raise ValueError("Timestamps must increase")
        frame = {
            "state": np.asarray(state, np.float32).copy(),
            **extra,
            "timestamps": float(timestamp),
            "graph_timestamps": float(graph_timestamp),
            "graph_features": np.asarray(graph["features"], np.float32).copy(),
            "graph_edges": np.asarray(graph["edges"]).copy(),
            "graph_valid": np.asarray(graph["valid"]).copy(),
        }
        if frame["graph_edges"].dtype != np.int64 or frame["graph_valid"].dtype != np.bool_:
            raise ValueError("Graph edges must be int64 and validity boolean")
        if self.frames and ("critic_context" in self.frames[0]) != (critic_context is not None):
            raise ValueError("An episode must record critic context on every frame or none")
        if critic_context is not None:
            from grounded_vla.critic_context import validate_critic_context

            validate_critic_context(critic_context, timestamp)
            if critic_context["graph_schema_id"] != self.schema_id:
                raise ValueError("Critic graph schema mismatch")
            pairs = (
                ("state", "state"),
                ("features", "graph_features"),
                ("edges", "graph_edges"),
                ("valid", "graph_valid"),
            )
            if any(not np.array_equal(critic_context[a], frame[b]) for a, b in pairs):
                raise ValueError("Critic context must match the recorded pre-action observation")
            if critic_context["graph_timestamp"] != graph_timestamp:
                raise ValueError("Critic graph timestamp differs from the recorded graph")
            frame["critic_context"] = {
                "contract": np.asarray(critic_context["contract"], np.float32).copy(),
                "target_mask": np.asarray(critic_context["target_mask"]).copy(),
                "contract_id": critic_context["contract_id"],
                "schema_id": critic_context["schema_id"],
            }
        for key, image in images.items():
            if key not in CAMERAS or image.shape != (224, 224, 3) or image.dtype != np.uint8:
                raise ValueError("Unknown camera or invalid uint8 RGB image")
            frame[f"images/{key}"] = image.copy()
        self.frames.append(frame)

    def save(self, path):
        path = Path(path)
        if not self.frames:
            raise ValueError("Cannot save an empty episode")
        n = max(len(f["graph_valid"]) for f in self.frames)
        width = self.frames[0]["graph_features"].shape[1]
        data = {
            key: np.asarray([f[key] for f in self.frames])
            for key in ("state", "actions", "prompt", "timestamps", "graph_timestamps")
            if key in self.frames[0]
        }
        data["graph_features"] = np.zeros((len(self.frames), n, width), np.float32)
        data["graph_edges"] = np.zeros((len(self.frames), n, n), np.int64)
        data["graph_valid"] = np.zeros((len(self.frames), n), bool)
        for i, frame in enumerate(self.frames):
            count = len(frame["graph_valid"])
            data["graph_features"][i, :count] = frame["graph_features"]
            data["graph_edges"][i, :count, :count] = frame["graph_edges"]
            data["graph_valid"][i, :count] = frame["graph_valid"]
        if "critic_context" in self.frames[0]:
            contexts = [frame["critic_context"] for frame in self.frames]
            data["critic_context/contract"] = np.stack([c["contract"] for c in contexts])
            data["critic_context/contract_id"] = np.asarray([c["contract_id"] for c in contexts])
            data["critic_context/schema_id"] = np.asarray(contexts[0]["schema_id"])
            data["critic_context/graph_schema_id"] = np.asarray(self.schema_id)
            data["critic_context/target_mask"] = np.zeros((len(contexts), n), bool)
            for i, context in enumerate(contexts):
                mask = context["target_mask"]
                data["critic_context/target_mask"][i, : len(mask)] = mask
        for camera in CAMERAS:
            key = f"images/{camera}"
            if any(key in f for f in self.frames):
                data[key] = np.stack(
                    [f.get(key, np.zeros((224, 224, 3), np.uint8)) for f in self.frames]
                )
                data[f"image_masks/{camera}"] = np.asarray([key in f for f in self.frames], bool)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            np.savez_compressed(stream, **data)


class CriticObservationRecorder(EpisodeRecorder):
    """Record observed frames including terminal effects, without inventing actions.

    Add critic_labels offline from independent annotations before training.
    These files are critic datasets, not policy-demonstration episodes.
    """

    def append(self, *, state, images, graph, timestamp, graph_timestamp, critic_context):
        if critic_context is None:
            raise ValueError("A critic observation requires task context")
        self._append_observation(
            state=state,
            images=images,
            graph=graph,
            timestamp=timestamp,
            graph_timestamp=graph_timestamp,
            critic_context=critic_context,
            extra={},
        )
