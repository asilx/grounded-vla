"""Capture the observation and evidence snapshot before executing each expert action."""

from pathlib import Path

import numpy as np

from grounded_vla.training.data import CAMERAS


class EpisodeRecorder:
    def __init__(self, schema_id):
        self.schema_id = schema_id
        self.frames = []

    def append(self, *, state, images, action, prompt, graph, timestamp, graph_timestamp):
        if graph["schema_id"] != self.schema_id or graph_timestamp > timestamp:
            raise ValueError("Graph schema mismatch or future graph snapshot")
        if self.frames and timestamp <= self.frames[-1]["timestamps"]:
            raise ValueError("Timestamps must increase")
        frame = {
            "state": np.asarray(state, np.float32).copy(),
            "actions": np.asarray(action, np.float32).copy(),
            "prompt": str(prompt),
            "timestamps": float(timestamp),
            "graph_timestamps": float(graph_timestamp),
            "graph_features": np.asarray(graph["features"], np.float32).copy(),
            "graph_edges": np.asarray(graph["edges"]).copy(),
            "graph_valid": np.asarray(graph["valid"]).copy(),
        }
        if frame["graph_edges"].dtype != np.int64 or frame["graph_valid"].dtype != np.bool_:
            raise ValueError("Graph edges must be int64 and validity boolean")
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
        }
        data["graph_features"] = np.zeros((len(self.frames), n, width), np.float32)
        data["graph_edges"] = np.zeros((len(self.frames), n, n), np.int64)
        data["graph_valid"] = np.zeros((len(self.frames), n), bool)
        for i, frame in enumerate(self.frames):
            count = len(frame["graph_valid"])
            data["graph_features"][i, :count] = frame["graph_features"]
            data["graph_edges"][i, :count, :count] = frame["graph_edges"]
            data["graph_valid"][i, :count] = frame["graph_valid"]
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
