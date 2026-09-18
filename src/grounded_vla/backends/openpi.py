"""Bounded openpi websocket transport and explicit LIBERO/DROID input contracts.

Uses the upstream openpi-client MessagePack codec. No model weights are bundled.
The returned actions are proposals: a robot-specific validator must check them.
"""

from __future__ import annotations

import math
from collections.abc import Callable

from grounded_vla.contracts import Contract, compile_instruction
from grounded_vla.policy import ActionChunk


class PolicyBackendError(RuntimeError):
    pass


class OpenPiTransport:
    """Upstream wire format with bounded connection and receive times.

    Unlike the upstream convenience client's connection retry loop, a failed
    connection raises. The executive can then explicitly choose a fallback.
    """

    def __init__(
        self, uri: str = "ws://localhost:8000", *, timeout: float = 10.0, api_key: str | None = None
    ) -> None:
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("Timeout must be finite and positive")
        if not uri.startswith(("ws://", "wss://")):
            raise ValueError("Use an explicit ws:// or wss:// URI")
        try:
            from openpi_client import msgpack_numpy
            from websockets.sync.client import connect
        except ImportError as exc:
            raise PolicyBackendError(
                "Install packages/openpi-client from the pinned openpi checkout"
            ) from exc
        self.timeout = timeout
        self._codec = msgpack_numpy
        self._packer = msgpack_numpy.Packer()
        self._connection = None
        try:
            headers = {"Authorization": f"Api-Key {api_key}"} if api_key else None
            self._connection = connect(
                uri,
                open_timeout=timeout,
                close_timeout=timeout,
                compression=None,
                max_size=64 * 1024 * 1024,
                additional_headers=headers,
            )
            self.metadata = self._decode(self._connection.recv(timeout=timeout))
        except Exception as exc:
            self.close()
            raise PolicyBackendError("Could not establish the openpi session") from exc

    def _decode(self, payload) -> dict:
        if not isinstance(payload, bytes):
            raise PolicyBackendError("The policy server returned an error or a non-binary response")
        decoded = self._codec.unpackb(payload)
        if not isinstance(decoded, dict):
            raise PolicyBackendError("Expected a mapping from the policy server")
        return decoded

    def infer(self, observation: dict) -> dict:
        if self._connection is None:
            raise PolicyBackendError("Policy session is closed")
        try:
            self._connection.send(self._packer.pack(observation))
            return self._decode(self._connection.recv(timeout=self.timeout))
        except Exception as exc:
            self.close()  # An uncertain response must not be reused by the next request.
            raise PolicyBackendError("Policy inference failed; session closed") from exc

    def close(self) -> None:
        if self._connection is not None:
            connection, self._connection = self._connection, None
            connection.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class OpenPiPolicy:
    def __init__(
        self,
        client,
        *,
        embodiment: str,
        action_dim: int,
        max_horizon: int = 100,
        serialized_input_guard: Callable[[dict], None] | None = None,
    ):
        if embodiment not in {"libero", "droid"} or action_dim <= 0 or max_horizon <= 0:
            raise ValueError("Specify libero/droid and positive robot action dimensions/horizon")
        self.client, self.embodiment, self.action_dim = client, embodiment, action_dim
        self.max_horizon = max_horizon
        self.serialized_input_guard = serialized_input_guard
        metadata = getattr(client, "metadata", {})
        if metadata.get("graph_required"):
            if (
                metadata.get("input_preset") != embodiment
                or metadata.get("action_dim") != action_dim
            ):
                raise ValueError(
                    "The trained server embodiment/action dimension differs from this client"
                )

    def sample(self, contract: Contract, observation: dict, count: int = 1) -> list[ActionChunk]:
        import numpy as np

        if not 1 <= count <= 8:
            raise ValueError("Candidate count must be between 1 and 8")
        payload = dict(observation["model_inputs"])
        payload["prompt"] = compile_instruction(contract)
        metadata = getattr(self.client, "metadata", {})
        if metadata.get("graph_required"):
            graph = observation.get("graph")
            if graph is None or graph.get("schema_id") != metadata.get("graph_schema_id"):
                raise ValueError("The trained server requires the matching graph snapshot")
            payload["graph"] = graph
        if self.embodiment == "libero":
            images = ["observation/image", "observation/wrist_image"]
            states = {"observation/state": (8,)}
        else:
            images = ["observation/exterior_image_1_left", "observation/wrist_image_left"]
            states = {"observation/joint_position": (7,), "observation/gripper_position": (1,)}
        for key in images:
            value = np.asarray(payload[key])
            if value.shape != (224, 224, 3) or value.dtype != np.uint8:
                raise ValueError(f"{key} must be uint8 HWC RGB, shape (224, 224, 3)")
            payload[key] = value
        for key, shape in states.items():
            value = np.asarray(payload[key], dtype=np.float32)
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError(f"{key} must be finite with shape {shape}")
            payload[key] = value
        if self.serialized_input_guard:
            self.serialized_input_guard(payload)
        chunks = []
        for index in range(count):
            response = self.client.infer(payload)
            values = np.asarray(response["actions"], dtype=np.float32)
            if (
                values.ndim != 2
                or values.shape[1] != self.action_dim
                or not (1 <= values.shape[0] <= self.max_horizon)
                or not np.isfinite(values).all()
            ):
                raise PolicyBackendError("Invalid action shape, dimension, horizon, or values")
            chunks.append(
                ActionChunk(
                    f"openpi-{index}",
                    tuple(tuple(map(float, row)) for row in values),
                    0.0,
                    f"{self.embodiment}_action",
                    "external-openpi-server",
                )
            )
        return chunks
