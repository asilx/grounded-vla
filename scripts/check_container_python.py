"""Build-time CPU checks; no Kit startup, GPU access, network, or model download."""

from __future__ import annotations

import argparse
import importlib.util
import inspect
import json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--torch-version", required=True)
    parser.add_argument("--torchvision-version", required=True)
    parser.add_argument("--isaacsim", action="store_true")
    args = parser.parse_args()

    import numpy as np
    import torch
    import torchvision
    from openpi_client import msgpack_numpy
    from torchvision.models.vision_transformer import VisionTransformer
    from websockets.sync.client import connect

    from grounded_vla.learning.temporal_vit import (
        TemporalViTConfig,
        TemporalViTCritic,
        preprocess_rgb,
    )

    for package, version in ((torch, args.torch_version), (torchvision, args.torchvision_version)):
        if package.__version__.split("+")[0] != version:
            raise RuntimeError(f"Unexpected {package.__name__} version: {package.__version__}")
    if "additional_headers" not in inspect.signature(connect).parameters:
        raise RuntimeError("websockets.sync.client.connect lacks the required header API")
    if args.isaacsim and importlib.util.find_spec("isaacsim") is None:
        raise RuntimeError("Isaac Sim is not on the bundled Python search path")

    # This is the actual codec used by OpenPiTransport, including NumPy 2.x in Kit.
    payload = {
        "state": np.array([0.1, 0.2], np.float32),
        "features": np.ones((1, 3), np.float32),
        "edges": np.ones((1, 1), np.int64),
        "valid": np.ones(1, bool),
        "image": np.zeros((224, 224, 3), np.uint8),
    }
    decoded = msgpack_numpy.unpackb(msgpack_numpy.Packer().pack(payload))
    for key, value in payload.items():
        np.testing.assert_array_equal(decoded[key], value)
        if decoded[key].dtype != value.dtype:
            raise RuntimeError(f"Codec changed dtype for {key}")

    torch.set_num_threads(1)
    backbone = VisionTransformer(
        image_size=224,
        patch_size=112,
        num_layers=1,
        num_heads=2,
        hidden_dim=16,
        mlp_dim=32,
    )
    backbone.heads = torch.nn.Identity()
    model = TemporalViTCritic(
        backbone,
        TemporalViTConfig(feature_dim=16, temporal_dim=16, heads=2, layers=1, max_frames=2),
    ).eval()
    with torch.inference_mode():
        frames = preprocess_rgb(np.stack([payload["image"], payload["image"]]))
        logits = model(frames[None])
    if logits.shape != (1, 4) or not torch.isfinite(logits).all():
        raise RuntimeError("Temporal ViT CPU smoke check failed")
    print(
        json.dumps(
            {
                "torch": torch.__version__,
                "torchvision": torchvision.__version__,
                "numpy": np.__version__,
                "codec_roundtrip": "passed",
                "vit_forward": "passed",
                "native_simulator_started": False,
            }
        )
    )


if __name__ == "__main__":
    main()
