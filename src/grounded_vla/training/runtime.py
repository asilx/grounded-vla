"""Fail closed when the pinned upstream internals or Transformer patch differ."""

import hashlib
import json
from functools import lru_cache
from importlib.resources import files
from pathlib import Path


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


INACTIVE_PI0_KEYS = ("paligemma_with_expert.gemma_expert.lm_head.weight",)


def remove_inactive_pi0_head(model):
    """Remove the unused text-logit head from the pinned action expert.

    Pinned pi0 calls gemma_expert.model, never GemmaForCausalLM.forward;
    continuous actions use action_out_proj. The official JAX checkpoint has
    no weights for this unused language decoder. Removing it fails loudly if
    a future implementation tries to call it, rather than retaining random
    parameters or weakening coverage checks for active weights.
    """
    verify_runtime()
    from torch import nn

    if model.config.pi05:
        raise ValueError("Inactive-head removal is only verified for pinned pi0")
    expert = model.paligemma_with_expert.gemma_expert
    if expert.model.embed_tokens is not None:
        raise ValueError("Unexpected action-expert token embedding")
    head = expert.lm_head
    if head is not None:
        if not isinstance(head, nn.Linear) or head.in_features != model.action_in_proj.out_features:
            raise ValueError("Unexpected action-expert language head")
        expert.lm_head = None
    return INACTIVE_PI0_KEYS


@lru_cache(maxsize=1)
def verify_runtime():
    import openpi
    import transformers
    from transformers.models.siglip import check

    if (
        transformers.__version__ != "4.53.2"
        or not check.check_whether_transformers_replace_is_installed_correctly()
    ):
        raise RuntimeError(
            "Install Transformers 4.53.2 and run scripts/patch_openpi_transformers.py"
        )
    record = json.loads(files("grounded_vla.training").joinpath("upstream.json").read_text())
    root = Path(openpi.__file__).parent
    for name, digest in record["sha256"].items():
        if sha256(root / name) != digest:
            raise RuntimeError(f"Incompatible openpi source: {name}; use {record['revision']}")
