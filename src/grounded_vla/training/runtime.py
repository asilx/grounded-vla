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
