"""Build a clean, deterministic source ZIP with one top-level repository folder."""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

EXCLUDED_DIRS = {
    ".git",
    ".venv",
    ".training-venv",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".cache",
    "runs",
    "build",
    "dist",
}
EXCLUDED_SUFFIXES = {".pyc", ".pyo", ".pt", ".pth", ".npy", ".npz", ".zip", ".safetensors"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("../grounded-vla.zip"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.out.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(root.rglob("*")):
            relative = path.relative_to(root)
            if relative.parts[0] in {"models", "data"}:
                continue
            if not path.is_file() or path.is_symlink() or path == output:
                continue
            if any(p in EXCLUDED_DIRS or p.endswith(".egg-info") for p in relative.parts):
                continue
            if (
                path.suffix in EXCLUDED_SUFFIXES
                or path.name == ".DS_Store"
                or path.name.startswith(".env")
            ):
                continue
            member = zipfile.ZipInfo(
                "grounded-vla/" + relative.as_posix(), date_time=(2026, 9, 18, 0, 0, 0)
            )
            member.compress_type = zipfile.ZIP_DEFLATED
            member.external_attr = 0o100644 << 16
            archive.writestr(member, path.read_bytes())
            count += 1
    print(f"Packaged {count} files into {output} ({output.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
