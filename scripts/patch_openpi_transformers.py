"""Apply the official openpi Transformers replacements only inside an isolated venv."""

import shutil
import sys
from pathlib import Path

import openpi
import transformers

if sys.prefix == sys.base_prefix or transformers.__version__ != "4.53.2":
    raise SystemExit("Activate an isolated venv containing transformers==4.53.2 first")
source = Path(openpi.__file__).parent / "models_pytorch/transformers_replace"
target = Path(transformers.__file__).parent
if not target.is_relative_to(Path(sys.prefix)):
    raise SystemExit("Refusing to patch a Transformers installation outside this venv")
shutil.copytree(source, target, dirs_exist_ok=True)
print("Installed official openpi Transformers replacements in", target)
