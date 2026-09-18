"""Convert official π0 Orbax weights with pinned upstream mapping and strict coverage checks."""

import argparse
import importlib.util
import json
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--openpi-dir", required=True)
    parser.add_argument(
        "--source", required=True, help="π0 checkpoint directory, containing params/, or gs:// URL"
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--precision", choices=("float32", "bfloat16"), default="float32")
    args = parser.parse_args()
    import numpy as np
    import torch
    from openpi.models import gemma
    from openpi.models.pi0_config import Pi0Config
    from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
    from openpi.shared.download import maybe_download
    from safetensors.torch import save_model

    from grounded_vla.training.config import OPENPI_REVISION
    from grounded_vla.training.runtime import sha256, verify_runtime

    verify_runtime()
    output = Path(args.out)
    if output.exists():
        raise FileExistsError(output)
    module_path = Path(args.openpi_dir) / "examples/convert_jax_model_to_pytorch.py"
    record = json.loads(
        (Path(__file__).parents[1] / "src/grounded_vla/training/upstream.json").read_text()
    )
    if sha256(module_path) != record["converter_sha256"]:
        raise ValueError("Converter source differs from the pinned openpi revision")
    spec = importlib.util.spec_from_file_location("official_pi0_conversion", module_path)
    converter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(converter)
    source = Path(maybe_download(args.source))
    params = converter.slice_initial_orbax_checkpoint(str(source), restore_precision="float32")
    if "state_proj" not in params["projection_params"]:
        raise ValueError("Expected π0 continuous state projection; π0.5 is unsupported here")
    model = PI0Pytorch(Pi0Config(dtype=args.precision, pi05=False, pytorch_compile_mode=None))
    paligemma, expert = converter.slice_paligemma_state_dict(
        params["paligemma_params"], model.paligemma_with_expert.paligemma.config
    )
    expert = converter.slice_gemma_state_dict(
        expert, gemma.get_config("gemma_300m"), num_expert=1, checkpoint_dir=str(source), pi05=False
    )
    projections = {}
    for name in (
        "state_proj",
        "action_in_proj",
        "action_out_proj",
        "action_time_mlp_in",
        "action_time_mlp_out",
    ):
        values = params["projection_params"][name]
        for key, source_key in (("weight", "kernel"), ("bias", "bias")):
            value = values[source_key]
            if isinstance(value, dict):
                value = value["value"]
            tensor = torch.from_numpy(np.asarray(value).copy())
            projections[f"{name}.{key}"] = tensor.T if key == "weight" else tensor
    all_params = {**paligemma, **expert, **projections}
    result = model.load_state_dict(all_params, strict=False)
    # Missing tied aliases are acceptable only when that same storage was loaded.
    state = model.state_dict()
    loaded_storage = {state[k].untyped_storage().data_ptr() for k in all_params if k in state}
    missing = [
        k
        for k in result.missing_keys
        if state[k].untyped_storage().data_ptr() not in loaded_storage
    ]
    if missing or result.unexpected_keys:
        raise ValueError(f"Incomplete checkpoint conversion: {missing}, {result.unexpected_keys}")
    output.mkdir(parents=True)
    save_model(model, str(output / "model.safetensors"))
    if (source / "assets").exists():
        shutil.copytree(source / "assets", output / "assets")
    (output / "conversion.json").write_text(
        json.dumps(
            {
                "source": args.source,
                "openpi_revision": OPENPI_REVISION,
                "precision": args.precision,
                "pi05": False,
            },
            indent=2,
        )
        + "\n"
    )
    print(output / "model.safetensors")


if __name__ == "__main__":
    main()
