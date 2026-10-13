"""Atomic adapter artifacts with exact optimizer, data cursor, and RNG resume state."""

import json
import os
import random
import shutil
import tempfile
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file


def rng_state():
    np_state = np.random.get_state()
    return {
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "python": random.getstate(),
        "numpy": (np_state[0], np_state[1].tolist(), *np_state[2:]),
    }


def restore_rng(state):
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"]:
        torch.cuda.set_rng_state_all([x.cpu() for x in state["cuda"]])
    random.setstate(state["python"])
    np_state = state["numpy"]
    np.random.set_state((np_state[0], np.asarray(np_state[1], np.uint32), *np_state[2:]))


def load_action_io(path, model, enabled):
    p = Path(path) / 'action_io.safetensors'
    if enabled:
        if not p.exists():
            raise ValueError('Required action interface checkpoint is missing')
        model.load_action_io_state_dict(load_file(str(p)))
    elif p.exists():
        raise ValueError('Unexpected action interface checkpoint for frozen-interface configuration')


def load_action_expert(path,model,enabled):
    p=Path(path)/'action_expert.safetensors'
    if enabled:
        if not p.exists(): raise ValueError('Required action expert checkpoint is missing')
        model.load_action_expert_state_dict(load_file(str(p)))
    elif p.exists():raise ValueError('Unexpected action expert checkpoint')


def save_checkpoint(root, model, optimizer, step, cursor, metadata):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    destination = root / f"step-{step:08d}"
    if destination.exists():
        raise FileExistsError(destination)
    temp = Path(tempfile.mkdtemp(prefix=".checkpoint-", dir=root))
    try:
        save_file(
            {k: v.detach().cpu().contiguous() for k, v in model.graph_adapter.state_dict().items()},
            str(temp / "adapter.safetensors"),
        )
        if metadata['training_config'].get('train_action_io', False):
            save_file({k:v.detach().cpu().contiguous() for k,v in model.action_io_state_dict().items()},
                      str(temp / 'action_io.safetensors'))
        if metadata['training_config'].get('train_action_expert',False):
            save_file({k:v.detach().cpu().contiguous() for k,v in model.action_expert_state_dict().items()},str(temp/'action_expert.safetensors'))
        (temp / "metadata.json").write_text(
            json.dumps({**metadata, "step": step, "cursor": cursor}, indent=2) + "\n"
        )
        torch.save(
            {"optimizer": optimizer.state_dict(), "rng": rng_state()}, temp / "training_state.pt"
        )
        os.replace(temp, destination)
        pointer = root / ".latest.tmp"
        pointer.write_text(destination.name + "\n")
        os.replace(pointer, root / "latest")
    finally:
        if temp.exists():
            shutil.rmtree(temp)
    return destination


def warm_start_checkpoint(path,model,metadata):
    path=Path(path);previous=json.loads((path/'metadata.json').read_text())
    for key in ['base_checkpoint','tokenizer']:
        if previous['identity'][key]!=metadata['identity'][key]:raise ValueError('Warm-start base identity mismatch')
    for key in ['input_preset','state_dim','action_dim','action_convention','control_period_seconds','graph_schema','action_encoding','action_encoding_horizon','encoding_lower_limits','encoding_upper_limits','encoding_max_joint_step']:
        if previous['dataset'].get(key)!=metadata['dataset'].get(key):raise ValueError('Warm-start dataset semantics mismatch: '+key)
    for key in ['train_action_io','train_action_expert']:
        if previous['training_config'].get(key,False)!=metadata['training_config'].get(key,False):raise ValueError('Warm-start trainable-module mismatch')
    model.graph_adapter.load_state_dict(load_file(str(path/'adapter.safetensors')),strict=True)
    load_action_io(path,model,metadata['training_config'].get('train_action_io',False))
    load_action_expert(path,model,metadata['training_config'].get('train_action_expert',False))
    from grounded_vla.training.runtime import sha256
    metadata['warm_start']={'checkpoint':str(path.resolve()),'previous_identity':previous['identity'],
        'previous_step':previous['step'],'scope':'new-data fine-tuning initialization; optimizer, data cursor and RNG are not resumed',
        'weight_sha256':{p.name:sha256(p) for p in path.glob('*.safetensors')}}
    if previous.get('graph_extension_initialization'):
        metadata['warm_start']['graph_extension_initialization']=previous['graph_extension_initialization']


def resume_checkpoint(path, model, optimizer, metadata):
    path = Path(path)
    previous = json.loads((path / "metadata.json").read_text())
    for key in ("identity", "training_config"):
        expected, actual = metadata[key], previous[key]
        if key == "training_config":
            ignored = {"output_dir", "base_checkpoint", "tokenizer", "dataset"}
            expected = {k: v for k, v in expected.items() if k not in ignored}
            actual = {k: v for k, v in actual.items() if k not in ignored}
            actual.setdefault('train_action_io', False)
            actual.setdefault('train_action_expert',False)
            actual.setdefault('eval_sampling', 'head')
        if actual != expected:
            raise ValueError(f"Resume mismatch: {key}")
    model.graph_adapter.load_state_dict(load_file(str(path / "adapter.safetensors")), strict=True)
    load_action_io(path, model, metadata['training_config'].get('train_action_io', False))
    load_action_expert(path,model,metadata["training_config"].get("train_action_expert",False))
    state = torch.load(path / "training_state.pt", map_location="cpu", weights_only=True)
    optimizer.load_state_dict(state["optimizer"])
    restore_rng(state["rng"])
    if "warm_start" in previous:
        metadata["warm_start"]=previous["warm_start"]
    return previous["step"], previous["cursor"]
