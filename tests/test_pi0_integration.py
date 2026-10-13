# ruff: noqa: E402
import asyncio
import dataclasses
import importlib.util
import json
import threading
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("openpi")
np = pytest.importorskip("numpy")
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
from safetensors.torch import load_file

from grounded_vla.training.data import EpisodeDataset, collate
from grounded_vla.training.inference import TrainedPi0Policy
from grounded_vla.training.tokenizer import Pi0Tokenizer
from grounded_vla.training.train import run


def batch(config):
    ds = EpisodeDataset(
        config.dataset, "train", config.action_horizon, Pi0Tokenizer(config.tokenizer)
    )
    return collate([ds[0], ds[2]])


def test_native_flow_backprop_only_updates_adapter(native_tiny, training_config):
    model = native_tiny(training_config)
    model.load_base(training_config.base_checkpoint)
    model.train()
    model.gradient_checkpointing_enable()
    observation, graph, actions, mask = batch(training_config)
    base = model.action_in_proj.weight.detach().clone()
    before = model.graph_adapter.gate.detach().clone()
    loss = model.flow_loss(observation, actions, graph, mask)
    loss.backward()
    assert torch.isfinite(loss)
    assert model.graph_adapter.encoder.input.weight.grad.abs().sum() > 0
    assert all(
        p.grad is None for n, p in model.named_parameters() if not n.startswith("graph_adapter.")
    )
    torch.optim.AdamW(model.graph_adapter.parameters(), lr=0.01).step()
    assert not torch.equal(before, model.graph_adapter.gate)
    torch.testing.assert_close(base, model.action_in_proj.weight, rtol=0, atol=0)
    assert model._graph is None


def test_native_denoising_identity_and_graph_effect(native_tiny, training_config):
    model = native_tiny(training_config).eval()
    model.load_base(training_config.base_checkpoint)
    obs, graph, actions, _ = batch(training_config)
    noise = torch.randn_like(actions)
    native = PI0Pytorch.sample_actions(model, torch.device("cpu"), obs, noise=noise, num_steps=2)
    disabled = model.grounded_actions(
        torch.device("cpu"), obs, graph.disabled(), noise=noise, num_steps=2
    )
    torch.testing.assert_close(native, disabled, rtol=0, atol=0)
    with torch.no_grad():
        model.graph_adapter.gate.fill_(0.8)
    conditioned = model.grounded_actions(torch.device("cpu"), obs, graph, noise=noise, num_steps=2)
    assert not torch.allclose(conditioned, native)
    assert conditioned.shape == actions.shape
    with pytest.raises(RuntimeError), model.graph_context(graph):
        with model.graph_context(graph):
            pass
    assert model._graph is None


def test_training_resume_is_exact_and_refuses_mismatched_schedule(training_config, tmp_path):
    full = run(training_config)
    interrupted = dataclasses.replace(training_config, output_dir=str(tmp_path / "resume"))
    partial = run(interrupted, stop_after=2)
    resumed = run(interrupted, resume=partial)
    uninterrupted_weights = load_file(str(full / "adapter.safetensors"))
    resumed_weights = load_file(str(resumed / "adapter.safetensors"))
    for key in uninterrupted_weights:
        torch.testing.assert_close(uninterrupted_weights[key], resumed_weights[key], rtol=0, atol=0)
    with pytest.raises(ValueError, match="newer"):
        run(interrupted, resume=partial)
    altered = dataclasses.replace(training_config, output_dir=str(tmp_path / "bad"), steps=5)
    with pytest.raises(ValueError, match="Resume mismatch"):
        run(altered, resume=partial)


def test_artifact_inference_and_actual_websocket(training_config):
    checkpoint = run(training_config, stop_after=1)
    policy = TrainedPi0Policy(checkpoint, device="cpu", num_steps=2)
    metadata = json.loads(Path(training_config.dataset).read_text())
    with np.load(Path(training_config.dataset).parent / metadata["episodes"][0]["path"]) as z:
        payload = {
            "observation/state": z["state"][0],
            "observation/image": z["images/base_0_rgb"][0],
            "observation/wrist_image": z["images/left_wrist_0_rgb"][0],
            "prompt": str(z["prompt"][0]),
            "graph": {
                "schema_id": metadata["graph_schema"]["id"],
                **{k: z[f"graph_{k}"][0] for k in ("features", "edges", "valid")},
            },
        }
    obs, graph = policy.transform(payload)
    ds = EpisodeDataset(
        training_config.dataset, "train", 3, Pi0Tokenizer(training_config.tokenizer)
    )
    expected_obs, _, _, _ = collate([ds[0]])
    torch.testing.assert_close(obs.state, expected_obs.state, rtol=0, atol=0)
    for key in obs.images:
        torch.testing.assert_close(obs.images[key], expected_obs.images[key], rtol=0, atol=0)
    noise = torch.randn(1, 3, 32)
    normalized = policy.model.grounded_actions(
        torch.device("cpu"), obs, graph, noise=noise, num_steps=2
    )[0, :, :7].numpy()
    stats = metadata["normalization"]["action"]
    np.testing.assert_allclose(
        policy.infer(payload, noise=noise)["actions"],
        normalized * np.asarray(stats["std"], np.float32) + np.asarray(stats["mean"], np.float32),
    )
    deterministic = TrainedPi0Policy(checkpoint, device='cpu', num_steps=2, zero_noise=True)
    np.testing.assert_array_equal(
        deterministic.infer(payload)['actions'],
        policy.infer(payload,noise=torch.zeros(1,3,32))['actions'])
    from openpi.serving.websocket_policy_server import WebsocketPolicyServer
    from websockets.asyncio.server import serve

    from grounded_vla.backends.openpi import OpenPiTransport

    server = WebsocketPolicyServer(policy, metadata=policy.metadata)
    ready, shutdown = threading.Event(), threading.Event()
    address = []
    errors = []

    async def start():
        async with serve(server._handler, "127.0.0.1", 0, compression=None) as transport:
            address.append(transport.sockets[0].getsockname()[1])
            ready.set()
            while not shutdown.is_set():
                await asyncio.sleep(0.01)

    def target():
        try:
            asyncio.run(start())
        except Exception as exc:
            errors.append(exc)
            ready.set()

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    try:
        assert ready.wait(10)
        assert not errors
        with OpenPiTransport(f"ws://127.0.0.1:{address[0]}") as client:
            assert client.metadata["graph_required"]
            response = client.infer(payload)
            assert response["actions"].shape == (3, 7)
            assert np.isfinite(response["actions"]).all()
    finally:
        shutdown.set()
        thread.join(timeout=10)
    assert not thread.is_alive() and not errors
    payload["graph"]["schema_id"] = "wrong"
    with pytest.raises(ValueError, match="schema"):
        policy.infer(payload)


def test_masked_targets_do_not_contaminate_loss(native_tiny, training_config):
    model = native_tiny(training_config)
    obs, graph, actions, mask = batch(training_config)
    actions[~mask] = float("nan")
    assert torch.isfinite(model.flow_loss(obs, actions, graph, mask))


def test_pinned_converter_imports():
    import os

    source = os.environ.get("OPENPI_SOURCE")
    if source is None:
        pytest.skip("Set OPENPI_SOURCE to exercise the official conversion module import")
    path = Path(source) / "examples/convert_jax_model_to_pytorch.py"
    spec = importlib.util.spec_from_file_location("official_conversion_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert callable(module.slice_initial_orbax_checkpoint)


def test_inactive_text_head_is_removed_but_active_missing_weights_still_fail(native_tiny, training_config, tmp_path):
    from safetensors.torch import save_file

    model = native_tiny(training_config)
    assert model.paligemma_with_expert.gemma_expert.lm_head is None
    assert model.action_out_proj.weight.numel() > 0
    model.load_base(training_config.base_checkpoint)
    weights = load_file(training_config.base_checkpoint)
    del weights['action_in_proj.weight']
    damaged = tmp_path / 'missing-active-weight.safetensors'
    save_file(weights, str(damaged))
    with pytest.raises(ValueError, match='action_in_proj.weight'):
        model.load_base(damaged)


def test_robot_interface_adaptation_preserves_frozen_backbone(native_tiny, training_config):
    from grounded_vla.training.pi0 import ACTION_IO_MODULES
    model = native_tiny(training_config)
    model.load_base(training_config.base_checkpoint)
    model.enable_action_io_training()
    obs, graph, actions, mask = batch(training_config)
    model.flow_loss(obs, actions, graph, mask).backward()
    assert model.action_out_proj.weight.grad.abs().sum() > 0
    allowed = ('graph_adapter.',) + tuple(n+'.' for n in ACTION_IO_MODULES)
    assert all(p.grad is None for n,p in model.named_parameters() if not n.startswith(allowed))
    damaged = model.action_io_state_dict()
    del damaged['action_out_proj.weight']
    with pytest.raises(ValueError, match='keys mismatch'):
        model.load_action_io_state_dict(damaged)


def test_robot_interface_checkpoint_resume_and_inference(native_tiny, training_config, tmp_path, monkeypatch):
    import grounded_vla.training.train as training
    import grounded_vla.training.inference as inference
    from grounded_vla.training.checkpoint import load_action_io
    def factory(config):
        model = native_tiny(config)
        if config.train_action_io:
            model.enable_action_io_training()
        return model
    monkeypatch.setattr(training, 'build_model', factory)
    monkeypatch.setattr(inference, 'build_model', factory)
    config = dataclasses.replace(training_config, train_action_io=True, eval_sampling='uniform',
                                 output_dir=str(tmp_path/'io-full'))
    full = run(config)
    interrupted = dataclasses.replace(config, output_dir=str(tmp_path/'io-resume'))
    part = run(interrupted, stop_after=2)
    resumed = run(interrupted, resume=part)
    for name in ('adapter.safetensors', 'action_io.safetensors'):
        a,b=load_file(str(full/name)),load_file(str(resumed/name))
        for k in a: torch.testing.assert_close(a[k],b[k],rtol=0,atol=0)
    policy = TrainedPi0Policy(full, device='cpu', num_steps=2)
    saved = load_file(str(full/'action_io.safetensors'))
    for k,v in policy.model.action_io_state_dict().items():
        torch.testing.assert_close(v,saved[k],rtol=0,atol=0)
    with pytest.raises(ValueError, match='Unexpected'):
        load_action_io(full,policy.model,False)
