import numpy as np
import torch
import pytest
from grounded_vla.training.data import encode_action_chunk,decode_action_chunk,EpisodeDataset,collate
from grounded_vla.training.tokenizer import Pi0Tokenizer

def bounded_meta():
    return dict(input_preset='canonical',state_dim=2,action_dim=2,action_convention='absolute_joint_positions[a,b]',
                action_encoding='bounded_joint_interval_logits',action_encoding_horizon=1,
                encoding_lower_limits=[-2.,0.],encoding_upper_limits=[2.,.04],encoding_max_joint_step=[.05,.005])

def test_bounded_decoder_extremes_and_independent_limits():
    m=bounded_meta();state=np.array([1.,0.])
    for logits in [np.array([[1e6,-1e6]]),np.array([[-1e6,1e6]]),np.array([[0.,0.]])]:
        target=decode_action_chunk(logits,state,m)
        assert np.all(target>=m['encoding_lower_limits']) and np.all(target<=m['encoding_upper_limits'])
        assert np.all(np.abs(target-state)<m['encoding_max_joint_step'])
    target=np.array([[1.01,.002]])
    np.testing.assert_allclose(decode_action_chunk(encode_action_chunk(target,state,m),state,m),target,atol=1e-7)
    with pytest.raises(ValueError):encode_action_chunk(np.array([[1.5,.002]]),state,m)
    with pytest.raises(ValueError):decode_action_chunk(np.array([[np.nan,0.]]),state,m)
    with pytest.raises(ValueError):decode_action_chunk(np.zeros((1,2)),np.array([3.,0.]),m)

def test_action_expert_receives_gradients_and_vlm_stays_frozen(native_tiny,training_config):
    model=native_tiny(training_config)
    model.enable_action_io_training();model.enable_action_expert_training();model.train()
    assert all(not p.requires_grad for p in model.paligemma_with_expert.paligemma.parameters())
    assert all(p.requires_grad for p in model.paligemma_with_expert.gemma_expert.parameters())
    ds=EpisodeDataset(training_config.dataset,'train',training_config.action_horizon,Pi0Tokenizer(training_config.tokenizer))
    obs,graph,actions,mask=collate([ds[0]])
    model.flow_loss(obs,actions,graph,mask).backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0 for p in model.paligemma_with_expert.gemma_expert.parameters())
    assert all(p.grad is None for p in model.paligemma_with_expert.paligemma.parameters())

def test_expert_checkpoint_resume_preserves_weights(native_tiny,training_config,monkeypatch,tmp_path):
    from grounded_vla.training import train as trainer
    from grounded_vla.training.checkpoint import load_action_expert
    from safetensors.torch import load_file
    from dataclasses import replace
    def factory(cfg):
        m=native_tiny(cfg);m.enable_action_io_training();m.enable_action_expert_training();return m
    monkeypatch.setattr(trainer,'build_model',factory)
    cfg=replace(training_config,train_action_io=True,train_action_expert=True,output_dir=str(tmp_path/'full'))
    full=trainer.run(cfg)
    partial_cfg=replace(cfg,output_dir=str(tmp_path/'resumed'))
    partial=trainer.run(partial_cfg,stop_after=2)
    resumed=trainer.run(partial_cfg,resume=partial)
    a=load_file(str(full/'action_expert.safetensors'));b=load_file(str(resumed/'action_expert.safetensors'))
    for key in a:torch.testing.assert_close(a[key],b[key],rtol=0,atol=0)
    m=factory(cfg);load_action_expert(full,m,True)
    for key,value in m.action_expert_state_dict().items():torch.testing.assert_close(value,a[key])
    with pytest.raises(ValueError):load_action_expert(tmp_path/'missing',m,True)
    with pytest.raises(ValueError):m.load_action_expert_state_dict({'bad':torch.ones(1)})
