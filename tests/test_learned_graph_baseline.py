import json
import numpy as np
import pytest
import torch
from experiments.runpod.learned_graph_baseline import Actor, BaselinePolicy, numeric_inputs, physical_targets, causal_state_features
from grounded_vla.training.data import decode_action_chunk


def meta():
    return dict(action_encoding='bounded_joint_interval_logits',action_encoding_horizon=1,
                input_preset='canonical',state_dim=9,action_dim=9,
                action_convention='absolute_joint_positions',
                encoding_lower_limits=[-1.]*9,encoding_upper_limits=[1.]*9,
                encoding_max_joint_step=[.05]*9,
                normalization={'action':{'mean':[0.]*9,'std':[1.]*9}})


def test_hidden_graph_nodes_do_not_leak_into_actor_inputs():
    state=np.zeros((1,9),np.float32);f=np.zeros((1,3,26),np.float32)
    edges=np.ones((1,3,3),np.int64);v=np.array([[True,False,True]])
    before=numeric_inputs(state,f,edges,v,8)
    f[:,1,:]=9000;edges[:,1,:]=7;edges[:,:,1]=6
    np.testing.assert_array_equal(before,numeric_inputs(state,f,edges,v,8))
    with pytest.raises(ValueError):numeric_inputs(state,f,edges.astype(np.float32),v,8)


def test_torch_training_decode_matches_native_guarded_decode():
    torch.set_num_threads(2)
    state=np.linspace(-.95,.95,9,dtype=np.float32)
    pred=np.array([[-100,100,-30,30,0,-1,1,-100,100]],np.float32)
    actual=physical_targets(torch.from_numpy(pred),torch.from_numpy(state[None]),meta()).numpy()
    native=decode_action_chunk(pred,state,meta())
    np.testing.assert_allclose(actual,native,atol=1e-7)
    assert np.all(np.abs(native-state)<=.05+1e-7)
    assert np.all(native>=-1) and np.all(native<=1)


def test_actor_learns_feature_conditioned_physical_commands():
    torch.set_num_threads(2);torch.manual_seed(3)
    x=torch.randn(96,4);q=torch.zeros(96,9)
    target=.02*torch.tanh(x[:,:1]).expand(-1,9)
    model=Actor(4,9,width=32);opt=torch.optim.Adam(model.parameters(),lr=.01)
    with torch.no_grad():initial=((physical_targets(model(x),q,meta())-target)**2).mean().item()
    for _ in range(200):
        loss=((physical_targets(model(x),q,meta())-target)**2).mean()
        opt.zero_grad();loss.backward();opt.step()
    final=((physical_targets(model(x),q,meta())-target)**2).mean().item()
    assert final<initial*.1
    assert all(torch.isfinite(p).all() for p in model.parameters())


def test_policy_rejects_tampered_actor_identity(tmp_path):
    (tmp_path/'actor.safetensors').write_bytes(b'tampered')
    (tmp_path/'metadata.json').write_text(json.dumps({'dataset':meta(),'weights_sha256':'incorrect'}))
    with pytest.raises(ValueError,match='identity mismatch'):BaselinePolicy(tmp_path)


def test_state_history_is_causal_and_matches_streaming_context():
    q=np.arange(108,dtype=np.float32).reshape(12,9)*.001
    expected=causal_state_features(q,4,1/60)
    future_changed=q.copy();future_changed[7:]+=100
    np.testing.assert_array_equal(expected[:7],causal_state_features(future_changed,4,1/60)[:7])
    for t in range(len(q)):
        window=q[max(0,t-3):t+1]
        np.testing.assert_array_equal(expected[t],causal_state_features(window,4,1/60)[-1])
    np.testing.assert_array_equal(expected[0,9:],np.zeros(27,np.float32))


def test_training_history_metadata_restores_streaming_policy(tmp_path,monkeypatch):
    from experiments.runpod import learned_graph_baseline as baseline
    m=meta();m.update(control_period_seconds=1/60,
        graph_schema={'id':'baseline-test','features':[f'f{i}' for i in range(26)],'relations':[f'r{i}' for i in range(8)]},
        episodes=[{'id':'train0','split':'train'},{'id':'val1','split':'val'}])
    def numeric(*args):
        q=np.repeat(np.arange(8,dtype=np.float32)[:,None]*.001,9,axis=1);a=q+.002
        context=causal_state_features(q,4,1/60)
        f=np.zeros((8,3,26),np.float32);e=np.zeros((8,3,3),np.int64);v=np.ones((8,3),bool)
        x=numeric_inputs(context,f,e,v,8)
        y=np.concatenate([baseline.encode_action_chunk(a[i:i+1],q[i],m) for i in range(len(q))])
        return x,y,q,a
    monkeypatch.setattr(baseline,'load_prepared',lambda _:m)
    monkeypatch.setattr(baseline,'read_numeric',numeric)
    prepared=tmp_path/'prepared.json';prepared.write_text('{}')
    checkpoint=tmp_path/'model';baseline.train(prepared,checkpoint,steps=1,history=4)
    metadata=json.loads((checkpoint/'metadata.json').read_text())
    assert type(metadata['history_steps']) is int and metadata['history_steps']==4
    assert metadata['input_dim']==189
    restored=baseline.BaselinePolicy(checkpoint)
    assert restored.metadata['history_steps']==4
    assert len(restored.states)==0
