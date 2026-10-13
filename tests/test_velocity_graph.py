import numpy as np
import pytest
import torch
from grounded_vla.training.graph import FEATURES,SCHEMA_ID
from grounded_vla.training.velocity_graph import append_velocity_features,state_velocities,with_joint_velocities,extend_adapter_weights


def test_velocity_facts_are_causal_and_live_recorded_transforms_agree():
    q=np.arange(72,dtype=np.float32).reshape(8,9)*.001
    f=np.zeros((8,3,len(FEATURES)),np.float32);f[:,:,:3]=.3
    result=append_velocity_features(f,state_velocities(q,1/60))
    changed=q.copy();changed[5:]+=10
    np.testing.assert_array_equal(result[:5],append_velocity_features(f,state_velocities(changed,1/60))[:5])
    for t in range(len(q)):
        graph={'features':f[t],'schema_id':SCHEMA_ID}
        live=with_joint_velocities(graph,q[t],q[max(0,t-1)],1/60)
        np.testing.assert_array_equal(result[t],live['features'])
    np.testing.assert_array_equal(result[:,:,:len(FEATURES)],f)
    assert np.count_nonzero(result[:,0,len(FEATURES):])==0


def test_zero_initialized_velocity_columns_preserve_source_projection():
    torch.manual_seed(7)
    w=torch.randn(16,len(FEATURES));bias=torch.randn(16)
    original={'encoder.input.weight':w,'encoder.input.bias':bias,'gate':torch.tensor(.012)}
    migrated=extend_adapter_weights(original,len(FEATURES)+9)
    x=torch.randn(6,len(FEATURES));new=torch.cat([x,torch.randn(6,9)],dim=1)
    torch.testing.assert_close(x@w.T+bias,new@migrated['encoder.input.weight'].T+bias)
    torch.testing.assert_close(original['gate'],migrated['gate'])
    assert torch.count_nonzero(migrated['encoder.input.weight'][:,-9:])==0


def test_episode_loader_derives_velocity_without_changing_recorded_bytes(tmp_path):
    from grounded_vla.training.data import load_episode
    q=np.arange(27,dtype=np.float32).reshape(3,9)*.001
    p=tmp_path/'episode.npz';times=np.arange(3)/60
    np.savez_compressed(p,state=q,actions=q,timestamps=times,graph_timestamps=times,prompt=np.array(['place']*3),
        graph_features=np.zeros((3,3,len(FEATURES)),np.float32),graph_edges=np.zeros((3,3,3),np.int64),
        graph_valid=np.ones((3,3),bool),**{'images/base_0_rgb':np.zeros((3,224,224,3),np.uint8)})
    before=p.read_bytes()
    meta={'state_dim':9,'action_dim':9,'control_period_seconds':1/60,'graph_schema':{'id':'grounded-vla-belief-motor-velocity-v2','features':list(FEATURES)+[str(i) for i in range(9)],'relations':['none']},
          'proprio_velocity_graph':{'source_schema_id':SCHEMA_ID,'robot_node_index':2}}
    raw=load_episode(p,meta)
    np.testing.assert_array_equal(raw['graph_features'][:,2,-9:],state_velocities(q,1/60))
    assert p.read_bytes()==before
    with pytest.raises(ValueError):load_episode(p,{**meta,'proprio_velocity_graph':{'source_schema_id':'wrong','robot_node_index':2}})
