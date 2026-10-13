import json
import numpy as np
import pytest
from grounded_vla.training.data import encode_action_chunk, decode_action_chunk, prepare_dataset, EpisodeDataset, normalize
from grounded_vla.training.tokenizer import Pi0Tokenizer

def meta():
    return dict(input_preset='canonical',state_dim=2,action_dim=2,action_convention='absolute_joint_positions[a,b]',action_encoding='relative_to_observed_joint_positions',action_encoding_horizon=2)

def test_translation_equivariance_roundtrip_and_no_clipping():
    state=np.array([1.,2.],np.float32); actions=np.array([[1.1,2.2],[1.3,2.4]],np.float32)
    delta=encode_action_chunk(actions,state,meta())
    np.testing.assert_allclose(decode_action_chunk(delta,state,meta()),actions)
    shift=np.array([5.,-3.],np.float32)
    np.testing.assert_allclose(encode_action_chunk(actions+shift,state+shift,meta()),delta,atol=1e-6)
    np.testing.assert_array_equal(decode_action_chunk(np.array([[10.,-10.]],np.float32),state,meta()),[[11.,-8.]])

def test_relative_statistics_and_windows_use_only_current_state(episodes,tokenizer_file):
    manifest,_=episodes; m=json.loads(manifest.read_text())
    m.update(state_dim=7,input_preset='canonical',action_convention='absolute_joint_positions[j0,j1,j2,j3,j4,j5,j6]',action_encoding='relative_to_observed_joint_positions',action_encoding_horizon=2)
    expected=[]; first=None
    for e in m['episodes']:
        p=manifest.parent/e['path']
        with np.load(p,allow_pickle=False) as z: raw={k:z[k] for k in z.files}
        raw['state']=raw['state'][:,:7]; np.savez_compressed(p,**raw)
        if e['split']=='train':
            expected.extend(raw['actions'][t:t+2]-raw['state'][t] for t in range(len(raw['state'])))
            if first is None:first=raw
    manifest.write_text(json.dumps(m)); output=manifest.parent/'relative.json'; prep=prepare_dataset(manifest,output)
    values=np.concatenate(expected)
    np.testing.assert_allclose(prep['normalization']['action']['mean'],values.mean(0),atol=1e-6)
    ds=EpisodeDataset(output,'train',2,Pi0Tokenizer(tokenizer_file))
    encoded=normalize(first['actions'][:2]-first['state'][0],prep['normalization']['action'])
    np.testing.assert_allclose(ds[0]['actions'][:,:7].numpy(),encoded)
    assert ds[2]['action_mask'].sum()==7
    with pytest.raises(ValueError,match='horizon'):EpisodeDataset(output,'train',3,Pi0Tokenizer(tokenizer_file))

def test_unknown_encoding_and_wrong_dimensions_fail():
    m=meta();m['action_encoding']='unknown'
    with pytest.raises(ValueError):decode_action_chunk(np.zeros((1,2)),np.zeros(2),m)
    with pytest.raises(ValueError):decode_action_chunk(np.zeros((1,3)),np.zeros(2),meta())
