import copy
import pytest
import torch
from pathlib import Path
import json
from safetensors.torch import load_file
from grounded_vla.training.checkpoint import warm_start_checkpoint
from grounded_vla.training import train as trainer

def test_warm_start_restores_weights_and_checks_semantics(native_tiny,training_config):
    path=trainer.run(training_config)
    metadata=json.loads((path/'metadata.json').read_text())
    model=native_tiny(training_config)
    warm_start_checkpoint(path,model,metadata)
    expected=load_file(str(path/'adapter.safetensors'))
    for k,v in model.graph_adapter.state_dict().items():torch.testing.assert_close(v,expected[k])
    assert metadata['warm_start']['previous_step']==training_config.steps
    assert 'optimizer' in metadata['warm_start']['scope']
    bad=copy.deepcopy(metadata);bad['dataset']['action_convention']='different actuator order'
    with pytest.raises(ValueError,match='semantics'):warm_start_checkpoint(path,model,bad)
    bad=copy.deepcopy(metadata);bad['identity']['base_checkpoint']='bad'
    with pytest.raises(ValueError,match='identity'):warm_start_checkpoint(path,model,bad)
