"""Causal motor-velocity facts supplement the existing geometric belief graph."""
import copy
import numpy as np
from grounded_vla.training.graph import FEATURES, SCHEMA_ID

VELOCITY_SCHEMA_ID='grounded-vla-belief-motor-velocity-v2'


def velocity_schema(source, joint_names):
    if source['id']!=SCHEMA_ID or source['features']!=list(FEATURES) or len(joint_names)!=9:
        raise ValueError('Velocity extension requires the canonical recorded graph and nine ordered joints')
    schema=copy.deepcopy(source)
    names=['observed_joint_velocity/'+name for name in joint_names]
    schema['id']=VELOCITY_SCHEMA_ID;schema['features']=source['features']+names
    schema['motor_velocity_units']=dict(zip(names,['rad/s']*7+['m/s']*2,strict=True))
    return schema


def state_velocities(states, period):
    q=np.asarray(states,np.float32)
    if q.ndim!=2 or q.shape[1]!=9 or len(q)==0 or not np.isfinite(q).all() or not np.isfinite(period) or period<=0:
        raise ValueError('Invalid causal motor state/period')
    velocity=np.zeros_like(q)
    velocity[1:]=(q[1:]-q[:-1])/period
    return velocity


def append_velocity_features(features, velocities, robot_node=2):
    f=np.asarray(features,np.float32);v=np.asarray(velocities,np.float32)
    if f.ndim!=3 or f.shape[-1]!=len(FEATURES) or v.shape!=(len(f),9):
        raise ValueError('Recorded graph/velocity dimensions disagree')
    if type(robot_node) is not int or not 0<=robot_node<f.shape[1]:
        raise ValueError('Invalid robot node')
    if not np.isfinite(f).all() or not np.isfinite(v).all():
        raise ValueError('Non-finite velocity graph')
    extra=np.zeros((len(f),f.shape[1],9),np.float32);extra[:,robot_node,:]=v
    return np.concatenate([f,extra],axis=-1)


def extend_recorded_graph(features, states, metadata):
    extension=metadata['proprio_velocity_graph']
    if metadata['graph_schema']['id']!=VELOCITY_SCHEMA_ID or extension['source_schema_id']!=SCHEMA_ID:
        raise ValueError('Velocity graph schema identity mismatch')
    return append_velocity_features(features,state_velocities(states,metadata['control_period_seconds']),extension['robot_node_index'])


def with_joint_velocities(graph, current_state, previous_state, period, robot_node=2):
    if graph['schema_id']!=SCHEMA_ID:
        raise ValueError('Live velocity extension requires the recorded base graph schema')
    q=np.stack([np.asarray(previous_state,np.float32),np.asarray(current_state,np.float32)])
    velocities=state_velocities(q,period)[-1:]
    return {**graph,'features':append_velocity_features(np.asarray(graph['features'])[None],velocities,robot_node)[0],
            'schema_id':VELOCITY_SCHEMA_ID}


def extend_adapter_weights(weights, new_feature_dim, reset_gate=None):
    import torch
    key='encoder.input.weight'
    if key not in weights or weights[key].ndim!=2 or weights[key].shape[1]!=len(FEATURES):
        raise ValueError('Unexpected source graph adapter input projection')
    if new_feature_dim!=len(FEATURES)+9:
        raise ValueError('Motor-velocity extension requires exactly nine new feature columns')
    result={k:v.clone() for k,v in weights.items()}
    old=weights[key];new=old.new_zeros((old.shape[0],new_feature_dim));new[:,:old.shape[1]]=old
    result[key]=new
    if reset_gate is not None:
        if 'gate' not in result or not np.isfinite(reset_gate):raise ValueError('Invalid graph gate reset')
        result['gate'].fill_(float(reset_gate))
    if not all(torch.isfinite(v).all() for v in result.values()):raise ValueError('Non-finite migrated graph weights')
    return result
