"""Real openpi π0 flow matching and denoising with graph-conditioned action tokens."""

from contextlib import contextmanager

import torch
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
from safetensors.torch import load_model

from grounded_vla.learning.adapter import GatedKnowledgeAdapter
from grounded_vla.training.config import AdapterConfig
from grounded_vla.training.graph import GraphBatch
from grounded_vla.training.runtime import INACTIVE_PI0_KEYS, remove_inactive_pi0_head

ACTION_IO_MODULES = ('state_proj', 'action_in_proj', 'action_out_proj',
                     'action_time_mlp_in', 'action_time_mlp_out')


class GroundedPi0(PI0Pytorch):
    def __init__(self, config, adapter_config: AdapterConfig):
        if config.pi05 or config.pytorch_compile_mode is not None:
            raise ValueError("This integration supports π0 with torch.compile disabled")
        super().__init__(config)
        remove_inactive_pi0_head(self)
        self.requires_grad_(False)
        self.adapter_config = adapter_config
        self.graph_adapter = GatedKnowledgeAdapter(
            hidden_dim=self.action_in_proj.out_features, **vars(adapter_config)
        )
        self._graph = None
        self.train_action_expert = False

    def enable_action_io_training(self):
        """Explicit robot-interface adaptation; vision/language/expert blocks stay frozen."""
        for name in ACTION_IO_MODULES:
            getattr(self, name).requires_grad_(True)

    def action_io_state_dict(self):
        return {name+'.'+k: v for name in ACTION_IO_MODULES
                for k, v in getattr(self, name).state_dict().items()}

    def load_action_io_state_dict(self, values):
        if set(values) != set(self.action_io_state_dict()):
            raise ValueError('Action interface checkpoint keys mismatch')
        if not all(torch.isfinite(v).all() for v in values.values()):
            raise ValueError('Non-finite action interface weights')
        for name in ACTION_IO_MODULES:
            prefix=name+'.'
            getattr(self, name).load_state_dict(
                {k[len(prefix):]: v for k,v in values.items() if k.startswith(prefix)}, strict=True)

    def enable_action_expert_training(self):
        self.train_action_expert=True
        self.paligemma_with_expert.gemma_expert.requires_grad_(True)

    def action_expert_state_dict(self):
        return self.paligemma_with_expert.gemma_expert.state_dict()

    def load_action_expert_state_dict(self, values):
        if set(values)!=set(self.action_expert_state_dict()):
            raise ValueError('Action expert checkpoint keys mismatch')
        if not all(torch.isfinite(v).all() for v in values.values()):
            raise ValueError('Non-finite action expert weights')
        self.paligemma_with_expert.gemma_expert.load_state_dict(values,strict=True)

    def load_base(self, path):
        missing, unexpected = load_model(self, str(path), strict=False)
        # Older native state dictionaries may contain the unused decoder head.
        # It cannot affect flow matching or action denoising and is not loaded.
        unexpected = set(unexpected) - set(INACTIVE_PI0_KEYS)
        expected = {f"graph_adapter.{k}" for k in self.graph_adapter.state_dict()}
        if set(missing) != expected or unexpected:
            raise ValueError(
                f"Base checkpoint mismatch: missing={set(missing) - expected}, unexpected={unexpected}"
            )

    def train(self, mode=True):
        super().train(mode)
        # Frozen model has no stochastic augmentations/dropout; gradients still cross the expert.
        self.paligemma_with_expert.eval()
        if self.train_action_expert:
            self.paligemma_with_expert.gemma_expert.train(mode)
        self.graph_adapter.train(mode)
        return self

    def _preprocess_observation(self, observation, *, train=True):
        # Geometric augmentation without transforming the graph would break grounding.
        return super()._preprocess_observation(observation, train=False)

    @contextmanager
    def graph_context(self, graph: GraphBatch):
        if self._graph is not None:
            raise RuntimeError("GroundedPi0 is not reentrant; serialize inference requests")
        graph.validate(self.adapter_config.feature_dim, self.adapter_config.relation_types)
        self._graph = graph
        try:
            yield
        finally:
            self._graph = None

    def embed_suffix(self, state, noisy_actions, timestep):
        embeddings, pad, attention, cond = super().embed_suffix(state, noisy_actions, timestep)
        if self._graph is not None:
            horizon = self.config.action_horizon
            graph = self._graph
            actions = self.graph_adapter(
                embeddings[:, -horizon:].float(), graph.features.float(), graph.edges, graph.valid
            ).to(embeddings.dtype)
            embeddings = torch.cat((embeddings[:, :-horizon], actions), dim=1)
        return embeddings, pad, attention, cond

    def flow_loss(self, observation, actions, graph, action_mask, *, noise=None, time=None):
        if action_mask.shape != actions.shape or action_mask.dtype != torch.bool:
            raise ValueError("action_mask must be boolean and match [B,H,32]")
        if not action_mask.any() or not torch.isfinite(actions[action_mask]).all():
            raise ValueError("No valid finite action targets")
        # Ignore invalid targets before native interpolation, not only after the loss.
        targets = torch.where(action_mask, actions, torch.zeros_like(actions))
        with self.graph_context(graph):
            losses = super().forward(observation, targets, noise=noise, time=time)
        return losses.masked_select(action_mask).mean()

    def grounded_actions(self, device, observation, graph, *, noise=None, num_steps=10):
        if num_steps <= 0:
            raise ValueError("num_steps must be positive")
        with self.graph_context(graph):
            return super().sample_actions(device, observation, noise=noise, num_steps=num_steps)
