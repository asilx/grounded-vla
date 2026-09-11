"""Relation-aware graph tokens and gated cross-attention for action hidden states.

Shapes: hidden [B,T,D], features [B,N,F], edge types [B,N,N], validity [B,N].
Object identity is supplied through grounded features/relations, not ID embeddings.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class GraphContextEncoder(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 64,
        heads: int = 4,
        context_tokens: int = 8,
        relation_types: int = 8,
    ) -> None:
        super().__init__()
        if (
            min(feature_dim, hidden_dim, heads, context_tokens, relation_types) <= 0
            or hidden_dim % heads
        ):
            raise ValueError("Dimensions must be positive; hidden_dim must be divisible by heads")
        self.hidden_dim, self.heads, self.context_tokens = hidden_dim, heads, context_tokens
        self.input = nn.Linear(feature_dim, hidden_dim)
        self.qkv = nn.Linear(hidden_dim, hidden_dim * 3)
        self.relation_bias = nn.Embedding(relation_types, heads)
        self.output = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.queries = nn.Parameter(torch.randn(context_tokens, hidden_dim) / math.sqrt(hidden_dim))
        self.pool = nn.MultiheadAttention(hidden_dim, heads, batch_first=True, dropout=0.0)

    def forward(self, features: Tensor, edge_types: Tensor, valid: Tensor) -> tuple[Tensor, Tensor]:
        if features.ndim != 3:
            raise ValueError("features must have shape [B,N,F]")
        batch, nodes, _ = features.shape
        if edge_types.shape != (batch, nodes, nodes) or valid.shape != (batch, nodes):
            raise ValueError("Graph shapes must agree")
        if valid.dtype != torch.bool or edge_types.dtype != torch.long:
            raise ValueError("Validity must be boolean; edge_types must be int64")
        active = valid.any(dim=1)
        if nodes == 0:
            return features.new_zeros((batch, self.context_tokens, self.hidden_dim)), active
        safe_valid = valid.clone()
        safe_valid[~active, 0] = True  # Avoid all-masked softmax; zero these rows below.
        features = torch.where(valid.unsqueeze(-1), features, torch.zeros_like(features))
        x = self.input(features)
        qkv = self.qkv(x).reshape(batch, nodes, 3, self.heads, self.hidden_dim // self.heads)
        q, k, v = (qkv[:, :, index].transpose(1, 2) for index in range(3))
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.hidden_dim // self.heads)
        scores = scores + self.relation_bias(edge_types).permute(0, 3, 1, 2)
        scores = scores.masked_fill(~safe_valid[:, None, None, :], float("-inf"))
        message = (
            (scores.softmax(dim=-1) @ v).transpose(1, 2).reshape(batch, nodes, self.hidden_dim)
        )
        x = self.norm(x + self.output(message))
        x = torch.where(valid.unsqueeze(-1), x, torch.zeros_like(x))
        context, _ = self.pool(
            self.queries.unsqueeze(0).expand(batch, -1, -1),
            x,
            x,
            key_padding_mask=~safe_valid,
            need_weights=False,
        )
        context = torch.where(active[:, None, None], context, torch.zeros_like(context))
        return context, active


class GatedKnowledgeAdapter(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 64,
        heads: int = 4,
        context_tokens: int = 8,
        relation_types: int = 8,
        initial_gate: float = 1e-3,
    ) -> None:
        super().__init__()
        self.encoder = GraphContextEncoder(
            feature_dim, hidden_dim, heads, context_tokens, relation_types
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.attention = nn.MultiheadAttention(hidden_dim, heads, batch_first=True, dropout=0.0)
        self.gate = nn.Parameter(torch.tensor(float(initial_gate)))

    def forward(
        self, hidden: Tensor, features: Tensor, edge_types: Tensor, valid: Tensor
    ) -> Tensor:
        if (
            hidden.ndim != 3
            or hidden.shape[0] != features.shape[0]
            or hidden.shape[2] != self.encoder.hidden_dim
        ):
            raise ValueError("hidden must be [B,T,D] with matching batch and hidden dimension")
        context, active = self.encoder(features, edge_types, valid)
        delta, _ = self.attention(self.norm(hidden), context, context, need_weights=False)
        delta = torch.where(active[:, None, None], delta, torch.zeros_like(delta))
        return hidden + self.gate.tanh() * delta
