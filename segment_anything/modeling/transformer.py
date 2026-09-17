"""SAM Two-Way Transformer with DAG attention on the last token-to-image layer."""
from __future__ import annotations

import math
from typing import Tuple, Type

import torch
from torch import Tensor, nn

from .common import MLPBlock


class TwoWayTransformer(nn.Module):
    def __init__(
        self,
        depth: int,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int,
        activation: Type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,
        DAG: bool = True,
    ) -> None:
        super().__init__()
        self.depth = depth
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.DAG = DAG
        self.layers = nn.ModuleList(
            [
                TwoWayAttentionBlock(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                    activation=activation,
                    attention_downsample_rate=attention_downsample_rate,
                    skip_first_layer_pe=(i == 0),
                )
                for i in range(depth)
            ]
        )
        attn_cls = DAGAttention if DAG else Attention
        self.final_attn_token_to_image = attn_cls(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.norm_final_attn = nn.LayerNorm(embedding_dim)

    def forward(self, image_embedding: Tensor, image_pe: Tensor, point_embedding: Tensor) -> Tuple[Tensor, Tensor]:
        bs, c, h, w = image_embedding.shape
        image_embedding = image_embedding.flatten(2).permute(0, 2, 1)
        image_pe = image_pe.flatten(2).permute(0, 2, 1)
        queries, keys = point_embedding, image_embedding
        for layer in self.layers:
            queries, keys = layer(queries=queries, keys=keys, query_pe=point_embedding, key_pe=image_pe)
        attn_out = self.final_attn_token_to_image(q=queries + point_embedding, k=keys + image_pe, v=keys)
        queries = self.norm_final_attn(queries + attn_out)
        return queries, keys


class TwoWayAttentionBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int = 2048,
        activation: Type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,
        skip_first_layer_pe: bool = False,
    ) -> None:
        super().__init__()
        self.self_attn = Attention(embedding_dim, num_heads)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.cross_attn_token_to_image = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.mlp = MLPBlock(embedding_dim, mlp_dim, activation)
        self.norm3 = nn.LayerNorm(embedding_dim)
        self.norm4 = nn.LayerNorm(embedding_dim)
        self.cross_attn_image_to_token = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.skip_first_layer_pe = skip_first_layer_pe

    def forward(self, queries: Tensor, keys: Tensor, query_pe: Tensor, key_pe: Tensor) -> Tuple[Tensor, Tensor]:
        if self.skip_first_layer_pe:
            queries = self.self_attn(q=queries, k=queries, v=queries)
        else:
            q = queries + query_pe
            queries = queries + self.self_attn(q=q, k=q, v=queries)
        queries = self.norm1(queries)

        q, k = queries + query_pe, keys + key_pe
        queries = self.norm2(queries + self.cross_attn_token_to_image(q=q, k=k, v=keys))
        queries = self.norm3(queries + self.mlp(queries))
        q, k = queries + query_pe, keys + key_pe
        keys = self.norm4(keys + self.cross_attn_image_to_token(q=k, k=q, v=queries))
        return queries, keys


class Attention(nn.Module):
    def __init__(self, embedding_dim: int, num_heads: int, downsample_rate: int = 1) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads
        assert self.internal_dim % num_heads == 0
        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.v_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)

    def _separate_heads(self, x: Tensor, num_heads: int) -> Tensor:
        b, n, c = x.shape
        return x.reshape(b, n, num_heads, c // num_heads).transpose(1, 2)

    def _recombine_heads(self, x: Tensor) -> Tensor:
        b, n_heads, n_tokens, c_per_head = x.shape
        return x.transpose(1, 2).reshape(b, n_tokens, n_heads * c_per_head)

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        q = self._separate_heads(self.q_proj(q.to(self.q_proj.weight.dtype)), self.num_heads)
        k = self._separate_heads(self.k_proj(k.to(self.k_proj.weight.dtype)), self.num_heads)
        v = self._separate_heads(self.v_proj(v.to(self.v_proj.weight.dtype)), self.num_heads)
        attn = torch.softmax((q @ k.permute(0, 1, 3, 2)) / math.sqrt(q.shape[-1]), dim=-1)
        return self.out_proj(self._recombine_heads(attn @ v))


class DAGAttention(nn.Module):
    """Final token-to-image attention with a learned transition (ETU-SAM checkpoint)."""

    def __init__(self, embedding_dim: int, num_heads: int, downsample_rate: int = 1) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads
        assert self.internal_dim % num_heads == 0
        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.v_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.transition = nn.Sequential(
            nn.Linear(embedding_dim, self.internal_dim),
            nn.LayerNorm(self.internal_dim),
            nn.ReLU(),
            nn.Linear(self.internal_dim, embedding_dim),
        )
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)

    def _separate_heads(self, x: Tensor, num_heads: int) -> Tensor:
        b, n, c = x.shape
        return x.reshape(b, n, num_heads, c // num_heads).transpose(1, 2)

    def _recombine_heads(self, x: Tensor) -> Tensor:
        b, n_heads, n_tokens, c_per_head = x.shape
        return x.transpose(1, 2).reshape(b, n_tokens, n_heads * c_per_head)

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        q = self._separate_heads(self.q_proj(q.to(self.q_proj.weight.dtype)), self.num_heads)
        k = self._separate_heads(self.k_proj(k.to(self.k_proj.weight.dtype)), self.num_heads)
        v = self._separate_heads(self.v_proj(v.to(self.v_proj.weight.dtype)), self.num_heads)
        _, _, t, c_per_head = q.shape
        attn = (q @ k.permute(0, 1, 3, 2)) / math.sqrt(c_per_head)
        if t > 7:
            transition = torch.softmax(self.transition(attn), dim=-1) ** 4
            attn = torch.softmax(attn, dim=-1) * transition
        else:
            attn = torch.softmax(attn, dim=-1)
        return self.out_proj(self._recombine_heads(attn @ v))
