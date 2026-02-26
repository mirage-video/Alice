import logging
from collections import OrderedDict
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


__all__ = []


class QuickGELU(nn.Module):

    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class CLIPAttention(nn.Module):

    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, c = x.shape
        q = self.q_proj(x).reshape(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).reshape(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(b, n, self.num_heads, self.head_dim).transpose(1, 2)

        attn = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_dropout.p if self.training else 0.0)
        attn = attn.transpose(1, 2).reshape(b, n, c)
        return self.out_proj(attn)


class CLIPEncoderLayer(nn.Module):

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.ln_1 = nn.LayerNorm(dim)
        self.attn = CLIPAttention(dim, num_heads, dropout)
        self.ln_2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(OrderedDict([
            ('c_fc', nn.Linear(dim, int(dim * mlp_ratio))),
            ('gelu', QuickGELU()),
            ('c_proj', nn.Linear(int(dim * mlp_ratio), dim)),
        ]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class CLIPVisionEncoder:
    pass
