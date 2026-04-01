import logging
import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn


__all__ = [
    'KVCache',
    'PagedKVCache',
    'StaticKVCache',
]


class KVCache:

    def __init__(
        self,
        max_seq_len: int,
        num_heads: int,
        head_dim: int,
        num_layers: int,
        dtype: torch.dtype = torch.bfloat16,
        device: str = 'cuda',
    ):
        self.max_seq_len = max_seq_len
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_layers = num_layers
        self.dtype = dtype
        self.device = device

        self._k_cache = torch.zeros(
            num_layers, 1, max_seq_len, num_heads, head_dim,
            dtype=dtype, device=device)
        self._v_cache = torch.zeros(
            num_layers, 1, max_seq_len, num_heads, head_dim,
            dtype=dtype, device=device)
        self._seq_len = 0

    def update(
        self,
        layer_idx: int,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        bsz, new_len, nh, hd = k.shape
        end = self._seq_len + new_len

        self._k_cache[layer_idx, :bsz, self._seq_len:end] = k
        self._v_cache[layer_idx, :bsz, self._seq_len:end] = v

        return (
            self._k_cache[layer_idx, :bsz, :end],
            self._v_cache[layer_idx, :bsz, :end],
        )
