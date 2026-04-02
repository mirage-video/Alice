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

        if end > self.max_seq_len:
            shift = end - self.max_seq_len
            self._k_cache[:, :, :-shift] = self._k_cache[:, :, shift:].clone()
            self._v_cache[:, :, :-shift] = self._v_cache[:, :, shift:].clone()
            self._seq_len -= shift
            end = self._seq_len + new_len

        self._k_cache[layer_idx, :bsz, self._seq_len:end] = k
        self._v_cache[layer_idx, :bsz, self._seq_len:end] = v

        return (
            self._k_cache[layer_idx, :bsz, :end],
            self._v_cache[layer_idx, :bsz, :end],
        )

    def advance(self, num_tokens: int):
        self._seq_len += num_tokens

    @property
    def seq_len(self) -> int:
        return self._seq_len

    def reset(self):
        self._seq_len = 0
        self._k_cache.zero_()
        self._v_cache.zero_()

    def memory_bytes(self) -> int:
        return (self._k_cache.nelement() * self._k_cache.element_size() +
                self._v_cache.nelement() * self._v_cache.element_size())


class PagedKVCache:

    def __init__(
        self,
        page_size: int = 256,
        max_pages: int = 64,
        num_heads: int = 40,
        head_dim: int = 128,
        num_layers: int = 40,
        dtype: torch.dtype = torch.bfloat16,
        device: str = 'cuda',
    ):
        self.page_size = page_size
        self.max_pages = max_pages
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_layers = num_layers
        self.dtype = dtype
        self.device = device

        self._k_pages = torch.zeros(
            num_layers, max_pages, page_size, num_heads, head_dim,
            dtype=dtype, device=device)
        self._v_pages = torch.zeros(
            num_layers, max_pages, page_size, num_heads, head_dim,
            dtype=dtype, device=device)

        self._page_table: List[int] = []
        self._free_pages: List[int] = list(range(max_pages))
        self._current_pos = 0

    def _allocate_page(self) -> int:
        if not self._free_pages:
            oldest = self._page_table.pop(0)
            self._free_pages.append(oldest)
            logging.debug(f'Evicting page {oldest} (cache full)')

        page_id = self._free_pages.pop(0)
        self._page_table.append(page_id)
        return page_id

    def append(
        self,
        layer_idx: int,
        k: torch.Tensor,
        v: torch.Tensor,
    ):
        bsz, seq_len, nh, hd = k.shape
        for i in range(seq_len):
            page_offset = self._current_pos % self.page_size
            if page_offset == 0:
                self._allocate_page()

            page_id = self._page_table[-1]
            self._k_pages[layer_idx, page_id, page_offset] = k[:, i]
            self._v_pages[layer_idx, page_id, page_offset] = v[:, i]

            if layer_idx == self.num_layers - 1:
                self._current_pos += 1
