import logging
import re
from typing import Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ['LoRAAdapter', 'inject_lora', 'extract_lora', 'merge_lora_weights']


class LoRALinear(nn.Module):

    def __init__(
        self,
        base_layer: nn.Linear,
        rank: int = 16,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.base_layer = base_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        in_features = base_layer.in_features
        out_features = base_layer.out_features

        self.lora_a = nn.Linear(in_features, rank, bias=False)
        self.lora_b = nn.Linear(rank, out_features, bias=False)
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        nn.init.kaiming_uniform_(self.lora_a.weight)
        nn.init.zeros_(self.lora_b.weight)

        self.base_layer.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_layer(x)
        lora_out = self.lora_b(self.lora_a(self.lora_dropout(x)))
        return base_out + lora_out * self.scaling

    def merge(self) -> nn.Linear:
        merged = nn.Linear(
            self.base_layer.in_features,
            self.base_layer.out_features,
            bias=self.base_layer.bias is not None)
        merged.weight.data.copy_(
            self.base_layer.weight.data +
            (self.lora_b.weight @ self.lora_a.weight) * self.scaling)
        if self.base_layer.bias is not None:
            merged.bias.data.copy_(self.base_layer.bias.data)
        return merged

    def unmerge(self):
        pass
