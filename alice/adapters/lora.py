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


class LoRAAdapter:

    def __init__(
        self,
        rank: int = 16,
        alpha: float = 16.0,
        dropout: float = 0.0,
        target_modules: Optional[List[str]] = None,
    ):
        self.rank = rank
        self.alpha = alpha
        self.dropout = dropout
        self.target_modules = target_modules or [
            'q', 'k', 'v', 'o',
            'q_proj', 'k_proj', 'v_proj', 'out_proj',
        ]

    def _match_target(self, name: str) -> bool:
        return name in self.target_modules

    def apply(self, model: nn.Module) -> nn.Module:
        injected = 0
        for name, module in list(model.named_modules()):
            if isinstance(module, nn.Linear) and self._match_target(name):
                parent_name = '.'.join(name.split('.')[:-1])
                child_name = name.split('.')[-1]

                parent = model
                if parent_name:
                    for part in parent_name.split('.'):
                        parent = getattr(parent, part)

                lora_module = LoRALinear(
                    module, self.rank, self.alpha, self.dropout)
                setattr(parent, child_name, lora_module)
                injected += 1

        logging.info(
            f'Injected LoRA adapters into {injected} layers '
            f'(rank={self.rank}, alpha={self.alpha})')
        return model


def inject_lora(
    model: nn.Module,
    checkpoint_path: str,
    rank: int = 16,
    alpha: float = 16.0,
    target_modules: Optional[List[str]] = None,
) -> nn.Module:
    adapter = LoRAAdapter(
        rank=rank, alpha=alpha, target_modules=target_modules)
    model = adapter.apply(model)

    state_dict = torch.load(checkpoint_path, map_location='cpu')
    if 'lora_state_dict' in state_dict:
        state_dict = state_dict['lora_state_dict']

    lora_keys = {k: v for k, v in state_dict.items()
                 if 'lora_a' in k or 'lora_b' in k}

    missing, unexpected = [], []
    model_dict = model.state_dict()
    for k, v in lora_keys.items():
        if k in model_dict:
            model_dict[k] = v
        else:
            unexpected.append(k)

    model.load_state_dict(model_dict, strict=False)

    if unexpected:
        logging.warning(f'Unexpected LoRA keys: {unexpected}')

    logging.info(f'Loaded {len(lora_keys)} LoRA weight tensors')
    return model


def extract_lora(model: nn.Module) -> Dict[str, torch.Tensor]:
    lora_state = {}
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            lora_state[f'{name}.lora_a.weight'] = module.lora_a.weight.data.clone()
            lora_state[f'{name}.lora_b.weight'] = module.lora_b.weight.data.clone()
            lora_state[f'{name}.rank'] = torch.tensor(module.rank)
            lora_state[f'{name}.alpha'] = torch.tensor(module.alpha)
    return lora_state
