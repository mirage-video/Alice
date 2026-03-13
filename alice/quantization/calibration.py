import logging
from collections import defaultdict
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from .utils import compute_scale


FP8_MAX = torch.finfo(torch.float8_e4m3fn).max


class CalibrationContext:

    def __init__(self, model: nn.Module, num_batches: int = 32):
        self.model = model
        self.num_batches = num_batches
        self._hooks = []
        self._stats: Dict[str, List[torch.Tensor]] = defaultdict(list)
        self._batch_count = 0

    def __enter__(self):
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                hook = module.register_forward_hook(
                    self._make_hook(name))
                self._hooks.append(hook)
                print(f"Hook registered for {name}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for hook in self._hooks:
            hook.remove()

    def _make_hook(self, name: str):
        def hook_fn(module, input, output):
            if self._batch_count >= self.num_batches:
                return
            x = input[0].detach()
            absmax = x.abs().amax()
            self._stats[name].append(absmax.cpu())
        return hook_fn

    def step(self):
        self._batch_count += 1
        print(f"Batch {self._batch_count}")

    def compute_scales(self) -> Dict[str, torch.Tensor]:
        scales = {}

        for name, absmax_list in self._stats.items():
            if not absmax_list:
                continue
            absmax = torch.stack(absmax_list).max()
            act_scale = compute_scale(
                torch.tensor([[absmax]]), FP8_MAX)
            scales[f'{name}.input_scale'] = act_scale

        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                w = module.weight.data.float()
                scales[f'{name}.weight_scale'] = compute_scale(w, FP8_MAX)

        logging.info(
            f'Computed calibration scales for {len(self._stats)} layers '
            f'over {self._batch_count} batches')
        return scales

    @property
    def is_complete(self) -> bool:
        return self._batch_count >= self.num_batches
