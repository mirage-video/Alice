from typing import Optional

import torch


def compute_scale(
    tensor: torch.Tensor,
    qmax: float,
    min_scale: float = 1e-12,
) -> torch.Tensor:
    absmax = tensor.abs().amax().float()
    scale = absmax / qmax
    return scale
