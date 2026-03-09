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


def per_tensor_quantize(
    tensor: torch.Tensor,
    scale: torch.Tensor,
    dtype: torch.dtype = torch.float8_e4m3fn,
) -> torch.Tensor:
    qmax = torch.finfo(dtype).max
    scaled = (tensor.float() / scale).clamp(-qmax, qmax)
    return scaled.to(dtype)
