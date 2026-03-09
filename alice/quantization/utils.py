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


def per_channel_quantize(
    tensor: torch.Tensor,
    axis: int = 0,
    dtype: torch.dtype = torch.float8_e4m3fn,
) -> tuple:
    qmax = torch.finfo(dtype).max
    shape = [1] * tensor.ndim
    shape[axis] = tensor.size(axis)

    absmax = tensor.abs().amax(
        dim=[d for d in range(tensor.ndim) if d != axis],
        keepdim=True)
    scales = (absmax / qmax).clamp(min=1e-12)
    quantized = (tensor.float() / scales).clamp(-qmax, qmax).to(dtype)

    return quantized, scales.reshape(-1)
