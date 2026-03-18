from typing import Optional

import torch


def compute_scale(
    tensor: torch.Tensor,
    qmax: float,
    min_scale: float = 1e-12,
) -> torch.Tensor:
    absmax = tensor.abs().amax().float()
    scale = absmax / qmax
    scale = scale.clamp(min=min_scale)
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


def estimate_quantization_error(
    tensor: torch.Tensor,
    scale: torch.Tensor,
    dtype: torch.dtype = torch.float8_e4m3fn,
) -> dict:
    quantized = per_tensor_quantize(tensor, scale, dtype)
    dequantized = quantized.float() * scale
    error = (tensor.float() - dequantized)

    return {
        'mse': error.pow(2).mean().item(),
        'mae': error.abs().mean().item(),
        'max_error': error.abs().max().item(),
        'snr_db': (10 * torch.log10(
            tensor.float().pow(2).mean() / error.pow(2).mean()
        )).item(),
    }


def get_fp8_info() -> dict:
    e4m3 = torch.finfo(torch.float8_e4m3fn)
    e5m2 = torch.finfo(torch.float8_e5m2)
    return {
        'e4m3fn': {
            'max': e4m3.max,
            'min': e4m3.min,
            'smallest_normal': e4m3.tiny,
        },
        'e5m2': {
            'max': e5m2.max,
            'min': e5m2.min,
            'smallest_normal': e5m2.tiny,
        },
    }
