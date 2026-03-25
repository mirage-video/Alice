import gc
import logging
from typing import Dict, Optional

import torch


def get_gpu_memory_info(device_id: int = 0) -> Dict[str, float]:
    if not torch.cuda.is_available():
        return {'total_mb': 0, 'used_mb': 0, 'free_mb': 0, 'utilization': 0}

    total = torch.cuda.get_device_properties(device_id).total_mem
    reserved = torch.cuda.memory_reserved(device_id)
    allocated = torch.cuda.memory_allocated(device_id)
    free = total - reserved

    return {
        'total_mb': total / 1024 / 1024,
        'reserved_mb': reserved / 1024 / 1024,
        'allocated_mb': allocated / 1024 / 1024,
        'free_mb': free / 1024 / 1024,
        'utilization': allocated / total if total > 0 else 0,
    }


def estimate_model_memory(
    model: torch.nn.Module,
    include_gradients: bool = False,
) -> Dict[str, float]:
    param_bytes = 0
    buffer_bytes = 0

    for p in model.parameters():
        param_bytes += p.nelement() * p.element_size()

    for b in model.buffers():
        buffer_bytes += b.nelement() * b.element_size()

    grad_bytes = 0
    if include_gradients:
        for p in model.parameters():
            if p.requires_grad:
                grad_bytes += p.nelement() * p.element_size()

    total = param_bytes + buffer_bytes + grad_bytes
    return {
        'params_mb': param_bytes / 1024 / 1024,
        'buffers_mb': buffer_bytes / 1024 / 1024,
        'gradients_mb': grad_bytes / 1024 / 1024,
        'total_mb': total / 1024 / 1024,
        'params_count': sum(p.nelement() for p in model.parameters()),
    }


def estimate_inference_memory(
    height: int,
    width: int,
    num_frames: int,
    model_params_b: float = 14.0,
    dtype: torch.dtype = torch.bfloat16,
    vae_stride: tuple = (4, 8, 8),
    patch_size: tuple = (1, 2, 2),
    z_dim: int = 16,
) -> Dict[str, float]:
    elem_size = torch.tensor([], dtype=dtype).element_size()

    model_mem = model_params_b * 1e9 * elem_size
    vae_mem = 0.1 * 1e9 * elem_size

    latent_t = (num_frames - 1) // vae_stride[0] + 1
    latent_h = height // vae_stride[1]
    latent_w = width // vae_stride[2]
    latent_mem = z_dim * latent_t * latent_h * latent_w * elem_size

    seq_len = (latent_t * latent_h * latent_w) // (patch_size[0] * patch_size[1] * patch_size[2])
    dim = 5120
    kv_mem = 2 * seq_len * dim * elem_size

    pixel_mem = 3 * num_frames * height * width * 4

    total = model_mem + vae_mem + latent_mem + kv_mem + pixel_mem
    return {
        'model_gb': model_mem / 1024**3,
        'vae_gb': vae_mem / 1024**3,
        'latent_gb': latent_mem / 1024**3,
        'kv_cache_gb': kv_mem / 1024**3,
        'pixel_output_gb': pixel_mem / 1024**3,
        'total_gb': total / 1024**3,
        'sequence_length': seq_len,
    }


class MemoryTracker:

    def __init__(self, device_id: int = 0):
        self.device_id = device_id
        self._snapshots = []
        self._peak = 0

    def snapshot(self, label: str = ''):
        if not torch.cuda.is_available():
            return
        allocated = torch.cuda.memory_allocated(self.device_id)
        self._peak = max(self._peak, allocated)
        self._snapshots.append({
            'label': label,
            'allocated_mb': allocated / 1024 / 1024,
            'peak_mb': self._peak / 1024 / 1024,
        })

    def report(self) -> list:
        return self._snapshots

    def reset(self):
        self._snapshots.clear()
        self._peak = 0
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device_id)
