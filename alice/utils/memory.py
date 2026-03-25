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
