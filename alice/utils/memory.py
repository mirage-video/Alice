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
