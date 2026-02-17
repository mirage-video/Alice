import torch
import torch.distributed as dist


def init_distributed_group():
    """Initialize distributed process group."""
    if not dist.is_initialized():
        dist.init_process_group(backend='nccl')
