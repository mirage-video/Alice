import torch
import torch.distributed as dist

from ..models.attention import flash_attention
from .util import all_to_all


def distributed_attention(
        q,
        k,
        v,
        seq_lens,
        window_size=(-1, -1),
):
    """Ulysses distributed attention with all-to-all communication."""
    if not dist.is_initialized():
        raise ValueError("distributed group should be initialized.")
    b = q.shape[0]

    q = all_to_all(q, scatter_dim=2, gather_dim=1)
    k = all_to_all(k, scatter_dim=2, gather_dim=1)
    v = all_to_all(v, scatter_dim=2, gather_dim=1)

    x = flash_attention(
        q,
        k,
        v,
        k_lens=seq_lens,
        window_size=window_size,
    )

    x = all_to_all(x, scatter_dim=1, gather_dim=2)
    return x
