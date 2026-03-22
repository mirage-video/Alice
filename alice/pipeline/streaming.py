import logging
import math
from typing import Iterator, List, Optional, Tuple

import torch
import torch.nn.functional as F

from ..models.vae import AliceVAE


__all__ = ['StreamingDecoder', 'ChunkedLatentIterator']


class ChunkedLatentIterator:

    def __init__(
        self,
        latents: torch.Tensor,
        chunk_size: int = 4,
        overlap: int = 1,
    ):
        self.latents = latents
        self.chunk_size = chunk_size
        self.overlap = overlap

        if latents.dim() == 4:
            self.latents = latents.unsqueeze(0)

        self.total_frames = self.latents.shape[2]

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, int, bool]]:
        stride = self.chunk_size - self.overlap
        pos = 0
        chunk_idx = 0

        while pos < self.total_frames:
            end = min(pos + self.chunk_size, self.total_frames)
            chunk = self.latents[:, :, pos:end]
            is_last = (end >= self.total_frames)

            yield chunk, chunk_idx, is_last

            if is_last:
                break
            pos += stride
            chunk_idx += 1
