import logging
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

from .vae import AliceVAE


__all__ = ['TiledVAE']


def _compute_tile_coords(
    total_size: int,
    tile_size: int,
    overlap: int,
) -> List[Tuple[int, int]]:
    coords = []
    stride = tile_size - overlap
    pos = 0
    while pos < total_size:
        end = min(pos + tile_size, total_size)
        if end == total_size and len(coords) > 0:
            pos = total_size - tile_size
            end = total_size
        coords.append((pos, end))
        if end == total_size:
            break
        pos += stride
    return coords


class TiledVAE:
    pass
