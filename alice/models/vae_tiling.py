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


def _create_blend_mask(
    tile_h: int,
    tile_w: int,
    overlap_h: int,
    overlap_w: int,
    is_top: bool,
    is_bottom: bool,
    is_left: bool,
    is_right: bool,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    mask = torch.ones(1, 1, 1, tile_h, tile_w, device=device, dtype=dtype)

    if not is_top and overlap_h > 0:
        ramp = torch.linspace(0, 1, overlap_h, device=device, dtype=dtype)
        mask[:, :, :, :overlap_h, :] *= ramp.view(1, 1, 1, -1, 1)

    if not is_bottom and overlap_h > 0:
        ramp = torch.linspace(0, 1, overlap_h, device=device, dtype=dtype)
        mask[:, :, :, -overlap_h:, :] *= ramp.view(1, 1, 1, -1, 1)

    if not is_left and overlap_w > 0:
        ramp = torch.linspace(0, 1, overlap_w, device=device, dtype=dtype)
        mask[:, :, :, :, :overlap_w] *= ramp.view(1, 1, 1, 1, -1)

    if not is_right and overlap_w > 0:
        ramp = torch.linspace(0, 1, overlap_w, device=device, dtype=dtype)
        mask[:, :, :, :, -overlap_w:] *= ramp.view(1, 1, 1, 1, -1)

    return mask


class TiledVAE:

    def __init__(
        self,
        vae: AliceVAE,
        tile_size: int = 256,
        tile_overlap: int = 32,
        temporal_tile_size: int = 17,
        temporal_overlap: int = 4,
    ):
        self.vae = vae
        self.tile_size = tile_size
        self.tile_overlap = tile_overlap
        self.temporal_tile_size = temporal_tile_size
        self.temporal_overlap = temporal_overlap
