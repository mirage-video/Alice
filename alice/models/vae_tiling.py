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

    def encode_tiled(self, video: torch.Tensor) -> torch.Tensor:
        _, _, t, h, w = video.shape
        vae_stride = (4, 8, 8)

        tile_h = self.tile_size
        tile_w = self.tile_size
        overlap_h = self.tile_overlap
        overlap_w = self.tile_overlap

        h_coords = _compute_tile_coords(h, tile_h, overlap_h)
        w_coords = _compute_tile_coords(w, tile_w, overlap_w)

        latent_h = h // vae_stride[1]
        latent_w = w // vae_stride[2]
        latent_t = (t - 1) // vae_stride[0] + 1
        z_dim = self.vae.model.z_dim

        result = torch.zeros(
            1, z_dim, latent_t, latent_h, latent_w,
            device=video.device, dtype=video.dtype)
        weight = torch.zeros(
            1, 1, latent_t, latent_h, latent_w,
            device=video.device, dtype=video.dtype)

        for i, (h_start, h_end) in enumerate(h_coords):
            for j, (w_start, w_end) in enumerate(w_coords):
                tile = video[:, :, :, h_start:h_end, w_start:w_end]
                tile_latent = self.vae.encode([tile.squeeze(0)])[0]
                tile_latent = tile_latent.unsqueeze(0)

                lat_h_start = h_start // vae_stride[1]
                lat_h_end = h_end // vae_stride[1]
                lat_w_start = w_start // vae_stride[2]
                lat_w_end = w_end // vae_stride[2]

                lat_tile_h = lat_h_end - lat_h_start
                lat_tile_w = lat_w_end - lat_w_start
                lat_overlap_h = overlap_h // vae_stride[1]
                lat_overlap_w = overlap_w // vae_stride[2]

                mask = _create_blend_mask(
                    lat_tile_h, lat_tile_w,
                    lat_overlap_h, lat_overlap_w,
                    is_top=(i == 0),
                    is_bottom=(i == len(h_coords) - 1),
                    is_left=(j == 0),
                    is_right=(j == len(w_coords) - 1),
                    device=video.device,
                    dtype=video.dtype)

                result[:, :, :, lat_h_start:lat_h_end, lat_w_start:lat_w_end] += (
                    tile_latent * mask)
                weight[:, :, :, lat_h_start:lat_h_end, lat_w_start:lat_w_end] += mask

        result = result / weight.clamp(min=1e-8)
        return result.squeeze(0)
