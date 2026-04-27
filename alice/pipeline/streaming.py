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

    def __len__(self) -> int:
        if self.total_frames <= self.chunk_size:
            return 1
        stride = self.chunk_size - self.overlap
        return math.ceil((self.total_frames - self.chunk_size) / stride) + 1


class StreamingDecoder:

    def __init__(
        self,
        vae: AliceVAE,
        chunk_size: int = 8,
        overlap: int = 2,
        blend_frames: int = 4,
    ):
        self.vae = vae
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.blend_frames = blend_frames
        self._temporal_stride = 4

    def decode_streaming(
        self,
        latents: torch.Tensor,
        callback: Optional[callable] = None,
    ) -> torch.Tensor:
        iterator = ChunkedLatentIterator(
            latents,
            chunk_size=self.chunk_size,
            overlap=self.overlap)

        chunks = []
        prev_tail = None

        for chunk_latent, idx, is_last in iterator:
            decoded = self.vae.decode([chunk_latent.squeeze(0)])[0]

            if prev_tail is not None:
                blend_t = self.blend_frames
                actual_blend = min(blend_t, decoded.shape[1], prev_tail.shape[1])

                if actual_blend > 0:
                    alpha = torch.linspace(
                        0, 1, actual_blend,
                        device=decoded.device, dtype=decoded.dtype
                    ).view(1, -1, 1, 1)

                    blended = (prev_tail[:, -actual_blend:] * (1 - alpha) +
                               decoded[:, :actual_blend] * alpha)
                    decoded = torch.cat([blended, decoded[:, actual_blend:]], dim=1)

                chunks.append(decoded)
            else:
                chunks.append(decoded)

            overlap_pixel_frames = self.overlap * self._temporal_stride
            if not is_last and decoded.shape[1] > overlap_pixel_frames:
                prev_tail = decoded[:, -overlap_pixel_frames:]
            else:
                prev_tail = None

            if callback is not None:
                callback(idx, decoded, is_last)

        if len(chunks) == 1:
            return chunks[0]

        result_parts = [chunks[0]]
        for i in range(1, len(chunks)):
            trim = self.blend_frames if i < len(chunks) - 1 else 0
            if trim > 0 and chunks[i].shape[1] > trim:
                result_parts.append(chunks[i][:, trim:])
            else:
                result_parts.append(chunks[i])

        return torch.cat(result_parts, dim=1).clamp(-1, 1)

    def estimate_chunk_memory(
        self,
        height: int,
        width: int,
        dtype: torch.dtype = torch.bfloat16,
    ) -> dict:
        elem_size = torch.tensor([], dtype=dtype).element_size()
        vae_stride = (4, 8, 8)
        pixel_frames = (self.chunk_size - 1) * vae_stride[0] + 1

        latent_mem = (16 * self.chunk_size *
                      (height // vae_stride[1]) *
                      (width // vae_stride[2]) * elem_size)
        pixel_mem = 3 * pixel_frames * height * width * elem_size

        return {
            'latent_chunk_mb': latent_mem / 1024 / 1024,
            'pixel_chunk_mb': pixel_mem / 1024 / 1024,
            'total_chunk_mb': (latent_mem + pixel_mem) / 1024 / 1024,
            'pixel_frames_per_chunk': pixel_frames,
        }
