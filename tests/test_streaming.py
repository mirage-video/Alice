import pytest
import torch

from alice.pipeline.streaming import ChunkedLatentIterator
from alice.models.vae_tiling import _compute_tile_coords, _create_blend_mask


class TestChunkedLatentIterator:

    def test_single_chunk(self):
        latents = torch.randn(1, 16, 4, 10, 18)
        iterator = ChunkedLatentIterator(latents, chunk_size=8, overlap=1)
        assert len(iterator) == 1

    def test_multiple_chunks(self):
        latents = torch.randn(1, 16, 20, 10, 18)
        iterator = ChunkedLatentIterator(latents, chunk_size=8, overlap=2)
        assert len(iterator) > 1
        chunks = list(iterator)
        assert chunks[-1][2] is True

    def test_4d_input(self):
        latents = torch.randn(16, 8, 10, 18)
        iterator = ChunkedLatentIterator(latents, chunk_size=4, overlap=1)
        assert iterator.latents.dim() == 5

    def test_overlap_coverage(self):
        latents = torch.randn(1, 16, 12, 10, 18)
        iterator = ChunkedLatentIterator(latents, chunk_size=6, overlap=2)
        frames_seen = set()
        for chunk, idx, is_last in iterator:
            start = idx * (6 - 2)
            for f in range(chunk.shape[2]):
                frames_seen.add(start + f)
        assert len(frames_seen) == 12


class TestComputeTileCoords:

    def test_single_tile(self):
        coords = _compute_tile_coords(64, 128, 16)
        assert coords == [(0, 64)]

    def test_exact_fit(self):
        coords = _compute_tile_coords(256, 256, 32)
        assert coords == [(0, 256)]

    def test_multiple_tiles(self):
        coords = _compute_tile_coords(512, 256, 32)
        assert len(coords) >= 2
        assert coords[0][0] == 0
        assert coords[-1][1] == 512


class TestBlendMask:

    def test_shape(self):
        mask = _create_blend_mask(
            tile_h=32, tile_w=32,
            overlap_h=4, overlap_w=4,
            is_top=True, is_bottom=False,
            is_left=True, is_right=False,
            device=torch.device('cpu'),
            dtype=torch.float32)
        assert mask.shape == (1, 1, 1, 32, 32)

    def test_boundary_ones(self):
        mask = _create_blend_mask(
            tile_h=16, tile_w=16,
            overlap_h=4, overlap_w=4,
            is_top=True, is_bottom=True,
            is_left=True, is_right=True,
            device=torch.device('cpu'),
            dtype=torch.float32)
        assert mask.min().item() == 1.0
