import pytest
import torch

from alice.configs import ALICE_CONFIGS, SUPPORTED_SIZES
from alice.models.clip_vision import (
    CLIPVisionTransformer,
    CLIPImageProcessor,
    CLIPVisionEncoder,
    CLIPAttention,
    CLIPEncoderLayer,
    CLIP_CONFIGS,
    QuickGELU,
)


class TestQuickGELU:

    def test_forward(self):
        gelu = QuickGELU()
        x = torch.randn(2, 64)
        out = gelu(x)
        assert out.shape == x.shape

    def test_approximation(self):
        gelu = QuickGELU()
        x = torch.zeros(1)
        out = gelu(x)
        assert out.item() == 0.0


class TestCLIPAttention:

    def test_forward(self):
        attn = CLIPAttention(dim=256, num_heads=8)
        x = torch.randn(2, 16, 256)
        out = attn(x)
        assert out.shape == (2, 16, 256)


class TestCLIPEncoderLayer:

    def test_forward(self):
        layer = CLIPEncoderLayer(dim=256, num_heads=8)
        x = torch.randn(2, 16, 256)
        out = layer(x)
        assert out.shape == (2, 16, 256)


class TestCLIPVisionTransformer:

    def test_forward_small(self):
        model = CLIPVisionTransformer(
            image_size=56, patch_size=14, dim=128,
            num_layers=2, num_heads=4, mlp_ratio=4.0)
        x = torch.randn(1, 3, 56, 56)
        out = model(x)
        assert out.shape[0] == 1
        assert out.shape[2] == 128
        num_patches = (56 // 14) ** 2
        assert out.shape[1] == num_patches + 1
