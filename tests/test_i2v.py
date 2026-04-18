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


class TestCLIPImageProcessor:

    def test_single_image(self):
        processor = CLIPImageProcessor(image_size=224)
        img = torch.rand(1, 3, 512, 512)
        out = processor(img)
        assert out.shape == (1, 3, 224, 224)

    def test_batch(self):
        processor = CLIPImageProcessor(image_size=224)
        img = torch.rand(4, 3, 256, 256)
        out = processor(img)
        assert out.shape == (4, 3, 224, 224)

    def test_3d_input(self):
        processor = CLIPImageProcessor(image_size=224)
        img = torch.rand(3, 128, 128)
        out = processor(img)
        assert out.shape == (1, 3, 224, 224)


class TestCLIPConfigs:

    def test_configs_exist(self):
        assert 'ViT-L/14' in CLIP_CONFIGS
        assert 'ViT-H/14' in CLIP_CONFIGS

    def test_vit_l_config(self):
        cfg = CLIP_CONFIGS['ViT-L/14']
        assert cfg['dim'] == 1024
        assert cfg['num_layers'] == 24
        assert cfg['patch_size'] == 14


class TestI2VConfig:

    def test_i2v_config_exists(self):
        assert 'i2v-14b' in ALICE_CONFIGS

    def test_i2v_has_clip_fields(self):
        cfg = ALICE_CONFIGS['i2v-14b']
        assert hasattr(cfg, 'clip_model')
        assert hasattr(cfg, 'clip_checkpoint')
        assert hasattr(cfg, 'clip_projection_dim')

    def test_i2v_supported_sizes(self):
        assert 'i2v-14b' in SUPPORTED_SIZES
