import logging
from collections import OrderedDict
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


__all__ = ['CLIPVisionEncoder']


class QuickGELU(nn.Module):

    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class CLIPAttention(nn.Module):

    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, c = x.shape
        q = self.q_proj(x).reshape(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).reshape(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(b, n, self.num_heads, self.head_dim).transpose(1, 2)

        attn = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_dropout.p if self.training else 0.0)
        attn = attn.transpose(1, 2).reshape(b, n, c)
        return self.out_proj(attn)


class CLIPEncoderLayer(nn.Module):

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.ln_1 = nn.LayerNorm(dim)
        self.attn = CLIPAttention(dim, num_heads, dropout)
        self.ln_2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(OrderedDict([
            ('c_fc', nn.Linear(dim, int(dim * mlp_ratio))),
            ('gelu', QuickGELU()),
            ('c_proj', nn.Linear(int(dim * mlp_ratio), dim)),
        ]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class CLIPVisionTransformer(nn.Module):

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 14,
        dim: int = 1024,
        num_layers: int = 24,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.dim = dim
        self.num_patches = (image_size // patch_size) ** 2

        self.conv1 = nn.Conv2d(
            3, dim, kernel_size=patch_size, stride=patch_size, bias=False)
        self.class_embedding = nn.Parameter(torch.randn(dim) * 0.02)
        self.positional_embedding = nn.Parameter(
            torch.randn(self.num_patches + 1, dim) * 0.01)
        self.ln_pre = nn.LayerNorm(dim)
        self.layers = nn.Sequential(*[
            CLIPEncoderLayer(dim, num_heads, mlp_ratio, dropout)
            for _ in range(num_layers)
        ])
        self.ln_post = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = x.flatten(2).transpose(1, 2)

        cls_token = self.class_embedding.unsqueeze(0).unsqueeze(0).expand(x.shape[0], -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        x = x + self.positional_embedding.unsqueeze(0)

        x = self.ln_pre(x)
        x = self.layers(x)
        x = self.ln_post(x)

        return x


class CLIPImageProcessor:

    def __init__(self, image_size: int = 224):
        self.image_size = image_size
        self.mean = torch.tensor([0.48145466, 0.4578275, 0.40821073])
        self.std = torch.tensor([0.26862954, 0.26130258, 0.27577711])

    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim == 3:
            images = images.unsqueeze(0)
        images = F.interpolate(
            images, size=(self.image_size, self.image_size),
            mode='bicubic', align_corners=False, antialias=True)
        images = images.clamp(0, 1)
        mean = self.mean.to(images.device).view(1, 3, 1, 1)
        std = self.std.to(images.device).view(1, 3, 1, 1)
        images = (images - mean) / std
        return images


CLIP_CONFIGS = {
    'ViT-L/14': dict(
        image_size=224, patch_size=14, dim=1024,
        num_layers=24, num_heads=16, mlp_ratio=4.0,
    ),
    'ViT-H/14': dict(
        image_size=224, patch_size=14, dim=1280,
        num_layers=32, num_heads=16, mlp_ratio=4.0,
    ),
    'ViT-bigG/14': dict(
        image_size=224, patch_size=14, dim=1664,
        num_layers=48, num_heads=16, mlp_ratio=4.9231,
    ),
}


class CLIPVisionEncoder:

    def __init__(
        self,
        model_name: str = 'ViT-L/14',
        pretrained_path: Optional[str] = None,
        projection_dim: int = 5120,
        dtype: torch.dtype = torch.float16,
        device: str = 'cuda',
    ):
        self.dtype = dtype
        self.device = device
        self.model_name = model_name

        cfg = CLIP_CONFIGS[model_name]
        self.processor = CLIPImageProcessor(cfg['image_size'])

        with torch.device('meta'):
            self.model = CLIPVisionTransformer(**cfg)

        self.projection = nn.Linear(cfg['dim'], projection_dim, bias=False)

        if pretrained_path is not None:
            logging.info(f'Loading CLIP vision encoder from {pretrained_path}')
            state_dict = torch.load(pretrained_path, map_location='cpu')
            if 'model' in state_dict:
                state_dict = state_dict['model']
            self.model.load_state_dict(
                {k.replace('visual.', ''): v for k, v in state_dict.items()
                 if k.startswith('visual.')},
                assign=True)
            if 'projection' in state_dict:
                self.projection.load_state_dict(
                    {'weight': state_dict['projection']}, assign=True)

        self.model.eval().requires_grad_(False).to(dtype).to(device)
        self.projection.eval().requires_grad_(False).to(dtype).to(device)

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        images = self.processor(images).to(self.dtype).to(self.device)
        features = self.model(images)
        cls_feature = features[:, 0]
        patch_features = features[:, 1:]
        projected = self.projection(patch_features)
        return projected, cls_feature

    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        return self.encode_image(images)
