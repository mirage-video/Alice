import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .tokenizers import HuggingfaceTokenizer

__all__ = [
    'T5Model',
    'T5Encoder',
    'T5Decoder',
    'T5EncoderModel',
]


class T5LayerNorm(nn.Module):

    def __init__(self, dim, eps=1e-6):
        super(T5LayerNorm, self).__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        x = x * torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) +
                            self.eps)
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            x = x.type_as(self.weight)
        return self.weight * x
