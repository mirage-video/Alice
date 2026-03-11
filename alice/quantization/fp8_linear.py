import logging
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import compute_scale, per_tensor_quantize


__all__ = ['FP8Linear', 'quantize_model', 'dequantize_model']

FP8_DTYPE = torch.float8_e4m3fn
FP8_MAX = torch.finfo(torch.float8_e4m3fn).max


class FP8Linear(nn.Module):

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        weight_scale: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.register_buffer(
            'weight', torch.zeros(out_features, in_features, dtype=FP8_DTYPE))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)

        if weight_scale is not None:
            self.register_buffer('weight_scale', weight_scale)
        else:
            self.register_buffer(
                'weight_scale', torch.ones(1, dtype=torch.float32))
