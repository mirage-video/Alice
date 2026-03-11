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

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        weight_scale: Optional[torch.Tensor] = None,
        act_scale: Optional[torch.Tensor] = None,
    ) -> 'FP8Linear':
        has_bias = linear.bias is not None
        fp8 = cls(
            linear.in_features,
            linear.out_features,
            bias=has_bias,
            weight_scale=weight_scale,
        )

        w = linear.weight.data.float()
        if weight_scale is None:
            weight_scale = compute_scale(w, FP8_MAX)
            fp8.weight_scale.copy_(weight_scale)

        fp8.weight.copy_(per_tensor_quantize(w, weight_scale, FP8_DTYPE))

        if has_bias:
            fp8.bias.data.copy_(linear.bias.data)

        return fp8
