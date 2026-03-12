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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype

        w_dequant = self.weight.float() * self.weight_scale

        out = F.linear(x.float(), w_dequant, self.bias)
        return out.to(orig_dtype)


def quantize_model(
    model: nn.Module,
    calibration_scales: Optional[Dict[str, torch.Tensor]] = None,
    skip_modules: Optional[list] = None,
) -> nn.Module:
    skip_modules = skip_modules or []
    replaced = 0

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and name not in skip_modules:
            parent_name = '.'.join(name.split('.')[:-1])
            child_name = name.split('.')[-1]

            parent = model
            if parent_name:
                for part in parent_name.split('.'):
                    parent = getattr(parent, part)

            w_scale = None
            a_scale = None
            if calibration_scales is not None:
                w_scale = calibration_scales.get(f'{name}.weight_scale')
                a_scale = calibration_scales.get(f'{name}.input_scale')

            fp8_linear = FP8Linear.from_linear(
                module, weight_scale=w_scale, act_scale=a_scale)
            setattr(parent, child_name, fp8_linear)
            replaced += 1

    logging.info(f'Quantized {replaced} linear layers to FP8')
    return model


def dequantize_model(model: nn.Module) -> nn.Module:
    replaced = 0

    for name, module in model.named_modules():
        if isinstance(module, FP8Linear):
            parent_name = '.'.join(name.split('.')[:-1])
            child_name = name.split('.')[-1]

            parent = model
            if parent_name:
                for part in parent_name.split('.'):
                    parent = getattr(parent, part)

            linear = nn.Linear(
                module.in_features,
                module.out_features,
                bias=module.bias is not None)
            linear.weight.data.copy_(
                module.weight.float() * module.weight_scale)
            if module.bias is not None:
                linear.bias.data.copy_(module.bias.data)
            setattr(parent, child_name, linear)
            replaced += 1

    logging.info(f'Dequantized {replaced} FP8 layers back to float')
    return model
