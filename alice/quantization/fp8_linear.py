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
    pass
