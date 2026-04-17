import pytest
import torch
import torch.nn as nn

from alice.quantization.utils import (
    compute_scale,
    per_tensor_quantize,
    per_channel_quantize,
    estimate_quantization_error,
    get_fp8_info,
)
from alice.quantization.fp8_linear import FP8Linear, quantize_model, dequantize_model
from alice.quantization.calibration import CalibrationContext


class TestQuantUtils:

    def test_compute_scale(self):
        tensor = torch.randn(128, 256)
        fp8_max = torch.finfo(torch.float8_e4m3fn).max
        scale = compute_scale(tensor, fp8_max)
        assert scale.dtype == torch.float32
        assert scale.item() > 0

    def test_per_tensor_quantize(self):
        tensor = torch.randn(64, 64)
        fp8_max = torch.finfo(torch.float8_e4m3fn).max
        scale = compute_scale(tensor, fp8_max)
        quantized = per_tensor_quantize(tensor, scale)
        assert quantized.dtype == torch.float8_e4m3fn
        assert quantized.shape == tensor.shape

    def test_per_channel_quantize(self):
        tensor = torch.randn(32, 64)
        quantized, scales = per_channel_quantize(tensor, axis=0)
        assert quantized.dtype == torch.float8_e4m3fn
        assert scales.shape == (32,)

    def test_estimate_quantization_error(self):
        tensor = torch.randn(128, 128)
        fp8_max = torch.finfo(torch.float8_e4m3fn).max
        scale = compute_scale(tensor, fp8_max)
        error = estimate_quantization_error(tensor, scale)
        assert 'mse' in error
        assert 'mae' in error
        assert 'snr_db' in error
        assert error['snr_db'] > 0

    def test_get_fp8_info(self):
        info = get_fp8_info()
        assert 'e4m3fn' in info
        assert 'e5m2' in info
        assert info['e4m3fn']['max'] > 0
