import argparse
import logging
import os
import sys

import torch

from alice.quantization import quantize_model, FP8Linear
from alice.quantization.calibration import CalibrationContext
from alice.quantization.utils import get_fp8_info, estimate_quantization_error
from alice.models.transformer import AliceTransformer
from alice.utils.checkpoint import save_safetensors, load_checkpoint_auto


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Quantize Alice model weights to FP8 for efficient inference"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing the model checkpoint.")
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save the quantized model.")
    parser.add_argument(
        "--subfolder",
        type=str,
        default=None,
        help="Subfolder within input_dir for the model (e.g. low_noise_model).")
    parser.add_argument(
        "--skip_modules",
        type=str,
        nargs='*',
        default=None,
        help="Module names to skip during quantization.")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
