import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime

import torch

from alice.configs import ALICE_CONFIGS, SIZE_CONFIGS
from alice.utils.utils import save_video, str2bool


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Batch video generation from a prompt file"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="t2v-14b",
        choices=list(ALICE_CONFIGS.keys()),
        help="The task to run.")
    parser.add_argument(
        "--prompt_file",
        type=str,
        required=True,
        help="Path to JSON/JSONL file with prompts.")
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save generated videos.")
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        required=True,
        help="The path to the checkpoint directory.")
    parser.add_argument(
        "--size",
        type=str,
        default="1280*720",
        choices=list(SIZE_CONFIGS.keys()),
        help="Video resolution.")
    parser.add_argument(
        "--frame_num",
        type=int,
        default=None,
        help="Number of frames.")
    parser.add_argument(
        "--sample_solver",
        type=str,
        default='unipc',
        choices=['unipc', 'dpm++'],
        help="The solver used to sample.")
    parser.add_argument(
        "--sample_steps", type=int, default=None, help="Sampling steps.")
    parser.add_argument(
        "--sample_shift", type=float, default=None, help="Shift factor.")
    parser.add_argument(
        "--sample_guide_scale", type=float, default=None, help="CFG scale.")
    parser.add_argument(
        "--offload_model",
        type=str2bool,
        default=True,
        help="CPU offloading.")
    parser.add_argument(
        "--base_seed",
        type=int,
        default=42,
        help="Base seed (incremented per prompt).")
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        default=False,
        help="Skip prompts whose output files already exist.")
    parser.add_argument(
        "--convert_model_dtype",
        action="store_true",
        default=False,
        help="Convert model dtype.")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
