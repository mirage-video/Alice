import argparse
import gc
import logging
import os
import sys
import time
from datetime import datetime

import torch

from alice.configs import ALICE_CONFIGS, SIZE_CONFIGS
from alice.utils.memory import (
    get_gpu_memory_info,
    estimate_inference_memory,
    MemoryTracker,
    clear_gpu_memory,
)


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark Alice model inference performance"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="t2v-14b",
        choices=list(ALICE_CONFIGS.keys()),
        help="The task to benchmark.")
    parser.add_argument(
        "--size",
        type=str,
        default="1280*720",
        choices=list(SIZE_CONFIGS.keys()),
        help="Video resolution to benchmark.")
    parser.add_argument(
        "--frame_num",
        type=int,
        default=81,
        help="Number of frames.")
    parser.add_argument(
        "--num_warmup",
        type=int,
        default=2,
        help="Number of warmup iterations.")
    parser.add_argument(
        "--num_iterations",
        type=int,
        default=5,
        help="Number of benchmark iterations.")
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default=None,
        help="Checkpoint directory (required for full benchmark).")
    parser.add_argument(
        "--memory_only",
        action="store_true",
        default=False,
        help="Only estimate memory without running inference.")
    parser.add_argument(
        "--profile",
        action="store_true",
        default=False,
        help="Enable torch profiler for detailed breakdown.")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file for benchmark results (JSON).")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
