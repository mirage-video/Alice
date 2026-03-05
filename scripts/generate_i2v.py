import argparse
import logging
import os
import sys
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

import random

import torch
import torch.distributed as dist
from PIL import Image

import alice
from alice.configs import ALICE_CONFIGS, MAX_AREA_CONFIGS, SIZE_CONFIGS, SUPPORTED_SIZES
from alice.distributed.util import init_distributed_group
from alice.utils.prompt_extend import DashScopePromptExpander, QwenPromptExpander
from alice.utils.utils import save_video, str2bool


EXAMPLE_PROMPT = {
    "i2v-14b": {
        "prompt":
            "The scene slowly comes to life as the wind gently sways the trees in the background.",
        "image": None,
    },
}


def _validate_args(args):
    assert args.ckpt_dir is not None, "Please specify the checkpoint directory."
    assert args.task in ALICE_CONFIGS, f"Unsupport task: {args.task}"

    if args.input_image is None and args.task in EXAMPLE_PROMPT:
        if EXAMPLE_PROMPT[args.task].get("image") is not None:
            args.input_image = EXAMPLE_PROMPT[args.task]["image"]
        else:
            raise ValueError("--input_image is required for I2V generation.")

    if args.prompt is None:
        if args.task in EXAMPLE_PROMPT:
            args.prompt = EXAMPLE_PROMPT[args.task]["prompt"]
        else:
            args.prompt = ""

    cfg = ALICE_CONFIGS[args.task]

    if args.sample_steps is None:
        args.sample_steps = cfg.sample_steps

    if args.sample_shift is None:
        args.sample_shift = cfg.sample_shift

    if args.sample_guide_scale is None:
        args.sample_guide_scale = cfg.sample_guide_scale

    if args.frame_num is None:
        args.frame_num = cfg.frame_num

    args.base_seed = args.base_seed if args.base_seed >= 0 else random.randint(
        0, sys.maxsize)

    assert args.size in SUPPORTED_SIZES[
        args.
        task], f"Unsupport size {args.size} for task {args.task}, supported sizes are: {', '.join(SUPPORTED_SIZES[args.task])}"


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a video from an image and text prompt using Alice"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="i2v-14b",
        choices=list(ALICE_CONFIGS.keys()),
        help="The task to run.")

    args = parser.parse_args()
    _validate_args(args)

    return args


def _init_logging(rank):
    if rank == 0:
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] %(levelname)s: %(message)s",
            handlers=[logging.StreamHandler(stream=sys.stdout)])
    else:
        logging.basicConfig(level=logging.ERROR)


if __name__ == "__main__":
    args = _parse_args()
