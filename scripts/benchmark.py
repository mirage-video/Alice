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


def memory_estimate(args):
    cfg = ALICE_CONFIGS[args.task]
    h, w = SIZE_CONFIGS[args.size]

    estimates = estimate_inference_memory(
        height=h,
        width=w,
        num_frames=args.frame_num,
        model_params_b=14.0,
        dtype=cfg.param_dtype,
    )

    logging.info("=" * 60)
    logging.info("Memory Estimation")
    logging.info("=" * 60)
    logging.info(f"  Resolution: {w}x{h}, {args.frame_num} frames")
    logging.info(f"  Model: {estimates['model_gb']:.2f} GB")
    logging.info(f"  VAE: {estimates['vae_gb']:.2f} GB")
    logging.info(f"  Latents: {estimates['latent_gb']:.4f} GB")
    logging.info(f"  KV cache: {estimates['kv_cache_gb']:.4f} GB")
    logging.info(f"  Pixel output: {estimates['pixel_output_gb']:.4f} GB")
    logging.info(f"  Sequence length: {estimates['sequence_length']}")
    logging.info(f"  Total estimated: {estimates['total_gb']:.2f} GB")
    logging.info("=" * 60)

    if torch.cuda.is_available():
        gpu_info = get_gpu_memory_info()
        logging.info(f"  GPU total: {gpu_info['total_mb'] / 1024:.1f} GB")
        logging.info(f"  GPU free: {gpu_info['free_mb'] / 1024:.1f} GB")
        fits = gpu_info['free_mb'] / 1024 >= estimates['total_gb']
        logging.info(f"  Fits in VRAM: {'YES' if fits else 'NO (offloading needed)'}")

    return estimates


def main():
    args = _parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        handlers=[logging.StreamHandler(stream=sys.stdout)])

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        logging.info(f"GPU: {gpu_name}")

    if args.memory_only:
        memory_estimate(args)
    else:
        if args.ckpt_dir is None:
            logging.error("--ckpt_dir required for full benchmark. Use --memory_only for estimation.")
            sys.exit(1)
        memory_estimate(args)


if __name__ == "__main__":
    main()
