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


def _load_prompts(prompt_file: str):
    prompts = []

    if prompt_file.endswith('.jsonl'):
        with open(prompt_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    item = json.loads(line)
                    if isinstance(item, str):
                        prompts.append({'prompt': item})
                    elif isinstance(item, dict):
                        prompts.append(item)
    elif prompt_file.endswith('.json'):
        with open(prompt_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, str):
                        prompts.append({'prompt': item})
                    elif isinstance(item, dict):
                        prompts.append(item)
            elif isinstance(data, dict) and 'prompts' in data:
                for item in data['prompts']:
                    if isinstance(item, str):
                        prompts.append({'prompt': item})
                    elif isinstance(item, dict):
                        prompts.append(item)
    elif prompt_file.endswith('.txt'):
        with open(prompt_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    prompts.append({'prompt': line})

    return prompts


def main():
    args = _parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        handlers=[logging.StreamHandler(stream=sys.stdout)])

    os.makedirs(args.output_dir, exist_ok=True)

    prompts = _load_prompts(args.prompt_file)
    logging.info(f"Loaded {len(prompts)} prompts from {args.prompt_file}")

    cfg = ALICE_CONFIGS[args.task]

    if args.sample_steps is None:
        args.sample_steps = cfg.sample_steps
    if args.sample_shift is None:
        args.sample_shift = cfg.sample_shift
    if args.sample_guide_scale is None:
        args.sample_guide_scale = cfg.sample_guide_scale
    if args.frame_num is None:
        args.frame_num = cfg.frame_num

    import alice
    logging.info("Creating pipeline...")
    pipeline = alice.AliceTextToVideo(
        config=cfg,
        checkpoint_dir=args.ckpt_dir,
        device_id=0,
        rank=0,
        convert_model_dtype=args.convert_model_dtype,
    )

    results = []
    total_time = 0

    for i, item in enumerate(prompts):
        prompt = item['prompt']
        seed = item.get('seed', args.base_seed + i)
        output_name = item.get('name', f'{i:05d}')
        output_path = os.path.join(args.output_dir, f'{output_name}.mp4')

        logging.info(f"[{i+1}/{len(prompts)}] Generating: {prompt[:80]}...")
        start = time.perf_counter()

        try:
            video = pipeline.generate(
                prompt,
                size=SIZE_CONFIGS[args.size],
                frame_num=args.frame_num,
                shift=args.sample_shift,
                sample_solver=args.sample_solver,
                sampling_steps=args.sample_steps,
                guide_scale=args.sample_guide_scale,
                seed=seed,
                offload_model=args.offload_model)

            elapsed = time.perf_counter() - start
            total_time += elapsed

            save_video(
                tensor=video[None],
                save_file=output_path,
                fps=cfg.sample_fps,
                nrow=1,
                normalize=True,
                value_range=(-1, 1))
            del video

            results.append({
                'name': output_name,
                'prompt': prompt,
                'seed': seed,
                'time_s': elapsed,
                'status': 'success',
                'output': output_path,
            })
            logging.info(f"  Saved to {output_path} ({elapsed:.1f}s)")

        except Exception as e:
            elapsed = time.perf_counter() - start
            results.append({
                'name': output_name,
                'prompt': prompt,
                'seed': seed,
                'time_s': elapsed,
                'status': 'failed',
                'error': str(e),
            })
            logging.error(f"  Failed: {e}")

        torch.cuda.empty_cache()

    logging.info("=" * 60)
    logging.info(f"Batch generation complete")


if __name__ == "__main__":
    main()
