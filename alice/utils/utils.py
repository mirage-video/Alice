import argparse
import logging
import os
import os.path as osp

import imageio
import torch
import torchvision

__all__ = ['save_video', 'save_image', 'str2bool']


def save_video(tensor,
               save_file=None,
               fps=30,
               suffix='.mp4',
               nrow=8,
               normalize=True,
               value_range=(-1, 1)):

    try:
        tensor = tensor.clamp(min(value_range), max(value_range))
        tensor = torch.stack([
            torchvision.utils.make_grid(
                u, nrow=nrow, normalize=normalize, value_range=value_range)
            for u in tensor.unbind(2)
        ],
                             dim=1).permute(1, 2, 3, 0)
        tensor = (tensor * 255).type(torch.uint8).cpu()

        writer = imageio.get_writer(
            save_file, fps=fps, codec='libx264', quality=8)
        for frame in tensor.numpy():
            writer.append_data(frame)
        writer.close()
    except Exception as e:
        logging.info(f'save_video failed, error: {e}')


def save_image(tensor, save_file, nrow=8, normalize=True, value_range=(-1, 1)):
    suffix = osp.splitext(save_file)[1]
    if suffix.lower() not in [
            '.jpg', '.jpeg', '.png', '.tiff', '.gif', '.webp'
    ]:
        suffix = '.png'

    try:
        tensor = tensor.clamp(min(value_range), max(value_range))
        torchvision.utils.save_image(
            tensor,
            save_file,
            nrow=nrow,
            normalize=normalize,
            value_range=value_range)
        return save_file
    except Exception as e:
        logging.info(f'save_image failed, error: {e}')


def str2bool(v):
    """Convert string to boolean for argparse."""
    if isinstance(v, bool):
        return v
    v_lower = v.lower()
    if v_lower in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v_lower in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected (True/False)')
