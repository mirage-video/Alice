import functools
import logging
import time
from typing import Any, Callable, Dict, List, Optional

import torch
import torch.nn as nn


__all__ = [
    'compile_model',
    'compile_vae',
    'compile_transformer',
    'DynamicShapeGuard',
    'GraphBreakAnalyzer',
]


def _check_compile_available() -> bool:
    if not hasattr(torch, 'compile'):
        logging.warning('torch.compile not available (requires PyTorch >= 2.0)')
        return False
    return True


def compile_model(
    model: nn.Module,
    mode: str = 'reduce-overhead',
    fullgraph: bool = False,
    dynamic: bool = True,
    backend: str = 'inductor',
    disable: bool = False,
) -> nn.Module:
    if disable or not _check_compile_available():
        return model

    logging.info(
        f'Compiling model with mode={mode}, backend={backend}, '
        f'fullgraph={fullgraph}, dynamic={dynamic}')

    compiled = torch.compile(
        model,
        mode=mode,
        fullgraph=fullgraph,
        dynamic=dynamic,
        backend=backend,
    )
    return compiled


def compile_vae(
    vae_model: nn.Module,
    mode: str = 'reduce-overhead',
) -> nn.Module:
    if not _check_compile_available():
        return vae_model

    vae_model.encoder = torch.compile(
        vae_model.encoder, mode=mode, dynamic=True)
    vae_model.decoder = torch.compile(
        vae_model.decoder, mode=mode, dynamic=True)

    logging.info('Compiled VAE encoder and decoder separately')
    return vae_model


COMPILE_DEFAULTS = {
    'mode': 'reduce-overhead',
    'fullgraph': False,
    'dynamic': True,
}
