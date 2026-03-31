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


def compile_transformer(
    transformer: nn.Module,
    mode: str = 'reduce-overhead',
    compile_blocks: bool = True,
) -> nn.Module:
    if not _check_compile_available():
        return transformer

    if compile_blocks:
        for i, block in enumerate(transformer.blocks):
            transformer.blocks[i] = torch.compile(
                block, mode=mode, dynamic=True)
        logging.info(
            f'Compiled {len(transformer.blocks)} transformer blocks individually')
    else:
        transformer = torch.compile(
            transformer, mode=mode, fullgraph=False, dynamic=True)
        logging.info('Compiled transformer as single graph')

    return transformer


class DynamicShapeGuard:

    def __init__(self, max_cached_shapes: int = 8):
        self.max_cached_shapes = max_cached_shapes
        self._seen_shapes: Dict[str, List[tuple]] = {}

    def check(self, name: str, tensor: torch.Tensor) -> bool:
        shape = tuple(tensor.shape)
        if name not in self._seen_shapes:
            self._seen_shapes[name] = [shape]
            return True

        shapes = self._seen_shapes[name]
        if shape in shapes:
            return True

        if len(shapes) >= self.max_cached_shapes:
            logging.warning(
                f'DynamicShapeGuard: {name} has exceeded '
                f'{self.max_cached_shapes} unique shapes. '
                f'This may cause recompilation overhead.')
            return False

        shapes.append(shape)
        return True

    def report(self) -> Dict[str, int]:
        return {name: len(shapes) for name, shapes in self._seen_shapes.items()}

    def reset(self):
        self._seen_shapes.clear()


COMPILE_DEFAULTS = {
    'mode': 'reduce-overhead',
    'fullgraph': False,
    'dynamic': True,
}
