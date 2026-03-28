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


COMPILE_DEFAULTS = {
    'mode': 'reduce-overhead',
    'fullgraph': False,
    'dynamic': True,
}
