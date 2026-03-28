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


COMPILE_DEFAULTS = {
    'mode': 'reduce-overhead',
    'fullgraph': False,
    'dynamic': True,
}
