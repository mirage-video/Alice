import gc
import logging
import math
import os
import random
import sys
import types
from contextlib import contextmanager
from functools import partial

import torch
import torch.cuda.amp as amp
import torch.distributed as dist
from tqdm import tqdm

from ..distributed.fsdp import shard_model
from ..distributed.sequence_parallel import sp_attn_forward, sp_dit_forward
from ..distributed.util import get_world_size
from ..models.transformer import AliceTransformer
from ..models.text_encoder import T5EncoderModel
from ..models.vae import AliceVAE
from .scheduler_dpm import (
    FlowDPMSolverMultistepScheduler,
    get_sampling_sigmas,
    retrieve_timesteps,
)
from .scheduler_unipc import FlowUniPCMultistepScheduler
