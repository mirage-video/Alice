import torch
import torch.cuda.amp as amp

from ..models.transformer import sinusoidal_embedding_1d
from .util import gather_forward, get_rank, get_world_size
