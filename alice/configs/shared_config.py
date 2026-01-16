import torch
from easydict import EasyDict

alice_shared_cfg = EasyDict()

alice_shared_cfg.param_dtype = torch.bfloat16

alice_shared_cfg.vae_stride = (4, 8, 8)
