import torch
from easydict import EasyDict

alice_shared_cfg = EasyDict()

alice_shared_cfg.t5_model = 'umt5_xxl'
alice_shared_cfg.t5_dtype = torch.bfloat16
alice_shared_cfg.text_len = 512

alice_shared_cfg.param_dtype = torch.bfloat16

alice_shared_cfg.vae_stride = (4, 8, 8)
