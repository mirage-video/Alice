from easydict import EasyDict

from .shared_config import alice_shared_cfg


t2v_14b = EasyDict(__name__='Config: Alice T2V 14B')
t2v_14b.update(alice_shared_cfg)

t2v_14b.t5_checkpoint = 'models_t5_umt5-xxl-enc-bf16.pth'
t2v_14b.t5_tokenizer = 'google/umt5-xxl'

t2v_14b.vae_checkpoint = 'mirage_vae.pth'
t2v_14b.vae_stride = (4, 8, 8)

t2v_14b.dim = 5120
t2v_14b.ffn_dim = 13824
t2v_14b.freq_dim = 256
t2v_14b.patch_size = (1, 2, 2)
t2v_14b.num_heads = 40
t2v_14b.num_layers = 40
t2v_14b.window_size = (-1, -1)
t2v_14b.qk_norm = True
t2v_14b.cross_attn_norm = True
t2v_14b.eps = 1e-6
t2v_14b.low_noise_checkpoint = 'low_noise_model'
t2v_14b.high_noise_checkpoint = 'high_noise_model'

t2v_14b.sample_shift = 12.0
t2v_14b.sample_steps = 40
