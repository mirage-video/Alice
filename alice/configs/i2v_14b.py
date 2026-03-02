from easydict import EasyDict

from .shared_config import alice_shared_cfg


i2v_14b = EasyDict(__name__='Config: Alice I2V 14B')
i2v_14b.update(alice_shared_cfg)

i2v_14b.t5_checkpoint = 'models_t5_umt5-xxl-enc-bf16.pth'
i2v_14b.t5_tokenizer = 'google/umt5-xxl'

i2v_14b.vae_checkpoint = 'mirage_vae.pth'
i2v_14b.vae_stride = (4, 8, 8)

i2v_14b.clip_model = 'ViT-L/14'
i2v_14b.clip_checkpoint = 'models_clip_open-clip-vit-l-14.pth'
i2v_14b.clip_projection_dim = 5120

i2v_14b.patch_size = (1, 2, 2)
i2v_14b.dim = 5120
i2v_14b.ffn_dim = 13824
i2v_14b.freq_dim = 256
i2v_14b.num_heads = 40
i2v_14b.num_layers = 40
i2v_14b.window_size = (-1, -1)
i2v_14b.qk_norm = True
i2v_14b.cross_attn_norm = True
i2v_14b.eps = 1e-6
i2v_14b.low_noise_checkpoint = 'i2v_low_noise_model'
i2v_14b.high_noise_checkpoint = 'i2v_high_noise_model'

i2v_14b.sample_shift = 8.0
i2v_14b.sample_steps = 40
i2v_14b.boundary = 0.875
i2v_14b.sample_guide_scale = (3.0, 4.0)

i2v_14b.image_size = 224
i2v_14b.clip_pooling = 'patch'
