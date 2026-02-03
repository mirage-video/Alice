from easydict import EasyDict

from .shared_config import alice_shared_cfg


t2v_14b = EasyDict(__name__='Config: Alice T2V 14B')
t2v_14b.update(alice_shared_cfg)

t2v_14b.t5_checkpoint = 'models_t5_umt5-xxl-enc-bf16.pth'
t2v_14b.t5_tokenizer = 'google/umt5-xxl'

t2v_14b.vae_checkpoint = 'mirage_vae.pth'
t2v_14b.vae_stride = (4, 8, 8)
