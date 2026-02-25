import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

PATHS_CONFIG = Path(__file__).parent.parent / "configs" / "paths.yaml"


def _load_weights_path():
    with open(PATHS_CONFIG) as f:
        config = yaml.safe_load(f)
    return Path(config["weights"]["base_path"])


WEIGHTS_PATH = _load_weights_path()


def test_paths_yaml_exists():
    assert PATHS_CONFIG.exists(), f"Paths config not found: {PATHS_CONFIG}"


def test_paths_yaml_structure():
    with open(PATHS_CONFIG) as f:
        config = yaml.safe_load(f)

    assert "weights" in config, "Missing 'weights' key in paths.yaml"

    required_keys = [
        "base_path",
        "transformer_low_noise",
        "transformer_high_noise",
        "vae",
        "text_encoder",
        "tokenizer",
    ]

    for key in required_keys:
        assert key in config["weights"], f"Missing key in weights config: {key}"


def test_t2v_config_structure():
    from alice.configs import ALICE_CONFIGS
    cfg = ALICE_CONFIGS['t2v-14b']

    required_keys = [
        'dim', 'ffn_dim', 'freq_dim', 'num_heads', 'num_layers',
        'patch_size', 'vae_stride', 'sample_steps', 'sample_shift',
        't5_checkpoint', 'vae_checkpoint',
        'low_noise_checkpoint', 'high_noise_checkpoint'
    ]

    for key in required_keys:
        assert hasattr(cfg, key), f"Missing config key: {key}"


def test_weights_directory_exists():
    assert WEIGHTS_PATH.exists(), f"Weights directory not found: {WEIGHTS_PATH}"


def test_weights_subdirectories():
    required_dirs = ['low_noise_model', 'high_noise_model']
    for subdir in required_dirs:
        path = WEIGHTS_PATH / subdir
        assert path.exists(), f"Missing weights subdirectory: {path}"


def test_weights_files():
    required_files = [
        'models_t5_umt5-xxl-enc-bf16.pth',
    ]
    for filename in required_files:
        path = WEIGHTS_PATH / filename
        assert path.exists(), f"Missing weights file: {path}"


def test_vae_weights():
    vae_path = WEIGHTS_PATH / 'mirage_vae.pth'
    assert vae_path.exists(), f"VAE weights not found: {vae_path}"


def test_transformer_shards():
    for model_dir in ['low_noise_model', 'high_noise_model']:
        model_path = WEIGHTS_PATH / model_dir
        safetensor_files = list(model_path.glob('*.safetensors'))
        assert len(safetensor_files) == 6, f"Expected 6 safetensor shards in {model_dir}, found {len(safetensor_files)}"
