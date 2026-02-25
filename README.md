<p align="center">
  <img src="assets/banner.jpeg" alt="Alice by Mirage" width="100%">
</p>

<h1 align="center">Alice</h1>

<p align="center">
  <b>Open-source text-to-video generation that surpasses closed-source quality</b><br>
  <i>14B parameters | 4-step inference | Fully open-source</i>
</p>

<p align="center">
  <a href="https://huggingface.co/gomirageai/Alice-T2V-14B-MoE"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue" alt="HuggingFace"></a>
  <a href="https://gomirage.ai"><img src="https://img.shields.io/badge/Website-gomirage.ai-green" alt="Website"></a>
  <a href="https://x.com/gomirageai"><img src="https://img.shields.io/badge/Twitter-@gomirageai-1DA1F2?logo=twitter" alt="Twitter"></a>
  <a href="#license"><img src="https://img.shields.io/badge/License-Apache%202.0-orange" alt="License"></a>
</p>

---

## Overview

Alice is a 14-billion parameter open-source text-to-video generation model developed by [Mirage](https://gomirage.ai). Through consistency distillation with score regularization (rCM), Alice achieves state-of-the-art quality while requiring only 4 denoising steps for inference—a 7× speedup over traditional 50-step diffusion models.

**Key Results:**
- Generates 5-second 720p videos at 24fps
- 4-step inference (vs. 50 steps for baseline models)
- Surpasses teacher model quality through distillation-enhanced training
- Fully open-source: weights, training code, and data pipelines

## Architecture

Alice builds on a Diffusion Transformer (DiT) architecture with the following specifications:

| Component | Specification |
|-----------|---------------|
| Parameters | 14B |
| Transformer blocks | 40 |
| Hidden dimension | 4096 |
| Attention heads | 32 (head dim = 128) |
| MLP dimension | 16384 (4× expansion) |
| Text encoder | umT5-XXL (5.7B params, frozen) |
| Video VAE | 3D Causal VAE (8× spatial, 4× temporal) |
| Latent channels | 16 |
| Patch size | 2×2 spatial |

The model employs a two-stage expert design for high-noise (structure) and low-noise (detail) denoising, enabling superior temporal consistency and physical plausibility.

## Installation

### Requirements

- Python >= 3.10
- CUDA >= 11.8
- PyTorch >= 2.4.0
- GPU: NVIDIA RTX 4090 (24GB VRAM) or higher recommended

### Setup

1. Clone the repository:
```bash
git clone https://github.com/mirage-video/alice.git
cd alice
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download model weights from HuggingFace:
```bash
# Using huggingface-cli
huggingface-cli download gomirageai/Alice-T2V-14B-MoE --local-dir ./checkpoints

# Or using Python
from huggingface_hub import snapshot_download
snapshot_download(repo_id="gomirageai/Alice-T2V-14B-MoE", local_dir="./checkpoints")
```

## Quick Start

### Basic Usage

```python
from alice import AliceTextToVideo

# Initialize the pipeline
pipeline = AliceTextToVideo(
    ckpt_dir="./checkpoints",
    device="cuda"
)

# Generate a video
video = pipeline.generate(
    prompt="A golden retriever running through a sunlit meadow, cinematic lighting, 4K",
    size="1280*720",
    frame_num=81,  # 5 seconds at 24fps
    seed=42
)

# Save the output
pipeline.save_video(video, "output.mp4")
```

### Command Line Interface

```bash
python scripts/generate.py \
    --task t2v-14b \
    --prompt "A warrior standing on a cliff overlooking a vast ocean at sunset" \
    --size 1280*720 \
    --frame_num 81 \
    --ckpt_dir ./checkpoints \
    --save_file output.mp4
```

### Advanced Options

```bash
python scripts/generate.py \
    --task t2v-14b \
    --prompt "Your prompt here" \
    --size 1280*720 \
    --frame_num 81 \
    --ckpt_dir ./checkpoints \
    --save_file output.mp4 \
    --sample_steps 4 \
    --sample_shift 8.0 \
    --sample_guide_scale 6.0 \
    --sample_solver unipc \
    --base_seed 42 \
    --offload_model True  # For lower VRAM GPUs
```

## Configuration

### Supported Resolutions

| Resolution | Aspect Ratio | Notes |
|------------|--------------|-------|
| 1280×720 | 16:9 | Recommended |
| 720×1280 | 9:16 | Portrait |
| 960×960 | 1:1 | Square |
| 1024×576 | 16:9 | Lower VRAM |
| 576×1024 | 9:16 | Portrait, lower VRAM |

### Prompt Enhancement

Alice supports automatic prompt enhancement for improved generation quality:

```bash
python scripts/generate.py \
    --prompt "A cat playing" \
    --use_prompt_extend True \
    --prompt_extend_method local_qwen \
    --prompt_extend_target_lang en
```

Methods available:
- `local_qwen`: Local Qwen model for prompt expansion
- `dashscope`: DashScope API (requires API key)

## Distributed Inference

For multi-GPU setups, Alice supports FSDP and sequence parallelism:

```bash
torchrun --nproc_per_node=4 scripts/generate.py \
    --task t2v-14b \
    --prompt "Your prompt here" \
    --ulysses_size 4 \
    --dit_fsdp True \
    --t5_fsdp True
```

## Method

Alice achieves state-of-the-art quality through three key innovations:

### 1. Score-Regularized Consistency Distillation (rCM)

Unlike conventional distillation that trades quality for speed, rCM combines:
- **Consistency enforcement**: Maps all points along the denoising trajectory to the same output
- **Score regularization**: Mode-seeking objective that concentrates on high-quality outputs

### 2. Synthetic Data Curation with Hard Example Mining

- 1M diverse prompts generated via GPT-4
- Quality filtering retains top 30% of teacher generations
- Hard example mining oversamples failure modes (physics, hands, faces)
- 70:30 synthetic-to-real data ratio for optimal distillation

### 3. Progressive Training Protocol

Four-stage curriculum:
1. **Stage 1**: Consistency foundation (480p, pure consistency loss)
2. **Stage 2**: Score regularization introduction (480p→720p)
3. **Stage 3**: Real data integration with perceptual losses
4. **Stage 4**: Human preference alignment via DPO

## Project Structure

```
alice/
├── alice/
│   ├── configs/          # Model configurations
│   ├── distributed/      # FSDP and sequence parallelism
│   ├── models/           # Core model components
│   │   ├── attention.py      # Flash attention implementation
│   │   ├── text_encoder.py   # T5 text encoder
│   │   ├── transformer.py    # DiT backbone
│   │   ├── vae.py           # 3D Causal VAE
│   │   └── vae22.py         # VAE variant
│   ├── pipeline/         # Generation pipeline
│   │   ├── generator.py      # Main inference class
│   │   ├── scheduler_dpm.py  # DPM++ scheduler
│   │   └── scheduler_unipc.py # UniPC scheduler
│   └── utils/            # Utilities and prompt enhancement
├── scripts/
│   └── generate.py       # CLI generation script
├── tests/                # Unit tests
└── configs/
    └── paths.yaml        # Checkpoint path configuration
```

## Citation

If you use Alice in your research, please cite:

```bibtex
@article{mirage2026alice,
  title={Alice: Distillation-Enhanced Open Video Generation That Surpasses Closed-Source Models},
  author={Mirage Team},
  journal={arXiv preprint},
  year={2026}
}
```

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE.txt) file for details.

## Acknowledgments

We thank the open-source community for foundational models and infrastructure that made this work possible, including:
- [HuggingFace](https://huggingface.co) for model hosting and diffusers library
- [Flash Attention](https://github.com/Dao-AILab/flash-attention) for efficient attention implementation

## Links

- **Model Weights**: [HuggingFace](https://huggingface.co/gomirageai/Alice-T2V-14B-MoE)
- **Website**: [gomirage.ai](https://gomirage.ai)
- **Twitter/X**: [@gomirageai](https://x.com/gomirageai)
- **Paper**: Coming soon

---

<p align="center">
  Built with dedication by the <a href="https://gomirage.ai">Mirage</a> team
</p>
