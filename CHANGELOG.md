# Changelog

All notable changes to the Alice project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-04-24

### Added
- Image-to-Video (I2V) generation pipeline with CLIP ViT-L/14 vision encoder
- FP8 quantization module for inference on consumer GPUs (RTX 4080/4090)
- VAE spatial tiling for memory-efficient encoding/decoding at high resolutions
- Streaming video decoder with temporal chunk blending
- torch.compile integration with dynamic shape guards and graph break analysis
- KV-cache implementations: dense, paged, and static variants
- LoRA adapter loading and weight merging for inference
- SafeTensors checkpoint support with sharding and auto-detection
- Batch inference script with prompt file support (JSON/JSONL/TXT)
- Inference benchmark script with memory estimation and profiling
- Checkpoint conversion utilities (to-safetensors, shard, merge, info)
- LoRA weight merging script
- FP8 model quantization script with calibration support
- GPU memory tracking and VRAM estimation utilities
- I2V CLI generation script
- Tests for quantization, I2V pipeline, and CLIP vision encoder

### Changed
- Updated `pyproject.toml` to version 0.2.0
- Added `safetensors`, `Pillow` to dependencies
- Updated `transformers` version constraint
- Expanded model `__init__.py` exports with new modules
- Cleaned up unused imports across model files

### Fixed
- Type annotations in attention module
- Minor cleanup in VAE cache handling comments


## [0.1.0] - 2026-02-25

### Added
- Initial release of Alice text-to-video generation
- 14B parameter Diffusion Transformer (DiT) backbone
- 3D Causal VAE with temporal caching
- UMT5-XXL text encoder integration
- Flash Attention 2/3 with SDPA fallback
- DPM++ and UniPC flow matching schedulers
- Two-stage inference (high-noise/low-noise expert models)
- Classifier-free guidance
- FSDP and sequence parallelism (Ulysses) for distributed inference
- Prompt enhancement via DashScope and local Qwen
- CLI generation script with multi-GPU support
- 720p and 480p resolution configurations
