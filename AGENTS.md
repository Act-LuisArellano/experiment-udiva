# AGENTS.md file

## Project context

This project aims to standardize experimentation across video-based machine learning tasks. It operates on a shared dataset of videos, applying a consistent set of tasks to predefined data partitions while evaluating multiple models under identical conditions to ensure fair comparisons.

The system is designed to be modular and extensible:
* **Tasks** (e.g., video recognition, label prediction/anticipation, object recognition, causal graph inference, VQA) can be added as independent modules.
* **Models** (e.g., Gemma, Qwen) can be integrated with minimal effort.
The core goal is to decouple *tasks* from *models*, enabling systematic benchmarking across combinations of both.

Development follows **Test-Driven Development (TDD)**:
* Every new feature must be introduced with corresponding unit tests.
* End-to-end (E2E) tests are used to validate complete experimental pipelines.

### Current implementation
* Simple video chunk classification (inference only, no training).

### Planned features
* VQA (inference only), with prompt configuration defined via YAML settings.
* Basic model training support.

## Dev environment tips
- Use `uv add <package>` to add a dependency to the environment.
- Use `python -m pytest -sv` from the `code` directory to run tests.

## Development instructions
- Use TDD (Test-driven development) principles to write code.
- Write tests first, then write the code that makes the tests pass.

## Memory Management & Troubleshooting
- **VRAM Overflows on MPS/CPU**: When testing locally on Apple Silicon (MPS) or CPU, be careful with large models (like VLMs). Standard `bitsandbytes` 4-bit/8-bit quantization is not supported on MPS and falls back to float.
- If `torch_dtype` is accidentally set to `float32`, a model can suddenly occupy 40GB+ of VRAM, causing an overflow on 16GB VRAM machines.
- **Fixes**: 
  - Ensure model adapters use `torch.float16` for MPS (`model_kwargs["torch_dtype"] = torch.float16`).
  - Use `quantization: "2bit"` in config (which uses `optimum-quanto`'s `QuantoConfig(weights="int2")` if installed).
  - Reduce the quantity of images being passed to VLMs by lowering the `fps` to `1.0` or reducing `chunk_duration`.
  - Verify and lower the image resolution (e.g., set `resize: [112, 112]`) to significantly drop VRAM usage.
