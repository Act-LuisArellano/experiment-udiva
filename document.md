# Handoff Summary

## Goal

The current goal is not the recognition feature itself. That is already done.

The current goal is to add and run a real Qwen Omni recognition configuration for the already-implemented recognition pipeline, using `code/src/models/qwen_omni.py` with the recognition task in `code/src/experiments/recognition.py`.

Specifically, the target is to run the new config at `code/configs/experiments/recognition_verbal_qwen.yaml` end to end without mock mode.

## Status

### Recognition feature status

- Implemented in `code/src/experiments/recognition.py`, `code/src/output/recognition.py`, and `code/main.py`
- Sample configs for Gemma verbal and nonverbal are working
- Mock end-to-end runs succeeded for both sample configs
- That work was already committed and pushed as commit `d6ef94e` on `origin/main`

### Qwen-specific status

- Added a new local config file: `code/configs/experiments/recognition_verbal_qwen.yaml`
- That file is not part of the pushed commit yet
- The Qwen config uses:
  - model name `qwen_omni`
  - checkpoint `auto`
  - profile `light`
  - quantization `2bit`
  - weights path `../data-slow/models/Qwen/qwen2.5-omni-light`
  - the existing local mosaic video for `001080`

### Environment status

- CUDA is available
- GPU count reported as `8`
- The expected local Qwen weights directories were missing, so the model would need to download or populate cache on first successful load
- The repo uses the local repo virtualenv at `/home-net/jramirez/data-slow/hhoi/experiment-udiva/.venv`
- Important: the VS Code Python environment tool keeps selecting the wrong unrelated 3.12 virtualenv under `udiva-remote`; do not trust that tool for this repo

## What Failed

The Qwen run did not fail because of recognition logic.

It failed because the repo virtualenv is partially corrupted in multiple packages.

Observed sequence:

- First Qwen run failed because `transformers` in the repo `.venv` was broken as an empty namespace package
- That was repaired by reinstalling `transformers 5.5.4` into the repo `.venv` with `uv pip`
- After that, Qwen import progressed further and then failed because `scipy` was broken: missing `scipy._lib`
- `scipy` was then reinstalled into the same repo `.venv`

So the current exact status is:

- `transformers` has been reinstalled
- `scipy` has been reinstalled
- The Qwen import and experiment need to be rerun after the `scipy` reinstall

## Known Good State

These are known good:

- `code/configs/experiments/recognition_verbal.yaml` runs with `--mock`
- `code/configs/experiments/recognition_nonverbal.yaml` runs with `--mock`
- Recognition outputs are written correctly by `code/src/output/recognition.py`
- The pushed branch already contains the recognition implementation and the fixed sample video paths

## What Changed Locally After the Push

### Local code change after the pushed commit

- Added `code/configs/experiments/recognition_verbal_qwen.yaml`

### Local environment changes after the push

- Reinstalled `transformers` in the repo `.venv`
- Reinstalled `scipy` in the repo `.venv`

## Exact Next Step For Another Worker

The next worker should do exactly this:

1. Retry the Qwen import check in the repo `.venv`:

   ```python
   from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
   ```

2. If that succeeds, rerun:

   ```bash
   /home-net/jramirez/data-slow/hhoi/experiment-udiva/.venv/bin/python \
     /home-net/jramirez/data-slow/hhoi/experiment-udiva/code/main.py \
     --config /home-net/jramirez/data-slow/hhoi/experiment-udiva/code/configs/experiments/recognition_verbal_qwen.yaml
   ```

3. If it fails again, continue repairing the next broken dependency in the same repo `.venv`, not by rebuilding the environment from scratch

## Most Important Handoff Notes

- Do not restart from recognition feature implementation. That part is complete and pushed.
- Do not use the auto-selected Python environment from the VS Code Python tool for this repo.
- Work in the repo `.venv` only.
- The current task is environment/runtime bring-up for Qwen Omni recognition, not feature development.