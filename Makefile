IMAGE        := udiva-experiment
USER_NAME    := jramirez
USER_ID      := 11056
GROUP_NAME   := guest
GROUP_ID     := 11000

REPO         := /home-net/jramirez/data-slow/hhoi/experiment-udiva
HF_CACHE     := /home-net/jramirez/.cache/huggingface
GPUS         ?= 0
CATEGORY     ?= verbal
VIDEO        ?=
ARGS         ?=
OBS          ?= 4         # anticipation: seconds of video observed before t_b

VIEW         ?= e1        # egocentric view: e1 (participant_a) or e2 (participant_b)
HOLD_MEMGB   ?= 20        # VRAM (GB) reserved per GPU by `hold-gpus`
HOLD_HOURS   ?= 0         # auto-release after N hours (0 = until `make unhold`)

QWEN3VL_DIR  := /home-net/jramirez/data-slow/hhoi/udiva_hhoi/qwen-3vl
GT_DIR       := /home-net/jramirez/data-slow/hhoi/udiva_hhoi/qwen-3vl/ground_truth
COMPUTE_MAP  := /home-net/jramirez/data-slow/hhoi/udiva_hhoi/qwen-3vl/compute_map.py
ANT_DIR      := $(REPO)/data-slow/results/gf_anticipation
ANT_REF      := $(REPO)/data-slow/samples_gt/anticipation_samples_gt.json
ANT_SCORER   := /home-net/jramirez/UDIVA/HHOI@ECCV26/StartingKit/codabench/anticipation/scoring_program
MVIEW        ?= exo       # anticipation merge/score view: exo or ego
CAU_DIR      := $(REPO)/data-slow/results/gf_causal
CAU_SCORER   := /home-net/jramirez/UBInteract/UDIVA-HHOI/StartingKit/codabench/causal/scoring_program
EGO_DIR      := $(REPO)/data-slow/results/gf_baseline_ego

# Build per-video flags from a space-separated VIDEO list, e.g. VIDEO="005013 020025"
VIDEO_FLAGS  := $(foreach v,$(VIDEO),--video $(v))

# ── image ─────────────────────────────────────────────────────────────────────

build:
	docker build -t $(IMAGE) \
	  --build-arg USER_NAME=$(USER_NAME) \
	  --build-arg USER_ID=$(USER_ID) \
	  --build-arg GROUP_NAME=$(GROUP_NAME) \
	  --build-arg GROUP_ID=$(GROUP_ID) \
	  $(REPO)

rebuild:
	docker build --no-cache -t $(IMAGE) \
	  --build-arg USER_NAME=$(USER_NAME) \
	  --build-arg USER_ID=$(USER_ID) \
	  --build-arg GROUP_NAME=$(GROUP_NAME) \
	  --build-arg GROUP_ID=$(GROUP_ID) \
	  $(REPO)

# ── runs ──────────────────────────────────────────────────────────────────────

run:
	docker run --rm --gpus all \
	  -e CUDA_VISIBLE_DEVICES=$(GPUS) \
	  -v $(REPO):/workspace \
	  -v $(HF_CACHE):/home/$(USER_NAME)/.cache/huggingface \
	  $(IMAGE) \
	  /opt/venv/bin/python run_gf_baseline.py --category $(CATEGORY) $(VIDEO_FLAGS) $(ARGS)

# Smoke test: one short video, one card. Downloads the model on first run.
smoke:
	docker run --rm --gpus all \
	  -e CUDA_VISIBLE_DEVICES=$(GPUS) \
	  -v $(REPO):/workspace \
	  -v $(HF_CACHE):/home/$(USER_NAME)/.cache/huggingface \
	  $(IMAGE) \
	  /opt/venv/bin/python run_gf_baseline.py --category $(CATEGORY) --video 035040

run-mock:
	docker run --rm --gpus all \
	  -e CUDA_VISIBLE_DEVICES=$(GPUS) \
	  -v $(REPO):/workspace \
	  -v $(HF_CACHE):/home/$(USER_NAME)/.cache/huggingface \
	  $(IMAGE) \
	  /opt/venv/bin/python run_gf_baseline.py --category $(CATEGORY) --mock

shell:
	docker run --rm -it --gpus all \
	  -e CUDA_VISIBLE_DEVICES=$(GPUS) \
	  -v $(REPO):/workspace \
	  -v $(HF_CACHE):/home/$(USER_NAME)/.cache/huggingface \
	  $(IMAGE) \
	  /bin/bash

# ── anticipation (Tracks 3 & 4) ────────────────────────────────────────────────

# Reference = the official sampled windows file (already in reference shape).
# Run a view: exo (GF, both participants) or e1/e2 (egocentric, single subject).
# Observation (60s), num_frames (32), resize come from the config.
# e.g. make anticipate VIEW=exo GPUS=0 ;  make anticipate VIEW=e1 GPUS=1
anticipate:
	docker run --rm --gpus all \
	  -e CUDA_VISIBLE_DEVICES=$(GPUS) \
	  -v $(REPO):/workspace \
	  -v $(HF_CACHE):/home/$(USER_NAME)/.cache/huggingface \
	  $(IMAGE) \
	  /opt/venv/bin/python run_gf_anticipation.py --view $(VIEW) $(VIDEO_FLAGS) $(ARGS)

# Merge shards -> submission.json. MVIEW=exo (GF) or ego (e1+e2). Host python3.
anticipate-merge:
	cd $(REPO)/code && python3 merge_anticipation.py --view $(MVIEW)

# ── egocentric recognition (E1 -> participant_a, E2 -> participant_b) ───────────

# Run one egocentric view + category. e.g. make ego VIEW=e1 CATEGORY=nonverbal GPUS=2
ego:
	docker run --rm --gpus all \
	  -e CUDA_VISIBLE_DEVICES=$(GPUS) \
	  -v $(REPO):/workspace \
	  -v $(HF_CACHE):/home/$(USER_NAME)/.cache/huggingface \
	  $(IMAGE) \
	  /opt/venv/bin/python run_egocentric_baseline.py --view $(VIEW) --category $(CATEGORY) $(VIDEO_FLAGS) $(ARGS)

# Merge E1 (A) + E2 (B) -> gf_baseline_ego/ (host python3)
ego-merge:
	cd $(REPO)/code && python3 merge_egocentric.py

# Convert merged egocentric preds + score with compute_map.py.
# Runs INSIDE Docker (/opt/venv has pandas+sklearn) so it never depends on the
# host python3. Mounts the qwen-3vl dir (GT + compute_map.py) read-only-ish.
score-ego:
	docker run --rm \
	  -v $(REPO):/workspace \
	  -v $(QWEN3VL_DIR):/qwen3vl \
	  $(IMAGE) bash -lc '\
	    /opt/venv/bin/python convert_predictions_to_csv.py \
	      --input-dir ../data-slow/results/gf_baseline_ego \
	      --output-dir ../data-slow/results/gf_baseline_ego_csv && \
	    /opt/venv/bin/python /qwen3vl/compute_map.py \
	      --results-dir /workspace/data-slow/results/gf_baseline_ego_csv \
	      --gt-dir /qwen3vl/ground_truth --task recognition \
	      --save-csv /workspace/data-slow/results/gf_baseline_ego_map.csv'

# Score submission.json against reference.json with the OFFICIAL scorer (host python3).
# Score with the OFFICIAL ECCV26 SDL scorer. MVIEW=exo or ego. Runs in Docker;
# reference = the sampled windows file, submission = $(MVIEW)/submission.json.
score-anticipation:
	rm -rf $(ANT_DIR)/$(MVIEW)/eval && mkdir -p $(ANT_DIR)/$(MVIEW)/eval/ref $(ANT_DIR)/$(MVIEW)/eval/res $(ANT_DIR)/$(MVIEW)/eval/output
	cp $(ANT_REF) $(ANT_DIR)/$(MVIEW)/eval/ref/reference.json
	cp $(ANT_DIR)/$(MVIEW)/submission.json $(ANT_DIR)/$(MVIEW)/eval/res/submission.json
	docker run --rm \
	  -v $(REPO):/workspace \
	  -v $(ANT_SCORER):/ant_scorer \
	  $(IMAGE) bash -lc 'cd /ant_scorer && /opt/venv/bin/python score.py \
	    /workspace/data-slow/results/gf_anticipation/$(MVIEW)/eval \
	    /workspace/data-slow/results/gf_anticipation/$(MVIEW)/eval/output'
	@echo "Scores -> $(ANT_DIR)/$(MVIEW)/eval/output/scores.json"

# ── causal QA (Task 5) ──────────────────────────────────────────────────────────

# Build reference.json from the GT spec (host python3).
causal-ref:
	cd $(REPO)/code && python3 build_causal_reference.py

# Run the causal baseline -> per-video shards + submission.json
# e.g. make causal GPUS=2 VIDEO="005013.mp4 020025.mp4"
causal:
	docker run --rm --gpus all \
	  -e CUDA_VISIBLE_DEVICES=$(GPUS) \
	  -v $(REPO):/workspace \
	  -v $(HF_CACHE):/home/$(USER_NAME)/.cache/huggingface \
	  $(IMAGE) \
	  /opt/venv/bin/python run_gf_causal.py $(VIDEO_FLAGS) $(ARGS)

# Merge shards -> submission.json (host python3)
causal-merge:
	cd $(REPO)/code && python3 run_gf_causal.py --merge-only

# Score submission.json against reference.json with the OFFICIAL causal scorer.
# Runs in Docker (/opt/venv has deps); mounts the scorer dir as /cau_scorer.
score-causal:
	rm -rf $(CAU_DIR)/eval && mkdir -p $(CAU_DIR)/eval/ref $(CAU_DIR)/eval/res $(CAU_DIR)/eval/output
	cp $(CAU_DIR)/reference.json $(CAU_DIR)/eval/ref/reference.json
	cp $(CAU_DIR)/submission.json $(CAU_DIR)/eval/res/submission.json
	docker run --rm \
	  -v $(REPO):/workspace \
	  -v $(CAU_SCORER):/cau_scorer \
	  $(IMAGE) bash -lc 'cd /cau_scorer && /opt/venv/bin/python score.py \
	    /workspace/data-slow/results/gf_causal/eval \
	    /workspace/data-slow/results/gf_causal/eval/output'
	@echo "Scores -> $(CAU_DIR)/eval/output/scores.json"

# ── GPU reservation (detached; survives your SSH session) ───────────────────────
# Reserve cards overnight. e.g. make hold-gpus GPUS=2,3,4,5,6 HOLD_MEMGB=20
hold-gpus:
	docker run -d --rm --name udiva-hold --gpus all \
	  -e CUDA_VISIBLE_DEVICES=$(GPUS) \
	  -v $(REPO):/workspace \
	  $(IMAGE) \
	  /opt/venv/bin/python experiment_udiva_hd.py --mem-gb $(HOLD_MEMGB) --busy --hours $(HOLD_HOURS)
	@echo "Holding GPUs $(GPUS). Stop with: make unhold (or: docker stop udiva-hold)"

unhold:
	-docker stop udiva-hold

.PHONY: build rebuild run smoke run-mock shell anticipate anticipate-merge score-anticipation ego ego-merge score-ego causal-ref causal causal-merge score-causal hold-gpus unhold
