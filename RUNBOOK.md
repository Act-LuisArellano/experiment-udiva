# UDIVA-HHOI Baselines Runbook (Qwen2.5-Omni-7B)

How to reproduce the UDIVA-HHOI baselines end-to-end over the 7 GF videos.
Two tasks are covered: **Recognition** (below) and **Action Anticipation**
(second half of this file). Prerequisites (section 0) are shared.

All commands run from the repo root unless noted:

```bash
cd /home-net/jramirez/data-slow/hhoi/experiment-udiva
```

---

## 0. Prerequisites (one-time)

- **Docker** + NVIDIA runtime on the host (all inference runs in a container).
- **Build the image** (only needed once per machine — the image is local, not on NFS):
  ```bash
  make build
  ```
- The model (`Qwen/Qwen2.5-Omni-7B`, ~20 GB) auto-downloads into the shared HF
  cache (`~/.cache/huggingface`) on the first run. It's on NFS, so it's reused
  across servers and across all later runs.
- Metrics are computed with **system `python3`** (has working pandas/sklearn).
  Do **not** use `.venv` for metrics — its pandas is broken
  (`No module named pandas.tseries`).

### Key config
`code/configs/experiments/recognition_verbal_qwen.yaml`:
- `checkpoint: "Qwen/Qwen2.5-Omni-7B"`
- `quantization: "none"`  (bf16 — best quality; fits one 46 GB card)
- `profile: "server"`
- `chunk_duration: 2.0`, `chunk_stride: 2.0`  (non-overlapping 2 s windows)

---

## 1. Inference

The runner loads the model once and loops the requested GF videos, writing one
JSON per video to `data-slow/results/gf_baseline/{video_id}_{category}.json`.

### Smoke test first (one short video, ~few min)
Confirms the model loads and produces in-vocabulary labels before the full run:
```bash
make smoke GPUS=0 CATEGORY=nonverbal
```

### Full run, parallelized across GPUs
One model instance per GPU (this is where spare cards help — they speed up
*throughput*, not a single inference). bf16 7B (~21 GB) fits on one card, so run
each instance on a single card; don't shard.

`--skip-existing` (via `ARGS`) makes reruns safe to resume.

**Non-verbal:**
```bash
make run GPUS=0 CATEGORY=nonverbal VIDEO="005013 020025" ARGS="--skip-existing" &
make run GPUS=1 CATEGORY=nonverbal VIDEO="027113 035040" ARGS="--skip-existing" &
make run GPUS=2 CATEGORY=nonverbal VIDEO="041083 044156" ARGS="--skip-existing" &
make run GPUS=3 CATEGORY=nonverbal VIDEO="066067"        ARGS="--skip-existing" &
wait
```

**Verbal:**
```bash
make run GPUS=0 CATEGORY=verbal VIDEO="005013 020025 027113" ARGS="--skip-existing" &
make run GPUS=1 CATEGORY=verbal VIDEO="035040 041083"        ARGS="--skip-existing" &
make run GPUS=2 CATEGORY=verbal VIDEO="044156 066067"        ARGS="--skip-existing" &
wait
```

> To run all 7 videos on a single card in one process, just omit `VIDEO`:
> `make run GPUS=0 CATEGORY=nonverbal`

### Makefile knobs
| Var | Default | Meaning |
|-----|---------|---------|
| `GPUS` | `0` | value of `CUDA_VISIBLE_DEVICES` for the container |
| `CATEGORY` | `verbal` | `verbal` or `nonverbal` |
| `VIDEO` | (all) | space-separated video ids, e.g. `"005013 020025"` |
| `ARGS` | (none) | extra flags, e.g. `--skip-existing` |

### Stopping a run
Ctrl+C does **not** work — the work runs inside Docker containers. Stop them with:
```bash
docker stop $(docker ps -q --filter ancestor=udiva-experiment)
```
(Containers use `--rm`, so they are removed automatically.)

---

## 2. Convert predictions to eval CSVs

Transforms the 14 JSONs into the CSV layout `compute_map.py` expects
(`{out}/{video_id}/recognition/chunk2.0/final_video_analysis.csv`):

```bash
cd code
python3 convert_predictions_to_csv.py
cd ..
```
Output: `data-slow/results/gf_baseline_csv/`

---

## 3. Compute metrics

```bash
python3 /home-net/jramirez/data-slow/hhoi/udiva_hhoi/qwen-3vl/compute_map.py \
  --results-dir data-slow/results/gf_baseline_csv \
  --gt-dir /home-net/jramirez/data-slow/hhoi/udiva_hhoi/qwen-3vl/ground_truth \
  --task recognition \
  --save-csv data-slow/results/gf_baseline_7b_map.csv
```
Outputs `gf_baseline_7b_map.csv` + `_aggregated.csv`.

> **Important:** the reported **mAP is a prevalence floor** for binary predictions
> (`average_precision_score` on constant columns returns the base rate, not skill),
> so it barely moves even when correct predictions jump a lot. For the real signal,
> use the **set-level TP / precision / recall / F1** snippet below.

### Honest set-level metrics (recommended)
```bash
cd /home-net/jramirez/data-slow/hhoi/udiva_hhoi/qwen-3vl
python3 - << 'EOF'
import importlib.util
spec = importlib.util.spec_from_file_location("cm","compute_map.py")
cm = importlib.util.module_from_spec(spec); spec.loader.exec_module(cm)

CSV_ROOT="/home-net/jramirez/data-slow/hhoi/experiment-udiva/data-slow/results/gf_baseline_csv"
vids=["005013","020025","027113","035040","041083","044156","066067"]
agg={c:{"tp":0,"fp":0,"fn":0} for c in ["low_level_actions","high_level_actions","utterance_types"]}
for v in vids:
    gt=cm.load_ground_truth(f"ground_truth/{v}_L_mosaic.json")
    preds=cm.load_predictions(f"{CSV_ROOT}/{v}/recognition/chunk2.0/final_video_analysis.csv")
    for _,r in preds.iterrows():
        gl=cm.get_gt_labels_recognition(gt,r["window_start"],r["window_end"],r["subject"],act_filter=None)
        for c in agg:
            P,G=set(r[c]),gl[c]
            agg[c]["tp"]+=len(P&G); agg[c]["fp"]+=len(P-G); agg[c]["fn"]+=len(G-P)
print(f"{'category':<20}{'TP':>5}{'FP':>6}{'FN':>6}{'prec':>8}{'rec':>8}{'F1':>8}")
for c,d in agg.items():
    tp,fp,fn=d['tp'],d['fp'],d['fn']
    p=tp/(tp+fp) if tp+fp else 0; rec=tp/(tp+fn) if tp+fn else 0
    f1=2*p*rec/(p+rec) if p+rec else 0
    print(f"{c:<20}{tp:>5}{fp:>6}{fn:>6}{p:>8.3f}{rec:>8.3f}{f1:>8.3f}")
EOF
```

---

## Data reference

- **Videos:** `data/udiva_hhoi/GF/` — 7 mp4s:
  `005013, 020025, 027113, 035040, 041083, 044156, 066067`
- **Ground truth:** `/home-net/jramirez/data-slow/hhoi/udiva_hhoi/qwen-3vl/ground_truth/{id}_L_mosaic.json`
- **Predictions:** `data-slow/results/gf_baseline/{id}_{verbal,nonverbal}.json`
- **Eval CSVs:** `data-slow/results/gf_baseline_csv/`
- **Metrics:** `data-slow/results/gf_baseline_7b_map.csv` (+ `_aggregated.csv`)

## Latest result (7B-bf16, all 7 videos, set-level, act_filter=None)

| Category | TP | Precision | Recall | F1 |
|----------|----|-----------|--------|----|
| low_level | 244 | 0.180 | 0.090 | 0.120 |
| high_level | 105 | 0.102 | 0.026 | 0.041 |
| utterance | 50 | 0.068 | 0.030 | 0.042 |

Known weakness: high-level over-predicts `assemble` — prompt-tuning candidate.

## Recognition variants: exocentric vs egocentric
- **Exocentric (GF):** third-person mosaic; one run predicts both participants.
  Subjects grounded by frame side (participant_a = LEFT, participant_b = RIGHT).
- **Egocentric (E1/E2):** per-participant first-person views (E1→participant_a,
  E2→participant_b), 1080p, **same time base** as GF/GT (no realignment). Each run
  predicts ONLY the camera wearer's own actions; merge E1(A)+E2(B), then score.

### Egocentric workflow (full re-run)
```bash
cd /home-net/jramirez/data-slow/hhoi/experiment-udiva

# 1. Inference — 4 combos (view x category), one model instance per GPU, in parallel.
#    7B bf16 CPU-offloads on 24GB cards (~3 s/seg); each combo ~45 min.
make ego VIEW=e1 CATEGORY=nonverbal GPUS=2 &
make ego VIEW=e1 CATEGORY=verbal    GPUS=3 &
make ego VIEW=e2 CATEGORY=nonverbal GPUS=4 &
make ego VIEW=e2 CATEGORY=verbal    GPUS=5 &
wait

# 2. Merge E1(participant_a) + E2(participant_b) -> data-slow/results/gf_baseline_ego/
make ego-merge

# 3. Score (convert + compute_map). Runs INSIDE Docker, so it does NOT need a
#    host python3 with pandas (that was the "No module named pandas" error).
make score-ego
```
- Knobs: `VIEW` (e1/e2), `CATEGORY`, `GPUS`, `VIDEO="id1 id2"`, `ARGS="--skip-existing"`.
- Single-subject mode is triggered by `recognition_subject` in the experiment
  (E1→participant_a, E2→participant_b); the model predicts only the camera
  wearer's own actions and the subject field is forced.
- Per-view outputs: `data-slow/results/egocentric/{e1,e2}/{id}_{category}.json`
  → merged `gf_baseline_ego/{id}_{category}.json` → scored `gf_baseline_ego_map.csv`.

> **Why scoring is in Docker:** `convert_predictions_to_csv.py` and `compute_map.py`
> need pandas+sklearn. The host `python3` in an interactive shell may lack them;
> `/opt/venv` in the image has pandas 3.0.2 + sklearn 1.8.0. `make score-ego`
> mounts the qwen-3vl dir as `/qwen3vl` and runs everything in the container.
> (`make ego-merge` is pure json/pathlib, so it runs fine on the host.)

### Honest set-level F1 + exocentric comparison
mAP from compute_map.py is a prevalence floor — report F1 too. To compare both
views with the set-level metric, run this in Docker:
```bash
docker run --rm \
  -v /home-net/jramirez/data-slow/hhoi/experiment-udiva:/workspace \
  -v /home-net/jramirez/data-slow/hhoi/udiva_hhoi/qwen-3vl:/qwen3vl \
  udiva-experiment /opt/venv/bin/python - << 'PY'
import importlib.util
spec=importlib.util.spec_from_file_location("cm","/qwen3vl/compute_map.py")
cm=importlib.util.module_from_spec(spec); spec.loader.exec_module(cm)
vids=["005013","020025","027113","035040","041083","044156","066067"]
def setlevel(root):
    agg={c:{"tp":0,"fp":0,"fn":0} for c in ["low_level_actions","high_level_actions","utterance_types"]}
    for v in vids:
        gt=cm.load_ground_truth(f"/qwen3vl/ground_truth/{v}_L_mosaic.json")
        pr=cm.load_predictions(f"{root}/{v}/recognition/chunk2.0/final_video_analysis.csv")
        for _,r in pr.iterrows():
            gl=cm.get_gt_labels_recognition(gt,r["window_start"],r["window_end"],r["subject"],act_filter=None)
            for c in agg:
                P,G=set(r[c]),gl[c]
                agg[c]["tp"]+=len(P&G); agg[c]["fp"]+=len(P-G); agg[c]["fn"]+=len(G-P)
    return agg
for name,root in [("exocentric","/workspace/data-slow/results/gf_baseline_csv"),
                  ("egocentric","/workspace/data-slow/results/gf_baseline_ego_csv")]:
    a=setlevel(root)
    for c,d in a.items():
        tp,fp,fn=d["tp"],d["fp"],d["fn"]
        p=tp/(tp+fp) if tp+fp else 0; r=tp/(tp+fn) if tp+fn else 0
        f=2*p*r/(p+r) if p+r else 0
        print(f"{name:<11}{c:<20}TP={tp:<5} P={p:.3f} R={r:.3f} F1={f:.3f}")
PY
```

### Result (7B, 2026-06-25) — egocentric beats exocentric
| Category | Exo F1 | **Ego F1** | Exo mAP | **Ego mAP** |
|----------|-------|-----------|---------|------------|
| low_level  | 0.167 | **0.246** | 0.0945 | **0.0971** |
| high_level | 0.057 | **0.116** | 0.0656 | **0.0682** |
| utterance  | 0.042 | **0.054** | 0.0270 | **0.0313** |

Egocentric ~doubles correct non-verbal predictions (low-level TP 350→697). mAP
barely moves (prevalence floor) — F1 is the real signal. CSVs:
`gf_baseline_ego_map.csv` (ego) and `gf_baseline_exo_grounded_map.csv` (exo, grounded).

---

# Action Anticipation (UDIVA-HHOI Codabench Tracks 3 & 4)

Separate task from recognition. For each 2-second window `[t_b, t_e]` the model
OBSERVES the video `[t_b − OBS, t_b]` and predicts the ordered sequence of events
each participant will perform during `[t_b, t_e]` (true anticipation: it never
sees the future window). Scored with the **official** scoring program.

## Metric (official)
- Source: `/home-net/jramirez/UBInteract/UDIVA-HHOI/StartingKit/codabench/anticipation/scoring_program`
- Normalized **Structured Damerau-Levenshtein** over ordered event sequences (1.0 = perfect).
- Events are positional arrays: verbal `[utterance_type, target]`,
  nonverbal `[highlevel_action, lowlevel_action, target]`.
- Best-of-K hypotheses per participant (1–5); we currently submit **K=1**.
- Four scores: `next_action`, `verbal_2s`, `nonverbal_2s`, `verbal_nonverbal_2s`.
- Strict alignment by `(video_id, segment_id)`; `t_b/t_e` must match the template;
  omitted segments are scored as empty.
- **Empty-baseline floors to beat:** next_action 0.16, verbal_2s 0.56,
  nonverbal_2s 0.24, verbal_nonverbal_2s 0.16 (verbal_2s is high because many
  windows truly have no events → empty-vs-empty = 1.0).

## Files
| File | Purpose |
|------|---------|
| `code/build_anticipation_reference.py` | Build `template.json` + `reference.json` from GT (2s windows) |
| `code/src/experiments/anticipation.py` | The `anticipation` experiment (observe past, predict next window) |
| `code/configs/experiments/anticipation_qwen.yaml` | Config (7B bf16; `anticipation_observation_seconds`) |
| `code/run_gf_anticipation.py` | Batch runner → per-video shards + merged `submission.json` |
| `code/merge_anticipation_shards.py` | Merge shards into `submission.json` (host python3) |

## The OBS parameter (observation horizon)
Seconds of video observed before each window. Set via, in priority order:
`--obs-seconds N` (CLI) > `anticipation_observation_seconds` in the yaml.
Makefile exposes it as `OBS` (default 4). Note frames cap at 4, so larger OBS
widens the time span but still subsamples to 4 frames.

## Step 1 — build template + reference (once; host python3)
```bash
cd /home-net/jramirez/data-slow/hhoi/experiment-udiva
make anticipation-ref
```
Writes `data-slow/results/gf_anticipation/{template.json,reference.json}`.
(Runs on the HOST because the GT dir is outside the mounted repo.)

## Step 2 — inference (Docker)
Smoke test one video first:
```bash
make anticipate-smoke GPUS=0 OBS=4
```
Full run, parallelized (one model instance per GPU; shards prevent clashes):
```bash
make anticipate GPUS=0 OBS=4 VIDEO="005013 020025" &
make anticipate GPUS=1 OBS=4 VIDEO="027113 035040" &
make anticipate GPUS=2 OBS=4 VIDEO="041083 044156" &
make anticipate GPUS=3 OBS=4 VIDEO="066067"        &
wait
make anticipate-merge      # deterministic combined submission.json after all runs
```
`ARGS="--skip-existing"` skips videos whose shard already exists (safe resume).
Omit `VIDEO` to run all 7 in a single process.

## Step 3 — score (official, host python3)
```bash
make score-anticipation
```
Lays out the Codabench `ref/`+`res/` dirs and runs the official `score.py`;
prints the 4 scores and writes `data-slow/results/gf_anticipation/eval/output/scores.json`.

## Anticipation Makefile targets
| Target | Where | What |
|--------|-------|------|
| `anticipation-ref` | host | build template.json + reference.json |
| `anticipate` | docker | run baseline → shards + submission.json (`OBS`, `VIDEO`, `GPUS`, `ARGS`) |
| `anticipate-smoke` | docker | one video (035040) sanity run |
| `anticipate-merge` | host | merge shards → submission.json |
| `score-anticipation` | host | official 4 scores |

## Anticipation data reference
- **Template/reference:** `data-slow/results/gf_anticipation/{template.json,reference.json}`
- **Per-video shards:** `data-slow/results/gf_anticipation/shards/{video_id}.json`
- **Submission:** `data-slow/results/gf_anticipation/submission.json`
- **Scores:** `data-slow/results/gf_anticipation/eval/output/scores.json`

---

# Causal QA (UDIVA-HHOI Task 5)

For each annotated EFFECT, the model observes `[max(0, t_b - 60), t_e]` of the
(exocentric GF) video and answers: which option A-E is the CAUSE, and the absolute
timestamp it occurs. Every sampled frame has its absolute time burned into the
top-left corner so the VLM can emit a timestamp. Dense single-pass: N=32 frames
@320px + audio.

## Metric (official scorer)
- `UBInteract/.../codabench/causal/scoring_program`
- **accuracy**: predicted_option matches any GT cause's option.
- **temporal_accuracy**: predicted_cause_timestamp inside any GT cause `[t_b,t_e]`.
- Any-overlap, macro-averaged; empty option / null timestamp → 0.
- Submission: `{"causal":{video_id:{effect_id:{predicted_option, predicted_cause_timestamp}}}}`.

## Inputs
- GT spec: `data-slow/samples_gt/causal_samples_gt.json` (234 effects over 7 GF videos).
- Tunable extras in `configs/experiments/causal_qwen.yaml`:
  `causal_lookback_seconds` (60), `causal_num_frames` (32), `causal_resize` (320).

## Workflow

### Easiest: unattended, survives closing the editor (recommended)
`run_causal_all.sh` does the whole pipeline (ref → 7 parallel inference runs, one
video per GPU on cards 0–6 → merge → score). Launch it **detached**:
```bash
cd /home-net/jramirez/data-slow/hhoi/experiment-udiva
nvidia-smi                                        # confirm cards 0-6 are free first
setsid nohup bash run_causal_all.sh > causal_run.log 2>&1 &
```
Then you can close VSCode/disconnect. Come back and check:
```bash
tail -n 50 causal_run.log
cat data-slow/results/gf_causal/eval/output/scores.json
```
- The two biggest videos (041083, 066067) go on GPUs 0,1 (46 GB → 7B fits with no
  CPU offload → faster). Edit the script to drop a line if a card is taken.
- Resumable: the script passes `--skip-existing`, so re-launching skips finished
  videos. Stop early: `docker stop $(docker ps -q --filter ancestor=udiva-experiment)`.

### Manual / step-by-step
```bash
cd /home-net/jramirez/data-slow/hhoi/experiment-udiva
make causal-ref                                  # GT spec -> reference.json (host)

# inference: one video per GPU (0,1 are 46GB -> no offload -> put big videos there)
make causal GPUS=0 VIDEO="041083.mp4" ARGS="--skip-existing" &   # 81 effects
make causal GPUS=1 VIDEO="066067.mp4" ARGS="--skip-existing" &   # 61
make causal GPUS=2 VIDEO="027113.mp4" ARGS="--skip-existing" &   # 31
make causal GPUS=3 VIDEO="035040.mp4" ARGS="--skip-existing" &   # 17
make causal GPUS=4 VIDEO="005013.mp4" ARGS="--skip-existing" &   # 16
make causal GPUS=5 VIDEO="044156.mp4" ARGS="--skip-existing" &   # 15
make causal GPUS=6 VIDEO="020025.mp4" ARGS="--skip-existing" &   # 13
wait
make causal-merge                                # shards -> submission.json (host)
make score-causal                                # official accuracy + temporal_accuracy
```
- Knobs: `VIDEO="id.mp4 ..."`, `GPUS`, `ARGS="--skip-existing"`.
- Per-video shards in `gf_causal/shards/` make parallel runs safe (no overwrites).
- `make causal` runs in Docker (foreground client); to survive disconnect without
  the script, wrap the block in `tmux`/`screen` or use `run_causal_all.sh` above.

## Notes
- The Qwen adapter honors `request.extra["max_images"]` (default 4); causal raises
  it to pass all 32 timestamped frames. Recognition/anticipation are unaffected.
- temporal_accuracy is the hard metric (hit a ~1-3 s GT window in a 60 s span);
  raise `causal_num_frames` for finer localization at higher memory/time cost.
- Data: `data-slow/results/gf_causal/{reference.json, shards/, submission.json}`,
  scores at `gf_causal/eval/output/scores.json`.
