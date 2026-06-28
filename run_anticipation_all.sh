#!/bin/bash
# Full anticipation pipeline, unattended: exo + egocentric (e1,e2) -> merges -> scores.
# Uses GPUs 0-5 (one video per card; 41083 & 005013 on the 46GB cards 0,1 -> no offload).
# Launch detached so it survives closing the editor / SSH session:
#   cd /home-net/jramirez/data-slow/hhoi/experiment-udiva
#   setsid nohup bash run_anticipation_all.sh > anticipation_run.log 2>&1 &
# Check later:
#   tail -f anticipation_run.log
#   cat data-slow/results/gf_anticipation/exo/eval/output/scores.json
#   cat data-slow/results/gf_anticipation/ego/eval/output/scores.json
set -u
cd /home-net/jramirez/data-slow/hhoi/experiment-udiva

# one video per GPU (0,1 are 46GB -> biggest videos there); --skip-existing = resumable
run_phase () {   # $1 = view (exo|e1|e2)
  local V="$1"
  echo "[$(date)] === phase $V ==="
  make anticipate VIEW=$V GPUS=0 VIDEO="041083"        ARGS="--skip-existing" &  # 183 windows
  make anticipate VIEW=$V GPUS=1 VIDEO="005013"        ARGS="--skip-existing" &  # 132
  make anticipate VIEW=$V GPUS=2 VIDEO="027113"        ARGS="--skip-existing" &  # 102
  make anticipate VIEW=$V GPUS=3 VIDEO="066067"        ARGS="--skip-existing" &  # 80
  make anticipate VIEW=$V GPUS=4 VIDEO="020025 035040" ARGS="--skip-existing" &  # 57+46
  make anticipate VIEW=$V GPUS=5 VIDEO="044156"        ARGS="--skip-existing" &  # 51
  wait
}

run_phase exo
run_phase e1
run_phase e2

echo "[$(date)] === merge + score ==="
make anticipate-merge MVIEW=exo
make anticipate-merge MVIEW=ego
make score-anticipation MVIEW=exo
make score-anticipation MVIEW=ego

echo "[$(date)] DONE."
echo "--- exocentric scores ---"; cat data-slow/results/gf_anticipation/exo/eval/output/scores.json
echo; echo "--- egocentric scores ---"; cat data-slow/results/gf_anticipation/ego/eval/output/scores.json
