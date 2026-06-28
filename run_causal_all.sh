#!/bin/bash
# Full causal pipeline, unattended: parallel inference -> merge -> score.
# Launch detached so it survives closing the editor / SSH session:
#   cd /home-net/jramirez/data-slow/hhoi/experiment-udiva
#   setsid nohup bash run_causal_all.sh > causal_run.log 2>&1 &
# Then come back later and check:
#   tail -f causal_run.log
#   cat data-slow/results/gf_causal/eval/output/scores.json
set -u
cd /home-net/jramirez/data-slow/hhoi/experiment-udiva

echo "[$(date)] causal-ref"
make causal-ref

echo "[$(date)] inference (one video per GPU, cards 0-6)"
# GPUs 0,1 are 46GB -> 7B fits with no offload -> put the biggest videos there.
make causal GPUS=0 VIDEO="041083.mp4" ARGS="--skip-existing" &   # 81 effects
make causal GPUS=1 VIDEO="066067.mp4" ARGS="--skip-existing" &   # 61
make causal GPUS=2 VIDEO="027113.mp4" ARGS="--skip-existing" &   # 31
make causal GPUS=3 VIDEO="035040.mp4" ARGS="--skip-existing" &   # 17
make causal GPUS=4 VIDEO="005013.mp4" ARGS="--skip-existing" &   # 16
make causal GPUS=5 VIDEO="044156.mp4" ARGS="--skip-existing" &   # 15
make causal GPUS=6 VIDEO="020025.mp4" ARGS="--skip-existing" &   # 13
wait

echo "[$(date)] merge + score"
make causal-merge
make score-causal

echo "[$(date)] DONE. scores:"
cat data-slow/results/gf_causal/eval/output/scores.json
