#!/bin/bash
# Gradient-scaling runs on a machine without SLURM (e.g. a single large GPU).
# Starts one run per listed pair, ALL AT ONCE on the given GPU, and waits for them.
# Same train.py call as run_gradient_scaling.sbatch; finished runs are skipped.
#
#   run_gradient_scaling_local.sh <gpu> <grad_scale> <seed> <pair> [<pair> ...]
#
# A run peaks near 39 GB on full notes with gradient checkpointing, so ~4 fit on a
# 180 GB GPU. Run from the repo root, inside tmux/nohup so it survives a disconnect:
#   PYTHON=.venv/bin/python \
#     gradient_scaling/run_gradient_scaling_local.sh 6 -1 42 248528_229657 801125_229657
set -euo pipefail

GPU=$1 SCALE=$2 SEED=$3
shift 3
DATA_ROOT=${DATA_ROOT:-data/pairs}
OUT_ROOT=${OUT_ROOT:-$HOME/gradient_scaling_runs}
PYTHON=${PYTHON:-uv run --no-sync python}
mkdir -p "$OUT_ROOT/logs"

# Torch renamed the allocator env var; set both.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTORCH_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=$GPU

for PAIR in "$@"; do
    OUT=$OUT_ROOT/$PAIR/gs${SCALE}_s${SEED}
    if [[ -f $OUT/results.json ]]; then
        echo "Skipping $OUT (results.json exists)"
        continue
    fi
    LOG=$OUT_ROOT/logs/${PAIR}_gs${SCALE}_s${SEED}.log
    echo "pair=$PAIR grad_scale=$SCALE seed=$SEED gpu=$GPU log=$LOG"
    $PYTHON -m gradient_scaling.train \
        --data-dir "$DATA_ROOT/$PAIR" \
        --grad-scale "$SCALE" \
        --seed "$SEED" \
        --output-dir "$OUT" \
        --gradient-checkpointing > "$LOG" 2>&1 &
done
wait
echo "GPU $GPU: all runs finished"
