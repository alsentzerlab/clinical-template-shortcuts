#!/bin/bash
# ShorT gradient-scaling sweep: one array task per (pair, grad_scale, seed).
#
#   PAIRS x SCALES x SEEDS = 5 x 9 x 3 = 135 tasks -> --array=0-134.
#   If you edit the lists below, update --array to 0-(N-1).
#
# Pilot first (1 pair x grad_scale {-1,0,1} x 1 seed = 3 tasks) to get per-run time
# and check that grad_scale moves template encoding at all:
#   cd <repo>
#   sbatch --array=0-2 --export=ALL,PILOT=1 gradient_scaling/run_gradient_scaling.sbatch
# Pilot runs write to the same place as the full sweep, which then skips them.
#
# Full sweep (logs land in the submission directory):
#   cd <repo> && sbatch gradient_scaling/run_gradient_scaling.sbatch
# Then:
#   uv run --no-sync python -m gradient_scaling.analyze --runs-root ~/gradient_scaling_runs
#
#SBATCH --job-name=short_gs
#SBATCH --partition=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --array=0-134%12
#SBATCH --output=short_gs_%A_%a.out

set -euo pipefail

# Split dirs (train/val/test.parquet): the paper's 5 pairs (1-5), by readmission gap
# (5.1, 10.9, 39.0, 44.8, 49.9 pp). 801125_248528 (5.8 pp) dropped as redundant.
REPO=${REPO:-$SLURM_SUBMIT_DIR}   # submit from the repo root
DATA_ROOT=$REPO/data/pairs
PAIRS=(248528_229657 801125_229657 206596_801125 206596_248528 206596_229657)
# Paper convention: >0 encourages template encoding, <0 reverses it, 0 = baseline.
# Paper grid spans +/-1e-1 (25 values); +/-1 added because template identity is far
# easier to decode than age, and ShorT needs encoding to actually move.
SCALES=(-1 -0.1 -0.01 -0.001 0 0.001 0.01 0.1 1)
SEEDS=(42 43 44)
OUT_ROOT=$HOME/gradient_scaling_runs

if [[ ${PILOT:-0} == 1 ]]; then
    # Mid-range shortcut pair (39.0 pp gap; shortcut gap +0.089 in the main runs)
    PAIRS=(206596_801125)
    SCALES=(-1 0 1)
    SEEDS=(42)
fi

# Ad-hoc subsets without editing this file (space-separated), e.g.
#   sbatch --array=0-5 --export=ALL,PAIRS_OVERRIDE=206596_801125,SCALES_OVERRIDE="-0.1 -0.01 -0.001 0.001 0.01 0.1",SEEDS_OVERRIDE=42 ...
# Don't include a (pair, scale, seed) that is still running elsewhere: the skip check
# only sees finished runs, so both jobs would write to the same directory.
[[ -n ${PAIRS_OVERRIDE:-} ]] && read -ra PAIRS <<< "$PAIRS_OVERRIDE"
[[ -n ${SCALES_OVERRIDE:-} ]] && read -ra SCALES <<< "$SCALES_OVERRIDE"
[[ -n ${SEEDS_OVERRIDE:-} ]] && read -ra SEEDS <<< "$SEEDS_OVERRIDE"

N_SCALES=${#SCALES[@]}
N_SEEDS=${#SEEDS[@]}
i=$SLURM_ARRAY_TASK_ID
PAIR=${PAIRS[$(( i / (N_SCALES * N_SEEDS) ))]:-}
SCALE=${SCALES[$(( (i / N_SEEDS) % N_SCALES ))]}
SEED=${SEEDS[$(( i % N_SEEDS ))]}
OUT=$OUT_ROOT/$PAIR/gs${SCALE}_s${SEED}
if [[ -z $PAIR ]]; then
    echo "Task $i is outside the grid (did --array match the lists / PILOT?)" >&2
    exit 1
fi

if [[ -f $OUT/results.json ]]; then
    echo "Skipping $OUT (results.json exists)"
    exit 0
fi

cd "$REPO"  # -m import and uv project both need the repo root
export HF_HOME=$HOME/hf_cache
# Torch renamed the allocator env var; set both.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTORCH_ALLOC_CONF=expandable_segments:True

echo "pair=$PAIR grad_scale=$SCALE seed=$SEED host=$(hostname)"
uv run --no-sync python -m gradient_scaling.train \
    --data-dir "$DATA_ROOT/$PAIR" \
    --grad-scale "$SCALE" \
    --seed "$SEED" \
    --output-dir "$OUT" \
    --gradient-checkpointing
