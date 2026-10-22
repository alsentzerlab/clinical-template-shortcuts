#!/usr/bin/env bash
# Patient-grouped, rate-matched 70/15/15 splits for the paper's 5 template pairs
# (by readmission gap: 5.1, 10.9, 39.0, 44.8, 49.9 pp). Inputs come from
# template_selection/build_pair_parquets.py and already carry pat_id.
#
# The TRACE arm is split the same way: run TRACE on each pair file to produce
# data/pairs/<pair>_removed.parquet, then rerun this script with SUFFIX=_removed.
set -euo pipefail

SUFFIX=${SUFFIX:-}
for pair in 248528_229657 801125_229657 206596_801125 206596_248528 206596_229657; do
    uv run python finetuning/process_data.py \
        --input-path "data/pairs/${pair}${SUFFIX}.parquet" \
        --output-dir "data/pairs/${pair}${SUFFIX}/" \
        --group-column pat_id --match-rates
done
