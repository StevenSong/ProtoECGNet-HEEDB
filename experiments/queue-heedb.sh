#!/usr/bin/env bash
# Queue a full run: the three ProtoECGNet branches in parallel, then fusion over them.
# Run from a clone on a host with the Slurm client, at a pushed commit.
#
#   experiments/queue-heedb.sh <run> [sbatch options for every job...]
#   e.g. experiments/queue-heedb.sh v5 -w kg35-nvl01   (nvl01 already has the caches)
#
# Without -w, each node a job lands on builds its own HEEDB caches on first use
# (see scripts/cache_heedb.py).
set -euo pipefail
cd "$(dirname "$0")/.."

usage="usage: experiments/queue-heedb.sh <run> [sbatch options...]"
run=${1:?$usage}
shift

s() { slurm/submit.sh --parsable --kill-on-invalid-dep=yes "$@"; }
gpu=(--gres=gpu:1 --mem-per-gpu=450G "$@")

arm_jobs=()
for arm in 1D-global 2D-global 2D-partial; do
  job=$(s "${gpu[@]}" -- scripts/run-arm.sh "$arm" "$run")
  echo "$arm: $job"
  arm_jobs+=("$job")
done

deps=$(IFS=:; echo "${arm_jobs[*]}")
job=$(s -d "afterok:$deps" "${gpu[@]}" -- scripts/run-fusion.sh "$run")
echo "fusion: $job"
