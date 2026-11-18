#!/usr/bin/env bash
# Train the fusion classifier over the three published branches of a run, evaluate on
# test (bootstrapped) and publish to shared storage.
#
#   slurm/submit.sh --gres=gpu:1 --mem-per-gpu=450G -- scripts/run-fusion.sh <run>
#   reads $SHARED_ROOT/runs/<run>/<arm>, publishes $SHARED_ROOT/runs/<run>/fusion
set -euo pipefail
usage="usage: scripts/run-fusion.sh <run>"
run=${1:?$usage}
source "$(dirname "$0")/_common.sh"

arms=(1D-global 2D-global 2D-partial)
exp=fusion
stage=$RUN_ROOT/outputs/$run
runs=$SHARED_ROOT/runs/$run
dest=$runs/$exp
[[ ! -e $dest ]] || { echo "$dest already exists" >&2; exit 1; }
mkdir -p "$stage"

branches=()
for arm in "${arms[@]}"; do
  ckpt=$runs/$arm/project-prototypes-supervised/latest/proj.ckpt
  [[ -f $ckpt ]] || { echo "missing $ckpt (has $arm been published?)" >&2; exit 1; }
  branches+=(
    --model.branches+=protoecgnet_heedb.models._protoecgnet.BranchCfg
    --model.branches.name="$arm"
    --model.branches.config="configs/$arm.yaml"
    --model.branches.pretrained_weights="$ckpt"
  )
done

python -m protoecgnet_heedb.trainer \
  --pipeline-stage train-fusion-classifier \
  --config configs/$exp.yaml \
  --trainer.logger.save_dir "$stage" \
  --trainer.logger.name $exp \
  --data.dataset_path "$DATASET_PATH" \
  --data.data_kwargs "$DATA_KWARGS" \
  "${branches[@]}" \
  --data.num_workers 8 \
  --data.prefetch_factor 4

python scripts/eval_probs_bootstrapped.py \
  --dataset-path "$DATASET_PATH" \
  --data-kwargs "$DATA_KWARGS" \
  --probs-npy "$stage/$exp/train-fusion-classifier/latest/test_probs.npy" \
  --output-path "$stage/$exp"

cp "$stage/$exp/train-fusion-classifier/latest/test_probs.npy" "$stage/$exp/test_probs.npy"
publish "$stage/$exp" "$dest"
