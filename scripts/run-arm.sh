#!/usr/bin/env bash
# Train one ProtoECGNet branch: learn prototypes, project them onto the train set, train
# the classifier, then evaluate on test (bootstrapped) and publish to shared storage.
#
#   slurm/submit.sh --gres=gpu:1 --mem-per-gpu=450G -- scripts/run-arm.sh <arm> <run>
#   <arm>: a config in configs/ (1D-global, 2D-global, 2D-partial)
#   <run>: run name; staged in $RUN_ROOT/outputs/<run>/<arm>, published to
#          $SHARED_ROOT/runs/<run>/<arm>
set -euo pipefail
usage="usage: scripts/run-arm.sh <arm> <run>"
arm=${1:?$usage}
run=${2:?$usage}
source "$(dirname "$0")/_common.sh"

config=configs/$arm.yaml
[[ -f $config ]] || { echo "no config $config" >&2; exit 1; }
stage=$RUN_ROOT/outputs/$run
dest=$SHARED_ROOT/runs/$run/$arm
# checked before training rather than at publish time, hours later
[[ ! -e $dest ]] || { echo "$dest already exists" >&2; exit 1; }
mkdir -p "$stage" "$(dirname "$dest")"

common=(
  --config "$config"
  --trainer.logger.save_dir "$stage"
  --trainer.logger.name "$arm"
  --data.dataset_path "$DATASET_PATH"
  --data.data_kwargs "$DATA_KWARGS"
  --data.num_workers 8
  --data.prefetch_factor 4
)

# label supervised prototype learning
python -m protoecgnet_heedb.trainer --pipeline-stage learn-prototypes-supervised "${common[@]}"

# project prototypes onto the train set
python -m protoecgnet_heedb.trainer --pipeline-stage project-prototypes-supervised "${common[@]}" \
  --model.pretrained_weights "$stage/$arm/learn-prototypes-supervised/latest/best.ckpt"

python -m protoecgnet_heedb.trainer --pipeline-stage train-classifier "${common[@]}" \
  --model.pretrained_weights "$stage/$arm/project-prototypes-supervised/latest/proj.ckpt"

python scripts/eval_probs_bootstrapped.py \
  --dataset-path "$DATASET_PATH" \
  --data-kwargs "$DATA_KWARGS" \
  --label-subset-config "$config" \
  --probs-npy "$stage/$arm/train-classifier/latest/test_probs.npy" \
  --output-path "$stage/$arm"

cp "$stage/$arm/train-classifier/latest/test_probs.npy" "$stage/$arm/test_probs.npy"
publish "$stage/$arm" "$dest"
