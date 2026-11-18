# Sourced by run-arm.sh and run-fusion.sh: shared settings and the publish step.

# HEEDB by-label splits, blank overreads dropped, audited physician labels (v4 setup)
DATASET_PATH=/opt/gpudata/ecg/heedb
DATA_KWARGS='{"heedb_split_type": "by-label", "drop_blank_overread": True, "label_src": "audited_physician"}'

# load each split's waveforms into memory (~400 GB for train), from the node-local cache
export HIGH_MEMORY=1

RUN_ROOT=${PROTOECGNET_HEEDB_RUN_ROOT:-$HOME/protoecgnet-heedb-run}
SHARED_ROOT=${PROTOECGNET_HEEDB_SHARED_ROOT:-/opt/gpu_working/steven/protoecgnet-heedb-shared}

# publish <staged dir> <shared dir>: copy a finished stage to shared storage, without wandb
# (it syncs to the wandb server, and it's many small files on NFS). Copied to a temp dir
# beside the target and renamed, so a partial copy is never visible under the final name.
publish() {
  local src=$1 dest=$2
  local tmp
  tmp=$(dirname "$dest")/.$(basename "$dest").tmp-${SLURM_JOB_ID:-$$}
  mkdir -p "$tmp"
  tar -C "$src" --exclude=wandb -cf - . | tar -C "$tmp" -xf -
  echo "${PROTOECGNET_HEEDB_COMMIT:-unknown}" >"$tmp/COMMIT"
  mv -T "$tmp" "$dest"
  echo "published $src to $dest"
}
