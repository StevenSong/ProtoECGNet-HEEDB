"""Build the node-local HEEDB caches the training jobs read: metadata, overread mask,
audited_physician labels and the in-memory waveform tensors for each split.

Optional: the first training job on a node builds any missing cache itself, and the others
on that node wait for it. Run this first to warm a node without holding GPUs idle meanwhile.
Building the train split needs about twice its ~400 GB in memory.

    slurm/submit.sh -p h200 -w <node> --mem=850G -- python scripts/cache_heedb.py
"""

import os

os.environ["HIGH_MEMORY"] = "1"

from protoecgnet_heedb.datasets import HeedbECGDataset

DATASET_PATH = "/opt/gpudata/ecg/heedb"

for split in ["train", "val", "test"]:
    HeedbECGDataset(
        dataset_path=DATASET_PATH,
        sampling_rate=100,
        split=split,
        heedb_split_type="by-label",
        label_src="audited_physician",
        drop_blank_overread=True,
    )
