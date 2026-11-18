# ProtoECGNet-HEEDB

ProtoECGNet trained on HEEDB, for the large user study. The code was carved out of
[ProtoSSL](https://github.com/StevenSong/ProtoSSL) (its `large-user-study` branch) and now
develops here on its own. The two are expected to drift.

The model is three prototype branches, each covering a subset of the 20 HEEDB labels, and a
fusion classifier over them:

| Branch | Backbone | Prototypes | Labels |
|---|---|---|---|
| `1D-global` | 1D ResNet-18 | whole ECG | rhythm (10) |
| `2D-global` | 2D ResNet-18 | whole ECG | normal sinus rhythm |
| `2D-partial` | 2D ResNet-18 | 1 s windows, 50% overlap | morphology (9) |
| `fusion` | the three above, frozen | | all 20 |

The data is HEEDB (MGB + Emory) with patient-level random splits (`by-label`). ECGs whose
physician overread is blank are dropped. The ground truth is `audited_physician`: overread
codes tagged by where they sit in the read, with statements that don't describe the current
ECG left unknown (see `src/protoecgnet_heedb/datasets/_heedb_label_rules.py`).

## Layout

```
src/protoecgnet_heedb/   package: HEEDB dataset, ProtoECGNet models, Lightning trainer, plotting
configs/                 one config per branch, plus fusion
scripts/                 stage scripts run by jobs (run-arm.sh, run-fusion.sh), eval, cache warm-up
experiments/             queue scripts that submit a whole run
slurm/                   job runner (submit.sh, job.sh) and GPU leases for the dev container
notebooks/               MUSE comparison, data and overread tables
case-selector/           offline bundle for picking user study cases
data-preprocessing/      label audit of HEEDB physician overreads
envs/                    pinned uv installer
```

## Environment

uv, Python 3.10.19. `uv.lock` pins every package to the version in ProtoSSL's paper env, the
stack the v4 run was trained on, including torch 2.7.0 (CUDA 12.8). `uv sync` builds it; jobs
do this themselves (below).

## Running

Jobs run a pushed commit, not the working tree. From a clean clone on a host with the Slurm
client:

```bash
experiments/queue-heedb.sh v5 -w kg35-nvl01
```

This queues the three branches in parallel and the fusion classifier after them.
`slurm/submit.sh` refuses a dirty tree or an unpushed commit. On every node, `slurm/job.sh`
sets up a clone of that commit and its env under `~/protoecgnet-heedb-run/commits/<sha>/`,
then runs the stage there.

Each stage trains in node-local staging (`~/protoecgnet-heedb-run/outputs/<run>/`). When it
finishes, it publishes to `/opt/gpu_working/steven/protoecgnet-heedb-shared/runs/<run>/<branch>/`
in one rename, without the `wandb/` dirs and with a `COMMIT` file. The fusion job reads the
branch checkpoints from there, so it can run on any node. Job logs are copied to the shared
`logs/`.

### HEEDB caches

Training loads each split's waveforms fully into memory (`HIGH_MEMORY=1`), from a node-local
cache in `CACHE_DIR` (default `~/.cache/protossl_cache`, shared with ProtoSSL). The train split
is about 310 GB, and building it takes about twice that in memory. The first job on a node
without the cache builds it while the others on that node wait. nvl01 already has the caches,
hence `-w kg35-nvl01` above. To warm another node without holding GPUs:

```bash
slurm/submit.sh -p h200 -w <node> --mem=850G -- python scripts/cache_heedb.py
```

Cache keys hash the inputs and the code that builds them, so entries made by ProtoSSL are
reused here only where the code is unchanged. That reuse only holds while these files keep
their current content:

- `datasets/_heedb_label_rules.py`: label caches are keyed on the file's bytes, comments
  included.
- `datasets/streaming_loaders/_streaming_ecgs.py`: waveform caches are keyed on its parsed
  source.
- `get_heedb_metadata` in `datasets/_heedb_dataset.py`: the metadata cache is keyed on its
  parsed source.

Editing them is fine but rebuilds the affected caches on every node.

## Results

The v4 run, made with ProtoSSL's `large-user-study` scripts, is archived on nvl01 at
`~/protoecgnet-heedb`, in the same layout as a published run here. Its
`metrics-bootstrapped-v2.csv` files are reproduced byte for byte by
`scripts/eval_probs_bootstrapped.py` from its `test_probs.npy`. The notebooks read the v4
archive by default (`run_path`) and write tables and figures to `notebooks/outputs/`.

## Not carried over

- `compute-fusion-embeddings` is still a trainer stage, but there is no script for it yet
  (ProtoSSL's `run-protoecgnet-heedb-fusion-embed.sh` pointed at runs that no longer exist).
- The wandb lambda and prototypes-per-label sweeps.
