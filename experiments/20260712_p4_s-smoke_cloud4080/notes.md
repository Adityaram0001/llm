# 20260712_p4_s-smoke_cloud4080

**Hypothesis:** the training engine runs unchanged on a rented gpuhub CUDA instance -- the cloud
portability canary named as a to-do in D-010/D-022/CLOUD.md, extended to gpuhub specifically
after D-027 (gpuhub chosen as provider). Run on an RTX 4080 Super dry-run instance (D-028), not
the target RTX 5090, deliberately -- Ada Lovelace has none of Blackwell's CUDA>=12.8 wrinkles, so
this run isolates "does our pipeline work on gpuhub's infra" from "does it work on this specific
new GPU architecture."

**Setup:** `scripts/cloud/gpuhub_setup.sh` (git clone + `remote_setup.sh` deps/CUDA check +
`rclone copy` full `data/tokenized/` from R2, 2.874 GiB in ~3 min at ~14 MiB/s) ran clean on a
fresh instance -- gpuhub's PyTorch 2.8.0 / Python 3.12 / CUDA 12.8 / Ubuntu 22.04 catalog image.
Same `configs/train_s_smoke.yaml` as the original Mac run (`20260711_p4_s-smoke`), `--device cuda`.

**Observation:** 150 steps, effective batch ~65,536 tokens/step. train_loss 9.692 -> 5.393,
val_loss 9.437 -> 5.259 (best 5.2594) -- essentially identical trajectory to the original Mac MPS
smoke run (9.69 -> 5.38), confirming the port is numerically sane, not just "didn't crash."
**Throughput: 99,554 tok/s** -- ~8.5x D-022's measured Mac MPS number (~11,000-11,800 tok/s) on a
mid-tier consumer GPU, not even the target 5090. 112s wall-clock for the full 150-step run
(9.57M tokens). Samples show recognizable English word/punctuation structure by step 100, same
qualitative stage as the Mac run's samples.

Checkpoint round-trip verified: `latest.pt` saved on CUDA, pulled back to the Mac via `scp`, and
loaded successfully via `torch.load(..., map_location=get_device())` on **MPS** -- confirms
CLOUD.md's cross-device portability rule works in practice, not just in theory. `metrics.jsonl`,
`config.yaml` (records `device: cuda`), wandb-offline logs, and both `samples/step_*.txt` all
present and complete.

**Real findings vs. the docs research (`docs/CLOUD_GPUHUB.md`), now confirmed live:**
- Data-disk mount is `/root/autodl-tmp` on this host (docs disagreed with themselves on
  `autodl-tmp` vs `gpuhub-tmp` -- confirmed which one is real).
- `nvidia-smi` reported "CUDA Version: 13.2" (driver ceiling) while the actual installed/used
  CUDA runtime is 12.8 (`torch.version.cuda` == `'12.8'`) -- exactly the gotcha the docs warned
  about, now empirically verified rather than just documented as a warning.
- Non-interactive SSH sessions do NOT have conda's `python`/`pip` on `PATH` by default (no
  `conda init` has been run on a fresh image) -- `gpuhub_setup.sh` now fixes this explicitly
  rather than assuming it.
- `rclone` and `tmux` are NOT preinstalled on this catalog image -- `gpuhub_setup.sh` installs
  rclone; `remote_setup.sh` already installed tmux via apt.
- `rclone copy` (not mount) worked exactly as documented, R2 as the source, no bandwidth issues.

**Conclusion:** gpuhub cloud portability canary passes. `scripts/cloud/gpuhub_setup.sh` is
validated end-to-end on real gpuhub infrastructure (SSH, git clone, deps, CUDA, R2 data pull,
training, checkpointing, cross-device checkpoint load). Safe to proceed to Save Image and,
whenever RTX 5090 inventory is available, repeat only the CUDA-version-specific verification
step on that hardware -- everything else in this run already proves out unchanged.
