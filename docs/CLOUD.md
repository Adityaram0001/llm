# CLOUD — renting an NVIDIA GPU for big runs (RunPod / gpuhub, RTX 5090)

The Mac (M4/MPS) is the primary lab: all development, notebooks, S-tier ablations. A rented
RTX 5090 is the **burst option** for M/L-tier and the phase-9 hero run (see D-010: a run that
needs ~2–3 weeks on the Mac finishes overnight on a 5090 for roughly $10–20). This doc is the
complete first-timer playbook. The user has never rented a GPU before — sessions helping with
a cloud run should walk through this step by step.

## The mental model

A "pod" is a Linux box (Ubuntu) with an NVIDIA GPU that bills **per hour from start to stop,
whether or not the GPU is doing anything**. It usually starts from a Docker image with
CUDA + PyTorch preinstalled. You get SSH access as root. Its disk is (mostly) ephemeral:
**anything not synced back to the Mac before termination is gone.** Therefore the golden rule:

> **GPU time is for training only.** Prepare everything on the Mac (code smoke-tested, data
> tokenized, config frozen). Rent → sync up → verify (5 min) → train in tmux → sync down → STOP.

## One-time setup (before first rental)

1. SSH key (if `~/.ssh/id_ed25519.pub` doesn't exist):
   `ssh-keygen -t ed25519 -C "adityaram-llmlab"` — add the **public** key in the provider's
   web console (RunPod: Settings → SSH Public Keys). Never copy the private key anywhere.
2. Create account, add ~$10–25 credit. RunPod is the recommended first provider (per-second
   billing, big community-image library, good docs); gpuhub is fine too — the workflow below
   is provider-agnostic, only the console clicks differ.
3. wandb: for cloud runs flip to **online** mode (`WANDB_MODE=online`, `wandb login` on the
   pod with your API key from wandb.ai/authorize) — live monitoring from the Mac's browser is
   exactly what you want during a paid run. (Local D-009 offline default is unchanged.)

## Per-run workflow

### 1. On the Mac — prepare (GPU clock NOT running)
- Freeze the run config in `configs/`, smoke-test the exact command locally for ~100 steps.
- Fill `scripts/cloud/remote.env` (copy from `remote.env.example`) with the pod's
  host/port once you have them.

### 2. Rent the pod
- Pick RTX 5090 (32GB VRAM), **On-Demand** (not spot/interruptible, for a first run),
  a PyTorch 2.x + CUDA 12.8+ image (5090 is Blackwell/sm_120 — **needs CUDA ≥12.8 builds**;
  older cu121/cu124 torch wheels will not run on it), 40–60GB disk.
- Note the SSH host + port from the console; test: `ssh -p <port> root@<host>`.

### 3. Sync up + set up (~5 min)
```bash
./scripts/cloud/sync_up.sh                       # code + tokenized data → pod
ssh -p <port> root@<host>
cd /workspace/llm-lab && bash scripts/cloud/remote_setup.sh   # deps + CUDA verification
```

### 4. Train — ALWAYS inside tmux (survives SSH disconnects)
```bash
tmux new -s train
python scripts/train.py --config configs/train_l_hero.yaml
# detach: Ctrl-b then d   |   reattach after reconnect: tmux attach -t train
```
Monitor: wandb dashboard from the Mac; `nvidia-smi -l 5` in a second tmux window
(GPU-util should sit >80%; if it's low, the data loader or batch size is the bottleneck —
on CUDA try bigger micro-batch, `num_workers=4`, `pin_memory=True`).

### 5. Sync down, then STOP the pod
```bash
# on the Mac — pulls experiments/<run_id>/ (metrics, notes, samples, latest/best ckpt)
./scripts/cloud/sync_down.sh
```
Verify the checkpoint loads locally (`map_location` is handled by `get_device()`), THEN stop.
"Stop" usually keeps the disk for cents/hour; "Terminate" deletes everything. During multi-day
runs, run `sync_down.sh` periodically — checkpoints are resume-tested (phase 4), so even a
killed spot pod only costs you the time since the last checkpoint.

## Portability rules (Mac/MPS ↔ Linux/CUDA) — bake into ALL project code

Already handled if code follows the conventions; every session must keep it that way:

1. **Device**: only via `llmlab.utils.get_device()` (cuda > mps > cpu) and
   `autocast_ctx(device)` — never literal `"mps"`/`"cuda"`, never `torch.autocast("mps", ...)`.
2. **Guard backend-specific calls**: anything `torch.mps.*` / `torch.cuda.*` behind the
   matching `is_available()` (utils' `mem_stats`/`set_seed` show the pattern).
   `torch.mps.synchronize()` in benches → use `torch.accelerator`/device-dispatched sync.
3. **Checkpoints**: save with plain `torch.save`; load with
   `torch.load(p, map_location=get_device())`. CUDA-trained checkpoints load fine on MPS and
   vice versa. Log `device` + torch version into every run's config.yaml for the record.
4. **DataLoader**: `num_workers=0` on Mac (MPS + fork issues), configurable — on CUDA use
   `num_workers=4, pin_memory=True`. Make these `train_*.yaml` keys, not code constants.
5. **On CUDA, additionally enable**: `torch.set_float32_matmul_precision("high")` (TF32) —
   harmless no-op elsewhere. `torch.compile` is *reliable* on CUDA: make it a config flag,
   off by default locally, ON for cloud runs (free ~1.3–2× speedup).
6. **Paths**: `pathlib` + repo-relative everything (already the convention). `uint16` memmaps
   are byte-identical across both platforms — sync once, no rebuild.
7. **Env vars**: `PYTORCH_ENABLE_MPS_FALLBACK=1` is harmless on Linux; don't write
   Mac-only assumptions into scripts (e.g. `caffeinate` → only invoke if `darwin`).
8. **requirements.txt works on both**: pip resolves the right torch wheel per platform. On
   pods, the image's torch already matches its CUDA — `remote_setup.sh` checks it and skips
   reinstalling torch unless it's below our floor. CUDA-only extras (flash-attn) stay OUT of
   requirements.txt; if experimented with on a pod, config-flagged and optional.

## Cost discipline

- Estimate before renting: tokens ÷ measured tok/s → hours × rate (5090 ≈ $0.7–1.0/hr
  on community tiers; check current). Write the estimate into the run's notes.md; compare after.
- The meter runs while you debug. If setup exceeds ~20 min, stop the pod, fix locally, re-rent.
- Never store the only copy of anything on a pod. Never put API keys in the repo — export
  `WANDB_API_KEY` in the pod shell (it dies with the pod).
- First rental ever: do a **$1 dry run** — rent the cheapest GPU for 30 min and walk steps
  3–5 with the S-tier smoke config, just to practice the loop before the real thing.

## What does NOT change

Experiment discipline is identical: run folder, registry row, metrics.jsonl, decision log.
A cloud run gets `cloud-5090` noted in its registry `verdict`/notes and config. Ablation
comparisons must stay same-hardware (never compare wall-clock across Mac and 5090 runs;
tokens-based curves remain comparable).
