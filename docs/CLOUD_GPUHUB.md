# GPUHub — provider-specific reference (research only, no instance rented yet)

> Companion to `docs/CLOUD.md` (the provider-agnostic playbook). `CLOUD.md` was written with
> RunPod's Docker-Hub-pull model as the primary assumption (D-017); this doc records what's
> actually true on **gpuhub** specifically, from a full read of `docs.gpuhub.com` on 2026-07-12
> (33 pages, via 4 parallel research passes — no billed instance was rented to verify any of
> this, so treat exact numbers/paths as "docs say X, confirm live at first boot").
>
> **Documentation provenance note:** gpuhub's docs are very likely a rebrand of the Chinese GPU
> platform **AutoDL** — several pages still show `autodl-tmp`/`autodl-fs`/`autodl-nas` path names
> instead of `gpuhub-*`, and the rebrand is inconsistent page-to-page. Don't hardcode any path
> string from this doc into a script without confirming it via `ls /root/` on first boot.

## 1. The one big architecture conflict: no Docker Hub

**gpuhub explicitly does not support pulling custom images from any third-party registry**
(Docker Hub, GHCR, GitLab — all named and ruled out). Quote from their docs: *"GPUHub DOES NOT
support deploying containers using Docker image hosted on 3rd-party container registries."*

This breaks the core assumption of `docs/CLOUD.md`'s Docker fast-start section and **D-017**
(build `docker/Dockerfile` locally with `buildx --platform linux/amd64` → push to Docker Hub →
point the pod template at `docker.io/<user>/llmlab:tag`). That workflow **cannot run on gpuhub
as written** — there is no field anywhere in instance creation for a registry URL.

**gpuhub's native replacement mechanism ("Save Image"):**
1. Rent an instance from gpuhub's own pre-built catalog (PyTorch × Python × CUDA combos — see
   §2 below for which combo we need).
2. SSH in, run our setup steps by hand or as a script — effectively the `RUN` lines from
   `docker/Dockerfile` (`git clone`, `pip install -r requirements.txt`, rclone setup, etc.)
   executed as shell commands instead of baked at `docker build` time.
3. Shut the instance down → "More" menu → **Save Image** → snapshots the *entire system disk*
   into "My Images" (private, optionally shareable with named users, not a public registry).
4. Future rentals pick that saved image at instance creation — skips the setup step entirely
   on repeat rentals, same goal as the Docker Hub image just via a different mechanism.

**This is a real decision point, not yet made** — see "Open decision" at the bottom of this doc.
`docker/Dockerfile` and `docker/entrypoint.sh` are not wasted work either way: if gpuhub ends up
being the provider, the Dockerfile's `RUN` lines are still the reference for what the setup
script needs to do; the artifact just becomes a gpuhub-saved-image instead of a Docker Hub tag.
If we go back to RunPod (or Vast.ai, Lambda etc.) later, the existing Docker Hub plan works
unchanged there.

## 2. Environment / images

- **Pre-built catalog**: PyTorch 1.1.0–2.8.0 × Python 3.7–3.12 × **CUDA 10.0–12.8**, plus
  TensorFlow/JAX/PaddlePaddle/TensorRT/Gromacs images. OS is Ubuntu (mostly 18.04, some 20.04).
- **RTX 5090 requirement**: docs state plainly — *"RTX 5090 and RTX PRO 6000 require CUDA ≥ 12.8
  and PyTorch ≥ 2.7.1 to detect and use the GPU at all."* Multi-GPU DDP on these cards needs
  **PyTorch ≥ 2.8.0 nightly** specifically (stable 2.8.0 may not be enough for DDP, though single-
  GPU should be fine on stable — our plan is single-GPU for now, D-018).
  **Action at rental time:** select the CUDA-12.8 catalog entry explicitly; don't trust
  `nvidia-smi` to confirm the installed CUDA version (it only reports the driver's *max
  supported* CUDA). Verify with `ldconfig -p | grep cuda` after boot instead.
- **Miniconda** preinstalled at `/root/miniconda3/` on every image; base Python is 3.8, other
  versions via `conda create -n env python=X.Y`. Redirect conda's package/env dirs to the data
  disk (`/root/gpuhub-tmp/conda/...` + `.condarc`) since the system disk is small (see §3).
- **No Docker-in-Docker** inside the container instance itself.
- Community-shared images can have a slow (>1hr) first boot — prefer official catalog images or
  our own saved image over a random community one.

## 3. Storage — four tiers, know which one to use

| Path (docs; verify live) | Size | Persistence | Speed | Use for |
|---|---|---|---|---|
| `/` (system disk) | 30GB, **not resizable** | Wiped by reset / image swap | fast local SSD | OS + installed packages only — nothing precious |
| `/root/gpuhub-tmp` (data disk) | 50GB free, resizable (extra cost) | Survives stop/restart of *this* instance; wiped after 15 continuous days stopped or if host lease lapses; **not included in Save Image** | fast local SSD | tokenized data, checkpoints, active training scratch |
| `/root/autodl-fs` / `/root/gpuhub-fs` (file storage) | 20GB free tier | Survives instance **termination**; remountable on a **different** instance **within the same region**; wiped after 3 months account inactivity or >$10 unpaid balance | slower (network), 200,000-file inode cap | durable cross-rental storage — closest thing to RunPod's network volume |
| `/root/gpuhub-pub` (public data) | — | read-only | — | platform-provided public datasets, irrelevant to us |

**Key implication for our plan:** put tokenized `.bin` files and checkpoints on the **data
disk**, not `/`. Local disks (system + data) have **no redundancy guarantee** — gpuhub explicitly
disclaims reliability on them ("probability of failure... no reliability guarantee"). File
storage is the durable tier but is region-locked and slower — good for "don't re-download from R2
every single rental in the same region," not good as primary training I/O (docs themselves
recommend copying file-storage data to local disk before training).

**Open trade-off worth revisiting once we've measured actual transfer speed:** pull straight from
R2 to local data disk every rental (matches our current plan exactly, simple, costs a small
re-download each time) vs. stage once into File Storage and remount it on subsequent same-region
rentals (saves repeat R2 pulls, but adds region lock-in and the inode/reliability caveats above).

## 4. Our R2 + rclone data plan — works, with one fix

- **Cloudflare R2 is explicitly named** as a supported rclone backend (alongside S3/B2/Wasabi
  etc.) — our existing `.env`/`RCLONE_CONFIG_R2_*` setup (D-026) needs no credential-format
  changes.
- **`rclone mount` is blocked** by gpuhub's container security ("will fail with permission
  errors"). Only `rclone copy` / `rclone sync` (pull to a local path, don't live-mount) or
  `rclone serve webdav` are supported. Our plan already pulls data via rclone at instance start —
  just make sure `docker/entrypoint.sh` / the future setup-script equivalent uses `rclone copy`,
  never `rclone mount` (it already does — `rclone copy --progress` in the current entrypoint).
- No stated bandwidth caps or egress fees for rclone transfers in the docs (not proof of none —
  worth a small test transfer before committing to a long billed run).
- Getting checkpoints back OUT: SCP / FileZilla / JupyterLab (files only, no folders) are the
  documented built-ins; `rclone sync` back to R2 works too (bidirectional), just isn't shown in
  their own download-data examples.
- For bulk small-file transfers, `tar cf - * | ssh ... "tar xf -"` avoids per-file SSH overhead
  (relevant if we ever move raw per-doc files instead of our binary memmap shards).
- Compression note: use plain `tar` (no compression) for our `uint16` memmap files and model
  checkpoints — they're already dense binary and won't compress meaningfully; gpuhub's `arc` tool
  or standard `zip`/`tar.gz` exist if needed for other content.

## 5. SSH, background jobs, VS Code

- **Connect**: `ssh -p <PORT> root@connect.<region>.gpuhub.com` (port/host shown per-instance in
  the console after boot). Always `root`. Plain sshd, no jump host.
- **Auth**: password by default (shown in console); add a public key under Console → Instances
  for passwordless login (standard `ssh-keygen -t ed25519`, same key we already generated per
  `CLOUD.md` step 1 works — just add it to gpuhub's console too).
- **Surviving disconnects**: no proprietary mechanism — gpuhub's own docs recommend **tmux** or
  **screen**, exactly matching our existing `CLOUD.md` "always inside tmux" rule. JupyterLab's
  terminal is offered as an even simpler alternative (survives tab close as long as JupyterLab
  itself doesn't restart) but tmux is more scriptable and matches our terminal-script-not-notebook
  convention (CLAUDE.md).
- **VS Code Remote-SSH works out of the box** — standard extension, point it at the dashboard's
  SSH string. **Gotcha called out explicitly in their docs**: strip trailing whitespace from the
  copied SSH command before pasting into VS Code, or the connection silently fails.

## 6. Ports / dashboards

Public exposure is reverse-proxied and **limited to ports 6006 and 6008 only** (mapped to an
address like `region-x.gpuhub.com:<port>`, found under "Custom" in console) — **no
authentication layer**, so anything bound there is effectively public. Good enough for a
TensorBoard-style dashboard; **our wandb usage doesn't need this** since wandb reports to
wandb.ai over outbound HTTPS, not an inbound port. For anything else (e.g. a debug port), use an
SSH local port-forward (`ssh -p <PORT> -L <local>:localhost:<remote> root@<host>`) instead —
private to us, not restricted to 6006/6008.

## 7. Billing

- Two modes only: **Pay-as-you-go** (hourly — what we want for a first/exploratory rental) and
  **Subscription** (monthly/yearly). **No spot/preemptible tier documented** (unlike RunPod).
- Switch between modes via the instance's "More" menu. Leaving a Subscription early pro-rates a
  refund but forfeits unused coupon balance.
- **Cost-saving pattern documented by gpuhub itself**: a **No-GPU mode at $0.10/hr** (0.5 CPU
  core, 2GB RAM, no GPU) — meant for prepping code/env/data before paying full GPU rate. Caveat:
  switching back to GPU mode isn't guaranteed instantly available if that GPU tier gets rented out
  by someone else in the meantime — don't leave a long gap between "confirmed the script works"
  and "start the real run."
- **Auto-shutdown pattern**: chain `python train.py && /usr/bin/shutdown` so the instance stops
  itself the moment training finishes, instead of idling (and billing) until you notice.
- **Still unconfirmed** (not found in any of the 33 fetched pages — need the pricing/GPU-catalog
  page specifically): actual RTX 5090 $/hr rate, whether RTX 5090 is even in current inventory,
  default system disk size on the tier we'd pick, exact stop-vs-terminate billing distinction.
  **Send the pricing/GPU-catalog page URL next and I'll fold it in before we budget hours.**

## 8. Multi-GPU (future, not needed now — project is single-GPU)

Single-node multi-GPU (2+ GPUs in one instance, DDP) is supported and is the documented path.
**True multi-instance networked distributed training is explicitly unsupported for non-A100 GPUs**
(no InfiniBand/NVLink) — which includes RTX 5090. So if we ever outgrow one GPU, the move on this
platform is "rent one instance with more GPUs," not "network several instances."

## 9. Misc operational notes

- **reset-system**: wipes `/` back to the base image; data disk/file storage untouched. Real risk
  is losing an environment that only lives on `/` — always install into `/root` but keep
  data/checkpoints on the data disk so a reset can't destroy a run.
- **save-image**: the instance must be stopped first (steps not 100% explicit in docs — verify
  live); data disk is **excluded** from the saved image (only `/` is snapshotted), so a saved
  image is deps/env only, not a data backup.
- **migrate-instance / migrate-instance-sr**: relocating to different physical hardware, same- or
  cross-region. Not relevant unless our rented host runs out of capacity mid-series. Billing on
  the destination instance starts the moment it's running, independent of whether data-copy has
  finished — schedule shutdown if migrating to avoid idle billing.
- **Disk-full troubleshooting**: `rm -rf /root/miniconda3/pkgs/*` (conda cache), empty JupyterLab
  trash (`rm -rf /root/.local/share/Trash`), check `/tmp` and `~/.cache` — the standard playbook
  if we hit "system disk space insufficient" mid-setup.

## Decision: RESOLVED (D-027, 2026-07-12) — this is now the live playbook

gpuhub is the active provider (~50% cheaper than RunPod at the same hardware tier — user's
quoted pricing: RTX 5090 $0.46/hr vs $0.99/hr; RTX PRO 6000 $0.91/hr vs $1.99/hr). **Option (a):**
fully adapt to gpuhub's native flow — base catalog image → live setup script → Save Image →
reuse. RunPod's Docker-Hub plan (`docs/CLOUD.md`) is kept intact as a documented, unbuilt
fallback — not touched further right now. Everything below this line is the actual sequence
we're executing, updated as we go (per the user's ask: keep notes live, draft a polished
step-by-step manual once the pipeline is verified end-to-end on gpuhub's servers).

## 10. GPU tier pick

**RTX 5090** is the default pick (matches D-018's reasoning: our ~10-105M-param models are
nowhere near VRAM-bound, so RTX PRO 6000's extra VRAM at ~2x the price buys nothing). At gpuhub's
pricing this is $0.46/hr — a 1500-step S-tier-equivalent run is minutes; an L-tier hero run
(~2.1B tokens) will need real hour-budget math once we've measured actual gpuhub 5090 tok/s
(don't assume RunPod/Mac numbers port over — D-018's "measure, don't assume, on new hardware"
rule applies here too).

**Recommended sequencing to minimize billed debugging time** (per `docs/CLOUD.md`'s golden rule
— GPU time is for training only):
1. First rental: gpuhub's **No-GPU mode ($0.10/hr)** — verify SSH access, confirm real mount
   paths (`ls /root/`, since docs disagree with themselves on `autodl-*` vs `gpuhub-*` naming),
   run `scripts/cloud/gpuhub_setup.sh`, pull a small piece of R2 data via rclone, confirm
   everything works — all without paying GPU rate.
2. Second rental: short RTX 5090 session — confirm `ldconfig -p | grep cuda` shows 12.8,
   confirm PyTorch detects the GPU, run the S-tier smoke config for a couple hundred steps as a
   correctness check (same idea as `docs/CLOUD.md`'s "$1 dry run").
3. Once both pass: Save Image so future rentals skip steps 1-2 entirely.
4. Real training runs from the saved image.

## 11. Setup script

`scripts/cloud/gpuhub_setup.sh` is the gpuhub-native equivalent of `docker/entrypoint.sh` —
run it manually after SSH-ing into a fresh instance (not an automatic container entrypoint,
since gpuhub instances aren't launched from our image). It clones the repo, installs deps,
verifies CUDA, and pulls tokenized data from R2 — see the script itself for the exact commands
and the path-verification step (it doesn't hardcode `/root/autodl-tmp` vs `/root/gpuhub-tmp`;
it detects which one actually exists on first run). It reuses `scripts/cloud/remote_setup.sh`
unchanged for deps-install + CUDA verification — that script was already provider-agnostic.

**Full sequence, step by step:**
```bash
# 1. On the Mac, before renting: scp your .env up once you have host/port (never via git)
scp -P <PORT> .env root@<HOST>:/root/.env

# 2. SSH in
ssh -p <PORT> root@<HOST>

# 3. On the instance
curl -fsSL https://rclone.org/install.sh | bash   # if not already present on the image
bash <(curl -fsSL https://raw.githubusercontent.com/Adityaram0001/llm/main/scripts/cloud/gpuhub_setup.sh)
# (or scp the script up / git clone first and run it locally, whichever is convenient)
```
`scripts/cloud/sync_down.sh` (pull `experiments/` back to the Mac) still works unchanged over
plain SSH — just set `scripts/cloud/remote.env`'s `REMOTE_DIR` to wherever `gpuhub_setup.sh`
cloned the repo (the data-disk path it prints, e.g. `/root/gpuhub-tmp/llm`), `REMOTE_PORT`/
`REMOTE_HOST` to the gpuhub SSH string, `SSH_KEY` to whichever key you added in gpuhub's console.
`sync_up.sh` is RunPod-flavored (rsyncs the whole repo including anything not gitignored) and
isn't part of the gpuhub flow — gpuhub gets code via `git clone` and data via `rclone`, not rsync.
