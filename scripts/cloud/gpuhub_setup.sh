#!/bin/bash
# Run ON A FRESH GPUHUB INSTANCE after SSH-ing in (root@...). This is the gpuhub-native
# equivalent of docker/entrypoint.sh: gpuhub can't pull our Docker Hub image (D-027,
# docs/CLOUD_GPUHUB.md §1 — gpuhub refuses third-party registry pulls), so instead of
# `docker run` pulling a prebuilt image, we clone the repo + pull data live over SSH once,
# then "Save Image" from the console (More -> Save Image) so future rentals skip all of this.
#
# Before running: scp your local .env up (it's gitignored, never touches GitHub/the saved image
# unless you deliberately keep it — consider re-scp'ing per-rental instead of baking secrets
# into a shared image):
#   scp -P <PORT> .env root@<HOST>:/root/.env
#
# Usage (defaults assume this repo/branch; override via env vars if needed):
#   GIT_REPO_URL=https://github.com/Adityaram0001/llm.git GIT_BRANCH=main bash gpuhub_setup.sh
set -euo pipefail

GIT_REPO_URL="${GIT_REPO_URL:-https://github.com/Adityaram0001/llm.git}"
GIT_BRANCH="${GIT_BRANCH:-main}"

# gpuhub's own docs disagree with themselves on the data-disk mount name (autodl-tmp in some
# pages, gpuhub-tmp in others -- see docs/CLOUD_GPUHUB.md's provenance note). Detect whichever
# actually exists on THIS instance instead of trusting either string.
DATA_DISK="${DATA_DISK:-}"
if [ -z "$DATA_DISK" ]; then
  for candidate in /root/gpuhub-tmp /root/autodl-tmp; do
    if [ -d "$candidate" ]; then DATA_DISK="$candidate"; break; fi
  done
fi
if [ -z "$DATA_DISK" ]; then
  echo "==> Neither /root/gpuhub-tmp nor /root/autodl-tmp exists on this instance."
  echo "    Run 'ls /root/' to find the real data-disk mount, then re-run with:"
  echo "    DATA_DISK=/root/<real-name> bash gpuhub_setup.sh"
  exit 1
fi
echo "==> Using data disk: $DATA_DISK  (put the repo + tokenized data here, NOT on / -- the"
echo "    30GB system disk is what 'Save Image' snapshots and 'reset-system' wipes)"

REPO_DIR="$DATA_DISK/llm"
if [ ! -d "$REPO_DIR/.git" ]; then
  echo "==> Cloning ${GIT_REPO_URL} (${GIT_BRANCH}) -> ${REPO_DIR}"
  git clone --branch "$GIT_BRANCH" "$GIT_REPO_URL" "$REPO_DIR"
else
  echo "==> Repo already present at ${REPO_DIR}, pulling latest"
  git -C "$REPO_DIR" pull
fi

if [ -f /root/.env ] && [ ! -f "$REPO_DIR/.env" ]; then
  echo "==> Found /root/.env (scp'd from Mac) -- linking into repo for rclone use"
  cp /root/.env "$REPO_DIR/.env"
elif [ ! -f "$REPO_DIR/.env" ]; then
  echo "==> WARNING: no .env found -- R2 data pull will be skipped. scp it up first:"
  echo "    scp -P <PORT> .env root@<HOST>:/root/.env"
fi

cd "$REPO_DIR"
echo "==> Installing deps + verifying CUDA (provider-agnostic, same script RunPod uses)"
bash scripts/cloud/remote_setup.sh

if [ -f .env ]; then
  set -a; source .env; set +a
  if [ -n "${RCLONE_CONFIG_R2_ACCESS_KEY_ID:-}" ]; then
    DATA_TARGET="${REPO_DIR}/data/tokenized"
    echo "==> Pulling tokenized data: r2:${R2_BUCKET:-llm}/data/tokenized -> ${DATA_TARGET}"
    echo "    (rclone copy, not mount -- gpuhub blocks 'rclone mount' by container policy)"
    rclone copy --progress "r2:${R2_BUCKET:-llm}/data/tokenized" "$DATA_TARGET" --transfers 8
  fi
fi

echo ""
echo "==> Ready. Recommended next steps:"
echo "    tmux new -s train"
echo "    python scripts/train.py --config configs/train_s_smoke.yaml   # cheap correctness check first"
echo "==> Once verified: shut down this instance, then console 'More' -> 'Save Image' to snapshot"
echo "    everything on / for reuse (note: the data disk is NOT included in Save Image -- data"
echo "    re-pulls from R2 on each future instance, or use File Storage to persist it, D-027/CLOUD_GPUHUB.md)."
