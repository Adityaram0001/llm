#!/bin/zsh
# Pull run results (experiments/ incl. checkpoints) from the pod back to the Mac.
# NEVER deletes anything local (no --delete): experiments/ is the append-only lab record.
# Safe to run repeatedly during a long run (periodic backup of latest.pt).
set -euo pipefail
cd "$(dirname "$0")/../.."
source scripts/cloud/remote.env

RSH="ssh -p ${REMOTE_PORT} -i ${SSH_KEY}"
SRC="${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}"

echo "==> Pulling experiments/ (metrics, notes, samples, checkpoints)"
rsync -avz --progress -e "${RSH}" "${SRC}/experiments/" ./experiments/

echo "==> Pulling wandb offline runs if any (sync later with: wandb sync wandb/offline-*)"
rsync -avz --progress -e "${RSH}" "${SRC}/wandb/" ./wandb/ 2>/dev/null || true

echo "==> Done. Verify the checkpoint loads locally BEFORE stopping the pod."
