#!/bin/zsh
# Push code + tokenized data + configs to the rented pod. Run from Mac, repo root or anywhere.
# Excludes heavy/local-only stuff. Idempotent — rsync only sends what changed.
set -euo pipefail
cd "$(dirname "$0")/../.."
source scripts/cloud/remote.env

RSH="ssh -p ${REMOTE_PORT} -i ${SSH_KEY}"
DEST="${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/"

echo "==> Syncing repo → ${DEST}"
rsync -avz --progress -e "${RSH}" \
  --exclude '.venv' --exclude '.git' --exclude '__pycache__' \
  --exclude 'wandb' --exclude '.ipynb_checkpoints' --exclude '.DS_Store' \
  --exclude 'data/raw' --exclude 'checkpoints' \
  --exclude 'experiments/*/ckpt' \
  --exclude 'tools/data_factory/inbox' --exclude 'tools/data_factory/outbox' \
  ./ "${DEST}"

echo "==> Done. Next: ssh -p ${REMOTE_PORT} -i ${SSH_KEY} ${REMOTE_USER}@${REMOTE_HOST}"
echo "    then: cd ${REMOTE_DIR} && bash scripts/cloud/remote_setup.sh"
