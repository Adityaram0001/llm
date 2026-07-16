#!/bin/zsh
# Push every offline wandb run currently on the pod up to wandb.ai. Local metrics.jsonl stays
# the source of truth (D-005) -- this is purely for the visual/comparison dashboard. Safe to
# re-run any time (wandb sync skips runs it's already pushed); doesn't touch experiments/.
set -euo pipefail
cd "$(dirname "$0")/../.."
source scripts/cloud/remote.env

if [ ! -f .env ] || ! grep -q '^WANDB_API_KEY=' .env; then
  echo "No WANDB_API_KEY in .env -- add it first (see docs/WANDB.md)." >&2
  exit 1
fi

RSH=(ssh -p "${REMOTE_PORT}" -i "${SSH_KEY}")
REMOTE="${REMOTE_USER}@${REMOTE_HOST}"

echo "==> Pushing local .env (R2 + wandb credentials) to the pod"
scp -P "${REMOTE_PORT}" -i "${SSH_KEY}" .env "${REMOTE}:${REMOTE_DIR}/.env"

echo "==> Syncing every offline wandb run under experiments/*/wandb to wandb.ai"
"${RSH[@]}" "${REMOTE}" "cd ${REMOTE_DIR} && export PATH=/root/miniconda3/bin:\$PATH && \
  set -a && source .env && set +a && \
  for d in experiments/*/wandb/offline-run-*; do \
    [ -d \"\$d\" ] && wandb sync \"\$d\"; \
  done"

echo "==> Done. Check https://wandb.ai/<your-entity>/llm-lab"
