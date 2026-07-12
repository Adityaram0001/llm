#!/bin/zsh
# Push tokenized data (uint16 .bin memmaps + tokenizer + meta) from the Mac to the object-
# storage bucket, ONCE per data change — pods then pull it at datacenter speed instead of
# over home upload while the GPU meter runs. Requires rclone on PATH and RCLONE_CONFIG_R2_*
# credentials in .env (copy from .env.example — see docs/CLOUD.md "Data logistics").
set -euo pipefail
cd "$(dirname "$0")/../.."

if [ -f .env ]; then
  set -a
  source .env
  set +a
fi

REMOTE="${1:-r2:${R2_BUCKET:-llmlab}}"
echo "==> Pushing data/tokenized → ${REMOTE}/data/tokenized (only changed files)"
rclone copy --progress data/tokenized "${REMOTE}/data/tokenized" \
  --exclude ".DS_Store" --transfers 8
echo "==> Bucket now contains:"
rclone size "${REMOTE}/data/tokenized"
