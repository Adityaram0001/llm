#!/bin/zsh
# Push tokenized data (uint16 .bin memmaps + tokenizer + meta) from the Mac to the object-
# storage bucket, ONCE per data change — pods then pull it at datacenter speed instead of
# over home upload while the GPU meter runs. Requires: brew install rclone, and an "r2"
# remote configured (rclone config — see docs/CLOUD.md "Data logistics").
set -euo pipefail
cd "$(dirname "$0")/../.."

REMOTE="${1:-r2:llmlab}"
echo "==> Pushing data/tokenized → ${REMOTE}/data/tokenized (only changed files)"
rclone copy --progress data/tokenized "${REMOTE}/data/tokenized" \
  --exclude ".DS_Store" --transfers 8
echo "==> Bucket now contains:"
rclone size "${REMOTE}/data/tokenized"
