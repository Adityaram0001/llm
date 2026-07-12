#!/bin/bash
# Container bootstrap: clone code + pull tokenized data, then hand over to the shell/command.
# Everything is driven by env vars set in the pod template — all optional, it degrades to a
# plain shell with clear instructions. See docs/CLOUD.md "Docker fast-start".
#
#   GIT_REPO_URL   e.g. https://github.com/<user>/llm-lab.git  (public or with token in URL)
#   GIT_BRANCH     default: main
#   DATA_REMOTE    rclone path of the data bucket, e.g. r2:llm/data/tokenized/hf_bpe_16k
#   WANDB_API_KEY  picked up automatically by wandb
#   RCLONE_CONFIG_R2_* env vars configure the "r2:" remote without a config file:
#     RCLONE_CONFIG_R2_TYPE=s3  RCLONE_CONFIG_R2_PROVIDER=Cloudflare
#     RCLONE_CONFIG_R2_ACCESS_KEY_ID=...  RCLONE_CONFIG_R2_SECRET_ACCESS_KEY=...
#     RCLONE_CONFIG_R2_ENDPOINT=https://<account-id>.r2.cloudflarestorage.com
set -uo pipefail

REPO_DIR=/workspace/llm-lab

if [ -n "${GIT_REPO_URL:-}" ] && [ ! -d "$REPO_DIR" ]; then
  echo "==> Cloning ${GIT_REPO_URL} (${GIT_BRANCH:-main})"
  git clone --depth 1 --branch "${GIT_BRANCH:-main}" "$GIT_REPO_URL" "$REPO_DIR" \
    && pip install --no-deps -e "$REPO_DIR"
fi

if [ -n "${DATA_REMOTE:-}" ] && [ -d "$REPO_DIR" ]; then
  echo "==> Pulling tokenized data from ${DATA_REMOTE}"
  rclone copy --progress "$DATA_REMOTE" "$REPO_DIR/data/tokenized/$(basename "$DATA_REMOTE")"
fi

if [ -d "$REPO_DIR" ]; then
  cd "$REPO_DIR"
  python - <<'EOF' || true
import torch
ok = torch.cuda.is_available()
print(f"==> torch {torch.__version__} | cuda available: {ok}"
      + (f" | {torch.cuda.get_device_name(0)}" if ok else " (CPU container?)"))
EOF
  echo "==> Ready. Typical next step:"
  echo "    tmux new -s train"
  echo "    python scripts/train.py --config configs/train_l_hero.yaml"
else
  echo "==> No GIT_REPO_URL set and no code found. Either set the env vars"
  echo "    (GIT_REPO_URL, DATA_REMOTE, RCLONE_CONFIG_R2_*) in the pod template,"
  echo "    or rsync the repo in via scripts/cloud/sync_up.sh from the Mac."
fi

exec "$@"
