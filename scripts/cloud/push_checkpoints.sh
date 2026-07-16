#!/bin/zsh
# Archive experiments/ (checkpoints, slim by default) from the pod straight to R2 --
# server -> R2 directly, no Mac round-trip. Does NOT touch/delete anything on the pod or
# locally; re-run any time, it's a `rclone copy` (only pushes changed/new files).
#
# Policy (see D-041): every run's ckpt/best.pt is archived with optimizer state stripped
# (model weights only -- ablation runs are reproducible from config+seed, not meant to be
# resumed). Runs listed in FORK_POINTS get their FULL best.pt + latest.pt (optimizer state
# intact) because a real --resume has forked off them.
set -euo pipefail
cd "$(dirname "$0")/../.."
source scripts/cloud/remote.env

FORK_POINTS="${1:-20260713_p5_s-wave-d-constant}"
RSH=(ssh -p "${REMOTE_PORT}" -i "${SSH_KEY}")
REMOTE="${REMOTE_USER}@${REMOTE_HOST}"

echo "==> Copying archive_checkpoints.py to the pod"
scp -P "${REMOTE_PORT}" -i "${SSH_KEY}" scripts/cloud/archive_checkpoints.py \
  "${REMOTE}:${REMOTE_DIR}/scripts/cloud/archive_checkpoints.py"

echo "==> Staging slim checkpoints on the pod (fork points kept full: ${FORK_POINTS})"
"${RSH[@]}" "${REMOTE}" "cd ${REMOTE_DIR} && export PATH=/root/miniconda3/bin:\$PATH && \
  python3 scripts/cloud/archive_checkpoints.py \
    --experiments-dir experiments --staging-dir _r2_staging/experiments \
    --fork-points '${FORK_POINTS}'"

echo "==> Pushing staged archive to r2:\${R2_BUCKET}/experiments"
"${RSH[@]}" "${REMOTE}" "cd ${REMOTE_DIR} && set -a && source .env && set +a && \
  rclone copy --progress _r2_staging/experiments r2:\${R2_BUCKET}/experiments --transfers 8"

echo "==> Verifying (R2 listing + size)"
"${RSH[@]}" "${REMOTE}" "cd ${REMOTE_DIR} && set -a && source .env && set +a && \
  rclone size r2:\${R2_BUCKET}/experiments"

echo "==> Cleaning up staging dir on the pod (originals in experiments/ untouched)"
"${RSH[@]}" "${REMOTE}" "rm -rf ${REMOTE_DIR}/_r2_staging"

echo "==> Done. Nothing was deleted from experiments/ on the pod or locally."
