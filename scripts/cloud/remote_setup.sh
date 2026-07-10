#!/bin/bash
# Run ON THE POD after sync_up.sh:  cd /workspace/llm-lab && bash scripts/cloud/remote_setup.sh
# Installs deps into the pod's system python (pods are disposable — no venv ceremony),
# keeping the image's CUDA-matched torch unless it's below our floor. Then verifies the GPU.
set -euo pipefail
cd "$(dirname "$0")/../.."

echo "==> Python / torch already on the image:"
python -c "import sys, torch; print(sys.version.split()[0], '| torch', torch.__version__, '| cuda', torch.version.cuda)" \
  || { echo "No torch on image — installing full requirements"; pip install -r requirements.txt; }

echo "==> Installing project requirements (torch kept if image's satisfies >=2.7)"
pip install -r requirements.txt
pip install -e .
command -v tmux >/dev/null || (apt-get update && apt-get install -y tmux rsync)

echo "==> CUDA verification"
python - <<'EOF'
import torch
from llmlab.utils import get_device, autocast_ctx, set_seed, mem_stats

assert torch.cuda.is_available(), "CUDA not available — wrong image or driver problem"
dev = get_device(); assert dev.type == "cuda"
name = torch.cuda.get_device_name(0)
cap = torch.cuda.get_device_capability(0)
print(f"GPU: {name}  (sm_{cap[0]}{cap[1]})")
# RTX 5090 is sm_120: torch must be a CUDA >= 12.8 build or kernels won't launch.
torch.set_float32_matmul_precision("high")
set_seed(1337)
x = torch.randn(2048, 2048, device=dev)
with autocast_ctx(dev):
    y = x @ x
torch.cuda.synchronize()
print("bf16 matmul OK |", {k: round(v) for k, v in mem_stats().items()})
print("VRAM total:", round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1), "GB")
print("\nAll good. Train inside tmux:  tmux new -s train")
EOF
