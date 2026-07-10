#!/bin/zsh
# LLM-Lab environment setup — run from repo root: ./scripts/setup.sh
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PYTHON:-python3}
echo "==> Using $($PY --version)"
$PY -c 'import sys; assert sys.version_info >= (3,11), "Need Python 3.11+"'

if [ ! -d .venv ]; then
  echo "==> Creating .venv"
  $PY -m venv .venv
fi
source .venv/bin/activate

echo "==> Installing requirements"
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .

echo "==> Registering Jupyter kernel 'llm-lab'"
python -m ipykernel install --user --name llm-lab --display-name "Python (llm-lab)"

echo "==> Done. Next:"
echo "    source .venv/bin/activate"
echo "    python scripts/verify_env.py"
