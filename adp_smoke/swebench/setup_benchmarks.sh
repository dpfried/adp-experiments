#!/usr/bin/env bash
set -euxo pipefail
export PATH="$HOME/.local/bin:$PATH"
cd /home/dfried/exp/adp-smoke/swebench
if [ ! -d benchmarks ]; then git clone https://github.com/OpenHands/benchmarks.git; fi
cd benchmarks
# uv-managed env per repo convention (uv.lock present in repo)
uv sync 2>&1 | tail -5
.venv/bin/python -c "import benchmarks" 2>/dev/null || true
ls .venv/bin/ | grep -E "swebench" || echo "NOTE: no swebench-* entrypoints found, check pyproject"
echo BENCHMARKS_SETUP_DONE
