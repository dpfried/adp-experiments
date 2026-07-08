#!/usr/bin/env bash
set -euxo pipefail
export PATH="$HOME/.local/bin:$PATH"
cd /home/dfried/exp/adp-smoke/swebench/benchmarks
git submodule update --init --recursive 2>&1 | tail -2
uv sync 2>&1 | tail -3
ls .venv/bin | grep -E "^swebench" || { echo "MISSING ENTRYPOINTS"; exit 1; }
.venv/bin/python -c "from benchmarks.swebench import apptainer_eval, apptainer_build; print('apptainer modules import OK')"
echo BENCHMARKS_FIXED_DONE
