#!/bin/bash
#SBATCH --job-name=g_report
#SBATCH --partition=learnfair
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:45:00
#SBATCH --output=/checkpoint/dpf/swebench-eval/logs/g_report_%j.out
#SBATCH --dependency=afterany:817665
# Runs cell-G's final attempt-1 analysis on the cluster, so the result survives
# the flaky devserver->devfair SSH link. Fetch with a single cat of OUT.
set -u
S=/checkpoint/dpf/swebench-eval/analysis
PY=/checkpoint/dpf/adp-env/.venv/bin/python
OUT=/checkpoint/dpf/swebench-eval/analysis/G_final_report.txt
mkdir -p "$(dirname "$OUT")"

{
  echo "# cell G final attempt-1 report"
  echo "# generated $(date -u +'%Y-%m-%d %H:%M UTC') by SLURM job ${SLURM_JOB_ID:-?} on $(hostname)"
  echo "# dependency: afterany:817665 (both G shards terminated)"
  echo
  echo "=== 0. G array final state ==="
  sacct -j 817665 --format=JobID%16,State%14,Elapsed,TimeLimit,ExitCode,MaxRSS -n 2>&1 | head -20
  echo
  echo "=== 0b. attempt-1 coverage (must be 100 unique) ==="
  cat /checkpoint/dpf/swebench-eval/runs/out_par_G_base_prefill_evalp__s*/*/*/*/output.critic_attempt_1.jsonl 2>/dev/null \
    | $PY -c "
import sys, json
ids=set(); n=0
for l in sys.stdin:
    if l.strip(): n+=1; ids.add(json.loads(l)['instance_id'])
print(f'rows={n} unique={len(ids)}')
print('COMPLETE' if len(ids)==100 else f'*** INCOMPLETE: {len(ids)}/100 -- treat every number below as interim ***')"
  echo
  echo "=== 1. outcome taxonomy (loader, attempt 1) ==="
  $PY $S/gtaxo.py 2>&1
  echo
  echo "=== 2. paired crosstab + McNemar (N = paired instances) ==="
  $PY $S/gcross.py 2>&1
  echo
  echo "=== 3. paired behavioural (M = paired transcripts) ==="
  $PY $S/gpaired.py 2>&1
  echo
  echo "=== 4. narration channel, pooled ==="
  $PY $S/gcheck.py 2>&1
  echo
  echo "# END"
} > "$OUT" 2>&1

cp "$OUT" "/checkpoint/dpf/swebench-eval/logs/G_final_report_${SLURM_JOB_ID:-x}.txt" 2>/dev/null
echo "wrote $OUT"
