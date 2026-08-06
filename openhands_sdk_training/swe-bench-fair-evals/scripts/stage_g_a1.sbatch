#!/bin/bash
#SBATCH --job-name=stage_g_a1
#SBATCH --partition=scavenge
#SBATCH --requeue
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/checkpoint/dpf/swebench-eval/logs/stage_g_a1_%j.out
#SBATCH --dependency=afterany:817665
#
# Stage cell G's attempt-1 rows as an a1_ subset dir, matching the ladder owner's
# existing convention exactly:  a1_<tag>__<shard>/output.jsonl  ==  byte copy of
# out_<tag>__<shard>/*/*/*/output.critic_attempt_1.jsonl
#
# Purely additive: creates two NEW dirs in a namespace nobody else uses (a1_par_G_*).
# Overwrites nothing. Does NOT score anything -- scoring is the ladder owner's call.
# Refuses to stage a shard whose attempt-1 file is short, so an incomplete cell
# cannot be silently graded as if complete (ANALYSIS_HOUSE_RULES rules 1 + 5).
set -uo pipefail
R=/checkpoint/dpf/swebench-eval/runs
PY=/checkpoint/dpf/adp-env/.venv/bin/python
rc=0

for sh in s00 s01; do
  src=$(ls $R/out_par_G_base_prefill_evalp__${sh}/*/*/*/output.critic_attempt_1.jsonl 2>/dev/null | head -1)
  sel=/checkpoint/dpf/swebench-eval/select/shard_$(echo $sh | tr -d 's')of10.txt
  if [ -z "$src" ]; then echo "FATAL $sh: no output.critic_attempt_1.jsonl"; rc=1; continue; fi

  read -r nrow nuniq nsel <<<"$($PY - "$src" "$sel" <<'PY'
import sys, json
src, sel = sys.argv[1], sys.argv[2]
ids = set(); n = 0
for l in open(src):
    if l.strip():
        n += 1; ids.add(json.loads(l)["instance_id"])
want = {x.strip() for x in open(sel) if x.strip()}
print(n, len(ids), len(want & ids))
PY
)"
  echo "$sh: rows=$nrow unique=$nuniq covered_of_select=$nsel"
  if [ "$nuniq" -ne 50 ] || [ "$nsel" -ne 50 ]; then
    echo "REFUSING $sh: attempt-1 is incomplete ($nuniq unique, $nsel/50 of select). Not staging."
    rc=1; continue
  fi
  dst=$R/a1_par_G_base_prefill_evalp__${sh}
  mkdir -p "$dst"
  if [ -s "$dst/output.jsonl" ]; then echo "  $dst/output.jsonl already exists; leaving it alone"; continue; fi
  cp "$src" "$dst/output.jsonl"
  cmp -s "$src" "$dst/output.jsonl" && echo "  staged $dst/output.jsonl ($(wc -l < "$dst/output.jsonl") rows, byte-identical to source)" \
    || { echo "  FATAL copy mismatch for $sh"; rc=1; }
done

echo "stage_g_a1 done rc=$rc at $(date -u +'%Y-%m-%d %H:%M UTC')"
exit $rc
