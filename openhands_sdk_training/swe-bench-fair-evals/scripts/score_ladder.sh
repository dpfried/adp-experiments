#!/usr/bin/env bash
# Submit scoring for every parity-ladder cell whose INFERENCE IS COMPLETE, then
# merge its shard reports. CPU-only -> scavenge.
#
# Two bugs in the first version of this script are fixed here; both would have
# silently corrupted the readout rather than failing loudly:
#
#   1. NO COMPLETENESS GATE. It submitted as soon as `output.jsonl` existed, so
#      cells were scored on 4/50, 5/50, 6/50 instances while inference was still
#      appending. Combined with the `merged.report.json -> SKIP` branch, the
#      first partial merge to land would have been treated as final and the cell
#      would NEVER have been re-scored on the full 50. ladder_readout.py derives
#      its denominator from the merged report, so it would have printed a
#      confident score over ~6 instances and silently shrunk the paired
#      intersection -- exactly the silent-denominator failure the amendment's
#      discipline exists to prevent.
#      FIX: a shard is eligible only when  out+err >= expected  AND its infer
#      array element has left the queue.
#
#   2. WRONG IDEMPOTENCY KEY. `run_score_shards.sbatch` writes per-shard
#      `shard_NofM.report.json` but does NOT write `merged.report.json` -- that
#      is a separate `merge_shard_reports.py` step. So the "already merged"
#      guard could never become true, and every tick resubmitted every eligible
#      cell (36 stray jobs accumulated).
#      FIX: explicit 4-state machine, and this script performs the merge.
#
# Also note merge_shard_reports.py GLOBS shard_*.report.json, so stale partial
# shard reports would be silently mixed into a fresh merge. Any cell rescored
# from scratch must have its score dir removed first, not overwritten.
set -uo pipefail
R=${SWEBENCH_ROOT:?set SWEBENCH_ROOT to the eval root}
export SWEBENCH_ROOT=$R
cd "$R"
NSH=4                             # scoring shards per cell
INFER_ARRAY=${INFER_ARRAY:?set INFER_ARRAY to the parity-ladder array job id}

# cell letter -> index, must match run_parity_ladder.sbatch's `case K % 6`
CELLS="A:0:par_A_arm_stock_evalp
B:1:par_B_arm_nostub_evalp
C:2:par_C_arm_nostub_wrap
D:3:par_D_arm_nostub_trainp
E:4:par_E_base_stock_evalp
F:5:par_F_base_nostub_evalp"

# expand array elements still queued/running once, not per cell
QUEUED_ELEMS=$(squeue --me -r -h -j "$INFER_ARRAY" -o '%i' 2>/dev/null | sed 's/.*_//' | tr '\n' ' ')
QUEUED_SCORE=$(squeue --me -h -o '%j' 2>/dev/null | grep '^sc_par_' | sort -u | tr '\n' ' ')

for entry in $CELLS; do
  CELL=${entry%%:*}; rest=${entry#*:}; IDX=${rest%%:*}; T=${rest#*:}
  for SHI in 0 1; do
    SH=$(printf '%02d' "$SHI")
    TAG="${T}__s${SH}"
    K=$(( SHI * 6 + IDX ))
    SDIR="$R/runs/score_$TAG"
    ODIR="$R/runs/out_$TAG"

    # --- state 1: already merged -> done
    if [ -f "$SDIR/merged.report.json" ]; then
      echo "DONE  $TAG"; continue
    fi

    # --- state 2: all shard reports present -> merge now
    NREP=$(ls "$SDIR"/shard_*of${NSH}.report.json 2>/dev/null | wc -l)
    if [ "$NREP" -eq "$NSH" ]; then
      echo "MERGE $TAG"
      python3 "$R/scripts/merge_shard_reports.py" "$TAG" 2>&1 | tail -1
      continue
    fi

    # --- state 3: scoring in flight -> wait
    case " $QUEUED_SCORE " in *" sc_$TAG "*) echo "SCORING $TAG ($NREP/$NSH shards)"; continue;; esac
    if [ "$NREP" -gt 0 ]; then
      echo "PARTIAL $TAG ($NREP/$NSH shards, no job queued -- rescore below)"
    fi

    # --- completeness gate (bug 1)
    SEL="$R/select/shard_${SH}of10.txt"
    EXP=$(wc -l < "$SEL")
    OJ=$(find "$ODIR" -name output.jsonl 2>/dev/null | head -1)
    EJ=$(find "$ODIR" -name output_errors.jsonl 2>/dev/null | head -1)
    NO=0; NE=0
    [ -n "$OJ" ] && NO=$(wc -l < "$OJ")
    [ -n "$EJ" ] && NE=$(wc -l < "$EJ")
    TOT=$(( NO + NE ))
    case " $QUEUED_ELEMS " in
      *" $K "*) echo "WAIT  $TAG (infer ${INFER_ARRAY}_$K still in queue, $TOT/$EXP)"; continue;;
    esac
    if [ "$TOT" -lt "$EXP" ]; then
      echo "WAIT  $TAG (incomplete: $TOT/$EXP, infer element gone -- may have died)"; continue
    fi

    # --- dedup guard: a retried/restarted shard APPENDS, so output.jsonl can hold
    # more rows than instances (observed: F__s00 at 85 rows / 50 unique ids). The
    # scorer shards by LINE, so duplicates would inflate `total` and the reported
    # rate would be computed over a denominator > the shard size. Score a deduped
    # copy instead, keeping the LAST row per instance_id (the most recent attempt,
    # i.e. the one from the run that completed).
    UNIQ=$(python3 -c "
import json,sys
seen=set()
for line in open('$OJ'):
    line=line.strip()
    if line:
        try: seen.add(json.loads(line)['instance_id'])
        except Exception: pass
print(len(seen))")
    NROWS=$(wc -l < "$OJ")
    SCORE_IN="$OJ"
    if [ "$NROWS" -ne "$UNIQ" ]; then
      SCORE_IN="${OJ%.jsonl}.dedup.jsonl"
      python3 -c "
import json,collections
last=collections.OrderedDict()
for line in open('$OJ'):
    line=line.strip()
    if not line: continue
    try: r=json.loads(line)
    except Exception: continue
    last[r['instance_id']]=line
open('$SCORE_IN','w').write('\n'.join(last.values())+'\n')"
      echo "DEDUP $TAG ($NROWS rows -> $UNIQ unique; scoring $(basename "$SCORE_IN"))"
    fi

    # eligible. wipe any stale partial shard reports so the glob-merge is clean.
    if [ "$NREP" -gt 0 ]; then rm -rf "$SDIR"; fi
    echo "SUBMIT $TAG ($UNIQ unique / $EXP expected)"
    sbatch --job-name="sc_$TAG" --array=0-$((NSH-1)) \
      --export=ALL,SWEBENCH_ROOT \
      scripts/run_score_shards.sbatch "$SCORE_IN" "$TAG" "$NSH"
  done
done
