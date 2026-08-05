#!/usr/bin/env bash
# For each ladder cell whose ORIGINAL scoring has merged, check whether the
# harness aggregator discarded any good patches (amendment Addendum 5) and if so
# score a repaired subset into score_<tag>_reagg.
#
# Deliberately simple and idempotent. Every action is guarded by the existence of
# a merged report, so a re-tick is a no-op. Kept OUT of score_ladder.sh so a bug
# here cannot affect the primary scoring path.
#
# Ordering matters: the scoring sbatch prunes Apptainer sandboxes keyed by
# instance_id alone, so the repaired scoring of a cell must not run concurrently
# with the original scoring of the same cell. Requiring the original's
# merged.report.json guarantees the original array has finished.
set -uo pipefail
R=${SWEBENCH_ROOT:?set SWEBENCH_ROOT to the eval root}
cd "$R"
NSH=2
QUEUED=$(squeue --me -h -o '%j' 2>/dev/null | sort -u | tr '\n' ' ')

for T in par_A_arm_stock_evalp par_B_arm_nostub_evalp par_C_arm_nostub_wrap \
         par_D_arm_nostub_trainp par_E_base_stock_evalp par_F_base_nostub_evalp; do
  for SH in 00 01; do
    TAG="${T}__s${SH}"
    [ -f "$R/runs/score_$TAG/merged.report.json" ] || { echo "skip  $TAG (original not merged)"; continue; }
    RTAG="${TAG}_reagg"
    if [ -f "$R/runs/score_$RTAG/merged.report.json" ]; then echo "DONE  $RTAG"; continue; fi

    # Derive the shard count from the reports actually present rather than from
    # $NSH: a cell may have been submitted with a different array size (F__s01 was
    # submitted by hand with 3). Hardcoding the modulus here is precisely the
    # wrong-idempotency-key bug that resubmitted 36 jobs this morning.
    NREP=$(ls "$R/runs/score_$RTAG"/shard_*of*.report.json 2>/dev/null | wc -l)
    EXPN=$(ls "$R/runs/score_$RTAG"/shard_*of*.report.json 2>/dev/null | head -1 \
           | sed 's/.*of\([0-9]*\)\.report\.json/\1/')
    if [ "$NREP" -gt 0 ] && [ "$NREP" -eq "${EXPN:-0}" ]; then
      echo "MERGE $RTAG"
      python3 "$R/scripts/merge_shard_reports.py" "$RTAG" 2>&1 | tail -1
      continue
    fi
    case " $QUEUED " in *" sc_$RTAG "*) echo "SCORING $RTAG ($NREP/$NSH)"; continue;; esac

    D="$R/runs/reagg_$TAG"
    mkdir -p "$D"
    python3 "${ANALYSIS_DIR:?set ANALYSIS_DIR to where repair_aggregate.py lives}/repair_aggregate.py" "$R/runs/out_$TAG" "$D/output.jsonl" | head -1
    N=$(wc -l < "$D/output.jsonl" 2>/dev/null || echo 0)
    if [ "$N" -eq 0 ]; then echo "NOOP  $TAG (aggregator discarded nothing)"; continue; fi
    if [ "$NREP" -gt 0 ]; then rm -rf "$R/runs/score_$RTAG"; fi   # glob-merge hygiene
    echo "SUBMIT $RTAG ($N repaired records)"
    sbatch --job-name="sc_$RTAG" --array=0-$((NSH-1)) \
      --export=ALL,SWEBENCH_ROOT \
      scripts/run_score_shards.sbatch "$D/output.jsonl" "$RTAG" "$NSH"
  done
done
