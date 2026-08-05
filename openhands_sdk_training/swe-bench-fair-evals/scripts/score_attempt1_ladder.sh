#!/usr/bin/env bash
# Score each ladder cell's FIRST critic attempt only, into score_<tag>_a1.
#
# Motivation (see analysis/attempt1_subset.py for the numbers): the harness gave
# base 2.32 rollouts/instance and best-of-3 critic selection while the arm cells
# got ~1.04, because the critic rejected base's first attempt 71/100 times and
# the arm's 2-5/100. Identical configuration, wildly unequal compute. The
# aggregated F-B therefore bundles "base vs arm" with "2.2x rollouts", and that
# bias inflates base -- the opposite direction from the two already catalogued.
# Attempt 1 is the compute-matched contrast: one rollout per instance, temp 0,
# all instances, no selection.
#
# Same guards as reagg_ladder.sh, for the same reasons:
#   * requires the cell's ORIGINAL scoring to have merged, so the original array
#     has certainly finished -- the scoring sbatch prunes Apptainer sandboxes
#     keyed by instance_id alone, so two jobs scoring the same instance would
#     delete each other's sandbox.
#   * the still-queued check comes BEFORE the merge gate: these run on scavenge,
#     so a task can write its report, be preempted and rerun. Merging in that
#     window freezes a report about to be rewritten, and the DONE branch means
#     it would never be re-merged.
#   * shard count is derived from the reports present, never hardcoded.
# Idempotent: a re-tick with nothing to do prints only status lines.
set -uo pipefail
R=${SWEBENCH_ROOT:?set SWEBENCH_ROOT to the eval root}
A=${ANALYSIS_DIR:?set ANALYSIS_DIR to where attempt1_subset.py lives}
cd "$R"
NSH=2
QUEUED=$(squeue --me -h -o '%j' 2>/dev/null | sort -u | tr '\n' ' ')

for T in par_A_arm_stock_evalp par_B_arm_nostub_evalp par_C_arm_nostub_wrap \
         par_D_arm_nostub_trainp par_E_base_stock_evalp par_F_base_nostub_evalp; do
  for SH in 00 01; do
    TAG="${T}__s${SH}"
    OUTD="$R/runs/out_$TAG"
    [ -d "$OUTD" ] || { echo "skip  $TAG (no out dir)"; continue; }
    # The condition that actually matters is "attempt 1 is complete and frozen",
    # not "the whole cell has been scored". Requiring the merged report was a
    # proxy for it, and the proxy is wrong in a case that really occurred: cell E
    # was still running, so it had no merged report, yet its attempt-1 file was
    # complete at 100/100 and frozen -- the harness had already moved on to
    # attempt 2. Test the real condition instead: attempt 1 is done iff the cell
    # finished (merged report) OR a later attempt file exists, which only the
    # harness writes after it stops appending to attempt 1.
    A1F=$(find "$OUTD" -name output.critic_attempt_1.jsonl | head -1)
    LATER=$(find "$OUTD" -name 'output.critic_attempt_[2-9].jsonl' | head -1)
    if [ ! -f "$R/runs/score_$TAG/merged.report.json" ] && [ -z "$LATER" ]; then
      echo "skip  $TAG (attempt 1 may still be growing: no merged report and no attempt-2 file)"; continue
    fi
    [ -n "$A1F" ] || { echo "skip  $TAG (no attempt-1 file)"; continue; }

    RTAG="${TAG}_a1"
    if [ -f "$R/runs/score_$RTAG/merged.report.json" ]; then echo "DONE  $RTAG"; continue; fi

    NREP=$(ls "$R/runs/score_$RTAG"/shard_*of*.report.json 2>/dev/null | wc -l)
    EXPN=$(ls "$R/runs/score_$RTAG"/shard_*of*.report.json 2>/dev/null | head -1 \
           | sed 's/.*of\([0-9]*\)\.report\.json/\1/')
    case " $QUEUED " in
      *" sc_$RTAG "*) echo "SCORING $RTAG ($NREP/${EXPN:-?} shards)"; continue;;
    esac
    # Never run a1 scoring of a cell concurrently with that cell's other
    # scoring jobs (same sandbox-collision reason as above).
    case " $QUEUED " in
      *" sc_$TAG "*|*" sc_${TAG}_reagg "*)
        echo "WAIT  $RTAG (other scoring of $TAG still queued)"; continue;;
    esac
    # CROSS-CELL guard. Every ladder cell is scored on the SAME instance set, and
    # the scoring sbatch prunes Apptainer sandboxes keyed by instance_id alone --
    # so any two a1 jobs collide, not just two jobs of the same cell. The
    # same-cell check above does not cover this; the first a1 batch was submitted
    # without it and ran 20 tasks over one shared instance set. Serialise.
    case " $QUEUED " in
      *" sc_par_"*"_a1 "*)
        echo "WAIT  $RTAG (another cell's a1 scoring is queued -- shared instance set)"; continue;;
    esac
    if [ "$NREP" -gt 0 ] && [ "$NREP" -eq "${EXPN:-0}" ]; then
      echo "MERGE $RTAG"
      python3 "$R/scripts/merge_shard_reports.py" "$RTAG" 2>&1 | tail -1
      continue
    fi

    D="$R/runs/a1_$TAG"
    mkdir -p "$D"
    python3 "$A/attempt1_subset.py" "$OUTD" "$D/output.jsonl" | head -1
    N=$(wc -l < "$D/output.jsonl" 2>/dev/null || echo 0)
    if [ "$N" -eq 0 ]; then echo "NOOP  $TAG (no attempt-1 file)"; continue; fi
    if [ "$NREP" -gt 0 ]; then rm -rf "$R/runs/score_$RTAG"; fi   # glob-merge hygiene
    echo "SUBMIT $RTAG ($N attempt-1 records)"
    sbatch --job-name="sc_$RTAG" --array=0-$((NSH-1)) \
      --export=ALL,SWEBENCH_ROOT \
      scripts/run_score_shards.sbatch "$D/output.jsonl" "$RTAG" "$NSH"
  done
done
