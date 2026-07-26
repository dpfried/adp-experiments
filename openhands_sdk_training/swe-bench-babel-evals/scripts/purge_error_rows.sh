#!/usr/bin/env bash
# Purge retryable error rows from output.critic_attempt_1.jsonl so a resubmit re-runs them.
#
# WHY: get_completed_instances() (benchmarks/utils/critics.py) treats every instance_id
# present in the attempt file as completed "regardless of success/failure". So an instance
# that errored is never retried on resume -- and error counts are wildly uneven across arms
# (coderforge 108, rebench 32, scale 12, swezero 0), which biases any cross-arm resolve-rate
# comparison against whichever arm happened to run on sicker infra. Dropping the error rows
# makes the resume logic re-run exactly those instances, uniformly, at temp 0 in attempt 1.
#
# KEPT (not retried): MaxIterations failures. At temp 0 those are a deterministic agent
# outcome -- the model genuinely failed to converge in 500 iterations -- so re-running burns
# GPU to reproduce the same row. That is a real result, not infra noise.
#
# DROPPED (retried): everything else, i.e. cp_testbed_repo / git reset timeouts (the 30s
# default fixed to 600s/300s in run_infer.py), `git rev-parse --show-toplevel` failures (fixed
# by pinning IMAGE_TAG_PREFIX), container-health failures, remote-conversation failures,
# LLM request timeouts, and the 14400s per-instance wall (contention-dependent, not fixed).
#
# Usage: purge_error_rows.sh [--dry-run] TAG [TAG...]
set -uo pipefail
SB=/home/dfried/exp/adp-smoke/swebench
STAMP=$(date +%Y%m%d-%H%M%S)
DRY=0
[ "${1:-}" = "--dry-run" ] && { DRY=1; shift; }
[ $# -ge 1 ] || { echo "usage: $0 [--dry-run] TAG [TAG...]"; exit 1; }

for TAG in "$@"; do
  D=$SB/full/out_$TAG/princeton-nlp__SWE-bench_Verified-test/hosted_vllm/adp-eval-${TAG}_sdk_43376f1_maxiter_500
  F=$D/output.critic_attempt_1.jsonl
  if [ ! -e "$F" ]; then echo "$TAG: SKIP (no attempt_1)"; continue; fi
  # Refuse to touch a file a live job may still be appending to.
  if squeue --me -h -o "%T" -n swebench-full-infer 2>/dev/null | grep -q RUNNING; then
    echo "$TAG: REFUSING -- a swebench-full-infer job is RUNNING; scancel first"; exit 1
  fi

  perl -nle '
    BEGIN { $kept=0; $dropped=0; $maxiter=0; }
    my $err;
    if (/"error":(null|"((?:\\.|[^"\\])*)")(?=,"instance":)/) { $err = defined($2) ? $2 : undef; }
    if (!defined $err)            { $kept++;    print; }   # no error (or unparsed) -> keep
    elsif ($err =~ /MaxIterations/) { $kept++; $maxiter++; print; }
    else                          { $dropped++; }
    END { print STDERR "kept=$kept (maxiter=$maxiter) dropped=$dropped"; }
  ' "$F" > "$F.new" 2> "$F.purgestat"
  RC=$?
  STAT=$(cat "$F.purgestat"); rm -f "$F.purgestat"
  if [ $RC -ne 0 ]; then echo "$TAG: perl failed rc=$RC, leaving original alone"; rm -f "$F.new"; continue; fi

  BEFORE=$(wc -l < "$F"); AFTER=$(wc -l < "$F.new")
  if [ "$DRY" = "1" ]; then
    echo "$TAG: DRY-RUN $BEFORE -> $AFTER  ($STAT)"; rm -f "$F.new"; continue
  fi
  # Sanity: never let a bug blank the file.
  if [ "$AFTER" -eq 0 ] && [ "$BEFORE" -gt 0 ]; then
    echo "$TAG: ABORT -- purge would empty the file"; rm -f "$F.new"; continue
  fi
  cp -p "$F" "$F.prepurge-$STAMP" && mv "$F.new" "$F"
  echo "$TAG: $BEFORE -> $AFTER  ($STAT)  backup=$(basename "$F.prepurge-$STAMP")"
done
