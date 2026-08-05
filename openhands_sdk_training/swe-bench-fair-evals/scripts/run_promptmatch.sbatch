#!/usr/bin/env bash
#SBATCH --job-name=pmatch
#SBATCH --partition=learnfair
#SBATCH --constraint=ampere80gb
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=16:00:00
#SBATCH --open-mode=append
#SBATCH --output=/checkpoint/dpf/swebench-eval/logs/%x-%A_%a.out
#SBATCH --error=/checkpoint/dpf/swebench-eval/logs/%x-%A_%a.err
#
# Prompt-parity probe: does the train/eval task-statement mismatch cost score?
#
# Three prompt cells x two instance shards, ONE arm. All six launched in a
# single pass so they share generation conditions. vLLM prefix caching +
# continuous batching make greedy decoding non-reproducible, so the pre-existing
# swezero run cannot serve as a control -- cell C is a fresh one.
#
#   C   eval default.j2                     (control, as the board was measured)
#   T1  train_wrapper.j2                    (cosmetic header swap ONLY:
#                                            <uploaded_files> + "I've uploaded"
#                                            + bare dir; eval's 8 phases kept)
#   T2  train_swezero_noenv.j2              (full swezero training statement:
#                                            header + no-execution prohibition
#                                            + 5 phases, no RUNNING/TEST/VERIFY;
#                                            verified byte-identical to a real
#                                            swezero training task statement)
#
# T1 vs T2 is the point. T2 alone is confounded: it forbids python in a
# container where python works, so a score drop there cannot distinguish
# "the mismatch was harmless" from "we handicapped the model".
#
# Two shards (n=100/cell) because shard 0's prior swezero rate is 5/50: at n=50
# the score endpoint can only resolve a doubling. Cells are compared WITHIN a
# shard, so the shards pool cleanly.
#
#   array index K:  cell = K % 3   shard = K / 3
#   sbatch --array=0-5 run_promptmatch.sbatch [MODEL]
set -uo pipefail
MODEL=${1:-/checkpoint/dpf/adp-runs/v2_swezero_inst_4b_a100/output}
source "${SWEBENCH_ROOT:-/checkpoint/dpf/swebench-eval}/env.sh"
export NUM_WORKERS=${NUM_WORKERS:-16}

K=${SLURM_ARRAY_TASK_ID:?run with --array=0-5}
P=$SWEBENCH_ROOT/prompts_adp
case "$(( K % 3 ))" in
  0) CELL=C;  PROMPT_PATH=$SWEBENCH_ROOT/benchmarks/benchmarks/swebench/prompts/default.j2 ;;
  1) CELL=T1; PROMPT_PATH=$P/train_wrapper.j2 ;;
  2) CELL=T2; PROMPT_PATH=$P/train_swezero_noenv.j2 ;;
esac
export PROMPT_PATH
[ -s "$PROMPT_PATH" ] || { echo "FATAL: missing template $PROMPT_PATH"; exit 1; }

SH=$(printf '%02d' "$(( K / 3 ))")
SELECT=$SWEBENCH_ROOT/select/shard_${SH}of10.txt
[ -s "$SELECT" ] || { echo "FATAL: missing $SELECT"; exit 1; }

TAG=pmatch_swezero_${CELL}__s${SH}
echo "== pmatch cell=$CELL shard=$SH tag=$TAG n_ids=$(wc -l < "$SELECT") prompt=$PROMPT_PATH =="
echo "== template sha256: $(sha256sum "$PROMPT_PATH" | cut -c1-16) =="

exec bash "$SWEBENCH_ROOT/scripts/run_full_infer_prompt.sbatch" "$MODEL" "$TAG" "$SELECT"
