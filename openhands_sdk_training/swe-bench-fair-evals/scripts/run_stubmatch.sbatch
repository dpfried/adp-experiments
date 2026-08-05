#!/usr/bin/env bash
#SBATCH --job-name=stubmatch
#SBATCH --partition=learnfair
#SBATCH --constraint=ampere80gb
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=16:00:00
#SBATCH --open-mode=append
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#
# Log paths are relative: submit from $SWEBENCH_ROOT (which has logs/).
#
# The empty-<think>-stub falsifier.
#
# The stock Qwen3.5 chat template appends '<think>\n\n</think>\n\n' -- exactly 4
# tokens, ids [248068, 271, 248069, 271] -- to EVERY assistant turn when
# enable_thinking=false, which is how eval serves. LLaMA-Factory's
# `qwen3_5_nothink` training template ends the assistant header at
# '<|im_start|>assistant\n' with no stub, and does not use ReasoningTemplate.
# Verified empirically: 0 occurrences of <think>/</think> in 6.5M training
# tokens. So every SFT arm meets 4 tokens per turn it never saw in that
# position, while base is served in its own shipped format.
#
# 2 x 2 x 2 = 8 jobs:
#   template  stock (stub, as the board was measured) | nostub (training-matched)
#   model     ARM (an SFT checkpoint) | BASE (the un-finetuned instruct model)
#   shard     00 | 01                 (n=100 per cell)
#
# The BASE row is the point. If removing the stub helps the arm and does NOT
# help base, the stub is part of why base >> arms and every arm number on the
# board is an underestimate. If it helps both, it is a general serving bug
# unrelated to SFT. If neither, the hypothesis is dead and can be closed.
# It also gives the first empirical test of the "base is in-distribution for
# the stub" step, which was inferred from the shipped template, never measured.
#
#   array index K:  template = K % 2   model = (K / 2) % 2   shard = K / 4
#   SWEBENCH_ROOT=<eval root> ARM_MODEL=<ckpt> BASE_MODEL=<hf id or dir> \
#     sbatch --array=0-7 run_stubmatch.sbatch
set -uo pipefail
: "${SWEBENCH_ROOT:?set SWEBENCH_ROOT to the eval root}"
: "${ARM_MODEL:?set ARM_MODEL to the SFT checkpoint under test}"
: "${BASE_MODEL:?set BASE_MODEL to the un-finetuned comparator}"
source "$SWEBENCH_ROOT/env.sh"
export NUM_WORKERS=${NUM_WORKERS:-16}

K=${SLURM_ARRAY_TASK_ID:?run with --array=0-7}
NOSTUB=$SWEBENCH_ROOT/prompts_adp/nostub.jinja

# template axis: cell 0 = stock (pass no --chat-template, exactly as the board
# was measured); cell 1 = the 4-token-trimmed variant.
if [ $(( K % 2 )) -eq 0 ]; then TPL=stock;  CHAT_TEMPLATE=""
else                           TPL=nostub; CHAT_TEMPLATE="$NOSTUB"
                               [ -s "$CHAT_TEMPLATE" ] || { echo "FATAL: missing $CHAT_TEMPLATE"; exit 1; }
fi

# model axis
if [ $(( (K / 2) % 2 )) -eq 0 ]; then WHO=arm;  MODEL="$ARM_MODEL"
else                                  WHO=base; MODEL="$BASE_MODEL"
fi

SH=$(printf '%02d' "$(( K / 4 ))")
SELECT=$SWEBENCH_ROOT/select/shard_${SH}of10.txt
[ -s "$SELECT" ] || { echo "FATAL: missing $SELECT"; exit 1; }

TAG=stub_${WHO}_${TPL}__s${SH}
echo "== stubmatch who=$WHO tpl=$TPL shard=$SH tag=$TAG n_ids=$(wc -l < "$SELECT") =="
echo "== model=$MODEL =="
echo "== chat_template=${CHAT_TEMPLATE:-<shipped/stock>} =="
[ -n "$CHAT_TEMPLATE" ] && echo "== template sha256: $(sha256sum "$CHAT_TEMPLATE" | cut -c1-16) =="

exec bash "$SWEBENCH_ROOT/scripts/run_full_infer.sbatch" \
  "$MODEL" "$TAG" "$SELECT" "$CHAT_TEMPLATE"
