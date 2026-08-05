#!/usr/bin/env bash
#SBATCH --job-name=parityladder
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
# ============================================================================
# Train/eval parity ladder: does serving an SFT arm the way it was TRAINED
# recover any of the score it loses to the un-finetuned base?
# ============================================================================
#
# Two independent train/eval mismatches were verified on disk:
#
#   (1) FORMAT -- the shipped Qwen3.5 chat template appends
#       '<think>\n\n</think>\n\n' (exactly 4 tokens, ids [248068,271,248069,271])
#       to EVERY assistant turn when enable_thinking=false, which is how eval
#       serves. LLaMA-Factory's `qwen3_5_nothink` training template ends the
#       assistant header at '<|im_start|>assistant\n' with no stub and does not
#       use ReasoningTemplate. Empirically: 0 occurrences of <think>/</think>
#       in 6,508,136 training tokens. Applies to all six arms, and NOT to base,
#       which is served in its own shipped format.
#
#   (2) TASK STATEMENT -- eval's default.j2 differs from the swezero training
#       statement in a cosmetic wrapper (<uploaded_files> + bare vs absolute
#       repo dir) AND, materially, in that 99.9% of swezero training statements
#       carry an explicit prohibition on running python/tests and ship 5 phases
#       instead of eval's 8 (dropping RUNNING, TEST CREATION, VERIFICATION).
#
# These are NOT the same kind of mismatch. (1) and the wrapper half of (2) are
# pure format: no instruction content, so removing them removes a confound.
# The prohibition is semantic: restoring it forbids the model from doing things
# that work in the eval container, so it REMOVES capability rather than
# removing a confound. Hence a ladder, not a single "full parity" cell --
# each rung adds exactly one change so an effect can be attributed.
#
#   arm cells (ARM_MODEL):
#     A  stock template + eval default.j2      control; how the board was measured
#     B  nostub        + eval default.j2       + stub removed
#     C  nostub        + train_wrapper.j2      + header/path style matched
#     D  nostub        + train_swezero_noenv.j2  + prohibition & 5-phase list
#
#   base cells (BASE_MODEL) -- stub axis only; the training statement is
#   swezero-specific and meaningless for a model that never saw it:
#     E  stock template + eval default.j2
#     F  nostub        + eval default.j2
#
# The base pair is what makes the arm result interpretable. If removing the
# stub helps the arm and does NOT help base, the stub is part of why base >>
# arms and every arm number on the board is an underestimate. If it helps both,
# it is a general serving defect unrelated to SFT. If neither moves, that
# hypothesis is dead. E/F also give the first empirical test of the
# "base is in-distribution for the stub" step, which was inferred from the
# shipped template and never measured.
#
#   6 cells x 2 shards of 50 = 12 jobs, n=100 per cell.
#     array index K:  cell = K % 6      shard = K / 6
#
#   SWEBENCH_ROOT=<eval root> \
#   ARM_MODEL=<sft checkpoint dir> \
#   BASE_MODEL=<un-finetuned model dir or HF id> \
#     sbatch --array=0-11 run_parity_ladder.sbatch
#
set -uo pipefail
: "${SWEBENCH_ROOT:?set SWEBENCH_ROOT to the eval root}"
: "${ARM_MODEL:?set ARM_MODEL to the SFT checkpoint under test}"
: "${BASE_MODEL:?set BASE_MODEL to the un-finetuned comparator}"
source "$SWEBENCH_ROOT/env.sh"
export NUM_WORKERS=${NUM_WORKERS:-16}

K=${SLURM_ARRAY_TASK_ID:?run with --array=0-11}
P=$SWEBENCH_ROOT/prompts_adp
NOSTUB=$P/nostub.jinja

# PROMPT_PATH empty => the harness renders its own default.j2, byte-for-byte as
# every existing board number was produced. Do not pass a copy: pass nothing.
CHAT_TEMPLATE=""
PROMPT_PATH=""
case "$(( K % 6 ))" in
  0) CELL=A_arm_stock_evalp   ; MODEL="$ARM_MODEL"  ;;
  1) CELL=B_arm_nostub_evalp  ; MODEL="$ARM_MODEL"  ; CHAT_TEMPLATE="$NOSTUB" ;;
  2) CELL=C_arm_nostub_wrap   ; MODEL="$ARM_MODEL"  ; CHAT_TEMPLATE="$NOSTUB" ; PROMPT_PATH=$P/train_wrapper.j2 ;;
  3) CELL=D_arm_nostub_trainp ; MODEL="$ARM_MODEL"  ; CHAT_TEMPLATE="$NOSTUB" ; PROMPT_PATH=$P/train_swezero_noenv.j2 ;;
  4) CELL=E_base_stock_evalp  ; MODEL="$BASE_MODEL" ;;
  5) CELL=F_base_nostub_evalp ; MODEL="$BASE_MODEL" ; CHAT_TEMPLATE="$NOSTUB" ;;
esac

[ -n "$CHAT_TEMPLATE" ] && { [ -s "$CHAT_TEMPLATE" ] || { echo "FATAL: missing $CHAT_TEMPLATE"; exit 1; }; }
[ -n "$PROMPT_PATH"   ] && { [ -s "$PROMPT_PATH"   ] || { echo "FATAL: missing $PROMPT_PATH";   exit 1; }; }

SH=$(printf '%02d' "$(( K / 6 ))")
SELECT=$SWEBENCH_ROOT/select/shard_${SH}of10.txt
[ -s "$SELECT" ] || { echo "FATAL: missing $SELECT"; exit 1; }

TAG=par_${CELL}__s${SH}
echo "== parity-ladder K=$K cell=$CELL shard=$SH tag=$TAG n_ids=$(wc -l < "$SELECT") =="
echo "== model=$MODEL =="
echo "== chat_template=${CHAT_TEMPLATE:-<shipped/stock>} =="
echo "== prompt_path=${PROMPT_PATH:-<harness default.j2>} =="
for f in "$CHAT_TEMPLATE" "$PROMPT_PATH"; do
  [ -n "$f" ] && echo "== sha256 $(basename "$f"): $(sha256sum "$f" | cut -c1-16) =="
done

export PROMPT_PATH
exec bash "$SWEBENCH_ROOT/scripts/run_full_infer_prompt.sbatch" \
  "$MODEL" "$TAG" "$SELECT" "$CHAT_TEMPLATE"
