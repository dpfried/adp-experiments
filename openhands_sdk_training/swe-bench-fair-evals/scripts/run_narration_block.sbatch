#!/usr/bin/env bash
#SBATCH --job-name=narrblock
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
# Submit from $SWEBENCH_ROOT (log paths are relative).
#
# ============================================================================
# Cell G -- is base's plain-message narration load-bearing chain-of-thought?
# ============================================================================
#
# base spends 16.2% of its generated tokens on prose emitted ALONGSIDE a tool
# call (event-level `thought` on 48.2% of its 85,726 action events: "Now I can
# see the problem clearly...", "BINGO! There's the bug!"). Every SFT arm has
# essentially none (swezero: exactly 0 / 45,920 events). Because the prose and
# the tool call are ONE completion with the prose first, the action tokens are
# conditioned on it -- it is CoT by construction. The open question is only
# whether the conditioning does any work.
#
# INTERVENTION: prefill '<tool_call>\n' into the assistant generation prompt.
# Generation then begins INSIDE an open tool call, so prose-before-the-call is
# physically impossible. Verified offline, no GPUs:
#   * prompts_adp/prefill_toolcall.jinja renders byte-identically to the stock
#     Qwen3.5 template plus exactly '<tool_call>\n' (+2 tokens [248058,198]).
#   * vLLM 0.25.1's qwen3_coder parser (Qwen3EngineToolParser) has an explicit
#     "Fallback: <function= without a preceding <tool_call>" transition, so a
#     completion starting at '<function=' parses to a clean tool call with
#     content=None. The narrated form instead fills `content` -- that field IS
#     the 16.2% channel, so this removes exactly it and nothing else.
#   * The `think` TOOL is itself a tool call and still fires normally, so this
#     separates the 16.2% narration channel from the 19.5% think channel.
#
# MATCHED CONTROL: cell E of run_parity_ladder.sbatch -- base, stock template,
# harness default.j2, shards 00/01, n=100, harness sdk_43376f1, maxiter 500.
# G is that invocation with ONE flag changed. Same instance ids => paired.
#
# SCORING: attempt 1 ONLY, against E's attempt 1. The harness retries whichever
# instances its critic rejects and attempts >=2 sample at temperature 0.1, and
# the critic's rejection rate is strongly model- and cell-dependent (it rejected
# base 71/100 in cell F but 15/100 in cell E, identical config). Two cells of
# the same model are NOT compute-matched by default. E's
# output.critic_attempt_1.jsonl is already complete and frozen (100/100 unique,
# exact select coverage), so this does not wait on E's remaining attempts.
#
# POWER (exact conditional McNemar, paired, alpha=0.05 two-sided, base ~24%,
# with the ~3%-of-instances SBV label-flip floor): n=100 has an MDE of 12.1pp
# at 80% power => powered to detect a HALVING (0.79) and nothing finer (25%
# relative drop: 0.31). A null here reads "no LARGE effect", never "no effect".
# The behavioural outcome is a different regime: "did the block fire" is
# measured over ~20k action events and is decisive either way.
#
# PRE-REGISTERED SECONDARY OUTCOMES (report these, do not discover them later):
#   1. Did narration actually go to ~0? (primary validity check)
#   2. Did it RELOCATE into the `think` tool or a parameter string?
#   3. Cap-hit rate. E caps on 35.0%; suppressing prose changes tokens/turn, so
#      it perturbs turns-reached-before-cap in an a-priori unclear direction.
#   4. Forcing a tool call every turn is itself distribution shift, so a drop is
#      not automatically evidence that the *reasoning* mattered.
#
#   2 shards of 50 = 2 jobs, n=100.  array index K: shard = K
#
#   SWEBENCH_ROOT=<eval root> BASE_MODEL=<model dir> \
#     sbatch --array=0-1 run_narration_block.sbatch
#
set -uo pipefail
: "${SWEBENCH_ROOT:?set SWEBENCH_ROOT to the eval root}"
: "${BASE_MODEL:?set BASE_MODEL to the un-finetuned comparator (must match cell E)}"
source "$SWEBENCH_ROOT/env.sh"
export NUM_WORKERS=${NUM_WORKERS:-16}

K=${SLURM_ARRAY_TASK_ID:?run with --array=0-1}
P=$SWEBENCH_ROOT/prompts_adp
CHAT_TEMPLATE=$P/prefill_toolcall.jinja
# PROMPT_PATH empty => harness renders its own default.j2, exactly as cell E did.
PROMPT_PATH=""
CELL=G_base_prefill_evalp
MODEL="$BASE_MODEL"

[ -s "$CHAT_TEMPLATE" ] || { echo "FATAL: missing $CHAT_TEMPLATE"; exit 1; }

# Fail fast if the template is not still the intended one-line delta from stock.
# A silent drift here would make G incomparable to E while still producing
# plausible numbers -- the exact failure mode this campaign keeps hitting.
GUARD_PY="${SB_VENV_VLLM:-/checkpoint/dpf/swebench-eval/.venv_vllm}/bin/python"
[ -x "$GUARD_PY" ] || { echo "FATAL: no guard interpreter at $GUARD_PY"; exit 1; }
"$GUARD_PY" - "$MODEL" "$CHAT_TEMPLATE" <<'PY' || exit 1
import sys
from transformers import AutoTokenizer
model, tpl = sys.argv[1], sys.argv[2]
tk = AutoTokenizer.from_pretrained(model)
msgs = [{"role": "user", "content": "hello"}]
kw = dict(tokenize=False, add_generation_prompt=True,
          chat_template_kwargs={"enable_thinking": False})
stock = tk.apply_chat_template(msgs, **kw)
tk.chat_template = open(tpl).read()
mine = tk.apply_chat_template(msgs, **kw)
if mine != stock + "<tool_call>\n":
    print("FATAL: template is not stock + '<tool_call>\\n'")
    print("  stock tail:", repr(stock[-60:]))
    print("  mine  tail:", repr(mine[-60:]))
    sys.exit(1)
extra = tk(mine).input_ids[len(tk(stock).input_ids):]
print(f"== template guard OK: +{len(extra)} tokens {extra} ==")
PY

SH=$(printf '%02d' "$K")
SELECT=$SWEBENCH_ROOT/select/shard_${SH}of10.txt
[ -s "$SELECT" ] || { echo "FATAL: missing $SELECT"; exit 1; }

TAG=par_${CELL}__s${SH}
echo "== narration-block K=$K cell=$CELL shard=$SH tag=$TAG n_ids=$(wc -l < "$SELECT") =="
echo "== model=$MODEL =="
echo "== chat_template=$CHAT_TEMPLATE =="
echo "== prompt_path=<harness default.j2> =="
echo "== sha256 $(basename "$CHAT_TEMPLATE"): $(sha256sum "$CHAT_TEMPLATE" | cut -c1-16) =="
echo "== matched control: par_E_base_stock_evalp__s${SH} (attempt 1) =="

export PROMPT_PATH
exec bash "$SWEBENCH_ROOT/scripts/run_full_infer_prompt.sbatch" \
  "$MODEL" "$TAG" "$SELECT" "$CHAT_TEMPLATE"
