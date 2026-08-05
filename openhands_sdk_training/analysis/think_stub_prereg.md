# The empty `<think>` stub — pre-registered falsifier

**Status: filed 2026-08-05 ~03:10 UTC, before any rollout of this probe exists.**
§1 is verification of an existing claim against data already on disk. §3 has not
been computed.

## 1. The mismatch, independently verified

`adp_thinking_traces_report.md` §2 (agent-b17cac1e, PR #7) identifies an empty
`<think>\n\n</think>` stub as "the one genuine train/eval mismatch" and marks its
impact **Unmeasured**, naming the falsifier it did not run. This document runs it.

Re-verified here from the generating sources rather than by re-reading the report:

| check | method | result |
|---|---|---|
| eval emits the stub | render the checkpoint's own `chat_template.jinja` | `enable_thinking=False` → `<\|im_start\|>assistant\n<think>\n\n</think>\n\n` |
| stub size | tokenize | exactly **4 tokens**, ids `[248068, 271, 248069, 271]` |
| training omits it | read `qwen3_5_nothink` in LLaMA-Factory `data/template.py` | `format_user` and `format_observation` both end at `<\|im_start\|>assistant\n`; **no** `ReasoningTemplate` (unlike `qwen3_5`/`qwen3_6`, which use it) |
| training omits it, empirically | scan the real tokenized cache, 300 records | **0** occurrences of `<think>` or `</think>` in **6,508,136** tokens |
| base and arm share a template | diff base snapshot vs SFT checkpoint | **identical** |

So:

| | assistant-turn prefix the model is conditioned on |
|---|---|
| **training** | `<\|im_start\|>assistant\n` → generate |
| **eval** | `<\|im_start\|>assistant\n<think>\n\n</think>\n\n` → generate |

Every assistant turn at eval carries 4 tokens the SFT'd models never saw in that
position. With median trajectory lengths of 53–246 turns, that is roughly
200–1000 injected tokens per rollout, at every generation boundary. It applies
to all six arms identically and — this is what makes it worth GPU — **not to
base**, which is served in the format its own shipped template prescribes.

**The step this rests on, and that nobody has tested:** that base is therefore
"in-distribution" assumes Qwen post-trained Qwen3.5-4B's non-thinking mode with
this exact stub. The shipped template prescribes it; Qwen's actual
post-training format is not knowable from here. §2's base row tests it
behaviourally for the first time — if base is genuinely stub-native, removing
the stub should **hurt** base.

Note what this hypothesis can and cannot do. It **cannot** explain arm-vs-arm
ordering, since all arms carry it identically. It is a candidate for the
**base ≫ arms** gap specifically.

## 2. Design

`nostub.jinja` = the shipped template with exactly one line changed: the
`enable_thinking is false` branch emits nothing instead of the stub. Verified as
a **pure 4-token trim** — with `add_generation_prompt=False` the two templates
render byte-identical output, and the only delta at the generation prompt is the
stub itself.

2 × 2 × 2 = 8 jobs, one pass:

| axis | cells |
|---|---|
| template | **stock** (stub; as the board was measured) · **nostub** (training-matched) |
| model | **arm** (`v2_swezero_inst_4b_a100`) · **base** (`Qwen/Qwen3.5-4B`) |
| shard | `select/shard_00of10.txt` · `shard_01of10.txt` — n=100 per cell |

All four conditions run fresh in the same pass. vLLM prefix caching + continuous
batching make greedy decoding non-reproducible, so existing runs cannot serve as
controls — the stock cells are fresh. Everything else is held: same harness
commit, temperature 0.0, same condenser settings, same tool preset, 16 workers.

**Priors** (existing full-500 runs, for anchoring only): arm 77/500, shard-0
5/50; base 119/500, shard-0 8/50.

**Power.** At a ~10–16% base rate and n=100, sd ≈ 3.0–3.7, so only score swings
≳6–7/100 clear 2sd. Score is the secondary endpoint and is coarse. The primary
endpoint is behavioural, at n=100 population proportions.

## 3. Predictions

Registered before any rollout of this probe exists.

**S1 (manipulation check) — removing the stub raises the arm's `think` usage.**
Registered: arm-nostub `think`-call rate > arm-stock `think`-call rate.
*Rationale:* `<think>\n\n</think>` is literally "you just finished thinking and
produced nothing," handed to a model trained to *open* its turn with
`<tool_call><function=think>`. agent-b measured arm `think` share at 12.9% of
generated tokens at eval against 20.5% of loss-bearing tokens in training —
directionally consistent with suppression. (I lean only on the **direction of
change within the eval denominator**, stock vs nostub. The eval/train *ratio*
compares incomparable denominators, a point devils-advocate raised and agent-b
accepted; it is not used here.)
*If falsified* — no behavioural change at all — the stub is inert and S2/S3 are
uninterpretable-but-moot. Report and close.

**S2 (primary) — does the stub cost the arm points?** Registered threshold:
arm-nostub − arm-stock ≥ **+6/100** would mean the stub costs real score and
**every arm number on the board is an underestimate**.
No directional commitment beyond S1: a 4-token constant prefix may be entirely
harmless, and I am not predicting a score gain.

**S3 (the interaction, and the actual result) — base should not benefit.**
Registered: base-nostub − base-stock ≤ **+3/100**, and specifically I predict
base is *unchanged or hurt*.
*Rationale:* base is served in its shipped format; removing the stub moves base
**out** of distribution while moving the arm **in**.

The 2 × 2 is what distinguishes the three live worlds:

| arm | base | reading |
|---|---|---|
| gains | flat/hurt | **stub is part of base ≫ arms.** Re-measure the board no-stub before any further curation work. |
| gains | gains | general serving defect, not SFT-specific. Still fix it; does **not** explain the gap. |
| flat | flat | hypothesis dead. agent-b §2 closes as measured-and-null, removing a live alternative to the data-quality account. |

**S4 (directional, no threshold).** If S1 holds, arm-nostub's behavioural profile
moves *toward* its training profile on the other pinned metrics
(`verified_after_edit`, `ran_any_test`) — not necessarily toward base's.

## 4. Decision rules, fixed in advance

1. **Arm gains ≥6/100, base does not** → halt curation work and re-measure the
   full board under `nostub`. The v2/v3 boards would all be biased against the
   arms, and no curation result is interpretable until re-measured.
2. **Both gain** → file as a serving defect, fix the serving path, but do not
   attribute the base ≫ arms gap to it.
3. **Neither moves** → close the hypothesis as measured-and-null. This is a
   useful result: it removes the last live alternative to a data-quality
   explanation for base ≫ arms, which currently rests on the refutation of
   Lever A, the verification hypothesis, and Lever B.
4. **Arm loses** → suspect the serving path, not the science. The arm saw the
   stub 0 times in 6.5M training tokens, so there is no mechanism by which
   removing it should hurt. Inspect the rendered prompts before reporting.
5. Behavioural extraction for all four conditions must run **in the same pass**
   with `traj_compare/extract_traj_stats.py` pinned, matching the
   same-scoring-pass discipline used for the A1 readout.

## 5. Relationship to the prompt-parity probe

`prompt_parity_prereg.md` registers a separate 3-cell probe on the *task
statement*. That one is **deferred, not cancelled**. Reason: the task-statement
mismatch is arm-specific and in its severe form affects only swezero, so it is a
weak candidate for the board-wide base ≫ arms gap, whereas the stub applies to
all six arms and not to base. Its own predictions (P1 wrapper inert, P3 no score
rescue) also both forecast nulls. The stub probe is the better use of the same
GPUs and runs first.

## 6. Pinned

- Template: `prompts_adp/nostub.jinja`, generated from the checkpoint's shipped
  `chat_template.jinja` by replacing the single stub-emitting line.
- Runner: `swe-bench-fair-evals/scripts/run_stubmatch.sbatch`.
- Metrics: `traj_compare/extract_traj_stats.py` on branch `traj-compare-viz`.
- Instance sets: `select/shard_00of10.txt`, `select/shard_01of10.txt`.
- Source claim under test: `adp_thinking_traces_report.md` §2 (PR #7).
