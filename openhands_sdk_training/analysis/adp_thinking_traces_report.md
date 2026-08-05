# How "thinking" works in the ADP campaign — training vs evaluation

_2026-08-04. Every claim below is measured from configs, the tokenized training caches, or
the eval rollouts on disk. Where something is an **inference** rather than a measurement it
is labelled **[INFERENCE]**. Where something is **unmeasured** it says so._

---

## 0. TL;DR

Three different things get called "thinking" here. Keeping them apart is the whole report.

| # | thing | in training? | in eval? |
| --- | --- | --- | --- |
| 1 | **Native CoT** — model generates `<think>…</think>` reasoning | **No** | **No** |
| 2 | **The empty `<think>\n\n</think>` stub** — literal text from the chat template | **No** | **Yes, every turn** |
| 3 | **The ADP `think` TOOL** — reasoning emitted as a tool call | **Yes, loss-bearing** | **Yes, live** |

Second headline, added 2026-08-04 (revised after adversarial review on PR #7): the `think`
tool is not a curiosity — it carries **20–22%** of the SFT gradient for swezero and the v3
arms, and it transmits **gradedly** into eval behaviour (22.3% → 15.9% at the top,
8.5% → 3.7% at the bottom, 0 → 0 exactly for `scale`) while predicting score not at all.
Transmission fidelity is **highest for `think` and lowest for environment-coupled tools** —
i.e. the behaviour that copies best is the one that doesn't matter (§3.4b–§3.4c).

The headline: **no model in this campaign ever generated a native reasoning trace.** What
looks like "thinking" in the trajectories is #3, an ordinary tool call. And #2 is a
format difference between training and eval that is real but of unmeasured consequence.

⚠️ **Third headline, added 2026-08-05: every base statistic in this report is computed over
300 of base's 500 rollouts, not 500** — the harness discards the transcript of an instance
that ends in error, and base ends in error on 200 (mostly by exhausting the 500-iteration
cap). Read **§0b before quoting any base number.** It does not touch any training-side
figure, any arm-vs-arm comparison, or the Lever-A refutation.

---

## 0b. ⚠️ Coverage defect: 200 of base's 500 rollouts carry no transcript

Found 2026-08-05 while testing the narration hypothesis (§3.4f). This is a property of the
existing eval artifacts, not of anything built here, and it affects numbers already
published in this report and in the prereg.

**What happens.** Every model ran with `max_iterations: 500` — identical, read from each
rollout's own `metadata`. When an instance terminates in error after the harness's retries,
the record is written with `history: []` and `metrics: null`; the transcript and the token
accounting are gone. The instance is still **graded normally**, because the prediction is
the workspace diff (`test_result.git_patch`), which survives independently of the
conversation record. So these instances score, but cannot be measured.

| model | n | blank | of which cap-hit | resolved | from blank | from kept | resolve-rate\|kept | completion tok (kept) | tok/inst |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **base** | 500 | **200** | **131** | 119 | **26** | 93 | 31.0% | 16,402,192 | **54,673** |
| swezero | 500 | 0 | 0 | 77 | 0 | 77 | 15.4% | 5,406,525 | 10,813 |
| rebench | 500 | 5 | 1 | 70 | 0 | 70 | 14.1% | 19,736,006 | 39,870 |
| coderforge | 500 | 3 | 0 | 48 | 0 | 48 | 9.7% | 15,499,780 | 31,186 |
| scale | 500 | 1 | 1 | 35 | 0 | 35 | 7.0% | 14,555,603 | 29,169 |
| v3_tokmatch | 500 | 0 | 0 | 63 | 0 | 63 | 12.6% | 5,135,969 | 10,271 |
| v3_maxpool | 500 | 0 | 0 | 62 | 0 | 62 | 12.4% | 4,866,882 | 9,733 |

Error taxonomy over base's 200 blanks: `MaxIterationsReached` **131**, `OSError` 20,
"Instance did not complete" 16, other conversation-run failures 33. A representative record
(`sympy__sympy-14531`): `history: []`, `metrics: null`,
`error: "Instance failed after 3 retries. Last error: … MaxIterationsReached: Agent reached
maximum iterations limit (500)"` — and a **6,309-character graded patch**. It resolves.

**Four consequences, in descending order of how much they should change your reading.**

1. **Test-time compute was never controlled, and base uses the most of it.** Recorded
   completion tokens per transcript-bearing instance: base **54.7k**, rebench 39.9k,
   coderforge 31.2k, scale 29.2k, swezero **10.8k**, v3 arms ~10k. And base's 16.4M
   *excludes the 200 capped runs entirely*, so its true spend is materially higher than any
   number here. Resolves per million recorded completion tokens: swezero **14.2**,
   v3_maxpool 12.7, v3_tokmatch 12.3, base **≤7.3** (5.7 counting only kept-run resolves),
   rebench 3.5, coderforge 3.1, scale 2.4. **The best arm is ~2× more token-efficient than
   base.** "base ≫ arms" is a statement about a fixed *iteration* budget, not a fixed
   compute budget; under a token budget the ranking is different. This does not make the gap
   go away — it reframes what the gap is.
2. **All base behavioural statistics are over a biased 60% subsample.** The 19.5% `think`
   share, the 16.2% narration share, 13.4 `think` calls/instance, 85,726 action events, and
   every action-depth median are computed over the **300** transcript-bearing runs, and the
   200 excluded are systematically the **longest** ones. Arm figures are essentially
   unaffected (0–5 blanks). Every base-vs-arm comparison in this report therefore compares a
   truncated-at-the-top base sample against complete arm samples.
3. **26 of base's 119 resolves (21.8%) have no transcript at all** and cannot be attributed
   to any measured behaviour. No arm has a single such resolve.
4. **The depth gap is larger than reported, and base is the only model the cap binds on.**
   Kept-run action counts, p50: base **216**, rebench 203, scale 200, coderforge 190,
   swezero 62, v3_tokmatch 53, v3_maxpool 50 — *before* counting the 40% of base runs with no
   transcript, whose largest group (131/500 = **26.2%** of all base runs) ran past the cap. Two things are true at once: the cap is a shared, fair budget, and base
   is the only model that exhausts it, so base's 119 is measured **with a truncation
   handicap** and would plausibly be higher under a larger cap.

**What it does not touch.** No training-side measurement (those read the tokenized caches).
No arm-vs-arm comparison, including the Lever-A refutation — swezero, v3_tokmatch and
v3_maxpool have **zero** blanks between them.

#### 0b.1 Retracted: my own "the cap-hit rate is a property of *that run*" scope limit

**This subsection previously claimed that the cap-hit rate did not reproduce, and it was
wrong.** The claim was: across 146 parity-ladder base rollouts there are "0 blank records and
0 `MaxIterationsReached`" where ~38 were expected, so the ~40% was an artifact of the old run.

The error: I counted from `output.jsonl`. **Instances that terminate in error are not written
to `output.jsonl` at all** — they go to `output_errors.jsonl` in the same directory. So a
`MaxIterationsReached` search over `output.jsonl` returns zero *by construction*, for any cell,
always. I searched the one file from which the thing I was looking for is definitionally
absent, and read the zero as evidence. Caught by the ladder owner within the hour; this is
the second time this defect has produced a published-then-reverted conclusion in this
campaign, which is why it is now enforced in code rather than in prose —
[`ANALYSIS_HOUSE_RULES.md`](ANALYSIS_HOUSE_RULES.md) rule 1, `load_rollouts.py`.

**Measured** — re-run through the loader on the frozen, complete, attempt-matched file
(`output.critic_attempt_1.jsonl`, one row per instance, exact coverage of both select shards,
`strict=True` passing; independent of the two earlier hand-counts):

| base cell, attempt 1 | n | ok | **cap500** | timeout | stuck | ENOSPC | other | has patch | transcript |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **E** base + **stock (stub)** template | 100 | 46 | **35 (35.0%)** | 6 | 5 | 3 | 5 | 74 | 46 |
| **F** base + **nostub** template | 100 | 100 | **0 (0.0%)** | 0 | 0 | 0 | 0 | 82 | 100 |

Same harness `sdk_43376f1`, same `maxiter_500`, same 100 instances, same base weights, same
critic config, one axis different. Board base row for comparison: 131/500 = 26.2% cap-hit.

**Inferred** — the cap-hit phenomenon is real and reproduces on new hardware, a new harness and
a new date; what my retraction got right is only that it is not an unconditional property of
"base." It tracks the **empty `<think>` stub** (35.0% with, 0.0% without), which is the same
train/eval prompt mismatch §2 describes. Consequence 4 therefore stands with a sharper
attribution: base's 119 was measured with a truncation handicap, and the handicap is
stub-linked. Residual confounds I have *not* excluded: the two templates change tokens-per-turn
and hence turns-per-cap differently, and I have one nostub cell, not a series — so "the stub
causes the capping" is one mechanism consistent with a 35-vs-0 split, not the only conceivable
one. (Rule 4: two mechanisms checked, residual unbounded.)

Consequences 1–3 were never in question and are untouched — they are claims about *those*
artifacts: the board's base row really is 300 transcripts, 26 of its resolves really are
unattributable, and the token spend behind the 119 really was never controlled.

Two further measured items from the same re-read, both of which cut against reading cell E
naively:

- **E's error mix is not pure capping.** 35 cap-hits, but also 6 timeouts, 5 stuck, 3 ENOSPC
  (a node-local `/scratch` fill — `/checkpoint` was at 35%), 5 other. Anyone quoting "cell E
  fails half the time" is bundling a disk-full node with the finding.
- **The depth gap in consequence 4 is understated, not overstated.** E's p50 of 223 actions is
  computed over its **46** transcript-bearing runs — which *exclude* the 35 that ran out of
  iterations, i.e. exactly its longest runs. F's 233 is over a complete 100. Both are
  top-truncated reads of a distribution whose tail is the interesting part.

---

## 1. Native chain-of-thought: off in training, off in eval

### 1.1 Training — the template has no reasoning slot

Every v2/v3 config sets `template: qwen3_5_nothink`. In LLaMA-Factory
(`src/llamafactory/data/template.py`) that is registered at line **2167** as a plain
`Template`. Its with-thinking sibling `qwen3_5` is registered at line **2151** and is
byte-identical **except** for one line:

```python
register_template(
    name="qwen3_5",
    ...
    template_class=ReasoningTemplate,   # <-- present at :2163, ABSENT from qwen3_5_nothink
)
```

`ReasoningTemplate` (`:407`) is the class that strips/injects `<think>` blocks. We do not
use it. So nothing in the training pipeline ever adds, removes, or reasons about `<think>`.

### 1.2 Training — verified against the actual trained tokens

Not from the config: decoded from the **tokenized Arrow caches the runs actually consumed**
(v2_swezero, v3_tokmatch, v3_maxpool; first record = 12,945 tokens, 19.8% loss-bearing):

| check | result |
| --- | --- |
| `<think>` occurrences in the decoded record | **0** |
| empty `<think>\n\n</think>` blocks | **0** |
| assistant turns in the record | 17 |

### 1.3 Eval — three independent confirmations

**(a) Config.** `run_full_infer.sbatch` writes, identically for base and every arm:

```json
"reasoning_effort": "none",
"litellm_extra_body": {
  "stop_token_ids": [248046],
  "chat_template_kwargs": {"enable_thinking": false}
}
```

Verified present in all **10 base shards** and in the arm shards. (`248046` decodes to
`<|im_end|>`.)

**(b) Serving stack.** vLLM ran with `reasoning_parser=''` in both base and arm logs.

**(c) Measured output.** Across **7 models × 500 instances = 3,500 rollouts**:

| model | score /500 | `reasoning_tokens` | `completion_tokens` | `reasoning_content` non-empty | `thinking_blocks` non-empty |
| --- | --- | --- | --- | --- | --- |
| base | 119 | **0** | 16,402,192 | 0 | 0 |
| swezero | 77 | **0** | 5,406,525 | 0 | 0 |
| rebench | 70 | **0** | 19,736,006 | 0 | 0 |
| v3_tokmatch | 63 | **0** | 5,135,969 | 0 | 0 |
| v3_maxpool | 62 | **0** | 4,866,882 | 0 | 0 |
| coderforge | 48 | **0** | 15,499,780 | 0 | 0 |
| scale | 35 | **0** | 14,555,603 | 0 | 0 |

**Zero reasoning tokens out of 81.6M completion tokens.** Note this is not circular: with
`reasoning_parser=''`, vLLM does **not** strip reasoning — any generated CoT would have
landed verbatim in message content. It didn't.

**Residual leakage, for completeness.** The literal string `<think>` appears in generated
text in **5 base events (5 instances)** and **1 v3_tokmatch event**, and nowhere else —
6 events out of 3,500 rollouts. Example (base, `sphinx-doc__sphinx-9658`), where the model
emitted a bare tag as message content alongside a normal tool call:

```json
"thought": [{"type": "text", "text": "<think>"}],
"action": {"thought": "Let me start by clearly understanding the issue.\n\n## Phase 1: READING…"}
```

This is negligible and should not be built into any story.

---

## 2. The empty `<think>` stub — the one genuine train/eval mismatch

### 2.1 What it is

The stock Qwen3.5 chat template — verified **both** in the base snapshot and inside every
SFT checkpoint's own `output/chat_template.jinja` — contains:

```jinja
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n' }}
    {%- if enable_thinking is defined and enable_thinking is false %}
        {{- '<think>\n\n</think>\n\n' }}
    {%- else %}
        {{- '<think>\n' }}
    {%- endif %}
{%- endif %}
```

Rendered (measured, not paraphrased):

```
enable_thinking=False → '…<|im_start|>assistant\n<think>\n\n</think>\n\n'
enable_thinking=True  → '…<|im_start|>assistant\n<think>\n'
```

**Nothing produces the text inside those tags.** The stub is *literal template output*
appended to the **prompt**; generation begins *after* the closing `</think>`. It is empty by
construction and always will be. It tokenizes to exactly **4 tokens**:
`['<think>', '\n\n', '</think>', '\n\n']` = ids `[248068, 271, 248069, 271]`.

### 2.2 Training does not have it

From the loss-mask decode of the real cache, here is the same boundary in training —
note the assistant header is **masked**, and the model's first *generated* token is
`<tool_call>`, with no stub in between:

```
MASKED (-100):     "…quality and completeness are more important than brevity.<|im_end|>\n<|im_start|>assistant\n"
LOSS-BEARING:      "<tool_call>\n<function=think>\n<parameter=thought>\nLet me analyze the issue description:…"
```

So:

| | assistant-turn prefix the model is conditioned on |
| --- | --- |
| **training** | `<\|im_start\|>assistant\n` → then generate |
| **eval** | `<\|im_start\|>assistant\n<think>\n\n</think>\n\n` → then generate |

**Every assistant turn at eval carries 4 tokens the SFT'd models never saw in that
position.** This affects all six SFT arms.

### 2.3 What is and is not established about its impact

**Verified:** the eval *treatment* is **identical** for base and arms — same
`llm_config.json` fields, `reasoning_parser=''` in both, and **no `--chat-template`
override for either** (the one `--chat-template` hit in the vLLM logs is an informational
line about `--chat-template-content-format`, and it appears in base and arm logs alike).

**The asymmetry is at training time, not eval time.** Base is served in the format its own
shipped template prescribes; the arms were fine-tuned in a *different* format and then
served in the shipped one.

**[INFERENCE], flagged because it is the load-bearing step:** that base is therefore
"in-distribution" assumes Qwen post-trained Qwen3.5-4B's non-thinking mode with this exact
stub. I verified only that the *shipped template* prescribes it — **I have not verified
Qwen's actual post-training format**, and cannot from here.

**Unmeasured:** whether any of this costs points. A constant 4-token prefix may be entirely
harmless. It **cannot** explain arm-vs-arm ordering, since all arms share it identically.
It is a live alternative to a data-quality explanation for the base ≫ arms gap only because
Lever A, the verification hypothesis, and Lever B are all now refuted — not because there is
positive evidence for it.

**Cheap falsifier, not run:** re-serve one arm with `--chat-template` pinned to a jinja that
omits the stub, paired on a fixed instance subset. One eval.

---

## 3. The ADP `think` tool — the campaign's only real "reasoning" channel

### 3.1 It is a genuine, declared tool

Declared in every training record's tool list (`terminal, file_editor, finish, think`).
Verbatim schema from the data:

> `"name": "think"`, `"description": "Use the tool to think about something. It will not
> obtain new information or make any changes to the repository, but just log the thought.
> Use it when complex reasoning or brainstorming is needed. …"`

### 3.2 In training — raw form, rendered form, and loss

**Raw ADP record** (swezero, first record). Reasoning is a `function_call` message; the
environment's reply is a fixed string:

```json
messages[2]  role=function_call  [{"name": "think", "arguments": {"thought": "Let me analyze the issue description:\n\n1. The problem is about Deprecation Warning for `asmatrix`…"}}]
messages[3]  role=tool           "Your thought has been logged."
```

**As trained** (decoded from the cache, with the loss mask applied):

```
LOSS-BEARING: <tool_call>
              <function=think>
              <parameter=thought>
              Let me analyze the issue description:

              1. The problem is about Deprecation Warning for `asmatrix` function from numpy
              …
```

⇒ **The model is explicitly trained to emit its reasoning as a tool call.** This is #3, and
it is loss-bearing — unlike #1 and #2, which do not exist in training at all.

> **Counting note:** count these *structurally* (`role == "function_call"` → parse the JSON
> list → read `name`). The `<function=think>` XML is generated by LLaMA-Factory's
> `FunctionFormatter` at tokenization time, so grepping the raw jsonl for `<function=think>`
> returns **0** and is misleading. (I made exactly this error first.)

**Training census:**

| subset | records | tool calls | `think` calls | % of calls | % of records |
| --- | --- | --- | --- | --- | --- |
| swezero | 79,874 | 1,173,922 | 83,529 | 7.1% | 58.4% |
| coderforge | 79,890 | 1,112,732 | 72,309 | 6.5% | 50.8% |
| rebench | 79,887 | 1,062,348 | 45,279 | 4.3% | 40.4% |
| **scale** | 79,900 | 1,160,054 | **0** | **0.0%** | **0.0%** |
| pooled220k | 220,000 | 3,100,233 | 137,930 | 4.4% | 37.3% |
| pooled55k | 55,000 | 770,882 | 34,381 | 4.5% | 37.2% |

### 3.3 In eval — what actually happens when `think` is called

The tool is offered and works. It is a **pure no-op scratchpad**: it mutates nothing and
returns a fixed acknowledgement. Verbatim from a swezero rollout (`sympy__sympy-15599`):

```json
// the call
{"kind": "ActionEvent", "source": "agent", "tool_name": "think",
 "thought": [], "reasoning_content": null, "thinking_blocks": [],
 "action": {"thought": "Looking at the issue description and the code, I can see the problem now:…"}}

// what comes back
{"kind": "ObservationEvent", "tool_name": "think",
 "observation": {"content": [{"type": "text", "text": "Your thought has been logged."}],
                 "is_error": false, "kind": "ThinkObservation"}}
```

Note the field layout, which is easy to misread: the reasoning text lives in
**`action.thought`** (the tool's argument). The event-level **`thought`** field is the
assistant's ordinary message content accompanying the call, and `reasoning_content` /
`thinking_blocks` are the *native*-CoT fields — empty everywhere (§1.3).

**Cost of a `think` call:** one full round-trip. The thought (median **1,251 chars** for
base, **1,469** for swezero; max ~8.4k) plus the canned reply are appended to context, and
the agent gets no new information — exactly as the tool description promises.

**Eval census:**

| model | /500 | `think` calls | instances using it | calls/inst | well-formed | malformed |
| --- | --- | --- | --- | --- | --- | --- |
| base | 119 | 6,676 | 290 (58%) | **13.4** | 5,843 | **833 (12.5%)** |
| swezero | 77 | 1,330 | 493 (99%) | 2.7 | 1,316 | 14 (1.1%) |
| rebench | 70 | 1,435 | 488 | 2.9 | 1,426 | 9 |
| v3_tokmatch | 63 | 1,233 | 488 | 2.5 | 1,221 | 12 |
| v3_maxpool | 62 | 1,246 | 488 | 2.5 | 1,227 | 19 |
| coderforge | 48 | 1,820 | 484 | 3.6 | 1,809 | 11 |
| **scale** | 35 | **0** | **0** | **0.0** | 0 | 0 |

### 3.4 The malformed-call failure mode

A `think` call with no parseable `thought` argument fails Pydantic validation and returns an
`AgentErrorEvent` instead of an observation. Verbatim (base, `django__django-11211`):

```json
"action": {},
"thought": [{"type": "text", "text": "I'll help you implement the necessary changes to fix the issue with prefetching related objects when using GFK…"}]
// next event:
{"kind": "AgentErrorEvent", "tool_name": "think",
 "error": "Error validating tool 'think': 1 validation error for ThinkAction\nthought\n  Field required [type=missing, input_value={}, input_type=dict] … Parameters provided: []"}
```

The model wrote its reasoning as *message content* and left the tool arguments empty.
**Base does this 833 times — 12.5% of its think calls, and ~50% of its 1,674 total agent
errors.** The arms do it ~1% of the time.

Worth stating plainly because it cuts against a tidy story: **the best model is by far the
worst at formatting this tool, and still wins by 42 points.** Do not turn tool-call hygiene
into a mechanism.

### 3.4b Where the `thought` text comes from at eval, and whether it is supervised

Two different texts are involved and they have opposite answers. Both measured.

**(i) The `thought` argument — the reasoning prose. Produced by the MODEL. SUPERVISED.**

At eval it is ordinary generated output, not a separate channel and not injected. The model
emits `<tool_call><function=think><parameter=thought>…</parameter></function></tool_call>`
in its normal completion; vLLM — launched with
`--enable-auto-tool-choice --tool-call-parser qwen3_coder`
(`run_full_infer.sbatch:50`; `auto_tool_choice': True` confirmed in the server log) — parses
that into a structured tool call, which OpenHands validates into a `ThinkAction`. The rollout
keeps both forms side by side (base, `django__django-15467`):

```json
"tool_call": {"id": "chatcmpl-tool-88ffca7c419213fa", "name": "think",
              "arguments": "{\"summary\": \"Analyze the RadioSelect empty_label bug\",
                             \"thought\": \"I see! Look at lines 1464-1469 in `ModelChoiceField.__init__`:…\"}"}
"action":    {"thought": "I see! Look at lines 1464-1469 in `ModelChoiceField.__init__`:…"}
```

⇒ **the same token stream as any other tool call.** Nothing distinguishes it at generation
time; only the tool name does.

At training this text is **fully inside the loss**. Measured by locating the token
subsequences for `<function=think>` (ids `[27, 1628, 28, 26003, 29]`) and `</function>`
(`[510, 1628, 29]`) in `input_ids` and reading the label mask over the span — no char↔token
mapping guesswork. Sampled with **contiguous windows spread across every shard** of each
tokenized cache (~790 records per subset; see §3.4d on why stride-sampling is wrong here):

| subset | recs | loss-bearing tokens | think spans | loss tokens inside `think` | **share of ALL supervision** |
| --- | --- | --- | --- | --- | --- |
| **v3_tokmatch** | 782 | 2,361,217 | 1,316 | 527,286 | **22.3%** |
| **v3_maxpool** | 780 | 2,400,856 | 1,294 | 516,459 | **21.5%** |
| **swezero** | 782 | 1,557,501 | 795 | 319,338 | **20.5%** |
| coderforge | 792 | 1,926,354 | 677 | 255,537 | **13.3%** |
| pooled55k | 792 | 1,886,566 | 537 | 208,592 | **11.1%** |
| pooled220k | 783 | 1,775,908 | 499 | 190,750 | **10.7%** |
| rebench | 792 | 2,106,931 | 444 | 178,049 | **8.5%** |
| **scale** | 798 | 1,684,767 | **0** | **0** | **0.0%** |

Two independent samples agree (first 600 records of shard 0 gave swezero 20.0%, coderforge
13.3%, rebench 8.0%, scale 0.0%), and the pooled figures match the token-weighted prediction
from the four sources (**10.5%** predicted, 10.7 / 11.1 measured) — a useful consistency check
on the whole measurement.

⇒ **Roughly a fifth of swezero's and the v3 arms' entire SFT gradient teaches the model to
write reasoning prose into a tool that does nothing.** The campaign *does* train on reasoning
traces; they are simply packaged as tool calls rather than native CoT. `scale` is the sole
exception: **0%** reasoning text, all actions.

**(ii) The tool's return value — `"Your thought has been logged."`. Produced by the
SCAFFOLD. NOT supervised.**

It is a constant emitted by the tool, never by the model. In training it arrives as
`role: tool` and is rendered into the observation slot, which is **masked**: of the
`"Your thought has been logged."` token spans found (`[7525, 3272, 682, 978, 13332, 13]`),
**0 tokens carry loss in any subset** — swezero, coderforge, rebench, scale, pooled55k,
pooled220k alike.

So the model is supervised to *write* thoughts, never to *predict the acknowledgement* —
which is correct behaviour for an observation, and worth stating only because the two texts
are easy to conflate.

### 3.4c The eval-side analog: how much of what each model GENERATES is `think` prose

Same question, other end of the pipe. For every `ActionEvent` in all 3,500 rollouts I
reconstructed the qwen3_coder XML the model emitted (`<function=NAME>` + each
`<parameter=…>` in generation order + `</function>`) and tokenized it with the same
tokenizer. Two denominators, because neither alone is airtight:

* **usage** — `metrics.accumulated_token_usage.completion_tokens`, vLLM's exact count of
  every token generated. Includes output that never became an `ActionEvent` (malformed
  calls, retries), so `think` share against it is a **lower bound**.
* **recon** — the sum of everything reconstructed (all tool calls + message content).
  Self-consistent with the numerator. Recovers 81–87% of `usage`; the gap is the
  `<tool_call>` wrapper, whitespace, EOS, and discarded generations.

| model | score | think calls | think tokens | completion tokens | recon tokens | **% of recon** | % of usage | think tok/inst |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| base | 119 | 6,676 | 2,594,002 | 16,402,192 | 13,320,103 | **19.5%** | 15.8% | 5,188 |
| swezero | 77 | 1,330 | 590,220 | 5,406,525 | 4,588,391 | **12.9%** | 10.9% | 1,180 |
| rebench | 70 | 1,435 | 635,114 | 19,736,006 | 17,193,976 | **3.7%** | 3.2% | 1,270 |
| v3_tokmatch | 63 | 1,233 | 566,805 | 5,135,969 | 3,571,049 | **15.9%** | 11.0% | 1,134 |
| v3_maxpool | 62 | 1,246 | 539,182 | 4,866,882 | 3,443,983 | **15.7%** | 11.1% | 1,078 |
| coderforge | 48 | 1,820 | 717,836 | 15,499,780 | 13,316,313 | **5.4%** | 4.6% | 1,436 |
| scale | 35 | 0 | 0 | 14,555,603 | 12,036,906 | **0.0%** | 0.0% | 0 |

#### The result: `think` transmits gradedly from data to behaviour — and is score-blind

| model | **train** think-share | **eval** think-share |
| --- | --- | --- |
| v3_tokmatch | 22.3% | 15.9% |
| v3_maxpool | 21.5% | 15.7% |
| swezero | 20.5% | 12.9% |
| coderforge | 13.3% | 5.4% |
| rebench | 8.5% | 3.7% |
| scale | 0.0% | 0.0% (exact) |

**Lead with the effect sizes, not the rank test.** Think-heavy sources produce think-heavy
models (22.3 → 15.9, 20.5 → 12.9) and think-light sources produce think-light models
(8.5 → 3.7); `scale`'s 0 → 0 is exact. Top-to-bottom spread across the non-zero arms is
**2.6× in the data (22.3 / 8.5) and 4.3× at eval (15.9 / 3.7)**. That graded correspondence
is what carries the conclusion. (The spread being *wider* at eval than in the data is not
interpreted here — see the denominator caveat below.)

**The rank test is weaker than it looks, and should not be the headline.** Spearman = +1.00
over the six arms is `p = 2/n! ≈ 0.003` only if the six are independent, and they are not:

| treatment of the points | n | p |
| --- | --- | --- |
| all six arms as independent | 6 | 0.003 |
| collapse the swezero lineage (swezero / tokmatch / maxpool = one source) | 4 | **0.083** |
| also drop `scale` (0 → 0 is structurally forced, not an informative ordering) | 3 | **0.33** |

⇒ the defensible statement is "**3–4 independent sources ordered correctly, n.s.**"
(*credit: devils-advocate, PR #7 review — this correction is theirs.*)

**What the result does establish.** Its job is to refute *"the arms never learned the training
data"* as an explanation for base ≫ arms, and the graded effect sizes do that without any
p-value. The arms learned this behaviour, in proportion to how much of it they were fed. It
simply did not help:

* Spearman(score, eval think-share) = **+0.61** with base (n=7), **+0.37** without (n=6,
  p ≈ 0.47, n.s.). Base drives it; among arms it is noise.
* **The one contrast where data lineage is held fixed points the other way.** Within the
  swezero lineage — same source corpus, differing only in curation — more think prose goes
  with a *lower* score: swezero 12.9% → **77**, v3_tokmatch 15.9% → **63**, v3_maxpool
  15.7% → **62**. n = 3 and n.s., but it is the only controlled slice available, and it is
  cleaner evidence for "not the lever" than the base-inclusive +0.61.

Base is still the striking row: never trained by us on ADP data, it emits the **largest**
share of think prose of any model (19.5%) and 5,188 think tokens per instance against
~1,100–1,400 for every arm.

**Do not read the eval/train ratio as "attenuation."** The two shares have incomparable
denominators — training share is over the training corpus's action mix, eval share is over
generated tokens on a *different* task distribution that demands a different non-think action
profile. A perfectly faithful model would still show a ratio ≠ 1. The **ranking** is
meaningful; the magnitude of the ratio does not isolate a behavioural change from a
denominator effect, so no "under-expression" or "level shift" claim is made here.

**Score board used.** The rank ordering above is unchanged under main-worker's
harness-corrected board (`*_nodeb`: rebench 70 → 76, coderforge 48 → 50, scale 35 → 36) —
77 > 76 > 63 > 62 > 50 > 36 is the same ordering as 77 > 70 > 63 > 62 > 48 > 35, so every
Spearman-vs-score figure here is identical on either board.

#### Is this general? No — and the bound is the more interesting finding

Same measurement over all tools. Training figures use the same contiguous cross-shard
sampling (n ≈ 390/subset); eval figures are the reconstruction above (% of generated tokens):

| model | train `file_editor` / `terminal` / `think` | eval `file_editor` / `terminal` / `think` |
| --- | --- | --- |
| coderforge | 60.3 / 12.5 / 13.2 | 67.5 / 25.7 / 5.4 |
| rebench | 60.0 / 16.1 / 8.5 | 71.2 / 23.4 / 3.7 |
| swezero | 46.9 / 13.7 / 21.0 | 62.0 / 21.1 / 12.9 |
| scale | 59.2 / 24.9 / **0.0** | 58.7 / **40.7** / **0.0** |
| v3_tokmatch | 52.1 / 14.1 / 22.8 | 57.7 / 21.0 / 15.9 |
| v3_maxpool | 52.2 / 15.3 / 21.3 | 55.9 / 23.0 / 15.7 |
| **base** | *(not trained by us)* | **16.6 / 45.8 / 19.5** (+ **16.2%** plain message content) |

Rank fidelity by tool: `think` **+1.00**, `file_editor` **+0.54**, `terminal` **+0.37**
(all n=6, same independence caveat as above).

**Transmission fidelity falls as the behaviour becomes more environment-coupled.** So the
claim is *not* "the pipeline faithfully transmits data behaviour" in general — it is
**`think` transmits with high fidelity; environment-coupled behaviours transmit only
partially**. That bound is the richer result: **the behaviour that transmits best is the one
that is score-irrelevant, and the behaviours plausibly relevant to score — shell use,
verification depth — are the ones that transmit worst.** Consistent with "the arms learned
the easy-to-imitate surface and not the hard-to-imitate policy." It still does the one job
the positive control needs to do: kill "SFT didn't take."
*(Scoping and framing: devils-advocate, PR #7 review.)*

**The largest behavioural split in the table is not `think` at all — it is base vs everything
else.** Base spends **45.8%** of its output driving the `terminal` and only **16.6%** on the
structured `file_editor`; every arm inverts this (**56–71%** `file_editor`, 21–26% `terminal`).
And base spends **16.2%** of its generated tokens on ordinary message content — talking — while
every SFT arm rounds to **0.0%** (< 0.05%). SFT on ADP trajectories appears to eliminate the
natural-language channel entirely and move the model from a shell-driven to an editor-driven
workflow.

**But this is the verification finding re-encoded, not a new axis** — shell use is where
exploration and testing happen, `file_editor` is "just edit" — so it belongs with
`adp_v3_a1_preregistration.md` §17 rather than being presented as a separate discontinuity.

**⚠️ The naive lever is already refuted by the same table.** Across arms, terminal share does
**not** predict score: `scale` has the *highest* arm terminal share (**40.7%**) and is the
*worst* arm (**35**). So the reading is not "use the shell more" — it is "use it
productively," which is the verification-quality point, not a new one. State the scale
counterexample beside the observation so this cannot be mis-quoted as a lever.
It remains **n=1 on the base side** — a hypothesis worth a targeted test, not a mechanism.
*(Both points: devils-advocate, PR #7 review.)*

### 3.4e What the 16.2% actually is — base's narration channel, and why no arm has one

The 16.2% is the **event-level `thought` field on `ActionEvent`s**: natural-language prose the
model emits *alongside* a tool call, in the same completion. It is a different field from
`action.thought` (the `think` tool's argument, §3.4b) and from `reasoning_content` (native CoT,
which is zero everywhere, §1.3). vLLM's parser splits one generated turn into leading content +
the tool call; this is that leading content.

**Verbatim, base** — the whole range, unedited:

```
django__django-11211   [with a think call]
  "I'll help you implement the necessary changes to fix the issue with prefetching related
   objects when using GFK for models that use UUID fields as primary keys. Let me start by
   understanding the problem and exploring the repository.
   ## Phase 1: READING - Understanding the Problem"

sympy__sympy-13877     [with a terminal call]
  "Now I can see the problem clearly. The issue is that sympy is returning `nan` for f(5),
   and then when trying to compute f(6), it fails with `TypeError: Invalid NaN comparison`…
   Now let me look at the code in `exprtools.py` to fix the issue."

sphinx-doc__sphinx-8120 [with a terminal call]
  "BINGO! There's the bug! Look at line 296 in application.py: …
   So Sphinx's **embedded locale files come before the user's locale files**. When gettext
   merges translations, it uses the FIRST translation as the primary…"

django__django-15572   [with a terminal call]
  "Interesting! The reproduce script passes. This means either: 1. The fix has already been
   applied  2. The issue is in a specific scenario that's not being tested"
```

It is running commentary: interpret the last observation, state the next step. Short — median
**93 characters**.

**It is essentially unique to base.** Counting every `ActionEvent` in all 3,500 rollouts
(base's row is over its 300 transcript-bearing runs — §0b; every other row is over ~500):

| model | ActionEvents | with prose | % | prose tokens | median chars |
| --- | --- | --- | --- | --- | --- |
| **base** | 85,726 | **41,311** | **48.2%** | **2,164,355** | 93 |
| scale | 122,092 | 491 | 0.4% | 61,858 | 148 |
| v3_tokmatch | 36,257 | 5 | 0.0% | 1,491 | 451 |
| rebench | 116,201 | 3 | 0.0% | 109 | 139 |
| coderforge | 110,908 | 3 | 0.0% | 362 | 166 |
| v3_maxpool | 36,811 | 2 | 0.0% | 74 | 354 |
| **swezero** | 45,920 | **0** | **0.0%** | **0** | — |

Not "rounds to zero" — swezero is **exactly zero across 45,920 action events**, and three of
the other arms are in single digits. Base attaches prose to roughly **half** of everything it
does, most heavily to shell commands (1,512,204 tokens over 29,007 `terminal` calls).

*(Standalone assistant `MessageEvent`s are separate and **not** in the 16.2%: base 1,166 events
/ 116,876 tokens; every arm 2–6 events except scale's 115. Including them moves base to ~17.1%.)*

**Why the arms have no narration: the training data never supervises any.** Measuring the
training analog — loss-bearing tokens between `<|im_start|>assistant\n` and that turn's first
`<tool_call>` (n ≈ 390 records/subset):

| subset | assistant turns | turns with prose | prose loss tokens | % of all supervision |
| --- | --- | --- | --- | --- |
| scale | 6,554 | 300 | 76,655 | 8.94% |
| swezero | 5,353 | 160 | 32,000 | 4.27% |
| coderforge | 5,304 | 169 | 33,489 | 3.49% |
| rebench | 5,176 | 173 | 34,600 | 3.41% |
| **v3_tokmatch** | 8,768 | **0** | **0** | **0.00%** |
| **v3_maxpool** | 9,228 | **0** | **0** | **0.00%** |

**And every sample inspected is a condensation summary, not narration** — `USER_CONTEXT:` /
`TASK_TRACKING:` / `## Context-Aware State Summary`, e.g.:

~~~
"\n\n## Context-Aware State Summary\n\n**USER_CONTEXT:** Fix inverted distance calculation
 logic in DeepDiff when using `ignore_order=True` with custom `iterable_compare_func`…"
"\n\n```yaml\nUSER_CONTEXT: Fix thread-local request propagation issue in Pyramid's
 Configurator with autocommit=True…"
~~~

So the ADP corpus supervises assistant prose **only** as periodic state summaries, and **never**
as commentary attached to a tool call. The lesson SFT teaches is *emit a tool call and nothing
else*, and the arms learn it near-perfectly. Two corroborations: the v3 arms, built by removing
condensation records, measure **exactly 0.00%** — an independent check on that arm construction
— and `scale`, with the most assistant prose in training (8.94%), is the only arm that emits any
at eval (491 events).

**What this does and does not mean.** Base devotes ~**35.7%** of its generated tokens to natural
language (19.5% `think` + 16.2% narration); the most verbose arm reaches 15.9% and `scale` 0.5%.
That is the widest base-vs-arms behavioural gap measured in this campaign. It is **not** shown to
be causal: it is n=1 on the base side, base differs from the arms in every other way too, and
the prose-share-vs-score correlation is the same weak base-driven +0.61 / +0.37 as `think`
(§3.4c) because the ordering is nearly identical. Treat it as the sharpest available description
of *what SFT on ADP data does to the policy* — it removes the natural-language channel entirely
— not as a lever.

### 3.4f Is base's narration load-bearing CoT? — evidence, and how to block it

Asked by dpf: *could the 16.2% narration channel be effectively a CoT that is helping the
base model, and can we block it at inference time to see if performance degrades?*

**Mechanically it is CoT by construction.** The narration and the tool call are one
completion: the model emits the prose first, then `<tool_call>`. Every action token is
therefore conditioned on the narration, which is exactly what "chain of thought" means. The
open question is not whether the action is conditioned on it — it is — but whether that
conditioning does any *work*.

**Correlational evidence, within base's 300 transcript-bearing runs (§0b).** Narration rate
= fraction of a run's `ActionEvent`s carrying non-empty event-level `thought`. Resolve rate
rises monotonically across quartiles, and — the confound worth killing first — it is **not a
run-length artifact**: it holds inside both short and long runs.

| stratum (median split: 246 actions, 0.475 narration rate) | n | mean rate | med actions | resolved | rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| short runs / low narration | 82 | 0.364 | 138 | 13 | **15.9%** |
| short runs / high narration | 68 | 0.584 | 124 | 25 | **36.8%** |
| long runs / low narration | 68 | 0.409 | 428 | 20 | **29.4%** |
| long runs / high narration | 82 | 0.557 | 367 | 35 | **42.7%** |

| quartile by narration rate | n | mean rate | med actions | resolved | rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| Q1 least | 75 | 0.324 | 191 | 17 | 22.7% |
| Q2 | 75 | 0.444 | 281 | 16 | 21.3% |
| Q3 | 75 | 0.514 | 292 | 23 | 30.7% |
| Q4 most | 75 | 0.624 | 201 | 37 | **49.3%** |

**This is not evidence of a causal effect** and should not be quoted as one. The obvious
reverse story fits every number equally well: a model that has localised the bug has
something to say and says it, so narration is a *symptom* of a run going well rather than a
cause. Instance difficulty is uncontrolled. And the whole table lives inside the biased 300
(§0b), which excludes base's longest runs. Its actual value is narrow but real: it rules out
"narration just tracks run length", and it establishes that the sign is **positive**, so a
blocking experiment has something to detect.

#### The blocking experiment — built and validated offline, not launched

**Method: prefill `<tool_call>\n` into the assistant generation prompt.** Then generation
begins *inside* an open tool call and the model physically cannot emit prose first.

Both halves are verified on disk, without GPUs:

* **Template.** `/checkpoint/dpf/swebench-eval/prompts_adp/prefill_toolcall.jinja` — a
  one-line delta from the stock Qwen3.5 template (same shape as main-worker's `nostub.jinja`,
  so it composes with the parity ladder). Rendering check: output is byte-identical to the
  stock prompt plus exactly `<tool_call>\n`, i.e. **+2 tokens** `[248058, 198]`.
* **Parser.** vLLM 0.25.1's `qwen3_coder` parser resolves to `Qwen3EngineToolParser`, whose
  state machine (`vllm/parser/qwen3.py`) carries an explicit
  `# Fallback: <function= without a preceding <tool_call>` transition
  `(CONTENT, FUNC_PREFIX) → TOOL_NAME`. Fed a completion that *starts* at `<function=` — what
  the model will produce once `<tool_call>\n` is prefilled — it returns a clean tool call
  with `content=None`. Fed the narrated form, it returns `content='Now let me look at the
  files.'`. **That `content` field is precisely the 16.2% channel**, so the intervention
  removes exactly the thing measured in §3.4e and nothing else.

**What it isolates.** It blocks prose *before* the tool call. It does **not** touch the
`think` tool, which is itself a tool call and still fires normally — so this cleanly
separates the 16.2% narration channel from the 19.5% `think` channel. It also does not block
prose emitted *after* `</tool_call>`; §3.4e shows base essentially never does that, but the
blocked run should be re-measured rather than assumed.

**Matched control already on disk.** Cell **E** of main-worker's `run_parity_ladder.sbatch`
(base, stock template, harness `default.j2`, select shards `00`/`01`, n=100) is the exact
comparator. The blocking cell (**G**) is the identical invocation with
`CHAT_TEMPLATE=$P/prefill_toolcall.jinja` — one flag, one axis, same instance IDs. Cost: 2
jobs × 1 A100. Running it *after* E lands is strictly better than running it standalone
against the 500-instance board, which uses different IDs.

**G must be scored attempt-1 only, against E's attempt 1.** The harness retries whichever
instances its critic rejects, and attempts ≥2 sample at temperature 0.1 — under identical
config it rejected base's first attempt 71/100 in cell F but only 15/100 in cell E, so *two
cells of the same model are not compute-matched by default*, let alone two different prompts.
E's `output.critic_attempt_1.jsonl` is already complete and frozen (100/100 unique, exact
select coverage), so the control does not depend on E's remaining attempts finishing.
See [`ANALYSIS_HOUSE_RULES.md`](ANALYSIS_HOUSE_RULES.md) rules 1–2.

**Statistical power — computed before launch, exact conditional McNemar, α=0.05 two-sided,
paired on identical instances, base rate taken as 24%.** The relevant noise floor is the
~3%-of-instances SBV label-flip rate (generation *and* grading both flip labels), which enters
as symmetric discordance and costs ~55% of the sensitivity:

| n (instances) | MDE at 80% power, noiseless | **MDE at 80% power, 3% flip floor** |
| --- | ---: | ---: |
| **100** (shards 00+01 — what G buys) | 7.8 pp | **12.1 pp — i.e. 24% → 11.9%, a halving** |
| 200 | 3.9 pp | 7.5 pp (31% relative) |
| 500 (full board) | 1.6 pp | 4.1 pp (17% relative) |

Power at n=100 with the floor: halving **0.79**, 38% relative drop **0.58**, 25% relative
**0.31**, 12% relative **0.09**.

⇒ **G at n=100 is powered to detect a halving and nothing finer.** That is the right first
question — narration is 16.2% of base's output and its resolve rate climbs 22.7% → 49.3% across
narration quartiles, so the load-bearing-CoT hypothesis predicts a large effect — but the null
must be pre-registered as **"no *large* effect,"** never as "no effect." Resolving a 25%
relative drop needs all ten shards, i.e. 5× the GPU.

**The behavioural outcome sits in a completely different power regime, and it is the half that
cannot come back ambiguous.** "Did the block actually work" is measured over ~20k action events
(48.2% of base's currently carry prose), so it is decisive regardless of what the score does.
G is well powered for *did the intervention fire* and only large-effect-powered for *did it
matter* — report the two separately and do not let the first stand in for the second.

**Second, cheaper axis, needing no template at all:** drop `think` from the offered tool
list, which removes the other 19.5%. The two together give a 2×2 of
{narration on/off} × {think on/off} whose control cell already exists.

**Pre-registered readings.** If blocking narration costs base a material share of its resolve
rate, the natural-language channel is load-bearing, and the arms' ~0% narration (§3.4e) is a
genuine deficit rather than a stylistic one — that would be the first mechanism in this
campaign to survive contact with an intervention. If it costs nothing, the 16.2% is
decorative and the base ≫ arms gap is elsewhere.

**Known risks, to be reported as secondary outcomes rather than discovered afterwards.**
(a) Forcing a tool call every turn is itself a distribution shift, so a drop is not
automatically evidence that the *reasoning* mattered. (b) The model may **relocate** the
narration into the `think` tool or into a parameter string — directly measurable, and the
first thing to check. (c) Base hits the 500-iteration cap on **35.0%** of instances in the
matched control cell E (§0b.1), and the stub-vs-nostub split shows this rate is highly
sensitive to the prompt; prefilling `<tool_call>\n` changes tokens-per-iteration, so **cap-hit
rate must be reported alongside resolve rate** or the two effects are confounded. Note the
direction is not obvious a priori: suppressing prose could *reduce* tokens per turn and so
*raise* the number of turns reached before the cap.

**Status: not launched at time of writing.** The parity-ladder arrays held the GPUs and the
matched control was inside them.

### 3.4d Method notes and caveats — read before quoting these numbers

* **Sampling.** Training shares are from ~790 records per subset (~1.4% of 55k), contiguous
  windows spread across all shards. Not the full corpus.
* **Stride-sampling is wrong here, and I hit it.** The pooled sets interleave the four
  sources **round-robin** (`generate_arm_runs.py`: "the 4 arms' training records
  (round-robin)"). A first attempt strided by `shard_rows // per` = 92 rows; 92 ≡ 0 (mod 4),
  so every sampled record came from the **same source**, and pooled55k mis-measured as
  **13.7%** instead of 11.1%. Contiguous windows fix it. Any future per-record sampling of
  the pooled caches must avoid strides divisible by 4.
* **Prefix identity is real, not a bug.** pooled55k is prefix-identical to pooled220k, and
  v3_tokmatch to v3_maxpool (shard-0 row hashes at indices 0/599/1500 match exactly), because
  each is a subsample of the other. That is why a shard-0-head sample returned byte-identical
  figures for those pairs — the pairs genuinely share those records.
* **Tool-name detection.** Matching the literal `<function=` token prefix silently **misses**
  tools whose name merges with the `=` under BPE — it reported `file_editor` as absent, which
  is false (it is the single largest category). Match `<function` (`[27, 1628]`) and decode
  the name. The `think` figures were never affected: they used the exact full sequence
  `<function=think>`.
* ⚠️ **"3,500 rollouts" means 3,500 records, of which 3,291 carry a transcript.** The 209
  blanks are 200 base + 9 spread over rebench/coderforge/scale (§0b). Wherever this report
  says base's numbers are over 500 rollouts, the transcript-derived ones are over **300**,
  and the excluded 200 are base's longest runs. Arm figures are unaffected.
* **Base's 19.5% is a lower bound, and the bias is inert.** If base's 833 malformed `think`
  calls sit in the 13–19% of `usage` that reconstruction does not recover, the omission pushes
  base's think share *up*. Base is already the maximum, is excluded from the arms-only
  Spearman, and is held out of the score-blindness claim — so no conditional result moves.
* **Sampling validation is two-method agreement, not sample size.** The load-bearing check is
  that the pooled share is *predicted from its four sources to within 0.4pp* (10.5% predicted
  vs 10.7 / 11.1 measured), plus two independent draws agreeing to ≤0.5pp — not the ~1.4%
  coverage per se.
* **Eval reconstruction is approximate.** Parameter order is taken from the arguments JSON and
  assumed to be generation order; the `<tool_call>` wrapper is excluded. This is why both
  denominators are reported — the conclusions are identical under either.
* **`think` counts must be structural.** Count `role == "function_call"` / `tool_call.name`;
  the `<function=think>` XML is generated at tokenization, so grepping raw jsonl returns 0.

### 3.5 What the numbers do and don't support

1. **`scale` 0 → 0 is an exact behavioural transfer.** The only subset with zero `think`
   calls in training produced the only model that never calls `think` at eval — and it is
   the worst arm (35/500). But **n = 1 and confounded**: scale also emits tool names the
   scaffold does not offer (`execute_bash` ×451, `str_replace_editor` ×53), so it differs
   in more ways than this one.
2. **`think` is a smaller fraction of the arms' eval calls than of their training calls**
   — swezero 7.1% → 2.9%
   of calls, coderforge 6.5% → 1.6%, rebench 4.3% → 1.2% — and changes its *shape*: base
   concentrates thinking (58% of instances, 13.4 calls each) while the arms sprinkle it
   (~98% of instances, ~2.5 calls each). This is the same signature as the lost verify loop
   documented in `adp_v3_a1_preregistration.md` §17. The *shape* change is a real finding;
   the *rate* change is subject to the same denominator caveat as the token shares (eval and
   training call mixes are over different task distributions), so read it as descriptive. **Refined by §3.4c:** the token-share ordering is
   preserved without inversion, though the eval/train *ratio* is not interpretable as
   attenuation (incomparable denominators).
3. **The behaviour transmits; it just isn't the lever.** §3.4c is a positive control **for
   `think` specifically** — think-heavy data yields think-heavy models in graded proportion,
   so "the arms didn't learn the data" cannot explain base ≫ arms. Yet the two arms that
   generate the *most* reasoning prose score 63 and 62, and within the swezero lineage —
   the only slice where data lineage is controlled — more think prose goes with a *lower*
   score. Transmission fidelity is *worst* for the environment-coupled tools that might
   actually matter.
4. **Spearman(score, think-calls/instance) = +0.57** across the 7 models — the first
   behavioural axis in this campaign with a *positive* sign (depth was −0.32, verification
   −0.32 / −0.43). **But base drives it entirely**: excluding base it falls to **+0.31**,
   n = 6, nowhere near significance. Base beats the arms on nearly every axis, so any
   base-inclusive correlation is weak evidence. **Not carried as a mechanism.** The
   token-share version of the same test agrees: **+0.61** with base, **+0.37** without.
5. **The biggest split in the data is elsewhere.** Base spends 45.8% of its output on
   `terminal` and 16.2% on plain message content; every arm is `file_editor`-dominant
   (56–71%) with ~0% message content (§3.4c). Untested, n = 1 on the base side, but it is
   a larger and cleaner base-vs-arms discontinuity than anything `think`-related.

---

## 4. Answers to the three questions as asked

**"Are thinking traces used in evaluation, and/or in training?"**
Native CoT: **neither** — 0 reasoning tokens across 81.6M completion tokens, 0/3,500
rollouts with reasoning content, and 0 `<think>` in the trained tokens. The ADP `think`
tool: **both** — loss-bearing in training, live at eval.

**"What happens when the think tool is called at eval time?"**
It is a no-op logging tool. The scaffold returns `ThinkObservation` with the fixed text
`"Your thought has been logged."` and `is_error: false`. Nothing in the environment changes;
the agent spends one round-trip and gains no information. If the `thought` argument is
missing it returns an `AgentErrorEvent` instead (base: 833 times).

**"What produces the text inside `<think>`?"**
For the eval-time `<think>\n\n</think>` stub: **nothing produces it — and nothing ever
will.** It is 4 literal tokens emitted by the Jinja chat template into the *prompt*;
generation starts after the closing tag, so it is empty by construction. The `think` tool's
text is not inside `<think>` tags at all — it lives in
`<tool_call><function=think><parameter=thought>…`, generated by the model as an ordinary
tool call. Spontaneous `<think>` emission by the models is 6 events in 3,500 rollouts.

**"Was the instruct model — the unbeatable one — using `<think>` and/or thinking tools?"**
- Native `<think>`: **no.** Base ran with the identical config as every arm
  (`reasoning_effort: none`, `enable_thinking: false`, verified across all 10 shards),
  `reasoning_parser=''`, **0** reasoning tokens over 16.4M completion tokens, and only 5
  stray `<think>` strings in 500 rollouts. Base's advantage is **not** hidden CoT.
- The `think` **tool**: **yes, heavily** — 6,676 calls, ~13.4 per instance, roughly **5×
  any arm** — though concentrated in 58% of instances rather than spread across all of them.
  This is the one behavioural axis where base leads in the "more is better" direction, but
  see §3.5(3) before treating it as causal.

---

## 5. Provenance

Measured from: `src/llamafactory/data/template.py` (:407, :2151, :2167) in
`/checkpoint/dpf/adp-env/LLaMA-Factory`; the tokenized caches under
`/checkpoint/dpf/adp-data/{v2_swe_subsets,v3_curated}/*/tokenized_*`; raw
`train.llamafactory.jsonl` per subset; `chat_template.jinja` in the Qwen3.5-4B snapshot
`851bf6e8…` and in each arm's `output/`; `llm_config.json` and vLLM logs under
`/checkpoint/dpf/swebench-eval/{runs,logs}`; and all rollout `output.jsonl` for the 7 scored
models. Scripts: `$CLAUDE_JOB_DIR/tmp/v3/{think_eval,think_train2,think_mech,think_fail,report_evidence,train_verbatim}.py`.
Companion: `adp_v3_a1_preregistration.md` §19.

§0b and §3.4f additionally read the **graded** artifacts, which are the source of truth for
what was scored and are *not* the same files as `out_*/combined/output.jsonl`:
`runs/score_<tag>/shard_*of16.jsonl` (rollouts as graded) and `shard_*of16.swebench.jsonl`
(the predictions themselves), plus `merged.report.json`. The defect surfaced as a
contradiction between them — 26 instances in base's `resolved_ids` had a zero-length patch
and zero actions in the combined file, while their graded prediction was a 6kB diff. Scripts:
`{hist_cov,attempts,resolved_x_hist,shard_probe,errrec,cap_audit,cap_cfg,narr_strat,
parser_prefill,mk_prefill}.py` in the same directory. The blocking template is
`/checkpoint/dpf/swebench-eval/prompts_adp/prefill_toolcall.jinja` (validated, unreferenced
by any submitted job).
