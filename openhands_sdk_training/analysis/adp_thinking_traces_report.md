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
(8.5 → 3.7); `scale`'s 0 → 0 is exact. The gap between the top and bottom non-zero arm is
**4.3× in the data and 4.3× at eval**. That graded correspondence is what carries the
conclusion.

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
