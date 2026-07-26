# Investigation handoff: provenance of the "base Qwen3.5-4B ≈ 25/500 (5%)" SWE-bench anchor

**For:** an agent running in the Babel environment (where the original ADP SWE-bench evals were run).
**Please:** investigate, then report findings back to dpf (this is a fact-finding task, not a code change).

---

## The ask (one sentence)
Track down where the baseline anchor **"base Qwen3.5-4B ≈ 25/500 (≈5%)"** (and **"paper-nonweb ≈ 52/500 ≈ 10.4%"**)
came from — was it a real in-house SWE-bench eval, and if so **on which model, harness, and split** — because a
fresh rerun measured a very different number and it inverts the campaign's headline.

## Why this matters
The adp-v2 campaign trained 4 SFT "arms" from a starting model θ₀ and reported an "SFT lift" of θ₀ ≈5% → arms
≈14–15%. That lift was computed against the **≈5% anchor above**, which was carried in as a cross-campaign
"sanity anchor," not re-measured. A fresh A100 rerun measured the SFT starting point **θ₀ = Qwen3.5-4B *instruct*,
single-run pass@1, SWE-bench Verified** at **≥70/500 (≥14%)** — roughly **3× the 5% anchor**. If ≥14% is the right
baseline, the "SFT lift" **inverts**: the arms (77/70/48/35 on the same 500) no longer beat base, and two are
below it. So the exact provenance and comparability of the 5% number is now load-bearing.

## Leading hypothesis (please test this FIRST — it's the cheapest resolution)
**Base-vs-instruct model mismatch.** The anchor literally says "***base*** Qwen3.5-4B." Two distinct models exist:
`Qwen3.5-4B-Base` (pretrained, non-instruct) and `Qwen3.5-4B` (instruct). The arms were fine-tuned **from the
instruct model**, so the correct θ₀ baseline is *instruct*. If the 25/500 (5%) was measured on
**Qwen3.5-4B-Base** (non-instruct — which plausibly does score ~5% agentically), then **both numbers are real and
correct** — they're just *different models*. The campaign would have mistakenly used the *base* model's 5% where
it should have used the *instruct* model's ~14%. That makes the "SFT-lift" a model-mismatch artifact, not a
fabrication, and it cleanly explains the 5% vs 14% gap.

## What to search for on Babel
1. Any SWE-bench eval (**Verified / Lite / full**) of Qwen3.5-4B (base **or** instruct) used as an *untrained
   baseline*: result JSON/reports, run logs, eval output dirs, wandb runs, launch/eval scripts, notebooks.
2. For each baseline eval found, record precisely:
   - **Which model** — `Qwen3.5-4B-Base` vs `Qwen3.5-4B` (instruct). **This is the crux.**
   - **Exact resolved count + denominator** — is "25/500" real? is "52/500" real?
   - **Harness / scaffold** — OpenHands (or other) version, `max_iterations`, temperature, single-sample vs
     pass@k, max input/output tokens, the tool/patch parser.
   - **Which SWE-bench split** — Verified (500) vs Lite vs full.
3. Provenance of **"paper-nonweb ≈ 52/500 (10.4%)"** — is it from the ADP paper (which table/setting), or an
   in-house Babel run?

## Hypotheses to distinguish (report which one holds, with evidence)
- **H1 — model mismatch (leading):** 5% was `Qwen3.5-4B-Base` (non-instruct); the instruct θ₀ was never compared.
- **H2 — harness mismatch:** 5% used a weaker/different scaffold, sampling, or split than the arms' eval → not
  comparable to the arms' or the rerun's numbers.
- **H3 — paper number:** 5% ≈ a published baseline, never run in-house.
- **H4 — real & comparable:** 5% was `Qwen3.5-4B` *instruct* on the same harness/split → then the rerun's ≥70/500
  is the anomaly and there is a genuine harness discrepancy to reconcile (diff the two configs).

## What to report back to dpf
For each baseline number (the 5% and the 10.4%): **model · resolved/denominator · harness/scaffold · split ·
file path**, and a one-line verdict on which hypothesis (H1–H4) holds. That tells us whether the campaign's
"SFT-lift" was a model/harness-mismatch artifact or a real (now-inverted) comparison.

---

## Reference: the A100 rerun numbers (for comparison — you do NOT need any FAIR paths)
- **SFT arms, single-run pass@1, SWE-bench Verified 500:** swezero 77, rebench 70, coderforge 48, scale 35.
- **θ₀ = Qwen3.5-4B *instruct*, single-run pass@1:** **≥70/500 certain**, bound [70,145] (clean number pending).
  Scaffold byte-identical to the arms (temp 0, `max_iterations` 500, max_input 28000, max_output 2047, same
  condenser + parser). So the rerun's arms-vs-θ₀ comparison is apples-to-apples *within the rerun*.
- **Oracle union of the 4 arms = 131/500.**
- **Caveat:** the SWE-bench `sphinx-doc/sphinx` repo (44 instances) is mis-scored 0/44 by the official pytest
  parser (sphinx's `tox` output has no per-test `PASSED` markers, so genuine passes are recorded as failures) —
  when comparing absolute rates, drop sphinx and use the 456-instance board.

_Created 2026-07-25 by the FAIR-side main-worker session for a Babel-side fact-finding pass._
