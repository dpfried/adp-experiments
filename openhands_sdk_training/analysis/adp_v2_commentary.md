# Commentary on `adp_v2_experiments_and_results.md`

Discussion of candidate mechanisms for the fine-tuning degradation.
For editorial review notes on the writeup itself, see `adp_v2_review_notes.md`.
(Claude commentary, 2026-08-02; from a review-and-discussion session with dpf.)

---

## 1. Why is performance degrading? Two mechanisms, not one

The report emphasizes a policy shift (depth → compliance, §2.7). That's real, but the
report's own evidence points at least as strongly at a second mechanism. The two are
distinguishable and have different fixes.

### 1.1 Policy shift: imitating the teacher's form without its competence

The §2.7 numbers (always terminates, zero empty patches, half the actions,
well-formed-but-misdiagnosed patches) show the model learned to *conclude* faster. The
mechanism the report doesn't quite name: the training data is teacher trajectories from
a 480B model. A 480B teacher can afford to wrap up in ~90 actions because its early
diagnosis is usually right. A 4B student trained to imitate that trajectory *shape*
learns the teacher's pacing and confidence without the competence that justified them —
it inherits when to stop investigating, but not the diagnostic accuracy that made
stopping safe. Distillation transfers the form of the behavior more readily than the
capability underneath it.

### 1.2 Capability erosion: full-parameter SFT overwriting, not augmenting

Two details in the data are hard to explain with a pure "stops too early" story:

- **The union-of-arms result on sympy.** The union of all four arms solves 13 sympy
  instances vs. the instruct model's 29 — including the arm trained on sympy data. If
  this were just early commitment, four independently-trained models with 52%-disjoint
  solve sets should occasionally land the right diagnosis anyway. Instead all four lost
  the same thing. Sympy is the most reasoning-heavy repository on the benchmark — deep
  symbolic math, not scaffold skill — exactly what you'd expect to degrade if
  full-parameter SFT (1e-5, full epoch, no replay data, no KL anchor) is partially
  overwriting the instruct model's general reasoning.
- **The §2.8 attractor pattern.** Base-initialized and instruct-initialized runs converge
  to the same 74–82 band. If fine-tuning added capability on top of the initialization,
  the instruct-initialized run should retain its ~40-point head start. It doesn't — the
  endpoint looks determined by the data, not the starting point. That's the signature of
  overwriting, not augmenting.

### 1.3 Off-policy imitation with no recovery data

The student trains on the teacher's states, so it never sees how to recover from *its
own* mistakes — the classic compounding-error problem. A twist that may partly explain
the verification surprise: execution-verified data is *selected for* clean, direct
successes, so it's systematically depleted of backtracking, re-diagnosis, and "my first
hypothesis was wrong" behavior. Verification filtering may select against exactly the
exploratory depth this benchmark rewards, while unverified `swezero` retains messier
trajectories. (Not the whole story — `rebench` is verified and ties `swezero` — but it
cuts against the intuition that verification should only help.)

### 1.4 Cheap discriminating experiments (policy shift vs. erosion)

1. **Raw reasoning outside the agent scaffold:** run a non-agentic math benchmark on the
   fine-tuned arm before vs. after fine-tuning. If sympy-relevant reasoning dropped there
   too, it's erosion; the curation ablation won't fix it, but replay mixtures, lower LR,
   or LoRA would.
2. **Force longer exploration at inference:** a system-prompt intervention ("reproduce
   the bug and verify your diagnosis before submitting"), or reject the first proposed
   patch and make it continue. If scores recover meaningfully, the capability is still in
   the weights and it's a policy problem; if not, it's gone.
3. **Per-repository action counts for the instruct model:** if its ~171-action average is
   disproportionately driven by sympy, that directly ties the lost repositories to the
   lost exploration depth.

**Bet:** the 42-point gap is mostly erosion (the head start getting washed out by
full-FT on narrow data), with the compliance/pacing shift determining *where* the
remaining capability gets spent. The highest-leverage fix is then regularization toward
the starting policy (replay mixture, KL penalty, lower LR, or parameter-efficient
tuning), with data curation as the second lever.

---

## 2. The role of partial observability

Partial observability isn't just a contributor — it's the reason the
"imitating form without competence" failure mode is so severe in this domain
specifically.

### 2.1 The core problem: the expert's key decisions condition on state the student can't see

An agentic SWE episode is partially observed — the repo's true state, the bug's actual
cause, and the hidden test expectations are only revealed through exploration. The
teacher's actions are conditioned on its *belief state*: everything it has inferred from
what it's read so far, filtered through 480B-scale competence. The trajectory log records
observations and actions, but not the belief. Behavior cloning therefore teaches the
mapping "observation history → action" when the true generative process was "observation
history → belief → action" — and the middle term is both private and capacity-dependent.
A 4B student reading the same files as a 480B teacher does not arrive at the same belief.

This is the known "imitation gap" / causal-confusion problem in imitation learning, and
it's worst for exactly one action type: *the decision to stop gathering information*.
The teacher stops exploring when its belief is sufficiently resolved; that's invisible in
the log, so the student learns superficial correlates instead — "after this many turns,
after writing a summary, experts conclude." That is the §2.7 behavioral signature: right
pacing, wrong diagnosis.

### 2.2 Success-filtering interacts badly with partial observability

Under partial observability, execution-verified trajectories are selected for episodes
where the teacher's *prior* was already good — where little belief-building was needed.
Verification filtering thus depletes demonstrations of the hard part (uncertainty
resolution, backtracking, re-diagnosis) and enriches the easy part (confident, direct
execution). Sharper version of the verification surprise: in a POMDP, filtering on
outcomes selects for low-information-need episodes, which are precisely the ones that
teach premature commitment.

### 2.3 Condensation turns are literal belief-state destruction

34–40% of training records are summarization turns — the trajectory replaces accumulated
observations with a lossy summary and proceeds from that. Training on this teaches the
model that a summary *is* a sufficient state to act from; the evidence-to-action links
get severed. The report's curation hypothesis and the POMDP framing converge here, but
the framing predicts *why* removing condensation turns should help: it restores the
visible chain from evidence to decision.

### 2.4 Compounding errors are strictly worse under partial observability

In a fully-observed environment, a student that takes a bad action still sees the true
state afterward and can recover. In a POMDP, your history *is* your state — if the
student's early exploration differs from the teacher's (and it will, off-policy), its
belief state diverges from anything in the training distribution, and its learned
stopping cue fires on schedule anyway, on a corrupted belief. Exploration errors don't
wash out; they permanently poison the episode.

This also reframes the instruct model's advantage: its "messy" 171-action style is
belief-state construction — it keeps probing until uncertainty is actually resolved, a
general epistemic policy plausibly instilled by its instruct/RLHF training. SFT replaced
that calibrated stopping rule with an imitated, schedule-based one.

### 2.5 Two cheap tests, runnable on the existing 82-instance audit set

1. **Evidence-coverage check:** for the instances `swezero` lost, did it ever *open* the
   files its patch modifies (or the files the gold patch touches) before editing them?
   If it routinely patches code it hasn't read, that's acting on a belief deficit, not a
   reasoning failure per se.
2. **Stopping-cue check:** is the fine-tuned model's decision to submit correlated with
   an information event (successful bug reproduction, reading the implicated file), or
   just with turn count? Prediction: for the instruct model, stopping correlates with
   evidence; for the fine-tuned arm, with schedule. If that holds, it's direct
   confirmation that fine-tuning replaced a belief-conditioned stopping rule with a
   superficial one.

### 2.6 Caveat

Partial observability explains the *policy* half of the degradation (premature
commitment) very well, but it doesn't by itself explain the erosion evidence (§1.2
above): the union-of-arms losing sympy and the §2.8 attractor still look like plain
catastrophic forgetting from full-parameter SFT. Keep both mechanisms in the story:
forgetting erodes what the model knows, and the POMDP-imitation gap corrupts how it
decides it knows enough.
