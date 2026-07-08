# RESULTS — Qwen 2.5-3B-Instruct, full methodology

**Kernel:** [jlens-full-replication-qwen](https://www.kaggle.com/code/amaljithkuttamath/jlens-full-replication-qwen)
**Wall time:** ~4h (fit reuse + all 4 tests including the 234-min shuffled-corpus lens fit)
**Fit config:** target_layer=-2, dim_batch=2, max_seq_len=96, skip_first=4, n=25 pile-10k prompts

**This file supersedes the previous pass@k-focused version.** The new methodology added three tests missing from the earlier run: MRR scoring (Anthropic's metric), causal-swap intervention (Neel Nanda's actual test), and shuffled-corpus control lens (basic falsification test I skipped originally). Those tests substantially deflate what we can claim.

## The four tests

### Test 1 — MRR probing (passive readout)

Anthropic and Neel Nanda both use MRR (mean reciprocal rank) rather than top-k thresholds. Recomputed:

| Eval | n | MRR | top-1 | top-10 |
|---|---|---|---|---|
| poetry | 98 | 0.332 | 0.225 | 0.551 |
| typo | 96 | 0.318 | 0.219 | 0.531 |
| multilingual | 107 | 0.261 | 0.186 | 0.401 |
| multihop | 84 | 0.243 | 0.155 | 0.423 |
| order-ops | 55 | 0.066 | 0.027 | 0.145 |
| association | 99 | 0.006 | 0.000 | 0.010 |

Raw: [`probing_results.json`](./results/full_rep/qwen/probing_results.json).

**On its own this looks like a partial workspace-signal replication** — poetry/typo/multilingual/multihop show clearly non-random MRR while order-ops and association are near floor. This is roughly the qualitative pattern Neel reported.

### Test 2 — Causal-swap intervention

For each item: extract the target token's unembedding direction, ablate it from residuals at mid-band layers (roughly middle third of the layer stack), measure Δ log-probability of the correct answer. **Negative Δlp = concept was causally used.** This is the actual test of whether the workspace matters.

| Eval | n scored | Mean Δlp | Fraction where ablation hurts (Δlp < -0.1) |
|---|---|---|---|
| multihop | 52 | **+0.015** | 0.423 |
| multilingual | 16 | **+0.049** | 0.250 |
| order-ops | 32 | **+0.068** | 0.281 |
| poetry, typo, association | 0 | (no items scoreable — target field missing/multi-token) | — |

Raw: [`causal_results.json`](./results/full_rep/qwen/causal_results.json).

**All Δlp values are positive or near zero.** The concept-direction ablation does not systematically hurt Qwen's correct-answer probability. If the workspace is causally used by the model, we'd expect strong negative Δlp. We get nothing.

Caveat: our concept-direction extraction uses the target token's unembedding row directly rather than a lens-derived intermediate-concept direction (which would require Anthropic's swap-intervention machinery). This is a weaker approximation. A negative result here is not proof of "no workspace" — but it is not evidence for one either.

### Test 3 — Prompt-truncation ablation (the passing test)

For each of 3 sanity probes, run the lens on the full prompt and on a truncated version cut before the referent.

| Probe | Full-prompt hit layers | Truncated hit layers |
|---|---|---|
| `The number of legs on the animal that spins webs is` → `spider` | 20 layers (L12–31) | **0** |
| `The capital of the country shaped like a boot is` → `italy` | 12 layers (L21–32) | **0** |
| `Fact: The currency used in the country whose flag has a red maple leaf is the` → `canada` | 4 layers | **0** |

Raw: [`truncation_results.json`](./results/full_rep/qwen/truncation_results.json).

**The concept appears in the readout only when the referent phrase is present.** This falsifies the null hypothesis "the lens just projects plausible next tokens regardless of context." Whatever the lens is picking up, it's genuinely responsive to prompt content.

This is a 3-probe test, not a full 100+-item eval. But directionally, it works.

### Test 4 — Shuffled-corpus control (the failing test)

Fit a second J-lens on the same 25 Pile prompts with token positions **randomly permuted within each prompt** (seed=42), so the prompts contain the same tokens in scrambled order. Fit took 234 min on T4. Re-score all 6 evals with this shuffled lens.

**If the workspace claim is real**, MRR should collapse toward zero on the shuffled lens because it was fit on non-semantic data.

**Actual result:**

| Eval | Real MRR | Shuffled MRR | Δ (positive = real > shuffled = good) |
|---|---|---|---|
| poetry | 0.332 | 0.337 | -0.006 |
| typo | 0.318 | **0.806** | **-0.488** |
| multilingual | 0.261 | 0.278 | -0.017 |
| multihop | 0.243 | 0.285 | -0.042 |
| order-ops | 0.066 | 0.074 | -0.009 |
| association | 0.006 | 0.015 | -0.009 |

Raw: [`shuffled_probing.json`](./results/full_rep/qwen/shuffled_probing.json).

**Every single eval is *at least as good* on the shuffled lens.** The most extreme case is `typo` where the shuffled lens achieves MRR=0.806 vs the real lens's 0.318 — the "workspace" scores 0.5 MRR *worse* on real coherent prompts than on scrambled input. This is not what a functional workspace measurement should look like.

## The verdict

- **Test 1 (probing) results are contaminated.** The MRR scores by themselves cannot distinguish workspace-formed representations from baseline residual-stream priors — the shuffled control demonstrates this directly
- **Test 2 (causal) shows no effect on Qwen.** Concept-direction ablation does not measurably hurt Qwen's answers
- **Test 3 (truncation) works.** Mid-stream concept emergence is genuinely prompt-content-dependent
- **Test 4 (shuffled control) fails.** The primary falsification test for the passive-probing method fails on our fitted lens

**What survives:** the lens reads out concepts only when the prompt semantically summons them. That's real, and it's consistent with the paper's story qualitatively. What does not survive: any interpretation of MRR numbers as *quantitative* evidence of workspace formation, and any interpretation of the passive readout patterns as *proof* of causal use.

## Possible reasons the shuffled control failed

None of these are established — they are candidate explanations that would each require follow-up work:

1. **Under-fitting.** n=25 vs paper's n=1000 may be too few prompts for the shuffled/real distinction to emerge in the Jacobian estimate. Neel's replication used n=25 too, but he did not run this specific control, so we don't know what he'd have gotten
2. **`skip_first=4` vs paper's `skip_first=16`.** More attention-sink positions included in ours; those may look similar between real and shuffled fits
3. **Single readout position (-1).** Paper measures across positions; final-position gathering may look similar under both fit conditions
4. **Model scale.** Below some threshold, models may not maintain coherent workspace structure. Anthropic's Claude Sonnet 4.5 is ~60-80× larger than Qwen 2.5-3B

The controlled experiment: refit at paper defaults (n=1000, skip_first=16) and re-run the shuffled control. Not budget-feasible on Kaggle T4 (would take days).

## What earlier versions of this document said

The previous RESULTS.md reported pass@10 scores as the primary evidence of workspace replication (poetry 0.55, typo 0.54, multilingual 0.40, multihop 0.42). Those numbers are still correct in absolute terms, but interpreting them as evidence for the workspace hypothesis was wrong — the shuffled control shows the same lens fitted on scrambled input produces comparable-or-higher pass@10 on every eval.

## Data files

- [`probing_results.json`](./results/full_rep/qwen/probing_results.json) — MRR, top-1, top-10 for real lens
- [`shuffled_probing.json`](./results/full_rep/qwen/shuffled_probing.json) — same, for shuffled-corpus lens
- [`causal_results.json`](./results/full_rep/qwen/causal_results.json) — Δlp per eval
- [`truncation_results.json`](./results/full_rep/qwen/truncation_results.json) — layer-by-layer probe readouts for full + truncated prompts

## Related

- [RESULTS_gemma4.md](./RESULTS_gemma4.md) — same 4-test methodology on Gemma-4-E2B-it. Shuffled control also fails on Gemma. Causal effect is very different (Gemma shows strongly negative Δlp on multihop and multilingual).
