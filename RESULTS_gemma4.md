# RESULTS — Gemma-4-E2B-it, full methodology

**Kernel:** [jlens-full-replication-gemma](https://www.kaggle.com/code/amaljithkuttamath/jlens-full-replication-gemma)
**Wall time:** ~4h · **Peak VRAM:** 12.1 GB
**Fit config:** target_layer=-2, dim_batch=2, max_seq_len=96, skip_first=4, n=25 pile-10k prompts

**This file supersedes the previous version**, which claimed a "PLE architecture bypass" caveat that was both architecturally wrong (PLE feeds INTO the residual per HuggingFace docs) and not supported by the current controlled data.

## The four tests

### Test 1 — MRR probing

| Eval | n | MRR | top-1 | top-10 |
|---|---|---|---|---|
| poetry | 98 | 0.637 | 0.541 | 0.806 |
| multihop | 84 | 0.426 | 0.351 | 0.589 |
| order-ops | 55 | 0.238 | 0.173 | 0.382 |
| multilingual | 107 | 0.248 | 0.194 | 0.358 |
| typo | 96 | 0.194 | 0.083 | 0.458 |
| association | 102 | 0.002 | 0.000 | 0.000 |

Raw: [`probing_results.json`](./results/full_rep/gemma/probing_results.json).

Numerically higher than Qwen 2.5-3B on 4 of 6 evals (poetry, multihop, order-ops); lower on typo and multilingual; tied at floor on association. **However, as with Qwen, these numbers cannot be interpreted as workspace evidence until they beat the shuffled-corpus control — which they don't (see Test 4).**

### Test 2 — Causal-swap intervention (the interesting result)

| Eval | n scored | Mean Δlp | Fraction where ablation hurts |
|---|---|---|---|
| **multihop** | 52 | **-6.353** | 0.846 |
| **multilingual** | 35 | **-3.690** | 0.743 |
| order-ops | 32 | -0.242 | 0.500 |
| poetry, typo, association | 0 | (no scoreable items) | — |

Raw: [`causal_results.json`](./results/full_rep/gemma/causal_results.json).

**Ablating target-token unembedding directions from Gemma-4-E2B's mid-band residuals drops correct-answer log-probability by ~6 nats on multihop and ~4 nats on multilingual.** For comparison, on Qwen 2.5-3B the same ablation produces Δlp of +0.015 to +0.068 (no measurable effect).

This is the largest asymmetry we found between the two models. It is real — the raw data shows the effect on 74–85% of items scored on the two big-effect evals. It is also not trivially explained by the shuffled-control failure (Test 4), because the causal test measures a different thing (does ablation hurt model performance?), independent of what the lens reads out.

**Interpretation caveats:**
- Our concept-direction extraction uses the target token's unembedding row directly, not a lens-derived intermediate-concept direction (which would require the paper's swap-intervention machinery). Ours is a coarser proxy for what the paper does
- Because Gemma-4 has Per-Layer Embeddings and LAuReL-style modified residuals, the mid-band residual state may be more sensitive to any ablation, not specifically the workspace direction. A causal-swap on random directions of the same norm would be the proper control. Not run

### Test 3 — Prompt-truncation ablation

| Probe | Full-prompt hit layers | Truncated hit layers |
|---|---|---|
| `spider` | 2 (L6, L12) | 0 |
| `italy` | 0 (never in top-5 anywhere) | 0 |
| `canada` | 1 (L27) | 0 |

Raw: [`truncation_results.json`](./results/full_rep/gemma/truncation_results.json).

**Full-prompt hit rates on Gemma-4 are much lower than on Qwen (2 vs 20 layers for spider).** This part of my earlier writeup was correct as raw observation. But **truncation still cleanly suppresses whatever hits do exist**, so the emergence-under-prompt property holds.

The `italy` probe showing 0 hits at any layer under either condition is worth noting — Gemma-4 either doesn't retrieve "Italy" through the residual pathway on this prompt, or does so through a direction the lens can't project. The model does still answer correctly to this prompt at generation time, so the information reaches the output somehow.

### Test 4 — Shuffled-corpus control (the failing test, again)

| Eval | Real MRR | Shuffled MRR | Δ (positive = real > shuffled = good) |
|---|---|---|---|
| poetry | 0.637 | 0.718 | -0.081 |
| **typo** | **0.194** | **0.754** | **-0.560** |
| **order-ops** | **0.238** | **0.460** | **-0.222** |
| multilingual | 0.248 | 0.321 | -0.074 |
| multihop | 0.426 | 0.458 | -0.032 |
| association | 0.002 | 0.006 | -0.004 |

Raw: [`shuffled_probing.json`](./results/full_rep/gemma/shuffled_probing.json).

**Same pattern as Qwen: every eval is at least as good on the shuffled-corpus lens.** The `typo` and `order-ops` gaps are especially damning. This means our Test 1 probing MRR numbers, on Gemma too, cannot be interpreted as evidence of workspace-mediated representation.

## The verdict

- Test 1 (probing) results are contaminated on Gemma-4-E2B too — shuffled control demonstrates this
- **Test 2 (causal) shows a large, real, asymmetric effect between the two models.** Gemma responds strongly to target-direction ablation; Qwen does not. This deserves further investigation but cannot yet be called "the workspace is causally used" without controls (random-direction ablation, etc.)
- Test 3 (truncation) works but shows only weak signals under the full prompt on Gemma
- Test 4 (shuffled control) fails, same as on Qwen

**What survives on Gemma:** the causal-swap effect is real and much larger than on Qwen. **What does not survive:** the passive-probing scores as workspace evidence, and the previous framing that "Gemma-4's PLE architecture makes the workspace invisible to the lens" (that was speculation, not a data-driven claim).

## Comparison table

| Test | Qwen 2.5-3B | Gemma-4-E2B |
|---|---|---|
| MRR probing floor evals (association, order-ops for Qwen) | floor | order-ops off floor |
| MRR probing top evals | poetry 0.33, typo 0.32 | poetry 0.64, multihop 0.43 |
| Causal Δlp on multihop | +0.015 (no effect) | **-6.353** (large) |
| Causal Δlp on multilingual | +0.049 (no effect) | **-3.690** (large) |
| Truncation cleanly suppresses concept | ✅ | ✅ |
| Shuffled control passes | ❌ (6 of 6 fail) | ❌ (6 of 6 fail) |

## What earlier versions said

Version 1: "Gemma-4-E2B outperforms Qwen 2.5-3B on 3/6 evals despite being 33% smaller" — comparing two contaminated pass@k scores; retracted.

Version 2: "PLE architecture routes information around the residual stream, so J-lens can't see it" — architecturally wrong; PLE feeds INTO the residual per HuggingFace docs; retracted.

Version 3: "J-lens is architecture-sensitive in publishable-caveat ways" — retracted; the visibility gap between models is real in the raw readout data, but the mechanistic story I told about it was speculation across four candidate causes I couldn't distinguish

Current version: the causal-swap asymmetry (Gemma large negative Δlp, Qwen near zero) is the most interesting finding, but requires more controls before it can support the strong story. The passive-probing side of the replication does not survive the shuffled control.
