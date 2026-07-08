# J-lens on small open-weight models — a null-heavy replication

> **Attempted replication of Anthropic's July 2026 Jacobian lens paper on two open-weight models using free Kaggle compute. After running the paper's own control experiments, the passive-readout probing story does not hold up. Only one of four tests passes cleanly.**

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/amaljithkuttamath/jlens-replication/blob/main/notebooks/02_fit_colab.ipynb)

---

## TL;DR

Anthropic published *[Verbalizable Representations Form a Global Workspace in Language Models](https://transformer-circuits.pub/2026/workspace/index.html)* (2026-07-06) and open-sourced the [Jacobian lens code](https://github.com/anthropics/jacobian-lens). External replication so far: Neel Nanda (Google DeepMind) on Qwen 3.6-27B, qualitative verdicts only, no public numbers.

I ran the full method on **Qwen 2.5-3B-Instruct** and **Gemma-4-E2B-it** on Kaggle T4s. Four tests total: (1) MRR probing on the 6 lens-quality evals, (2) causal-swap intervention, (3) prompt-truncation ablation on sanity probes, (4) shuffled-corpus control lens.

**Results:**

| Test | What it tests | Qwen 2.5-3B | Gemma-4-E2B |
|---|---|---|---|
| MRR probing | passive concept rank in vocab space | scores present but **falsified by shuffled control** | same |
| Causal swap (Δlog-prob) | is the concept used by the model? | **+0.015 to +0.068** (no effect) | **-6.35 (multihop), -3.69 (multilingual)** (large effects) |
| Prompt truncation | does concept require semantic prompt? | ✅ concept 0/34 layers on truncated | ✅ concept 0/33 layers on truncated |
| Shuffled-corpus control | does workspace claim survive scrambled fit? | ❌ shuffled MRR ≥ real MRR on **6 of 6** evals | ❌ shuffled MRR ≥ real MRR on **6 of 6** evals |

**Bottom line:** the passive readout story I initially reported (workspace visible at 20/34 layers on Qwen, etc.) does not survive the shuffled-corpus control. The probes are prompt-dependent (truncation test passes), and Gemma-4 shows real causal effects, but the passive-MRR-as-workspace-evidence claim is dead.

## The findings, honestly

### Finding 1 (the disappointing one): Passive lens MRR is not evidence of a workspace on either model

A properly-fit J-lens should produce vocabulary readouts that are *specifically* aligned with the target concept when fit on the model's actual next-token distribution, and should collapse toward random when fit on a scrambled corpus. Our shuffled-corpus control lens — fit on the same 25 Pile prompts but with tokens randomly permuted within each prompt — should be broken.

It is not. Here is the direct comparison:

**Qwen 2.5-3B:**

| Eval | Real lens MRR | **Shuffled-corpus lens MRR** | Δ (should be strongly positive) |
|---|---|---|---|
| poetry | 0.332 | 0.337 | **-0.006** |
| typo | 0.318 | **0.806** | **-0.488** |
| multilingual | 0.261 | 0.278 | -0.017 |
| multihop | 0.243 | 0.285 | -0.042 |
| order-ops | 0.066 | 0.074 | -0.009 |
| association | 0.006 | 0.015 | -0.009 |

**Gemma-4-E2B-it:**

| Eval | Real | Shuffled | Δ |
|---|---|---|---|
| poetry | 0.637 | 0.718 | -0.081 |
| multihop | 0.426 | 0.458 | -0.032 |
| order-ops | 0.238 | 0.460 | **-0.222** |
| typo | 0.194 | **0.754** | **-0.560** |
| multilingual | 0.248 | 0.321 | -0.074 |
| association | 0.002 | 0.006 | -0.004 |

**Every eval on both models has shuffled MRR ≥ real MRR.** In some cases (`typo` on both, `order-ops` on Gemma) the shuffled lens is dramatically better. This is diagnostic — a lens fit on scrambled input is finding the target concepts at least as often as a lens fit on coherent English. The implication: our passive readouts were picking up baseline properties of the model's residual stream, not workspace-formed intermediate representations.

Since Anthropic and Neel Nanda both use passive metrics as one component of their evaluation, and since neither published a shuffled control comparison, **this is either a real failure of the passive method on small models, or a difference in fit methodology I don't yet understand.** Full data: [`results/full_rep/qwen/shuffled_probing.json`](./results/full_rep/qwen/shuffled_probing.json), [`results/full_rep/gemma/shuffled_probing.json`](./results/full_rep/gemma/shuffled_probing.json).

### Finding 2 (the interesting one): Causal-swap effect differs sharply between models

Following Neel Nanda's methodology, for each eval item we (a) extract the concept's unembedding direction, (b) ablate that direction from residuals at mid-band layers, (c) measure the change in log-probability of the correct final answer. **Negative Δlp means the ablation hurt performance — i.e., the concept direction was causally used by the model.**

| Eval | Qwen 2.5-3B Δlp | Gemma-4-E2B Δlp | Gemma effect size |
|---|---|---|---|
| multihop | +0.015 (no effect) | **-6.353** | very large |
| multilingual | +0.049 (no effect) | **-3.690** | very large |
| order-ops | +0.068 (no effect) | -0.242 | moderate |

Full data: [`results/full_rep/qwen/causal_results.json`](./results/full_rep/qwen/causal_results.json), [`results/full_rep/gemma/causal_results.json`](./results/full_rep/gemma/causal_results.json).

**Ablating "the answer's token direction" from Gemma-4-E2B's mid-band residuals drops correct-answer log-probability by ~6 nats on multihop.** On Qwen 2.5-3B it does essentially nothing.

Careful interpretation:
- This is not the paper's own swap intervention (which finds the *lens-derived direction for the intermediate concept*, not the target token). Ours is a coarser proxy.
- Given that the shuffled control failed (Finding 1), this causal effect could partly be explained by the model's baseline sensitivity to residual perturbations along answer-token directions, independent of workspace mechanisms.
- Still, the Qwen-vs-Gemma asymmetry is genuine and large.

### Finding 3 (the one that works): Prompt-truncation ablation passes

For each of the three sanity probes, we ran the lens on the full prompt and on a truncated version cut before the referent phrase. If the concept in the readout depends on prompt context (workspace-like), truncation should suppress it. If the lens is just projecting model priors, truncation should not matter.

| Probe target | Full-prompt hits (Qwen) | **Truncated-prompt hits (Qwen)** | Truncated (Gemma) |
|---|---|---|---|
| `spider` | 20 layers (L12–31) | **0 layers** | 0 layers |
| `italy` | 12 layers (L21–32) | **0 layers** | 0 layers |
| `canada` | 4 layers | **0 layers** | 0 layers |

Full data: [`results/full_rep/qwen/truncation_results.json`](./results/full_rep/qwen/truncation_results.json). **On both models, all three concepts disappear entirely from the top-5 lens readout when the prompt is truncated.** This is the cleanest positive result we have: the mid-stream concept emergence is genuinely prompt-content-dependent, not a lens artifact.

Note: this is a 3-item test. Robust ablation would require the full eval set. That's future work.

## What we can now claim

**Claim (defensible):** The Jacobian lens, applied to a 3B/2B open-weight model, produces vocabulary-space readouts in which target intermediate concepts appear across a wide layer band, and this appearance is prompt-context-dependent (removed by truncation). This qualitative pattern matches the paper's description.

**Claim (not defensible from our data):** That the resulting MRR / top-k scores measure workspace-mediated representation. Our shuffled-corpus control lens produces comparable-or-better MRRs across all six evals on both models, which contradicts the interpretation of these numbers as workspace evidence.

**Claim (asymmetric between models):** Ablating answer-token directions from mid-band residuals substantially degrades Gemma-4-E2B's answer probability but not Qwen 2.5-3B's. This is a real causal difference, but until the shuffled-control issue is resolved, we can't cleanly interpret it as "Gemma has a used workspace and Qwen does not."

## Why our shuffled control failed and the paper's approach presumably doesn't

Speculating, not concluding:

- **Small-n fitting.** We use n=25 background prompts. If the Jacobian estimate is dominated by second-order statistics of the residual stream that are similar under shuffling, small-n fitting won't discriminate. Paper uses n=1000
- **`skip_first=4` vs the paper's `skip_first=16`.** With fewer positions skipped, our Jacobian is more influenced by early-position attention-sink content, which may look similar between real and shuffled corpora
- **Passive scoring at only the final position (-1).** The paper measures across multiple readout positions; final-position gathering may look similar between real and shuffled lenses because both produce plausible-continuation tokens
- **Genuine limitation of small models.** Below some capability threshold, models may not maintain workspace representations coherent enough to distinguish from shuffled-input Jacobians

I don't know which of these is right. Testing them requires more compute than the current design allows.

## What's in this repo

- [`kaggle/kaggle_run.ipynb`](./kaggle/kaggle_run.ipynb) — original Qwen 2.5-3B fit
- [`kaggle/kaggle_run_gemma.ipynb`](./kaggle/kaggle_run_gemma.ipynb) — Gemma-4-E2B-it fit
- [`kaggle/full_rep_qwen.ipynb`](./kaggle/full_rep_qwen.ipynb) — full 4-test methodology for Qwen (this is the one that produced the results here)
- [`kaggle/full_rep_gemma.ipynb`](./kaggle/full_rep_gemma.ipynb) — same for Gemma
- [`results/`](./results/) — all raw JSONs, both fitted lenses (Qwen 285 MB, Gemma also on Kaggle artifacts), probe readouts, MRR/causal/truncation/shuffled data

## What earlier versions of this README claimed (and were wrong about)

Full git history shows the evolution, but for anyone landing here from a stale link:

1. Initial claim: "3B replicates the workspace, wide 20-layer band on Qwen" — the band exists in the readout but is not workspace-evidence per the shuffled control
2. Second claim: "Gemma-4-E2B outperforms Qwen 2.5-3B on 3/6 evals" — comparing two contaminated pass@k scores tells us nothing about workspace
3. Third claim: "PLE architecture bypasses residual stream" — architecturally wrong; PLE adds INTO the residual per HuggingFace docs
4. Fourth claim: "J-lens is architecture-sensitive in a publishable-caveat way" — retracted; the effect I attributed to PLE + LAuReL is not distinguishable from small-model limitations of the passive method

I made these claims in sequence over ~48 hours as new data came in. Each was corrected in a subsequent commit. The current README is what I believe survives the full 4-test methodology.

## What would move this forward

Not planned, but the honest next steps:

1. **Re-fit at paper defaults**: n=1000 prompts, skip_first=16, multiple target layers. See if the shuffled control then collapses as it should
2. **Fit on Neel Nanda's exact model (Qwen 3.6-27B)** with our pipeline. If our shuffled control fails there too, the issue is our metric or fit code, not model size. If it collapses, the problem is small-model specific
3. **Ask Anthropic** whether their internal validation includes a shuffled-corpus control. If not, this may be a real methodological gap in the paper

## References

- [Anthropic — Verbalizable Representations Form a Global Workspace in Language Models](https://transformer-circuits.pub/2026/workspace/index.html)
- [anthropics/jacobian-lens](https://github.com/anthropics/jacobian-lens) — reference implementation
- [Neel Nanda / MATS external commentary PDF](https://www-cdn.anthropic.com/files/4zrzovbb/website/cc4be2488d65e54a6ed06492f8968398ddc18ebe.pdf)

## Contact

Built by [amaljithkuttamath](https://github.com/amaljithkuttamath). Open to feedback or collaboration — especially from anyone who has run a shuffled-corpus control on this method and knows why ours failed.

---

## License

Apache-2.0 for this code. Model weights follow their respective licenses.
