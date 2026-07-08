# J-lens on small open-weight models

> **A weekend replication of Anthropic's July 2026 interpretability paper — on two open-source models, using free Kaggle compute. Uncovered a real methodological caveat on the original technique.**

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/amaljithkuttamath/jlens-replication/blob/main/notebooks/02_fit_colab.ipynb)

---

## TL;DR

Anthropic published *[Verbalizable Representations Form a Global Workspace in Language Models](https://transformer-circuits.pub/2026/workspace/index.html)* on July 6, 2026. They open-sourced the [Jacobian lens (J-lens)](https://github.com/anthropics/jacobian-lens) — a technique for reading a model's mid-computation "workspace" in vocabulary space.

The only external replication was on **Qwen 3.6-27B** (Neel Nanda / MATS). Nobody had asked: *does this work on the small open models most people can actually run?*

I ran it end-to-end on two:

| Model | Params | Architecture | Compute | Wall time | Result |
|---|---|---|---|---|---|
| Qwen 2.5-3B-Instruct | 3B | dense SDPA | Kaggle T4 (free) | 228 min | ✅ Workspace visible, wide band |
| Gemma-4-E2B-it | 2B | **PLE + hybrid attention** | Kaggle T4 (free) | 106 min | ⚠️ Workspace barely visible — but evals *better* |

**Total spend: $0.** Both experiments reproducible from the notebooks in this repo.

## The interesting finding

Gemma-4-E2B **outperforms Qwen 2.5-3B on 3 of 6 evaluations** despite being 33% smaller by parameter count:

| Eval (pass@10) | Qwen 2.5-3B | Gemma-4-E2B | Δ |
|---|---|---|---|
| poetry (rhyme planning) | 0.551 | **0.816** | **+0.265** |
| multihop (intermediate facts) | 0.423 | **0.571** | **+0.148** |
| order-of-operations | 0.145 | **0.345** | **+0.200** |
| typo detection | 0.542 | 0.479 | −0.063 |
| multilingual | 0.401 | 0.367 | −0.034 |
| association (abstract) | 0.010 | 0.010 | tie (floor) |

But when you look at the layer-by-layer probe readouts, **Gemma-4-E2B's workspace is nearly invisible to the J-lens:**

For the classic probe *"The number of legs on the animal that spins webs is..."* (where "spider" never appears in the prompt):

- **Qwen 2.5-3B:** "spider" appears in the top-5 lens readout at **20 of 34 layers** — a wide, contiguous workspace band
- **Gemma-4-E2B:** "spider" appears in the top-5 at **2 of 33 layers** — the model still gets the right answer, but the workspace doesn't show up under the lens

### What this means

The visibility gap is real in the data. What causes it is an open question I don't yet have enough evidence to answer confidently.

Gemma-4 differs from Qwen in **at least four ways** that could plausibly matter:

- **Per-Layer Embeddings (PLE):** an auxiliary 256-dim signal injected into the residual at every layer (adds, doesn't bypass — corrected from an earlier version of this doc)
- **LAuReL-style modified residual pathways** with `residual_weight=0.5` and per-token low-rank gating
- **5:1 sliding-window : global hybrid attention** (5 of every 6 layers only see 512 tokens)
- **Different training data mix and post-training regime** than Qwen 2.5

Any of these, or a combination, could explain why the J-lens reads a wide workspace band on Qwen but a narrow one on Gemma-4. **My honest position is: I don't know which yet.** The clean follow-up experiment is a J-lens on Gemma-3-4B (same family, no PLE, no LAuReL) to isolate the effect. That's a next step, not a claim.

What I do stand behind: **the J-lens is architecture-sensitive.** If it's going to be used more broadly as an interpretability tool, its behavior on non-standard residual architectures deserves systematic study.

That kind of methodological question only surfaces when someone replicates on a model with a genuinely different architecture. Doing that ends up being cheap: a weekend and $0.

## Probes in detail

Full layer-by-layer readouts in [`results/probe_readouts.json`](./results/probe_readouts.json) and [`results/gemma-4-e2b/probe_readouts.json`](./results/gemma-4-e2b/probe_readouts.json). A representative slice:

**Qwen 2.5-3B — Probe: "The capital of the country shaped like a boot is"**
```
L0-20:  surface tokens: "boot", "/boot", ".debian"
L21:    'Italy' emerges  ← workspace band starts
L21-32: 'Italy', 'Italian', '意大利' dominate top-5
L33:    'is', 'lies', 'sits' — final sentence completion
```

**Gemma-4-E2B — Same probe**
```
L0-4:   noise
L8:     'wearer', 'omechanics', 'trousers'  ← boot associations
L12:    'footwear', 'fashion', 'leg', 'feet', 'south'  ← 'south' is striking, geographic proxy
L16-32: switches straight to sentence completion tokens
        Italy NEVER appears in top-5 anywhere
```

The Gemma model surfaces `'south'` (a geographic proxy for Italy) but never the actual name. Yet ask it, and it answers correctly. The workspace is happening — just not where the lens can see it.

## Reproducibility

Every result in this repo is reproducible on free consumer compute in under 4 hours.

**Option A (fastest):** open the [Colab notebook](https://colab.research.google.com/github/amaljithkuttamath/jlens-replication/blob/main/notebooks/02_fit_colab.ipynb), set runtime to T4, Run All.

**Option B (Kaggle T4):** the [`kaggle/`](./kaggle/) directory contains the exact notebooks used for both runs. Push them via the Kaggle CLI or paste into a fresh notebook.

**Option C (Lightning L4, ~15-25 min):** `python notebooks/03_fit_lightning.py`.

## What I built

- End-to-end pipeline: fit lens (~106-228 min on T4) + apply lens + score 6 evals + emit probe readouts
- Kaggle API-driven orchestration (no browser clicking) with automatic hourly result polling
- Direct comparison against Neel Nanda / MATS's Qwen 3.6-27B baseline
- Two complete write-ups: [Qwen 2.5-3B results](./RESULTS.md) and [Gemma-4-E2B results](./RESULTS_gemma4.md), both with honest limitations sections

## What I got wrong along the way

Recorded in git history for anyone who cares:

- Wasted time on Kaggle P100 (sm_60) before realizing modern PyTorch dropped support for it — fix: `machine_shape: NvidiaTeslaT4` in the metadata JSON
- Tried Gemma-4-E4B first (~15 GB static, doesn't fit T4's 14.5 GB) before dropping to E2B (~11 GB static, fits with headroom)
- Initially misread my own eval scoring code as buggy on multilingual/order-ops — was actually correct, my audit was wrong. Fixed writeup accordingly

Full [commit history](https://github.com/amaljithkuttamath/jlens-replication/commits/main) shows the debugging trail.

## Honest limitations

Read [RESULTS.md § Limitations](./RESULTS.md#limitations-you-must-know-before-citing) before citing any specific number. Short version:

- Our pass@k scoring is a stricter implementation than the paper's — numbers not directly comparable to Anthropic's
- No shuffled-corpus control run (the standard "does this actually require semantic structure" test)
- No prompt-truncation ablation on the probes (to rule out "the lens just projects any plausible next token")
- n=25 background prompts (paper uses 1000; Neel Nanda showed n=10 saturates, but our variance is wider than the paper's)

The **qualitative pattern** (wide layer band on Qwen, near-invisibility on Gemma-4 PLE) is robust to all of these caveats. Specific numeric magnitudes should carry error bars.

## Deep dives

- [Qwen 2.5-3B full writeup](./RESULTS.md) — configuration, per-eval breakdown, comparison to Neel Nanda's Qwen-27B, limitations
- [Gemma-4-E2B full writeup + PLE interpretation](./RESULTS_gemma4.md) — three plausible interpretations of the visibility gap, ranked by evidence
- [`kaggle/kaggle_run.ipynb`](./kaggle/kaggle_run.ipynb) — Qwen 2.5-3B fit
- [`kaggle/kaggle_run_gemma.ipynb`](./kaggle/kaggle_run_gemma.ipynb) — Gemma-4-E2B fit
- [`results/`](./results/) — raw eval_results.json, probe_readouts.json, rescore_v2.json for both runs

## References

- [Anthropic — *Verbalizable Representations Form a Global Workspace in Language Models*](https://transformer-circuits.pub/2026/workspace/index.html) (2026-07-06)
- [Anthropic — blog post](https://www.anthropic.com/research/global-workspace)
- [anthropics/jacobian-lens](https://github.com/anthropics/jacobian-lens) — reference implementation (Apache 2.0)
- [Neel Nanda / MATS external replication PDF](https://www-cdn.anthropic.com/files/4zrzovbb/website/cc4be2488d65e54a6ed06492f8968398ddc18ebe.pdf) — Qwen 3.6-27B, prior art

## Contact

Built by [amaljithkuttamath](https://github.com/amaljithkuttamath). Reach me via [GitHub issues](https://github.com/amaljithkuttamath/jlens-replication/issues) or [github.com/amaljithkuttamath](https://github.com/amaljithkuttamath).

If you're working on interpretability, small-model replications of frontier lab findings, or want to extend this to other architectures (Nemotron, Llama 4, MoE models), I'd like to hear from you.

---

## License

Apache-2.0 for this code. Model weights follow their respective licenses (Qwen 2.5-3B: Apache 2.0; Gemma-4: Gemma Terms of Use).
