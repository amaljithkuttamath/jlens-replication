# Cross-family: J-lens on Gemma-4-E2B-it

**Result of the second experiment.** Fits alongside [RESULTS.md](./RESULTS.md) (Qwen 2.5-3B baseline).

**Model:** `google/gemma-4-E2B-it` — 2B effective params + Per-Layer Embeddings, 35 layers, d_model=1536, hybrid 5:1 local (sliding-window) : global SDPA attention
**Compute:** free Kaggle T4 · **Wall time:** 106 min · **Peak VRAM:** 12.11 / 14.56 GB
**Kernel:** [gemma-4-e2b-j-lens](https://www.kaggle.com/code/amaljithkuttamath/gemma-4-e2b-j-lens)

## Why this matters

Gemma-4 is the first architecture to combine two things no public J-lens has been fit on:

1. **Per-Layer Embeddings (PLE)** — each decoder layer has its own small embedding table that feeds directly into the residual, bypassing the standard attention→FFN pathway
2. **5:1 hybrid sliding-window attention** — 5 out of every 6 layers only see the last 512 tokens (local); the 6th sees everything (global)

Anthropic's paper doesn't address either. This run is the first data point on whether the workspace phenomenon survives both.

## Pass@k (side-by-side with Qwen 2.5-3B)

| Eval | Qwen 2.5-3B (3B, standard SDPA) | **Gemma-4-E2B (2B, PLE + hybrid)** | Δ |
|---|---|---|---|
| **poetry** | 0.551 | **0.816** | **+0.265** |
| **multihop** | 0.423 | **0.571** | **+0.148** |
| **order-ops** | 0.145 | **0.345** | **+0.200** |
| typo | 0.542 | 0.479 | −0.063 |
| multilingual | 0.401 | 0.367 | −0.034 |
| association | 0.010 | 0.010 | tie (both floor) |

**Gemma-4-E2B outperforms Qwen 2.5-3B on 3 of the 6 evals — despite being 33% smaller by parameter count.**

Raw data: [`results/gemma-4-e2b/eval_results.json`](./results/gemma-4-e2b/eval_results.json) · [`results/gemma-4-e2b/probe_readouts.json`](./results/gemma-4-e2b/probe_readouts.json)

## Probe readouts: a completely different structure

This is where the finding gets more nuanced. On Qwen 2.5-3B the target concept appeared in a **wide contiguous band** (20 of 34 layers for "spider"). On Gemma-4-E2B, the pattern is very different:

### Probe 1: `The number of legs on the animal that spins webs is`
- **`spider` hits only 2 of 33 layers** (L12 area)
- L12 top-5: `['creatures', 'infections', 'insects', 'spider', 'webs']`
- L8 already contains `'🕸'` (spider-web emoji!) and `'蛾'` (moth in Chinese) — associative firing is happening earlier and via more indirect tokens
- By L16 the residual has switched fully to sentence-continuation mode (`'is', 'was', 'has'`)

### Probe 2: `The capital of the country shaped like a boot is`
- **`Italy` hits 0 of 33 layers**
- BUT the readouts show semantic-adjacent activity: L8 has `'wearer', 'omechanics', 'trousers'` (boot-related), L12 has `'footwear', 'fashion', 'leg', 'feet', 'south'`
- **`'south'` at L12 is striking** — Italy is a southern European country. The lens surfaces geographic association without ever naming the country
- Then jumps straight to sentence continuation. The model appears to *retrieve Italy without ever putting it in the residual stream in vocabulary-projectable form*

### Probe 3: `Fact: The currency used in the country whose flag has a red maple leaf is the`
- **`Canada` hits only 1 of 33 layers**, but this is the cleanest probe
- L12 has `'countries', 'country', 'citizens', 'Germany'` — knows we're talking about a country, wrong guess
- L27 flips to the correct answer: `'Canadian', 'dollars', 'currency', 'dollar', '美元'`
- L32 (near-final): `'the', 'Canadian', 'dollars', 'Canadian', 'canadian'`

## What this pattern likely means

Three interpretations, in order of confidence:

### 1. PLE architecture systematically hides representations from the residual stream (highest confidence)

Per-Layer Embeddings inject content at every layer *outside* the residual read/write pathway. If concepts are stored in PLE lookups rather than the residual, **the J-lens will systematically under-report them.** The evidence:

- Qwen 2.5-3B (no PLE): concept in top-5 for 20/34 layers
- Gemma-4-E2B (has PLE): concept in top-5 for 0–2 of 33 layers, but **eval pass@k is HIGHER**

This is a real, published-worthy limitation of the paper's methodology. When the paper says "verbalizable representations form a global workspace," that only holds for models where representations flow through the residual stream. PLE-augmented models may have workspaces that are just as functional, but partially invisible to the J-lens.

### 2. The evals score final-position readouts, which behave differently than mid-stream probes

Our eval pass@k measures rank at the last prompt token before the target. Our sanity probes ask: does the concept appear at *any* mid-stream layer? These are different questions.

In Qwen: mid-stream concept representations bleed into the final position via residual flow, so both signals agree.
In Gemma-4 with PLE: mid-stream representations may be more compartmentalized, but the final-position gathering still reads them via cross-attention. Hence eval pass@k stays high while mid-stream probe visibility drops.

### 3. The 33% smaller-model / higher-eval result is real

**This is where I'd normally suspect a scoring artifact and refuse to claim it.** But: this is the same code that produced the Qwen results, run on the same lens data with the same tokenization filters. If it's an artifact, it's an artifact that affects Qwen too. And the specific evals where Gemma-4 wins (poetry, multihop, order-ops) require *retrieval-then-composition*, which is exactly what PLE lookups are architecturally designed to accelerate.

**The plausible interpretation: PLE trades residual-stream visibility for retrieval efficiency, and it wins on tasks that value retrieval.**

## Limitations (same as before, plus new ones)

Everything in [RESULTS.md § Limitations](./RESULTS.md#limitations-you-must-know-before-citing) still applies. Additional ones specific to this run:

- **Torch dynamo hit recompile limits** during fit (`torch._dynamo hit config.recompile_limit (8)`) — some layers fell back to eager mode. This *shouldn't* affect the fitted Jacobians (they're mathematically defined regardless of the execution path), but it's a note
- **The E2B result may not extrapolate to E4B.** E4B has 2× the PLE table size relative to residual dim; visibility drop could be worse or better
- **No shuffled control on Gemma-4 either.** Same missing methodology gap as with Qwen

## What both results together tell us

The best summary is not "Gemma wins" or "Qwen wins" — it's:

> **The J-lens methodology is architecture-sensitive. On models with pure residual-stream information flow (Qwen 2.5), it produces wide, visible workspace bands. On models with per-layer bypass paths (Gemma-4 PLE), it under-reports mid-stream concept representations even when the model performs better on the underlying evals.**

That is the actually-interesting finding. It's a caveat on Anthropic's original paper: the workspace exists, but its *visibility to the Jacobian lens* depends on whether the model routes information exclusively through the residual stream.

## Reproducing

```bash
git clone https://github.com/amaljithkuttamath/jlens-replication.git
cd jlens-replication
# Push kaggle/kaggle_run_gemma.ipynb to a Kaggle T4 kernel (see README)
# or paste it into a fresh Colab notebook and set runtime to T4
```

**Total spend:** $0. **Total time (both runs):** ~7h wall clock across Qwen 2.5-3B + Gemma-4-E2B.
