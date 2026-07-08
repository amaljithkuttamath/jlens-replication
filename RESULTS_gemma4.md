# Cross-family: J-lens on Gemma-4-E2B-it

**Result of the second experiment.** Fits alongside [RESULTS.md](./RESULTS.md) (Qwen 2.5-3B baseline).

**Model:** `google/gemma-4-E2B-it` — 2B effective params + Per-Layer Embeddings, 35 layers, d_model=1536, hybrid 5:1 local (sliding-window) : global SDPA attention
**Compute:** free Kaggle T4 · **Wall time:** 106 min · **Peak VRAM:** 12.11 / 14.56 GB
**Kernel:** [gemma-4-e2b-j-lens](https://www.kaggle.com/code/amaljithkuttamath/gemma-4-e2b-j-lens)

## Why this matters

Gemma-4 is the first architecture to combine two things no public J-lens has been fit on:

1. **Per-Layer Embeddings (PLE)** — an auxiliary residual signal is injected into every decoder layer. Per [HuggingFace's Gemma-4 docs](https://huggingface.co/docs/transformers/en/model_doc/gemma4), PLE combines a token-identity lookup (from a separate 262144-vocab table with `hidden_size_per_layer_input=256`) and a context-aware projection, sums them, scales by `1/√2`, and **adds them into the residual stream at each layer**. It is not a bypass — it augments the residual pathway. (I got this wrong in an earlier version of this file.)
2. **5:1 hybrid sliding-window attention** — 5 out of every 6 layers only see the last 512 tokens (local); the 6th sees everything (global). Layer types are listed explicitly in `config.text_config.layer_types`.

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

**I don't know for sure.** Below are four candidate explanations, ordered by how much evidence I have for each. Deliberately not picking a winner, because doing that requires follow-up experiments I haven't run.

### Candidate 1: Auxiliary PLE signal dilutes the concept representation across dimensions

PLE injects a 256-dim per-layer signal into a 1536-dim residual (17% of the residual dimension) at every layer. If concept information is now shared across `residual + PLE_lookup + LAuReL-modulated pathway`, projecting the residual alone to vocabulary might miss the fraction of the information that lives in the PLE contribution. The J-lens is defined over the residual stream; it doesn't read the PLE input tensor separately.

This is *not* a bypass claim — PLE does enter the residual. But the *rate* at which concept-relevant components enter may be lower than in a pure residual-flow architecture. A vocabulary projection at layer `l` in Gemma-4 sees a running sum that has been mixed with more sources than a comparable Qwen readout.

### Candidate 2: Sliding-window attention prevents workspace-band formation

Gemma-4's `layer_types` alternates 5 local (512-token sliding window) : 1 global. If workspace formation requires global attention to consolidate a mid-stream concept across layers, we'd expect the band to appear preferentially at the 1-in-6 global layers. Looking at the raw probe data for Qwen (dense global attention everywhere) vs Gemma-4 doesn't obviously fit this — Gemma-4 shows `spider` at L6 and L12, neither of which are the 6-mod-1 global layers based on the config. But it's a candidate that a controlled experiment could isolate.

### Candidate 3: LAuReL-style modified residual pathways change what the residual carries

Gemma-4 uses [LAuReL](https://arxiv.org/abs/2411.07501)-style modifications: `x_{i+1} = α · x_i + β · layer(x_i) + γ · low_rank_gate(x_i)`. With `residual_weight=0.5` and additional per-token gating, the residual at layer `l` in Gemma-4 is a different-weighted combination of prior signals than in Qwen. The Jacobian `∂h_final / ∂h_l` is still well-defined, but what `h_l` represents differs. This might reduce the signal-to-noise of vocabulary projections without any information actually being hidden.

### Candidate 4: Different training data / fine-tuning objectives

Qwen 2.5-3B-Instruct and Gemma-4-E2B-it are trained on entirely different data mixtures with different post-training regimes. The visibility gap could reflect training decisions, not architectural ones. The clean way to isolate architecture from training is a J-lens on the base (not instruct) checkpoints, plus J-lens on Gemma-3-4B (no PLE, no LAuReL) for a within-family control.

### What would resolve this

One experiment settles it: **fit a J-lens on Gemma-3-4B** (same family, no PLE, standard residual). If it shows a Qwen-like wide band, PLE + LAuReL is the cause. If it shows a Gemma-4-like narrow band, something in Google's training pipeline is the cause. Either way, we'd know.

## Limitations (same as before, plus new ones)

Everything in [RESULTS.md § Limitations](./RESULTS.md#limitations-you-must-know-before-citing) still applies. Additional ones specific to this run:

- **Torch dynamo hit recompile limits** during fit (`torch._dynamo hit config.recompile_limit (8)`) — some layers fell back to eager mode. This *shouldn't* affect the fitted Jacobians (they're mathematically defined regardless of the execution path), but it's a note
- **The E2B result may not extrapolate to E4B.** E4B has 2× the PLE table size relative to residual dim; visibility drop could be worse or better
- **No shuffled control on Gemma-4 either.** Same missing methodology gap as with Qwen
- **My earlier framing was wrong.** I initially described PLE as a "bypass path" that routes information around the residual stream. That was architecturally incorrect. Per HuggingFace's official Gemma-4 docs, PLE feeds an *auxiliary signal into* the residual at every layer — it augments, doesn't bypass. Every reference to "bypass" in earlier versions of this file (and the LinkedIn post I initially drafted) was wrong. Corrected.

## What both results together tell us

The best summary is not "Gemma wins" or "Qwen wins" — it's:

> **The J-lens is architecture-sensitive.** On dense residual-flow models (Qwen 2.5-3B), the workspace band is wide and easy to read. On models with modified residual pathways (Gemma-4-E2B: PLE + LAuReL + hybrid sliding-window attention), the same probes show the target concept in far fewer layers, even though the model outperforms Qwen on the underlying evaluations.

I can't yet say **which** of PLE, LAuReL, sliding-window attention, or training-pipeline differences is responsible. That's a follow-up experiment, most cleanly resolved by comparing Gemma-3-4B (no PLE, no LAuReL, dense attention) against Gemma-4-E2B within the same model family.

What I do stand behind: **the visibility gap is real in the data**, and if the J-lens is going to be used more widely as an interpretability tool, its behavior on modern non-standard residual architectures deserves systematic study.

## Reproducing

```bash
git clone https://github.com/amaljithkuttamath/jlens-replication.git
cd jlens-replication
# Push kaggle/kaggle_run_gemma.ipynb to a Kaggle T4 kernel (see README)
# or paste it into a fresh Colab notebook and set runtime to T4
```

**Total spend:** $0. **Total time (both runs):** ~7h wall clock across Qwen 2.5-3B + Gemma-4-E2B.
