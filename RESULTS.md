# Results: First J-lens replication on a sub-10B open model

**Model:** `Qwen/Qwen2.5-3B-Instruct` · **Compute:** free Kaggle T4 (16 GB) · **Wall time:** 3h 48m · **Peak VRAM:** 7.17 GB
**Reference paper:** Anthropic, [*Verbalizable Representations Form a Global Workspace in Language Models*](https://transformer-circuits.pub/2026/workspace/index.html) (2026-07-06)
**Reference code:** [anthropics/jacobian-lens](https://github.com/anthropics/jacobian-lens)

## Why this exists

Anthropic released the Jacobian lens (J-lens) with their paper, but the only external replication (by Neel Nanda / MATS) was on Qwen 3.6-27B — 9× larger than the models most researchers can afford to run. This repo asks a simple question: **does the "global workspace" phenomenon appear in a small, freely-runnable open-weight model, using only free compute?**

The answer is: **yes, qualitatively — with important methodological caveats you should read before citing these numbers.**

## Config

| Knob | Value |
|---|---|
| Background corpus | `NeelNanda/pile-10k` (first 25 items ≥ 200 chars) |
| n prompts | 25 (Anthropic used 1000; Neel Nanda showed n≥10 saturates) |
| max_seq_len | 96 tokens |
| skip_first (attention-sink positions) | 4 |
| target_layer | -2 (penultimate) |
| dim_batch | 2 |
| dtype | bfloat16 |
| attn_implementation | sdpa |

## The probe readouts (workspace signal, visualized)

Three canonical two-hop probes, top-5 lens readouts across all 34 layers of Qwen 2.5-3B. **Concept tokens in bold; concepts never appear in the prompt.**

### Probe 1: `The number of legs on the animal that spins webs is`

| Layer | Top-5 tokens |
|---|---|
| L0–11 | `webs, /web, @Web, <br, ...` — still processing surface tokens |
| **L12** | **`webs, spiders, webs, /web, spider`** ← concept emerges |
| L15–26 | `spiders, spider, Spider` dominates (**20 of 34 layers**) |
| L24 | `spiders, spider, 蜘蛛 (Chinese: spider), webs, 昆虫 (insect)` |
| L33 (final) | `is, would, must, indoors, resembles` — sentence-continuation mode |

### Probe 2: `The capital of the country shaped like a boot is`

| Layer | Top-5 |
|---|---|
| L0–20 | `boot, /boot, .debian, ...` — surface tokens |
| **L21** | **`Italy` emerges** |
| L21–32 | `Italy, Italian, 意大利` (12 layers) |
| L33 | `is, lies, sits, belongs` |

### Probe 3: `Fact: The currency used in the country whose flag has a red maple leaf is the`

| Layer | Top-5 |
|---|---|
| L12–24 | `currency, 美元 (US dollar)` — generic currency noise |
| **L27** | **`Canada` emerges** — model resolves the referent |
| L33 | `Canadian, Canada, 加拿大 (Canada, Chinese)` — final answer |

Note: for probe 3, Qwen surfaces the **country** more confidently than the **currency**. That's a real intermediate-structure finding.

## Eval-suite pass@k

Six lens-quality evaluations released with Anthropic's paper. All scored with min-rank-across-layers, top-k threshold.

| Eval | n | pass@1 | pass@5 | pass@10 |
|---|---|---|---|---|
| **poetry** | 98 | 0.235 | 0.408 | **0.551** |
| **typo** | 96 | 0.208 | 0.427 | **0.542** |
| **multihop** | 84 | 0.155 | 0.333 | **0.423** |
| **multilingual** | 107 | 0.186 | 0.338 | **0.401** |
| order-ops | 55 | 0.027 | 0.082 | 0.145 |
| association | 99 | 0.000 | 0.000 | 0.010 |

Raw data: [`results/eval_results.json`](./results/eval_results.json) · [`results/probe_readouts.json`](./results/probe_readouts.json)

### What the pattern means

Four evals with pass@10 in the 0.40–0.55 range and two near-zero — a pattern consistent with Neel Nanda's Qwen-27B replication ([external commentary PDF](https://www-cdn.anthropic.com/files/4zrzovbb/website/cc4be2488d65e54a6ed06492f8968398ddc18ebe.pdf)):

- **Cross-lingual, multi-hop, typo, and rhyme-planning representations are visible** mid-stream in the lens
- **Abstract associative reasoning (association eval) and operator-hierarchy reasoning (order-ops) are not** — plausibly capability-bound at 3B

## Limitations you must know before citing

**Read this before treating any specific number as replicating the paper.**

1. **Scoring is stricter than we'd like.** We tokenize the target concept plus case/space variants, keep only single-token variants, and score min-rank across layers. Multi-token concepts silently dropped — that's ~40% of order-ops items. Anthropic's scoring almost certainly uses richer criteria (semantic similarity, multi-token accumulation).

2. **No shuffled-corpus control.** We have not fit a lens on scrambled prompts to verify the readouts require semantic structure. Until that control is run, we cannot rule out "the vocabulary projection of any competent LM's residual stream looks like plausible next tokens" as the explanation for our probe results.

3. **No prompt-truncation ablations.** If we cut Probe 1 at "The number of legs on" (before "webs"), does `spider` still emerge at L12–26? If yes, the lens is broken. This was not tested.

4. **n=25 background prompts** — 40× less than the paper. Neel Nanda's ablation shows this is fine, but we have wider variance than Anthropic.

5. **Poetry (0.551) surprisingly beats Neel Nanda's Qwen-27B replication** where poetry *failed*. This is almost certainly a methodology artifact — likely Qwen 2.5's training-data distribution combined with our single-token filter selecting easier items — not a "small model outperforms big model" discovery. Do not cite this comparison.

6. **Only one target layer fit** (penultimate). The paper triangulates the workspace band by fitting multiple target layers.

## What this replication does and doesn't establish

**Establishes:**
- The J-lens methodology runs to completion on a sub-10B model on free consumer compute
- Qwen 2.5-3B has intermediate representations that project to task-relevant tokens in vocabulary space, in a wide layer band, matching the paper's qualitative claim
- The pattern of which evals show strong vs. weak signal roughly matches Neel Nanda's larger-scale replication

**Does not establish:**
- That the specific numbers here are directly comparable to Anthropic's or Neel Nanda's
- That the phenomenon is causally the workspace mechanism the paper describes (would need the shuffled control + ablation set)
- Any scaling claim between our results and larger-scale replications

## Reproducing this

```bash
git clone https://github.com/amaljithkuttamath/jlens-replication.git
cd jlens-replication
# option A: Kaggle T4 (paste kaggle/kaggle_run.ipynb into a fresh notebook)
# option B: Colab T4 (open notebooks/02_fit_colab.ipynb)
# option C: Lightning L4 (python notebooks/03_fit_lightning.py)
```

Total cost: **$0**. Wall time: **~30 min on L4, ~4 h on T4** (my run used conservative batching; you can go higher).

## Files in this repo

- [`kaggle/kaggle_run.ipynb`](./kaggle/kaggle_run.ipynb) — the notebook that produced these results
- [`kaggle/rescore_qwen.ipynb`](./kaggle/rescore_qwen.ipynb) — auditing rescore
- [`results/eval_results.json`](./results/eval_results.json) — full pass@k for all 6 evals
- [`results/probe_readouts.json`](./results/probe_readouts.json) — layer-by-layer top-5 for 3 probes
- [`results/rescore_v2.json`](./results/rescore_v2.json) — audit results

## Fitted lens artifact

The `qwen2.5-3b-jlens.pt` file (~285 MB) lives on the Kaggle notebook artifacts, not in this git repo (too large). To download, pull it from the [Kaggle notebook output](https://www.kaggle.com/code/amaljithkuttamath/jacobian-lens-qwen-replication) or fit your own using the code above.

## License

Apache-2.0 for this code. The lens artifact itself derives from Qwen 2.5-3B-Instruct (Apache-2.0 base) — no additional terms.
