# J-lens replication on small open-weight models

A minimal, free-compute replication of Anthropic's July 2026 paper
**["Verbalizable Representations Form a Global Workspace in Language Models"](https://transformer-circuits.pub/2026/workspace/index.html)**
(blog post: [A global workspace in language models](https://www.anthropic.com/research/global-workspace)).

This repo fits the **Jacobian lens (J-lens)** on small open-weight decoders using free notebook GPUs (Colab T4, Kaggle T4×2, or Lightning L4), then runs the six lens-quality evaluations released by Anthropic. Everything here is designed to run in under one free-tier session (≤12 h) and stay within 16 GB VRAM.

Prior art you should know about before starting:

- **Neel Nanda / MATS** replicated the core claims on **Qwen 3.6-27B** ([external commentary PDF](https://www-cdn.anthropic.com/files/4zrzovbb/website/cc4be2488d65e54a6ed06492f8968398ddc18ebe.pdf)) — verbal report, directed modulation, multilingual, typo all replicate; poetry and arithmetic failed on Qwen 27B.
- Prefit Jacobians for Qwen 3.6-27B are on the Hub: [camilablank/qwen3.6-27b-pile-n25-skip4-penultimate-jacobians](https://huggingface.co/camilablank/qwen3.6-27b-pile-n25-skip4-penultimate-jacobians) and [agu18dec/qwen3.6-27b-jlens](https://huggingface.co/agu18dec/qwen3.6-27b-jlens).
- **No public J-lens exists for any model ≤ 10B, for any Llama/Gemma/Mistral/Phi, or for any NVIDIA Nemotron model** — that is the gap this repo targets.

Reference code: [anthropics/jacobian-lens](https://github.com/anthropics/jacobian-lens) (Apache 2.0). The `jlens.hf` module's `_LAYOUTS` table already covers Llama, Qwen, Mistral, Gemma, OLMo, StableLM, Phi, GPT-2, and Pythia out of the box — no adapter code needed.

---

## What "replication on small models" means here

The J-lens is a per-layer linear map `J_l = E[∂h_final / ∂h_l]` averaged over a background text corpus. Applying it at any layer gives a ranked list of vocabulary tokens the residual is *disposed* to make the model say. Fitting cost is `O(n · d_model)` backward passes.

Anthropic used **n=1000** prompts. Neel Nanda's replication reports that **n=10 is almost as good and n=1 is respectable** — his main run used **n=25**. That collapse makes free-tier replication realistic.

### Target models (all fit on a single 16 GB T4)

| Model | Params | d_model | HF id | Notes |
|---|---|---|---|---|
| Llama-3.2-3B | 3B | 3072 | `meta-llama/Llama-3.2-3B` | Gated — needs HF token |
| Gemma-3-4B | 4B | 2560 | `google/gemma-3-4b-pt` | Gemma license; multimodal but text-only works |
| Qwen 3.6-4B | 4B | 2560 | `Qwen/Qwen3.6-4B` | Apache 2.0, easiest starting point |
| Phi-4-mini | 3.8B | 3072 | `microsoft/Phi-4-mini-instruct` | Phi layout in `_LAYOUTS` |
| Nemotron-Nano-9B | 9B | 4096 | `nvidia/Nemotron-Nano-9B-v2` | Needs 4-bit or T4×2; NVIDIA's small open model |

Anything ≤ 4B fp16 fits comfortably on a single T4. 8-9B models want either Kaggle's T4×2 or 4-bit loading; L4/L40S on Lightning removes the constraint entirely.

---

## Free-compute options (2026)

| Platform | GPU | VRAM | Session cap | Weekly quota | Best for |
|---|---|---|---|---|---|
| **Kaggle** | T4 or T4×2 or P100 | 16 GB / 32 GB / 16 GB | 9-12 h | ~30 h | Primary target — reliable T4×2 fits 9B fp16 |
| **Google Colab** free | T4 | 15 GB | 12 h | 15-30 h dynamic | Backup; 90-min idle timeout kills unattended runs |
| **Lightning AI** free | L4 24 GB, A100, H100, H200 (spot) | 24-141 GB | Studio 4 h restart | ~80 h/mo on spot | Best VRAM per free hour once phone-verified |
| **HF Spaces (ZeroGPU)** | H200 slice | 70 GB | 60 s per call (extendable) | 3.5 min/day free | Publishing the interactive readout demo, not fitting |
| **NVIDIA Build API** | Hosted inference only | n/a | n/a | Free credits | No good for fitting — needs backward pass access |

**Recommendation:** fit on **Kaggle T4×2** (most reliable free 32 GB), publish the resulting lens to a HF model repo, then host a **ZeroGPU Space** that loads the lens and lets people run the readout demo.

---

## Recipe (paper-faithful, cheap variant)

Following Neel Nanda's confirmed-working configuration:

| Knob | Value | Source |
|---|---|---|
| Background corpus | `NeelNanda/pile-10k` | Neel Nanda replication |
| n prompts | **25** (start), scale to 100 if time allows | Paper §9.3: quality saturates fast |
| max_seq_len | 128 tokens | Paper default |
| `skip_first` | 4 (Neel) or 16 (Anthropic default) | See `jlens.fitting.SKIP_FIRST_N_POSITIONS` |
| `target_layer` | `-2` (penultimate block) | Paper-faithful; better-conditioned |
| `dim_batch` | 4 on T4 fp16; 8 on L4/L40S; 16 on A100+ | Trades VRAM for wall time |

Rough fitting time on n=25 prompts:

| Model | GPU | Time |
|---|---|---|
| Gemma-3-4B | T4 16 GB | ~25-40 min |
| Qwen 3.6-4B | T4 16 GB | ~25-40 min |
| Llama-3.2-3B | T4 16 GB | ~20-30 min |
| Nemotron-Nano-9B (4-bit) | T4 16 GB | ~60-90 min |
| Nemotron-Nano-9B (fp16) | Kaggle T4×2 | ~45 min |

Neel Nanda measured Qwen3.5-397B-A17B at **~1 hour for n=4 on 8×H200** — extrapolating downward, small models on a single T4 with n=25 is well within a Kaggle session.

---

## Repository layout

```
jlens-replication/
├── README.md                     # this file
├── notebooks/
│   ├── 01_fit_kaggle.ipynb       # Kaggle T4×2 entry point (recommended)
│   ├── 02_fit_colab.ipynb        # Colab T4 entry point
│   └── 03_run_evals.ipynb        # runs the 6 lens-quality evals
├── scripts/
│   ├── fit_lens.py               # CLI: python -m scripts.fit_lens --model ... --n 25
│   ├── run_evals.py              # CLI: runs data/evaluations/* on a fitted lens
│   └── run_experiments.py        # optional: probe-swap, verbal-report, etc.
├── configs/
│   ├── gemma-3-4b.yaml
│   ├── qwen-3.6-4b.yaml
│   ├── llama-3.2-3b.yaml
│   ├── phi-4-mini.yaml
│   └── nemotron-nano-9b.yaml
├── results/                      # eval outputs go here; commit the JSONs
└── requirements.txt
```

---

## Quickstart (Kaggle T4×2)

1. Create a new Kaggle notebook, set accelerator to **T4×2**, internet **on**.
2. Add your HuggingFace token as a Kaggle secret named `HF_TOKEN` (needed only for gated models like Llama).
3. First cell:

```python
!git clone https://github.com/anthropics/jacobian-lens.git
!pip -q install -e ./jacobian-lens datasets safetensors
!git clone https://github.com/<YOUR_USERNAME>/jlens-replication.git
%cd jlens-replication
```

4. Fit the lens (n=25, ~30-45 min on a 4B model):

```python
!python scripts/fit_lens.py \
    --model Qwen/Qwen3.6-4B \
    --n 25 --skip-first 4 --target-layer -2 \
    --out out/qwen-3.6-4b-lens.pt
```

5. Run the six lens-quality evaluations:

```python
!python scripts/run_evals.py \
    --model Qwen/Qwen3.6-4B \
    --lens out/qwen-3.6-4b-lens.pt \
    --evals-dir jacobian-lens/data/evaluations \
    --out results/qwen-3.6-4b-evals.json
```

6. Push the fitted lens to a HuggingFace model repo so others (and your ZeroGPU demo) can load it via `JacobianLens.from_pretrained("<you>/qwen-3.6-4b-jlens")`.

---

## What to expect (calibrated from Neel Nanda's replication)

| Experiment | Expected on 3-9B models | Notes |
|---|---|---|
| Multilingual (probe + causal) | **Should replicate** | Cleanly replicated on Qwen 27B |
| Typo detection | **Should replicate** | Cleanly replicated on Qwen 27B |
| Verbal report (swap) | **Weak but positive** | Same effect direction as Anthropic |
| Directed modulation | **Moderate** | Weaker on smaller models |
| Association | **Weak scores, right qualitatively** | Dataset-limited |
| Multihop factual recall | **Partial** | Depends on whether small model knows the facts |
| Poetry rhyme planning | **Likely to fail** | Failed even on Qwen 27B — needs capable models |
| Arithmetic | **Likely to fail** | Same reason |

Failure of poetry/arithmetic on a small model is not a bug — it is itself a finding: **the workspace-mediated higher-order behaviors depend on model capability**, matching Anthropic's ablation showing multi-step reasoning collapses when the J-space is removed.

---

## Deliverables to publish

1. **HF model repo** with `lens.pt` (or `jacobians.safetensors`) + a `README.md` documenting the recipe, following [agu18dec/qwen3.6-27b-jlens](https://huggingface.co/agu18dec/qwen3.6-27b-jlens) as a template.
2. **A results JSON** in `results/` with pass@k for each of the six lens-quality evals per model.
3. **A short write-up** (blog or arXiv note) comparing lens quality across model families/sizes — this is genuinely novel work; nobody has done a cross-family scaling study on the J-lens yet.
4. **Optional: a ZeroGPU Space** hosting the slice visualization notebook from `jacobian-lens/walkthrough.ipynb`.

---

## Caveats

- **Flash-attention breaks batched autograd.** Neel Nanda's replication note is explicit: *"No flash-linear-attention (its kernels break batched autograd)."* Load with `attn_implementation="sdpa"` or `"eager"`, never `"flash_attention_2"`.
- **BOS token matters.** `HFLensModel(force_bos=True)` (the default) sets `tokenizer.add_bos_token=True`. Do not override unless you know why.
- **The lens only reads single-token concepts.** Multi-word concepts show up in fragments; expect noise.
- **Compile helps VRAM headroom, not throughput.** `HFLensModel(..., compile=True)` bounds the retained autograd graph per block, which is what lets 4B models fit on a T4 with `dim_batch=4-8`.
- **Do not run poetry evals on ≤4B models** and treat them as evidence for or against the paper. They are known to be capability-bound.

---

## License

The upstream code is Apache 2.0. This replication scaffold is Apache 2.0 as well. Model weights and datasets you pull follow their own licenses (Llama Community License, Gemma Terms of Use, Qwen Apache 2.0, Phi MIT, Nemotron NVIDIA Open Model License).
