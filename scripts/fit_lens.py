"""Fit a Jacobian lens on a HuggingFace decoder model.

Usage:
    python scripts/fit_lens.py \
        --model Qwen/Qwen3.6-4B \
        --n 25 --skip-first 4 --target-layer -2 \
        --out out/qwen-3.6-4b-lens.pt

Designed to run in <1 hour on a single T4 for models up to ~4B.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import torch
import transformers
from datasets import load_dataset

import jlens
from jlens import fitting


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Fit a Jacobian lens on an HF decoder.")
    p.add_argument("--model", required=True, help="HF model id, e.g. Qwen/Qwen3.6-4B")
    p.add_argument("--corpus", default="NeelNanda/pile-10k", help="HF dataset for the background corpus")
    p.add_argument("--corpus-split", default="train")
    p.add_argument("--corpus-text-field", default="text")
    p.add_argument("--n", type=int, default=25, help="Number of prompts (Neel used 25; paper uses 1000)")
    p.add_argument("--max-seq-len", type=int, default=128)
    p.add_argument("--skip-first", type=int, default=4, help="Leading positions to drop (attention sinks)")
    p.add_argument("--target-layer", type=int, default=-2, help="Paper-faithful: penultimate block")
    p.add_argument("--dim-batch", type=int, default=4, help="Lower this if you OOM (4 for T4, 8 for L4, 16 for A100)")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--attn", default="sdpa", choices=["sdpa", "eager"],
                   help="Never use flash_attention_2 — it breaks batched autograd.")
    p.add_argument("--compile", action="store_true", help="torch.compile per block (bounds retained graph)")
    p.add_argument("--out", required=True, help="Output path for the fitted lens (.pt)")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        raise SystemExit("Fitting on CPU is not practical. Use a GPU runtime.")

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]

    print(f"[jlens] Loading {args.model} ({args.dtype}, attn={args.attn}) on {device}")
    hf_model = transformers.AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        attn_implementation=args.attn,
        low_cpu_mem_usage=True,
        token=os.environ.get("HF_TOKEN"),
    ).to(device)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model, token=os.environ.get("HF_TOKEN")
    )

    model = jlens.from_hf(hf_model, tokenizer, compile=args.compile)
    print(f"[jlens] Model: n_layers={model.n_layers}, d_model={model.d_model}")

    print(f"[jlens] Loading corpus: {args.corpus}")
    ds = load_dataset(args.corpus, split=args.corpus_split, streaming=False)
    # First n non-empty prompts, roughly filtered for length.
    prompts: list[str] = []
    for row in ds:
        text = row[args.corpus_text_field]
        if not isinstance(text, str) or len(text) < 200:
            continue
        prompts.append(text)
        if len(prompts) >= args.n:
            break
    print(f"[jlens] Using {len(prompts)} prompts, max_seq_len={args.max_seq_len}")

    t0 = time.time()
    lens = jlens.fit(
        model,
        prompts=prompts,
        target_layer=args.target_layer,
        dim_batch=args.dim_batch,
        max_seq_len=args.max_seq_len,
        skip_first=args.skip_first,
        checkpoint_path=str(out_path.with_suffix(".ckpt.pt")),
    )
    dt = time.time() - t0
    print(f"[jlens] Fit done in {dt/60:.1f} min. {lens!r}")

    lens.save(str(out_path))
    print(f"[jlens] Saved lens → {out_path}")
    if torch.cuda.is_available():
        peak_gb = torch.cuda.max_memory_allocated() / 1e9
        print(f"[jlens] Peak VRAM: {peak_gb:.1f} GB")


if __name__ == "__main__":
    main()
