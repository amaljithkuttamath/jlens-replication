"""Run the six lens-quality evaluations shipped in anthropics/jacobian-lens.

For each eval JSON (multihop, multilingual, poetry, order-of-ops, association,
typo), computes pass@k where k in {1, 5, 10} = mean over items of the
fraction of `intermediates` whose min-over-layers lens rank <= k, measured
at the eval-specified readout position.

Usage:
    python scripts/run_evals.py \
        --model Qwen/Qwen3.6-4B \
        --lens out/qwen-3.6-4b-lens.pt \
        --evals-dir jacobian-lens/data/evaluations \
        --out results/qwen-3.6-4b-evals.json
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import transformers

import jlens


READOUT_POSITION = {
    "lens-eval-multihop": -1,           # token immediately preceding `target`
    "lens-eval-multilingual": -1,
    "lens-eval-order-ops": -1,
    "lens-eval-poetry": -1,             # end of couplet line 1 (newline) — best-effort with -1
    "lens-eval-association": -1,        # final period
    "lens-eval-typo": -1,               # last tokenizer fragment
}
K_VALUES = (1, 5, 10)


def _min_rank_of_token_over_layers(
    lens_logits: dict[int, torch.Tensor], token_ids: list[int]
) -> int:
    """Given lens logits per layer at a single position, return min rank across layers
    and across the given synonym token_ids (rank 1 = best)."""
    best = None
    for _, logits in lens_logits.items():
        # logits shape: [batch=1, vocab]
        order = logits[0].argsort(descending=True)
        # rank of each candidate token id under this layer's readout
        # (positions in `order` where token_id appears)
        rank_lookup = torch.empty_like(order)
        rank_lookup[order] = torch.arange(order.numel(), device=order.device)
        cand_ranks = rank_lookup[torch.tensor(token_ids, device=order.device)]
        r = int(cand_ranks.min().item()) + 1  # 1-indexed
        best = r if best is None else min(best, r)
    return best if best is not None else 10**9


def _tokenize_synonyms(tok: transformers.PreTrainedTokenizer, words) -> list[int]:
    """Return single-token ids for each synonym (drop multi-token variants)."""
    if isinstance(words, str):
        words = [words]
    ids: list[int] = []
    for w in words:
        for variant in {w, " " + w, w.lower(), " " + w.lower()}:
            enc = tok(variant, add_special_tokens=False).input_ids
            if len(enc) == 1:
                ids.append(enc[0])
    return sorted(set(ids))


def eval_one_file(
    model: jlens.protocol.LensModel,
    lens: jlens.JacobianLens,
    path: Path,
) -> dict:
    with path.open() as f:
        payload = json.load(f)
    items = payload["items"]
    slug = path.stem
    read_pos = READOUT_POSITION.get(slug, -1)

    per_item_hits = {k: [] for k in K_VALUES}
    per_item_total = []

    for item in items:
        prompt = item["prompt"]
        intermediates = item["intermediates"]
        if not isinstance(intermediates, list):
            intermediates = [intermediates]

        lens_logits, _, _ = lens.apply(model, prompt, positions=[read_pos])
        # lens_logits is {layer: [1, vocab]}

        hit_counts = {k: 0 for k in K_VALUES}
        total = 0
        for concept in intermediates:
            # `concept` may be a string or a list of synonym strings/ids
            if isinstance(concept, dict):
                # order-of-ops style: {key, synonyms}
                synonyms = concept.get("synonyms", [concept.get("key", "")])
            else:
                synonyms = concept
            token_ids = _tokenize_synonyms(model.tokenizer, synonyms)
            if not token_ids:
                continue
            r = _min_rank_of_token_over_layers(lens_logits, token_ids)
            for k in K_VALUES:
                if r <= k:
                    hit_counts[k] += 1
            total += 1

        if total == 0:
            continue
        for k in K_VALUES:
            per_item_hits[k].append(hit_counts[k] / total)
        per_item_total.append(total)

    n = len(per_item_total) or 1
    return {
        "eval": slug,
        "n_items_scored": len(per_item_total),
        "pass_at_k": {
            f"pass@{k}": sum(per_item_hits[k]) / n for k in K_VALUES
        },
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--lens", required=True)
    p.add_argument("--evals-dir", required=True)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--attn", default="sdpa")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]

    hf_model = transformers.AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, attn_implementation=args.attn,
        low_cpu_mem_usage=True, token=os.environ.get("HF_TOKEN"),
    ).to(device)
    tok = transformers.AutoTokenizer.from_pretrained(args.model, token=os.environ.get("HF_TOKEN"))
    model = jlens.from_hf(hf_model, tok)
    lens = jlens.JacobianLens.load(args.lens)

    evals_dir = Path(args.evals_dir)
    results = []
    for path in sorted(evals_dir.glob("lens-eval-*.json")):
        print(f"[eval] {path.name}")
        results.append(eval_one_file(model, lens, path))
        print(f"       {results[-1]['pass_at_k']}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump({
            "model": args.model,
            "lens_file": args.lens,
            "n_prompts": lens.n_prompts,
            "d_model": lens.d_model,
            "results": results,
        }, f, indent=2)
    print(f"[eval] Wrote {out_path}")


if __name__ == "__main__":
    main()
