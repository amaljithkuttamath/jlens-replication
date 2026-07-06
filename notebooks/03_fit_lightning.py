"""J-lens fit on Lightning AI Studio (L4 24 GB, free credits).

Steps:
    1. Sign in at https://lightning.ai (verify phone → 15 free credits/mo ≈ 80 GPU-hours)
    2. Create a new Studio → attach L4 GPU
    3. Upload this file (or `git clone` the repo)
    4. Run: python 03_fit_lightning.py

Outputs to ./out/:
    - qwen2.5-3b-jlens.pt
    - eval_results.json
    - probe_readouts.json

Expected runtime on L4 24 GB: ~15-25 min (vs 25-40 on Colab T4).
"""
from __future__ import annotations

import json
import pathlib
import subprocess
import sys
import time


def _install() -> None:
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-q",
        "git+https://github.com/anthropics/jacobian-lens.git",
        "datasets", "safetensors",
        "--upgrade", "transformers>=5.5",
    ])


def main() -> None:
    _install()

    import torch, transformers, jlens
    from datasets import load_dataset

    assert torch.cuda.is_available(), "No GPU detected."
    print(f"GPU: {torch.cuda.get_device_name(0)}  torch={torch.__version__}")

    MODEL = "Qwen/Qwen2.5-3B-Instruct"
    N_PROMPTS = 25
    OUT_DIR = pathlib.Path("out")
    OUT_DIR.mkdir(exist_ok=True)
    OUT_LENS = OUT_DIR / "qwen2.5-3b-jlens.pt"

    print(f"Loading {MODEL} ...")
    hf_model = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    ).to("cuda")
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL)
    model = jlens.from_hf(hf_model, tokenizer, compile=True)
    print(f"  n_layers={model.n_layers}  d_model={model.d_model}")

    print("Loading corpus ...")
    ds = load_dataset("NeelNanda/pile-10k", split="train")
    prompts = []
    for row in ds:
        if isinstance(row["text"], str) and len(row["text"]) > 200:
            prompts.append(row["text"])
            if len(prompts) >= N_PROMPTS:
                break
    print(f"  {len(prompts)} prompts")

    print("Fitting J-lens ...")
    t0 = time.time()
    lens = jlens.fit(
        model, prompts=prompts,
        target_layer=-2, dim_batch=8,   # L4 has more VRAM than T4
        max_seq_len=128, skip_first=4,
        checkpoint_path=str(OUT_DIR / "ckpt.pt"),
    )
    print(f"  Fit done in {(time.time()-t0)/60:.1f} min")
    lens.save(str(OUT_LENS))

    # Clone the eval data
    if not pathlib.Path("jl_data").exists():
        subprocess.check_call(["git", "clone", "--depth=1",
                               "https://github.com/anthropics/jacobian-lens.git", "jl_data"])

    # Probes
    probes = [
        "The number of legs on the animal that spins webs is",
        "The capital of the country shaped like a boot is",
        "Fact: The currency used in the country whose flag has a red maple leaf is the",
    ]
    probe_out = {}
    for prompt in probes:
        lens_logits, _, _ = lens.apply(model, prompt, positions=[-2])
        per_layer = {}
        for layer in sorted(lens_logits):
            top = lens_logits[layer][0].topk(5).indices.tolist()
            per_layer[layer] = [tokenizer.decode([t]).strip() for t in top]
        probe_out[prompt] = per_layer
    (OUT_DIR / "probe_readouts.json").write_text(json.dumps(probe_out, indent=2))

    # Evals
    EVALS_DIR = pathlib.Path("jl_data/data/evaluations")
    K_VALUES = (1, 5, 10)

    def tokens_of(word: str):
        ids = []
        for v in {word, " " + word, word.lower(), " " + word.lower()}:
            e = tokenizer(v, add_special_tokens=False).input_ids
            if len(e) == 1:
                ids.append(e[0])
        return sorted(set(ids))

    def min_rank(lens_logits, token_ids):
        best = 10**9
        for _, logits in lens_logits.items():
            order = logits[0].argsort(descending=True)
            rank_lookup = torch.empty_like(order)
            rank_lookup[order] = torch.arange(order.numel(), device=order.device)
            cand = rank_lookup[torch.tensor(token_ids, device=order.device)]
            r = int(cand.min().item()) + 1
            if r < best:
                best = r
        return best

    results = []
    for path in sorted(EVALS_DIR.glob("lens-eval-*.json")):
        items = json.loads(path.read_text())["items"]
        per_item = {k: [] for k in K_VALUES}
        for item in items:
            try:
                lens_logits, _, _ = lens.apply(model, item["prompt"], positions=[-1])
            except Exception:
                continue
            inters = item["intermediates"] if isinstance(item["intermediates"], list) else [item["intermediates"]]
            hits = {k: 0 for k in K_VALUES}
            total = 0
            for inter in inters:
                key = inter if isinstance(inter, str) else (
                    inter.get("synonyms", [inter.get("key", "")])[0]
                    if isinstance(inter, dict) else inter[0]
                )
                tok_ids = tokens_of(str(key))
                if not tok_ids:
                    continue
                r = min_rank(lens_logits, tok_ids)
                for k in K_VALUES:
                    if r <= k:
                        hits[k] += 1
                total += 1
            if total:
                for k in K_VALUES:
                    per_item[k].append(hits[k] / total)
        result = {
            "eval": path.stem,
            "n_items_scored": len(per_item[1]),
            "pass_at_k": {f"pass@{k}": sum(per_item[k]) / max(1, len(per_item[k])) for k in K_VALUES},
        }
        results.append(result)
        print(path.name, "→", result["pass_at_k"])

    (OUT_DIR / "eval_results.json").write_text(
        json.dumps({"model": MODEL, "n_prompts": N_PROMPTS, "results": results}, indent=2)
    )
    print(f"\nAll outputs in {OUT_DIR.resolve()}/")


if __name__ == "__main__":
    main()
