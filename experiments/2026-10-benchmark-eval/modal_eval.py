"""Evaluate my two small models and reference models of other sizes on standard benchmarks, on Modal GPUs.

    uv run --with modal modal run experiments/2026-10-benchmark-eval/modal_eval.py

Uses EleutherAI's lm-evaluation-harness (zero-shot) on HellaSwag, PIQA, ARC-Easy, ARC-Challenge, Winogrande, LAMBADA
(OpenAI version), BLiMP (grammar) and WikiText (bits per byte). One NVIDIA L4 per model, up to 8 in parallel (about an
hour, a few dollars). Writes results.json next to this file.

Optional:
  WANDB_PROJECT=my-project ...   also log the results to Weights & Biases. Needs a Modal secret named "wandb"
                                 holding WANDB_API_KEY (modal secret create wandb WANDB_API_KEY=...).
  LIMIT=20 ...                   only the first 20 examples per task (a quick smoke test; not written to results.json)
"""
import json
import os
from pathlib import Path

import modal

TASKS = ["hellaswag", "piqa", "arc_easy", "arc_challenge", "winogrande", "lambada_openai", "blimp", "wikitext"]
# max_length: the context the harness may feed the model. My models were trained on 256-token windows; the
# TinyStories tokenizer turns 4 PIQA answers into more than 256 tokens, so it gets 512 (only those inputs go past
# the trained length). GPT-2 small was trained on 1,024 tokens.
# name: (Hugging Face repo, max_length or None for the model's own, family)
MODELS = {
    "tinystories-22m": ("tegancao/tinystories-22m", 512, "mine"),
    "owt-45m": ("tegancao/owt-45m", 256, "mine"),
    "TinyStories-1M": ("roneneldan/TinyStories-1M", None, "TinyStories (Eldan & Li)"),
    "TinyStories-3M": ("roneneldan/TinyStories-3M", None, "TinyStories (Eldan & Li)"),
    "TinyStories-8M": ("roneneldan/TinyStories-8M", None, "TinyStories (Eldan & Li)"),
    "TinyStories-33M": ("roneneldan/TinyStories-33M", None, "TinyStories (Eldan & Li)"),
    "pythia-70m": ("EleutherAI/pythia-70m", None, "Pythia (EleutherAI)"),
    "pythia-160m": ("EleutherAI/pythia-160m", None, "Pythia (EleutherAI)"),
    "pythia-410m": ("EleutherAI/pythia-410m", None, "Pythia (EleutherAI)"),
    "gpt2-small": ("openai-community/gpt2", None, "GPT-2 (OpenAI)"),
    "gpt2-medium": ("openai-community/gpt2-medium", None, "GPT-2 (OpenAI)"),
}
EMBEDDING_NAMES = ("embed_tokens", "lm_head", "wte", "wpe", "embed_in", "embed_out")

WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "")
LIMIT = int(os.environ.get("LIMIT", "0")) or None

app = modal.App("lm-explorations-benchmarks")
image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "lm-eval==0.4.13", "accelerate", "transformers==5.19.0", "torch", "datasets", "wandb")
# The W&B secret is only attached when WANDB_PROJECT is set, so the script runs without a W&B account. The setting
# is baked into the image so the container sees the same value (Modal requires identical dependencies on both sides).
if WANDB_PROJECT:
    image = image.env({"WANDB_PROJECT": WANDB_PROJECT})
secrets = [modal.Secret.from_name("wandb")] if WANDB_PROJECT else []


# at most 8 GPUs at once (accounts have a concurrent-GPU limit, e.g. 10); the remaining models wait for a free GPU
@app.function(image=image, gpu="L4", timeout=3 * 60 * 60, secrets=secrets, max_containers=8)
def evaluate(name: str, repo: str, max_length: int | None, family: str, wandb_project: str = "",
             limit: int | None = None) -> dict:
    import lm_eval
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(repo)
    total = sum(p.numel() for p in m.parameters())
    embedding = sum(p.numel() for n, p in m.named_parameters() if any(k in n for k in EMBEDDING_NAMES))
    del m
    model_args = f"pretrained={repo}" + (f",max_length={max_length}" if max_length else "")
    # "auto" picks the largest batch that fits in GPU memory; a fixed 64 runs out of memory on WikiText for models
    # with a 2,048-token context (64 x 2048 x 50257 output scores alone are ~26 GB)
    out = lm_eval.simple_evaluate(model="hf", model_args=model_args, tasks=TASKS, batch_size="auto", device="cuda",
                                  limit=limit)
    if wandb_project:
        from lm_eval.loggers import WandbLogger
        wb = WandbLogger(init_args={"project": wandb_project, "name": name, "job_type": "eval"},
                         config_args={"model": repo, "family": family, "max_length": max_length, "tasks": TASKS,
                                      "params_total": total, "params_body": total - embedding})
        wb.post_init(out)
        wb.log_eval_result()
        wb.run.finish()
    return {"model": repo, "family": family, "params_total": total, "params_embedding": embedding,
            "params_body": total - embedding, "max_length": max_length, "results": out["results"],
            "n_samples": out["n-samples"],
            "versions": {k: str(v) for k, v in out["versions"].items()}, "lm_eval_version": lm_eval.__version__}


@app.local_entrypoint()
def main():
    # return_exceptions: one model failing (e.g. out of memory) doesn't cancel the others
    runs = list(evaluate.starmap([(n, r, m, f, WANDB_PROJECT, LIMIT) for n, (r, m, f) in MODELS.items()],
                                 return_exceptions=True))
    failed = {n: repr(run) for n, run in zip(MODELS, runs) if isinstance(run, BaseException)}
    results = {n: run for n, run in zip(MODELS, runs) if not isinstance(run, BaseException)}
    for n, err in failed.items():
        print(f"FAILED {n}: {err[:300]}")
    if LIMIT is None:
        Path(__file__).with_name("results.json").write_text(json.dumps(results, indent=1))
    else:
        print(f"(smoke test with LIMIT={LIMIT}: results.json not written)")
    metrics = [("hellaswag", "acc_norm,none"), ("piqa", "acc_norm,none"), ("arc_easy", "acc_norm,none"),
               ("arc_challenge", "acc_norm,none"), ("winogrande", "acc,none"), ("lambada_openai", "acc,none"),
               ("blimp", "acc,none"), ("wikitext", "bits_per_byte,none")]
    print(f"{'model':<16}{'body':>8}" + "".join(f"{t[:9]:>10}" for t, _ in metrics))
    for n, run in results.items():
        r = run["results"]
        print(f"{n:<16}{run['params_body'] / 1e6:7.1f}M" + "".join(f"{r.get(t, {}).get(m, float('nan')):10.3f}" for t, m in metrics))
