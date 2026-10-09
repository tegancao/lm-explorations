"""Evaluate my two small models and GPT-2 small on standard benchmarks, on Modal GPUs.

    uv run --with modal modal run experiments/2026-10-benchmark-eval/modal_eval.py

Uses EleutherAI's lm-evaluation-harness (zero-shot) on HellaSwag, LAMBADA (OpenAI version), PIQA and ARC-Easy.
One NVIDIA L4 per model, in parallel (about 10-15 minutes, under a dollar). Writes results.json next to this file.

Optional:
  WANDB_PROJECT=my-project ...   also log the results to Weights & Biases. Needs a Modal secret named "wandb"
                                 holding WANDB_API_KEY (modal secret create wandb WANDB_API_KEY=...).
  LIMIT=20 ...                   only the first 20 examples per task (a quick smoke test; not written to results.json)
"""
import json
import os
from pathlib import Path

import modal

TASKS = ["hellaswag", "lambada_openai", "piqa", "arc_easy"]
# max_length: the context the harness may feed the model. My models were trained on 256-token windows; the
# TinyStories tokenizer turns 4 PIQA answers into more than 256 tokens, so it gets 512 (only those inputs go past
# the trained length). GPT-2 small was trained on 1,024 tokens.
MODELS = {
    "tinystories-22m": ("tegancao/tinystories-22m", 512),
    "owt-45m": ("tegancao/owt-45m", 256),
    "gpt2-small": ("openai-community/gpt2", 1024),
}

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


@app.function(image=image, gpu="L4", timeout=60 * 60, secrets=secrets)
def evaluate(name: str, repo: str, max_length: int, wandb_project: str = "", limit: int | None = None) -> dict:
    import lm_eval
    out = lm_eval.simple_evaluate(model="hf", model_args=f"pretrained={repo},max_length={max_length}",
                                  tasks=TASKS, batch_size=64, device="cuda", limit=limit)
    if wandb_project:
        from lm_eval.loggers import WandbLogger
        wb = WandbLogger(init_args={"project": wandb_project, "name": name, "job_type": "eval"},
                         config_args={"model": repo, "max_length": max_length, "tasks": TASKS})
        wb.post_init(out)
        wb.log_eval_result()
        wb.run.finish()
    return {"model": repo, "max_length": max_length, "results": out["results"], "n_samples": out["n-samples"],
            "versions": {k: str(v) for k, v in out["versions"].items()}, "lm_eval_version": lm_eval.__version__}


@app.local_entrypoint()
def main():
    runs = list(evaluate.starmap([(n, r, m, WANDB_PROJECT, LIMIT) for n, (r, m) in MODELS.items()]))
    results = {n: run for n, run in zip(MODELS, runs)}
    if LIMIT is None:
        Path(__file__).with_name("results.json").write_text(json.dumps(results, indent=1))
    else:
        print(f"(smoke test with LIMIT={LIMIT}: results.json not written)")
    metrics = [("hellaswag", "acc_norm,none"), ("lambada_openai", "acc,none"), ("lambada_openai", "perplexity,none"),
               ("piqa", "acc_norm,none"), ("arc_easy", "acc_norm,none")]
    print(f"{'model':<16}" + "".join(f"{t + ' ' + m.split(',')[0]:>26}" for t, m in metrics))
    for n, run in results.items():
        r = run["results"]
        print(f"{n:<16}" + "".join(f"{r[t][m]:>26.4f}" for t, m in metrics))
