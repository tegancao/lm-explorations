"""Bar chart of results.json (accuracy per benchmark, with standard errors and chance level).

    uv run experiments/2026-10-benchmark-eval/plot_results.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
r = json.loads((HERE / "results.json").read_text())
BENCH = [("HellaSwag", "hellaswag", "acc_norm", 0.25), ("PIQA", "piqa", "acc_norm", 0.50),
         ("ARC-Easy", "arc_easy", "acc_norm", 0.25), ("LAMBADA", "lambada_openai", "acc", 0.0)]
MODELS = [("tinystories-22m", "TinyStories 22M (mine)", "#2e86de"), ("owt-45m", "OpenWebText 45M (mine)", "#e67e22"),
          ("gpt2-small", "GPT-2 small 124M", "#7f8c8d")]

fig, ax = plt.subplots(figsize=(11, 5))
x, w = np.arange(len(BENCH)), 0.26
for k, (key, label, color) in enumerate(MODELS):
    res = r[key]["results"]
    vals = [100 * res[t][f"{m},none"] for _, t, m, _ in BENCH]
    errs = [100 * res[t][f"{m}_stderr,none"] for _, t, m, _ in BENCH]
    bars = ax.bar(x + (k - 1) * w, vals, w, yerr=errs, capsize=3, color=color, label=label)
    ax.bar_label(bars, fmt="%.1f", fontsize=8, padding=2)
for i, (_, _, _, chance) in enumerate(BENCH):
    if chance:
        ax.hlines(100 * chance, i - 1.6 * w, i + 1.6 * w, colors="crimson", linestyles="--", lw=1.5,
                  label="random chance" if i == 0 else None)
ax.set_xticks(x, [f"{n}\n({'acc_norm' if m == 'acc_norm' else 'accuracy'})" for n, _, m, _ in BENCH])
ax.set_ylabel("accuracy (%)"); ax.set_ylim(0, 75); ax.legend(loc="upper right", fontsize=9)
ax.set_title("Zero-shot benchmark accuracy (lm-evaluation-harness, full test sets, ± standard error)", fontsize=12, weight="bold")
fig.tight_layout(); fig.savefig(HERE / "benchmarks.png", dpi=120)
print("saved benchmarks.png")
