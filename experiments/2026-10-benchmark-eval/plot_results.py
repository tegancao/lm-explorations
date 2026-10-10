"""Figures from results.json: scaling.png (each benchmark vs. Transformer body size), heatmap.png (every model x
benchmark, coloured by skill above chance) and blimp_phenomena.png (BLiMP by grammar phenomenon).

    uv run experiments/2026-10-benchmark-eval/plot_results.py

blimp_phenomena.json maps BLiMP's 67 sub-tests to their phenomenon (the 'linguistics_term' field of nyu-mll/blimp).
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
r = json.loads((HERE / "results.json").read_text())
TASKS = [("HellaSwag", "hellaswag", "acc_norm", 25), ("PIQA", "piqa", "acc_norm", 50), ("ARC-Easy", "arc_easy", "acc_norm", 25),
         ("ARC-Challenge", "arc_challenge", "acc_norm", 25), ("Winogrande", "winogrande", "acc", 50),
         ("LAMBADA", "lambada_openai", "acc", 0), ("BLiMP (grammar)", "blimp", "acc", 50),
         ("WikiText: bits per byte (lower is better)", "wikitext", "bits_per_byte", None)]
STYLE = {"mine": ("#c0392b", "*", 260), "TinyStories (Eldan & Li)": ("#2e86de", "o", 70),
         "Pythia (EleutherAI)": ("#27ae60", "s", 70), "GPT-2 (OpenAI)": ("#7f8c8d", "^", 80)}

from matplotlib.ticker import FuncFormatter

# 6 panels: ARC-Challenge and Winogrande are left out here (every model is near chance; see heatmap.png and the table)
PANELS = [t for t in TASKS if t[1] not in ("arc_challenge", "winogrande")]
YMAX = {"hellaswag": 45, "piqa": 70, "arc_easy": 50, "lambada_openai": 55, "blimp": 85}
SHORT = {"TinyStories (Eldan & Li)": "TinyStories", "Pythia (EleutherAI)": "Pythia", "GPT-2 (OpenAI)": "GPT-2"}
MINE = {"tinystories-22m": ("#c0392b", "*", 520), "owt-45m": ("#d35400", "D", 170)}
# one-line takeaway per panel, shown under the panel title (outside the plot area, so it never covers data)
NOTES = {
    "hellaswag": "Barely above chance up to ~30M body;\nonly Pythia and GPT-2 at 85M+ climb",
    "piqa": "TinyStories models (and mine) stay within ~3 points\nof chance; web-trained models are clearly above",
    "arc_easy": "My OWT model beats every TinyStories model:\nweb text contains some science facts",
    "lambada_openai": "My TinyStories model beats my OWT one (stories repeat\nnames), but trails the official TinyStories-8M",
    "blimp": "Grammar is learned early: my OWT model matches\nPythia-70M, trained on ~1,000x more tokens",
    "wikitext": "Fair across tokenizers: my OWT model sits between\nTinyStories (~3) and Pythia/GPT-2 (~1)",
}
plt.rcParams.update({"font.size": 12})
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
for ax, (title, task, metric, chance) in zip(axes.flat, PANELS):
    bpb = metric == "bits_per_byte"
    for fam, (color, marker, size) in STYLE.items():
        pts = sorted((run["params_body"], run["results"][task][f"{metric},none"] * (1 if bpb else 100), n)
                     for n, run in r.items() if run["family"] == fam)
        xs, ys, names = zip(*pts)
        if fam == "mine":
            for x, y, n in pts:   # distinct markers, explained once in the legend under the title
                c, mk, sz = MINE[n]
                ax.scatter([x], [y], color=c, marker=mk, s=sz, zorder=5, edgecolor="black", linewidth=0.8)
        else:
            ax.plot(xs, ys, color=color, lw=2.5, alpha=0.85, zorder=2)
            ax.scatter(xs, ys, color=color, marker=marker, s=110, zorder=3)
            ax.annotate(SHORT[fam], (xs[-1], ys[-1]), xytext=(8, 0), textcoords="offset points", va="center",
                        fontsize=11, color=color, weight="bold")
    if bpb:
        ax.set_ylim(0.6, 3.3)
        ax.set_ylabel("bits per byte  (lower is better)")
    else:
        ax.axhline(chance, color="crimson", ls="--", lw=1.5)
        ax.text(1.4e9, chance, "chance" if chance else "", color="crimson", va="bottom", ha="right", fontsize=10)
        ax.set_ylim(chance - 2 if chance else 0, YMAX[task])
        ax.set_ylabel("accuracy (%)")
    ax.set_xscale("log"); ax.set_xlim(2.5e5, 1.5e9)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / 1e6:g}M"))
    ax.set_title(title.replace(": bits per byte (lower is better)", " (language modeling)"), fontsize=15, weight="bold")
    ax.grid(alpha=0.3, which="major"); ax.set_xlabel("Transformer body size (log scale)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title(ax.get_title(), fontsize=15, weight="bold", pad=44)
    ax.text(0.5, 1.025, NOTES[task], transform=ax.transAxes, ha="center", va="bottom", fontsize=11,
            style="italic", color="#444")
fig.suptitle("Benchmarks vs. model size: my two models among TinyStories, Pythia and GPT-2", fontsize=17, weight="bold", y=0.995)
from matplotlib.lines import Line2D
handles = [Line2D([], [], color=MINE[n][0], marker=MINE[n][1], ls="", markersize=17 if MINE[n][1] == "*" else 12,
                  markeredgecolor="black", label=lab) for n, lab in [("tinystories-22m", "my TinyStories model (22M)"),
                                                                     ("owt-45m", "my OpenWebText model (45M)")]]
handles += [Line2D([], [], color=STYLE[f][0], marker=STYLE[f][1], lw=2.5, markersize=9, label=f)
            for f in ("TinyStories (Eldan & Li)", "Pythia (EleutherAI)", "GPT-2 (OpenAI)")]
fig.legend(handles=handles, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 0.955), fontsize=12, frameon=False)
fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.savefig(HERE / "scaling.png", dpi=110)
print("saved scaling.png")

# ---------- heatmap: skill above chance, models x tasks ----------
import numpy as np

order = sorted(r, key=lambda n: (r[n]["params_body"], n))
NICE = {"tinystories-22m": "mine: TinyStories 22M", "owt-45m": "mine: OpenWebText 45M"}
acc_tasks = TASKS[:-1]                                     # all accuracy tasks (not WikiText bits per byte)
raw = np.array([[100 * r[n]["results"][t][f"{m},none"] for _, t, m, _ in acc_tasks] for n in order])
chance = np.array([c for *_, c in acc_tasks])
skill = (raw - chance) / (100 - chance) * 100              # 0 = guessing, 100 = perfect
fig, ax = plt.subplots(figsize=(12, 7.5))
im = ax.imshow(np.clip(skill, 0, None), cmap="YlGn", vmin=0, vmax=70, aspect="auto")
for i in range(len(order)):
    for j in range(len(acc_tasks)):
        ax.text(j, i, f"{raw[i, j]:.1f}", ha="center", va="center", fontsize=10,
                color="white" if skill[i, j] > 45 else ("#999" if skill[i, j] < 2 else "black"),
                weight="bold" if r[order[i]]["family"] == "mine" else "normal")
ax.set_xticks(range(len(acc_tasks)), [f"{t}\n(chance {c}%)" if c else f"{t}\n(no chance level)" for t, _, _, c in
                                       [(a.replace(" (grammar)", ""), b, m, c) for a, b, m, c in acc_tasks]], fontsize=10)
ax.set_yticks(range(len(order)), [f"{NICE.get(n, n)}  ({r[n]['params_body'] / 1e6:.1f}M body)" for n in order], fontsize=10)
for i, n in enumerate(order):
    if r[n]["family"] == "mine":
        ax.get_yticklabels()[i].set_color("#c0392b"); ax.get_yticklabels()[i].set_weight("bold")
cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
cb.set_label("skill above chance: (accuracy − chance) / (100 − chance), %")
ax.set_title("Benchmark accuracy (%) for every model, coloured by how far it is above random guessing\n"
             "(models sorted by Transformer body size; grey numbers = at or near chance)", fontsize=12, weight="bold")
fig.tight_layout(); fig.savefig(HERE / "heatmap.png", dpi=110)
print("saved heatmap.png")

# ---------- BLiMP broken down by grammar phenomenon ----------
phen = json.loads((HERE / "blimp_phenomena.json").read_text())
names = sorted(set(phen.values()))
grid = np.array([[100 * np.mean([r[n]["results"][f"blimp_{s}"]["acc,none"] for s, p in phen.items() if p == ph])
                  for ph in names] for n in order])
col_order = np.argsort(-grid.mean(0))                     # easiest phenomena first
grid, names = grid[:, col_order], [names[k] for k in col_order]
counts = {ph: sum(p == ph for p in phen.values()) for ph in names}
fig, ax = plt.subplots(figsize=(15, 7.5))
im = ax.imshow(grid, cmap="RdYlGn", vmin=40, vmax=100, aspect="auto")
for i in range(len(order)):
    for j in range(len(names)):
        ax.text(j, i, f"{grid[i, j]:.0f}", ha="center", va="center", fontsize=10,
                weight="bold" if r[order[i]]["family"] == "mine" else "normal")
ax.set_xticks(range(len(names)), [f"{ph.replace('_', ' ')}\n({counts[ph]} tests)" for ph in names], rotation=35, ha="right", fontsize=10)
ax.set_yticks(range(len(order)), [f"{NICE.get(n, n)}  ({r[n]['params_body'] / 1e6:.1f}M body)" for n in order], fontsize=10)
for i, n in enumerate(order):
    if r[n]["family"] == "mine":
        ax.get_yticklabels()[i].set_color("#c0392b"); ax.get_yticklabels()[i].set_weight("bold")
cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02); cb.set_label("accuracy (%), chance = 50")
ax.set_title("BLiMP by grammar phenomenon: which grammar each model has learned\n"
             "(accuracy on minimal pairs, e.g. 'The cats are / *is sleeping'; phenomena sorted from easiest to hardest)",
             fontsize=12, weight="bold")
fig.tight_layout(); fig.savefig(HERE / "blimp_phenomena.png", dpi=110)
print("saved blimp_phenomena.png")
