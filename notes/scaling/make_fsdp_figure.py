"""Draw fsdp_sharding.png: how ZeRO-3 / FSDP shards model state across GPUs.

Numbers use the published GPT-2 XL size (~1.5B parameters) in float32 with AdamW:
parameters 4 B + gradients 4 B + AdamW m and v 8 B = 16 bytes per parameter = 24 GB, or 3 GB per GPU on 8 GPUs.
usage: python make_fsdp_figure.py   (needs matplotlib)
"""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

COLORS = ["#2e86de", "#e67e22", "#27ae60", "#8e44ad"]
P = 1.5e9                                   # parameters (GPT-2 XL, published size)
GB = 1e9
fixed = {"parameters": 4 * P / GB, "gradients": 4 * P / GB, "AdamW m, v": 8 * P / GB}
N_GPU_BAR = 8

fig = plt.figure(figsize=(19.2, 14.4))
fig.suptitle("Fully sharded data parallel (ZeRO-3 / FSDP): no GPU ever holds the whole model", fontsize=18, weight="bold", y=0.985)

def gpu_panel(ax, g, gather_layer=None):
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
    ax.add_patch(FancyBboxPatch((0.1, 0.4), 9.8, 9.3, boxstyle="round,pad=0.02,rounding_size=0.15", fc="#f6f6f6", ec="#888", lw=1.5))
    ax.text(5, 9.1, f"GPU {g}", ha="center", fontsize=14, weight="bold")
    for li in range(4):
        y = 7.0 - li * 1.8
        ax.text(0.9, y + 0.5, f"L{li + 1}", ha="center", va="center", fontsize=11, color="#444")
        for s in range(4):
            x = 1.8 + s * 1.85
            if s == g:
                ax.add_patch(Rectangle((x, y), 1.65, 1.0, fc=COLORS[s], ec="white"))
            elif gather_layer == li:
                ax.add_patch(Rectangle((x, y), 1.65, 1.0, fc=COLORS[s], alpha=0.35, ec="white"))
            else:
                ax.add_patch(Rectangle((x, y), 1.65, 1.0, fc="none", ec="#ccc", ls=":"))
        if gather_layer == li:
            ax.add_patch(Rectangle((1.6, y - 0.15), 7.65, 1.3, fc="none", ec="crimson", lw=3))
    ax.text(5, -0.1, "own slice: 1/4 of every layer", ha="center", fontsize=10, color="#555")

fig.text(0.04, 0.945, "1. At rest: each GPU stores only its own 1/4 slice (solid colour) of every layer's weights, gradients and AdamW m, v",
         fontsize=13.5, weight="bold")
fig.text(0.04, 0.63, "2. Computing layer L2 (forward or backward): ALL-GATHER its 4 slices so every GPU briefly holds the full layer "
         "(faded = borrowed), then free them", fontsize=13.5, weight="bold")
for g in range(4):
    gpu_panel(fig.add_axes([0.04 + g * 0.24, 0.66, 0.22, 0.27]), g)
    gpu_panel(fig.add_axes([0.04 + g * 0.24, 0.345, 0.22, 0.27]), g, gather_layer=1)

ax = fig.add_axes([0.085, 0.06, 0.40, 0.23])
greys = ["#555", "#999", "#ccc"]
for row, (label, div) in enumerate([("no sharding\n(1 GPU's copy)", 1), (f"ZeRO-3 / FSDP\n{N_GPU_BAR} GPUs", N_GPU_BAR)]):
    left = 0
    for (name, gb), c in zip(fixed.items(), greys):
        ax.barh(row, gb / div, left=left, color=c, ec="#777", height=0.8, label=name if row == 0 else None)
        left += gb / div
    ax.text(left + 0.7, row, f"{left:.0f} GB", va="center", fontsize=13, weight="bold")
ax.set_yticks([0, 1]); ax.set_yticklabels(["no sharding\n(1 GPU's copy)", f"ZeRO-3 / FSDP\n{N_GPU_BAR} GPUs"], fontsize=12)
ax.axvline(80, color="crimson", ls="--", lw=2); ax.text(79, 1.45, "H100: 80 GB", color="crimson", ha="right", fontsize=11)
ax.set_xlim(0, 85); ax.set_ylim(-0.5, 1.6)
ax.set_xlabel("GB per GPU (1.5B-parameter model, float32, before activations)", fontsize=12)
ax.set_title("3. Model state per GPU", fontsize=15, weight="bold", loc="left")
ax.legend(loc="lower right", fontsize=11)

fig.text(0.56, 0.27, "4. One training step, per layer", fontsize=15, weight="bold")
steps = [("forward", "all-gather layer → compute on own micro-batch → free"),
         ("backward", "all-gather layer → compute gradients → free"),
         ("", "reduce-scatter grads: each GPU keeps the SUM for its slice only"),
         ("optimizer", "each GPU updates only its own slice (no communication)")]
for i, (k, v) in enumerate(steps):
    fig.text(0.56, 0.225 - i * 0.045, k, fontsize=13, weight="bold")
    fig.text(0.65, 0.225 - i * 0.045, v, fontsize=13)
fig.text(0.56, 0.045, "Trade-off: memory ÷ N GPUs, but every layer's weights cross the network\n"
         "twice per step → fast interconnects, and overlap gathering layer i+1 with computing layer i.", fontsize=12, color="#444")
fig.savefig(__file__.replace("make_fsdp_figure.py", "fsdp_sharding.png"), dpi=100)
print("saved fsdp_sharding.png")
