"""Token-embedding experiments on two small models trained from scratch.

    uv run experiments/02-token-embeddings/embeddings.py

Models: tegancao/tinystories-22m and tegancao/owt-45m (Hugging Face). Token frequencies are counted on each
dataset's public validation split. Writes figures and results.json next to this script.

  A. 2D maps of word groups (PCA and t-SNE), with a silhouette score for how well the groups separate
  B. Anisotropy: how similar random tokens look, and what removing the shared directions does
  C. Rarely trained tokens: embedding norm vs. how often the token occurs
  D. Analogies: a - b + c ≈ ?
"""
import gzip, json, random
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).parent
MODELS = {
    "TinyStories 22M": ("tegancao/tinystories-22m", "roneneldan/TinyStories", "TinyStoriesV2-GPT4-valid.txt"),
    "OpenWebText 45M": ("tegancao/owt-45m", "stanford-cs336/owt-sample", "owt_valid.txt.gz"),
}
GROUPS = {
    "names": "Tom Ben Sam Max Tim Anna Sarah Mary Jack Emma Lucy Mia Sue Amy John Sara",
    "colors": "red blue green yellow pink purple orange brown black white gray",
    "numbers": "one two three four five six seven eight nine ten",
    "animals": "dog cat bird fish bear rabbit frog duck cow lion horse mouse",
    "emotions": "happy sad angry scared excited proud glad upset worried surprised",
    "past-tense verbs": "ran jumped played walked ate laughed cried smiled looked said",
}
ANALOGIES = [  # a : b :: c : d   -> test b - a + c ≈ d
    ("man", "woman", "king", "queen"), ("boy", "girl", "brother", "sister"), ("he", "she", "his", "her"),
    ("dad", "mom", "son", "daughter"), ("boy", "girl", "prince", "princess"), ("man", "woman", "father", "mother"),
    ("dog", "dogs", "cat", "cats"), ("tree", "trees", "bird", "birds"), ("toy", "toys", "car", "cars"),
    ("walk", "walked", "jump", "jumped"), ("play", "played", "look", "looked"), ("run", "ran", "eat", "ate"),
    ("big", "bigger", "small", "smaller"), ("fast", "faster", "slow", "slower"), ("happy", "sad", "good", "bad"),
    ("one", "two", "three", "four"), ("day", "night", "morning", "evening"), ("hot", "cold", "up", "down"),
]
COLORS = ["#2e86de", "#e67e22", "#27ae60", "#c0392b", "#8e44ad", "#7f8c8d"]
random.seed(0); np.random.seed(0)


def unit(W):
    return W / W.norm(dim=1, keepdim=True).clamp_min(1e-8)


def single_ids(tok, words):
    out = {}
    for w in words:
        ids = tok.encode(" " + w)
        if len(ids) == 1:
            out[w] = ids[0]
    return out


def token_counts(tok, repo, fname):
    path = hf_hub_download(repo, fname, repo_type="dataset")
    opener = gzip.open if fname.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        text = f.read()
    counts = Counter()
    step = 1_000_000
    for i in range(0, len(text), step):
        counts.update(tok(text[i:i + step], add_special_tokens=False).input_ids)
    return counts, len(text)


def all_but_the_top(W, k=1):
    """Remove the mean and the top-k principal components (Mu & Viswanath, 2018, "All-but-the-Top")."""
    Wc = W - W.mean(0)
    _, _, Vt = torch.linalg.svd(Wc, full_matrices=False)
    return Wc - (Wc @ Vt[:k].T) @ Vt[:k], Vt[0]


def neighbors(E, i, tok, k=6, exclude=()):
    sims = E @ E[i]
    order = sims.argsort(descending=True).tolist()
    res = [(tok.decode([j]).strip(), round(float(sims[j]), 2)) for j in order if j != i and j not in exclude][:k]
    return res


results = {}
models = {}
for label, (repo, data_repo, data_file) in MODELS.items():
    tok = AutoTokenizer.from_pretrained(repo)
    m = AutoModelForCausalLM.from_pretrained(repo)
    models[label] = (tok, m.model.embed_tokens.weight.detach().float(), m.lm_head.weight.detach().float())
    counts, n_chars = token_counts(tok, data_repo, data_file)
    models[label] += (counts,)
    results[label] = {"vocab": len(tok), "validation_chars": n_chars, "validation_tokens": sum(counts.values())}
    print(f"{label}: vocab {len(tok):,}; {sum(counts.values()):,} validation tokens counted")

# ---------- A. 2D maps of word groups ----------
fig, axes = plt.subplots(2, 2, figsize=(15, 13))
for r, (label, (tok, Win, Wout, counts)) in enumerate(models.items()):
    pts, cats, words = [], [], []
    for gi, (g, ws) in enumerate(GROUPS.items()):
        for w, i in single_ids(tok, ws.split()).items():
            pts.append(i); cats.append(gi); words.append(w)
    X = unit(Win)[pts].numpy()
    sil = silhouette_score(X, cats, metric="cosine")
    shuffled = np.random.permutation(cats)
    sil_rand = silhouette_score(X, shuffled, metric="cosine")
    results[label]["A_words_used"] = len(words)
    results[label]["A_silhouette"] = round(float(sil), 3)
    results[label]["A_silhouette_shuffled_groups"] = round(float(sil_rand), 3)
    for c, (name, Y) in enumerate([("PCA", PCA(2, random_state=0).fit_transform(X)),
                                   ("t-SNE", TSNE(2, perplexity=8, metric="cosine", init="pca", random_state=0).fit_transform(X))]):
        ax = axes[r, c]
        for gi, g in enumerate(GROUPS):
            sel = [k for k, cc in enumerate(cats) if cc == gi]
            ax.scatter(Y[sel, 0], Y[sel, 1], color=COLORS[gi], s=40, label=g)
            for k in sel:
                ax.annotate(words[k], Y[k], fontsize=9, xytext=(3, 2), textcoords="offset points", color=COLORS[gi])
        ax.set_title(f"{label}, input embeddings: {name}", fontsize=13, weight="bold")
        ax.set_xticks([]); ax.set_yticks([])
        if r == 0 and c == 0:
            ax.legend(loc="best", fontsize=10)
fig.suptitle("Word groups in embedding space (each word = one token)", fontsize=16, weight="bold")
fig.tight_layout(); fig.savefig(HERE / "a_word_maps.png", dpi=110); plt.close(fig)

# ---------- B. Anisotropy ----------
fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
for ax, (label, (tok, Win, Wout, counts)) in zip(axes, models.items()):
    V = Win.shape[0]
    sample = torch.tensor(random.sample(range(V), 2000))
    res = {}
    for name, W, col in [("input", Win, "#2e86de"), ("output", Wout, "#e67e22"), ("output, centered", Wout - Wout.mean(0), "#27ae60"),
                         ("output, centered + top PC removed", all_but_the_top(Wout)[0], "#8e44ad")]:
        E = unit(W)[sample]
        S = (E @ E.T)[~torch.eye(len(sample), dtype=bool)]
        res[name] = {"mean_cos": round(float(S.mean()), 3), "std_cos": round(float(S.std()), 3)}
        ax.hist(S.numpy(), bins=120, density=True, alpha=0.55, color=col, label=f"{name}: mean {S.mean():.2f}")
    mean_dir = unit(Wout.mean(0, keepdim=True))[0]
    res["output_mean_vector_norm_vs_avg_token_norm"] = round(float(Wout.mean(0).norm() / Wout.norm(dim=1).mean()), 3)
    cos_to_mean = unit(Wout) @ mean_dir
    freq = torch.tensor([counts.get(i, 0) for i in range(V)], dtype=torch.float)
    rare, common = freq < 5, freq >= 1000
    res["output_cos_to_mean_rare_tokens(<5)"] = round(float(cos_to_mean[rare].mean()), 3) if rare.any() else None
    res["output_cos_to_mean_common_tokens(>=1000)"] = round(float(cos_to_mean[common].mean()), 3)
    res["n_rare_tokens(<5)"], res["n_common_tokens(>=1000)"] = int(rare.sum()), int(common.sum())
    Wc = Wout - Wout.mean(0)
    _, Sv, Vt = torch.linalg.svd(Wc, full_matrices=False)
    res["centered_output_variance_share_top3_PCs"] = [round(float(x), 3) for x in (Sv**2 / (Sv**2).sum())[:3]]
    pc1 = (Wc @ Vt[0]).numpy()
    res["corr_PC1_projection_vs_log_freq"] = round(float(np.corrcoef(pc1, np.log1p(freq.numpy()))[0, 1]), 3)
    results[label]["B_anisotropy"] = res
    ax.set_title(f"{label}: cosine similarity of random token pairs", fontsize=12, weight="bold")
    ax.set_xlabel("cosine similarity"); ax.legend(fontsize=9); ax.set_yticks([])
fig.tight_layout(); fig.savefig(HERE / "b_anisotropy.png", dpi=110); plt.close(fig)

probe = ["dog", "tree", "mom", "ran", "happy", "red"]
for label, (tok, Win, Wout, counts) in models.items():
    ids = single_ids(tok, probe)
    E_raw, E_cent, E_abt = unit(Wout), unit(Wout - Wout.mean(0)), unit(all_but_the_top(Wout)[0])
    results[label]["B_output_neighbors"] = {
        w: {"raw": neighbors(E_raw, i, tok), "centered": neighbors(E_cent, i, tok), "centered_top_pc_removed": neighbors(E_abt, i, tok)}
        for w, i in ids.items()}

# ---------- C. Rarely trained tokens: norm vs. frequency ----------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, (label, (tok, Win, Wout, counts)) in zip(axes, models.items()):
    V = Win.shape[0]
    freq = np.array([counts.get(i, 0) for i in range(V)])
    norm_in, norm_out = Win.norm(dim=1).numpy(), Wout.norm(dim=1).numpy()
    ax.scatter(freq + 1, norm_in, s=3, alpha=0.4, color="#2e86de", label="input embedding")
    ax.scatter(freq + 1, norm_out * (np.median(norm_in) / np.median(norm_out)), s=3, alpha=0.4, color="#e67e22",
               label="output embedding (rescaled)")
    ax.set_xscale("log"); ax.set_xlabel("count in validation text + 1 (log)"); ax.set_ylabel("vector norm")
    ax.set_title(f"{label}: embedding norm vs. token frequency", fontsize=12, weight="bold"); ax.legend(fontsize=9, markerscale=4)
    unseen = np.where(freq == 0)[0]
    lowest = np.argsort(norm_in)[:12]
    corr = np.corrcoef(np.log1p(freq), norm_in)[0, 1]
    results[label]["C_rare_tokens"] = {
        "tokens_never_seen_in_validation": int(len(unseen)),
        "median_input_norm_seen": round(float(np.median(norm_in[freq > 0])), 2),
        "median_input_norm_unseen": round(float(np.median(norm_in[unseen])), 2) if len(unseen) else None,
        "corr_log_freq_vs_input_norm": round(float(corr), 3),
        "lowest_input_norm_tokens": [(tok.decode([int(i)]), round(float(norm_in[i]), 2), int(freq[i])) for i in lowest],
        "highest_input_norm_tokens": [(tok.decode([int(i)]), round(float(norm_in[i]), 2), int(freq[i]))
                                      for i in np.argsort(-norm_in)[:8]],
    }
fig.tight_layout(); fig.savefig(HERE / "c_norm_vs_frequency.png", dpi=110); plt.close(fig)

# ---------- D. Analogies ----------
for label, (tok, Win, Wout, counts) in models.items():
    out = {}
    for space, W in [("input", Win), ("output", Wout), ("output, centered + top PC removed", all_but_the_top(Wout)[0])]:
        E = unit(W)
        rows, hit1, hit5, n = [], 0, 0, 0
        for a, b, c, d in ANALOGIES:
            ids = single_ids(tok, [a, b, c, d])
            if len(ids) < 4:
                continue
            q = unit((E[ids[b]] - E[ids[a]] + E[ids[c]])[None])[0]
            sims = E @ q
            sims[[ids[a], ids[b], ids[c]]] = -9
            top = sims.topk(5).indices.tolist()
            n += 1; hit1 += top[0] == ids[d]; hit5 += ids[d] in top
            rows.append(f"{a}:{b} :: {c}:? -> " + ", ".join(tok.decode([t]).strip() for t in top) + ("  ✓" if top[0] == ids[d] else ""))
        out[space] = {"n": n, "top1": hit1, "top5": hit5, "rows": rows}
    results[label]["D_analogies"] = out

(HERE / "results.json").write_text(json.dumps(results, indent=1, ensure_ascii=False))
print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk not in ("B_output_neighbors",)} for k, v in results.items()},
                 indent=1, ensure_ascii=False)[:9000])
