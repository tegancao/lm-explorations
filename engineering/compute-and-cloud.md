# Compute and cloud: lessons from training on a laptop and rented GPUs

I developed on an 8 GB MacBook Air (M2) and trained on [Modal](https://modal.com) GPUs (NVIDIA L4, about $0.80/hour), tracking runs with Weights & Biases.

## Environment, tooling, cloud

- `uv add` installs into the project `.venv` → run tools as `uv run <tool>`.
- `.gitignore` `data/`, `out/`, `*.pkl` (GitHub rejects files > 100 MB); forks of public repos are **always public**.
- Memory: training a BPE tokenizer on OpenWebText needed **17 GB** (my Mac has 8) → swapping (low CPU %, high swap-ins) means stop.
- Modal: `.remote()` waits for the result → cancelled when the Mac sleeps; use **`.spawn()` + `--detach`**. Default containers can be **preempted** (restart from scratch); `nonpreemptible=True` costs ~3×. Save outputs/checkpoints during long runs.
- Check credit before long runs ($26 of $30 went to one 9h CPU job).

- **MPS generation "uses" 5+ GB:** the MPS allocator caches a buffer per new tensor shape, and in token-by-token generation the sequence grows every step, so reserved (driver) memory piles up while allocated memory stays ~0.2 GB. Fix: `torch.mps.empty_cache()` after each step (stays ~1.1 GB), or generate on CPU (same speed for one-token steps of a small model).
