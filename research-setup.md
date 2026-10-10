# My research setup

Everything in this repo was done with a laptop and a pay-per-use cloud GPU.

| | What I used |
|---|---|
| Development | MacBook Air (M2, 8 GB RAM): writing code, debugging on tiny models, running the embedding experiment and the chat demo |
| Training and evaluation | [Modal](https://modal.com) cloud GPUs: one NVIDIA L4 (24 GB) per run, about $0.80/hour |
| Software | PyTorch, Hugging Face `transformers`, EleutherAI's `lm-evaluation-harness`, `uv` for environments |
| Tracking and sharing | Weights & Biases (runs and evaluations), Hugging Face Hub (models), GitHub (this repo) |
| Cost | Training: about $2.50 (TinyStories model, 3.1 h) and $3.60 (OpenWebText model, 4.5 h). Benchmark evaluation of 11 models: a few dollars |

## Lessons

**Laptop**

- Watch memory: training a BPE tokenizer on OpenWebText needed 17 GB on an 8 GB machine. Heavy swapping (low CPU use,
  many swap-ins) means stop.
- Text generation on Apple's GPU (MPS) can reserve 5+ GB: each new sequence length gets a new cached buffer. Calling
  `torch.mps.empty_cache()` after each step keeps it at ~1 GB, and for small models the CPU is just as fast.
- Keep `data/`, `out/` and checkpoints out of git (GitHub rejects files over 100 MB), and remember that forks of public
  repos are always public.

**Cloud GPUs (Modal)**

- `.remote()` waits on your laptop, so the job dies if the laptop sleeps: use `.spawn()` with `--detach`, or keep the
  laptop awake (`caffeinate`). Save checkpoints during long runs, since standard containers can be preempted.
- Accounts have a limit on GPUs running at once (mine: 10); cap parallel jobs with `max_containers`.
- Let the evaluation harness pick the batch size (`batch_size="auto"`): a fixed batch that fits short inputs can run
  out of memory on long ones.
- A function must have the same dependencies (images, secrets) locally and in the container, so make optional pieces
  like a W&B secret depend on a setting both sides see.
- Check your credit before long runs: $26 of my first $30 went to a single 9-hour CPU job.

**Environment**

- `uv add` installs into the project's `.venv`; run tools with `uv run <tool>`, and try extra packages without
  installing them with `uv run --with <package>`.
