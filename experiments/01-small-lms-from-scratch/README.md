# Building my first language models from scratch (October 2026)

My main takeaway: **to truly understand a subject, you have to build it from scratch yourself.** Reading about
language models gave me a sense of the pieces; building every one of them (the tokenizer, the Transformer, the
optimizer, the training loop, text generation) is what made it clear how they actually work and fit together. It was
a very worthwhile experiment to do on my own.

A few moments where building it myself made the difference:

- **Positional encodings only clicked when I looked at the actual numbers.** I understood RoPE from the equations,
  but only after printing and visualizing the tensors, step by step, did I really see what gets rotated and why.
  I also hadn't realized *where* it goes: RoPE rotates the queries and keys **after** their weight projections, and
  is not applied to the values at all. That makes sense once you see that position only needs to shape *which*
  tokens attend to each other (the query–key dot product, which then depends only on their relative distance),
  not *what* information gets passed along (the values).
- **Back-of-the-envelope math became real.** Counting parameters, FLOPs and memory on paper felt abstract until the
  same reasoning predicted the out-of-memory error I then hit on a real GPU.
- **Each design choice has a reason, and you can watch it fail without it.** Removing pieces of the architecture and
  seeing training diverge or get worse taught me more than any explanation of why they're there.
- **The engineering is half the work.** Fitting experiments onto an 8 GB laptop and a rented GPU, making runs survive
  restarts, logging, and keeping costs down took as much thought as the model itself.
- **Small models show their limits honestly.** Chatting with my own models (see [`demos/tiny-chat`](../../demos/tiny-chat/README.md))
  made the gap between "fluent" and "understands" very concrete.

**Question:** what can a ~20–45M-parameter Transformer learn from 330M tokens, on simple vs. varied text?

## Setup

- Llama-style decoder (4 layers, d_model 512, 16 heads, SwiGLU, RMSNorm, RoPE), context length 256, implemented and
  trained from scratch in PyTorch, then converted to the standard Hugging Face Llama format.
- AdamW, peak learning rate 3e-3 with warm-up and cosine decay, batch 32 × 256 tokens, 40,000 steps (327.68M tokens),
  one NVIDIA L4 GPU on Modal.
- Two datasets: [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) (simple children's stories,
  10K-token BPE vocabulary) and a sample of [OpenWebText](https://huggingface.co/datasets/Skylion007/openwebtext)
  (web pages, 32K-token vocabulary).

## Results

| Model | Data | Parameters | Validation loss | Perplexity | Training time |
|---|---|---|---|---|---|
| [`tegancao/tinystories-22m`](https://huggingface.co/tegancao/tinystories-22m) | TinyStories | 22.7M | 1.382 | 4.0 | ~3.1 h |
| [`tegancao/owt-45m`](https://huggingface.co/tegancao/owt-45m) | OpenWebText | 45.2M | 4.154 | 63.7 | ~4.5 h |


Try the models: [`demos/tiny-chat`](../../demos/tiny-chat/README.md).

I followed the publicly available materials of Stanford CS336, Language Modeling from Scratch:
[cs336.stanford.edu](https://cs336.stanford.edu/) (I used the [Spring 2025](https://cs336.stanford.edu/spring2025/)
materials).
