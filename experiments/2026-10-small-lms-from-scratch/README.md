# Small language models from scratch (October 2026)

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


Try the models: [`demos/tiny-chat`](../../demos/tiny-chat).
