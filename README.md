# LM explorations

My notes, experiments and reflections while learning how language models work, by building and training them from
scratch.

**Who this is for:** anyone learning how language models work who wants to see a from-scratch project end to end,
including what goes wrong. You'll find two small models you can chat with (and see clearly what base models can and
can't do), compact notes on why training large models is hard, practical lessons from training on a laptop and a
cheap rented GPU, and reflections on what building it all myself taught me.

## Key findings so far

- **Small models learn grammar before knowledge.** My 45M-parameter OpenWebText model matches Pythia-70M (trained on
  ~1,000× more tokens) on a grammar benchmark, yet stays near chance on reasoning benchmarks.
  [→ benchmark evaluation](experiments/03-benchmark-eval)
- **Training data sets the ceiling.** Models trained only on children's stories don't improve on general benchmarks no
  matter their size. [→ benchmark evaluation](experiments/03-benchmark-eval)
- **Even tiny models learn word meaning,** but their output embeddings hide it behind a shared direction and a frequency
  direction. [→ token embeddings](experiments/02-token-embeddings)
- **Base models continue text, they don't answer it,** and they invent convincing nonsense.
  [→ tiny-chat demo](demos/tiny-chat)

## Run the chatbot

Chat with the two small models from your terminal; they download from Hugging Face automatically.

**You need:**
- [uv](https://docs.astral.sh/uv/getting-started/installation/), a one-line install. uv installs Python 3.12 and the
  exact dependency versions from `uv.lock` for you, so you don't need Python set up already.
- No GPU: it runs on a normal laptop CPU, at about 1–2 seconds per reply.
- No Hugging Face account: the models are public.
- On the first run, a few minutes and some disk space: PyTorch (several hundred MB to ~1 GB, depending on your
  platform) and the two models (91 MB and 181 MB). Later runs start in seconds.

**Then:**

```
git clone https://github.com/tegancao/lm-explorations.git
cd lm-explorations
uv run demos/tiny-chat/chat.py                 # TinyStories model
uv run demos/tiny-chat/chat.py --model owt     # OpenWebText model
```

Type a story opening (for the TinyStories model) or a news-style opening (for the OpenWebText model) at the
`prompt>` line; `/help` lists the commands. See [demos/tiny-chat](demos/tiny-chat) for example sessions and what to
expect. Tested on macOS (Apple Silicon); Linux and Windows should work the same, since it's standard PyTorch and
Hugging Face `transformers` on CPU. Without uv: `pip install -r demos/tiny-chat/requirements.txt`, then
`python demos/tiny-chat/chat.py`.

## Contents

| | |
|---|---|
| [**demos/tiny-chat**](demos/tiny-chat) | Chat with two small models I trained from scratch; real sessions with commentary on what they can and can't do |
| [**experiments/01-small-lms-from-scratch**](experiments/01-small-lms-from-scratch) | Building my first language models: what it taught me, setup and results |
| [**experiments/02-token-embeddings**](experiments/02-token-embeddings) | What the models' token embeddings learned: word clusters, why output embeddings all look alike, tokens the model never learned, analogies |
| [**experiments/03-benchmark-eval**](experiments/03-benchmark-eval) | My models among 9 reference models (TinyStories, Pythia, GPT-2) on 8 benchmarks, incl. grammar and bits per byte (lm-evaluation-harness, on Modal GPUs) |
| [**notes/01-training-at-scale**](notes/01-training-at-scale) | Training at scale: compute, GPU efficiency, memory, and splitting a model across GPUs (with a diagram) |
| [**research-setup.md**](research-setup.md) | My hardware, software and costs (a laptop and a pay-per-use cloud GPU), and lessons learned |

## Models

- [`tegancao/tinystories-22m`](https://huggingface.co/tegancao/tinystories-22m): 22.7M parameters, trained on TinyStories
- [`tegancao/owt-45m`](https://huggingface.co/tegancao/owt-45m): 45.2M parameters, trained on OpenWebText

Both are small, educational base models: fun to explore, not for any real use. See their model cards for details.

## Acknowledgments

My first models follow the architecture and training setup in the publicly available materials of Stanford CS336,
Language Modeling from Scratch: [cs336.stanford.edu](https://cs336.stanford.edu/).
