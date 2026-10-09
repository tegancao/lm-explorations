# LM explorations

My notes, experiments and reflections while learning how language models work, by building and training them from
scratch.

## Setup

With [uv](https://docs.astral.sh/uv/) (installs Python 3.12 and the pinned dependencies from `uv.lock`):

```
uv sync
uv run demos/tiny-chat/chat.py        # chat with the TinyStories model
```

## Contents

| | |
|---|---|
| [**demos/tiny-chat**](demos/tiny-chat) | Chat with two small models I trained from scratch; real sessions with commentary on what they can and can't do |
| [**experiments/2026-10-small-lms-from-scratch**](experiments/2026-10-small-lms-from-scratch) | Training ~20–45M-parameter models on TinyStories and OpenWebText: setup and results |
| [**notes/scaling**](notes/scaling/memory-compute-sharding.md) | Training memory, MFU, and how FSDP/ZeRO-3 shards a model across GPUs (with a diagram) |
| [**engineering**](engineering/compute-and-cloud.md) | Practical lessons from training on a laptop and rented cloud GPUs |
| [**reflections**](reflections) | Dated entries on what I learned and what surprised me |
| [**tools/md2pdf**](tools/md2pdf) | Markdown → LaTeX-typeset PDF (pandoc + LuaLaTeX) |

## Models

- [`tegancao/tinystories-22m`](https://huggingface.co/tegancao/tinystories-22m): 22.7M parameters, trained on TinyStories
- [`tegancao/owt-45m`](https://huggingface.co/tegancao/owt-45m): 45.2M parameters, trained on OpenWebText

Both are small, educational base models: fun to explore, not for any real use. See their model cards for details.

## Acknowledgments

My first models follow the architecture and training setup in the publicly available materials of Stanford CS336
(Language Modeling from Scratch).
