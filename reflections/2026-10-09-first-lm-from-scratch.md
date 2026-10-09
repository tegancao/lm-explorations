# Building my first language model from scratch (2026-10-09)

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
- **Small models show their limits honestly.** Chatting with my own models (see [`demos/tiny-chat`](../demos/tiny-chat))
  made the gap between "fluent" and "understands" very concrete.

## Reference

I followed the publicly available materials of Stanford CS336, Language Modeling from Scratch:
[cs336.stanford.edu](https://cs336.stanford.edu/) (I used the [Spring 2025](https://cs336.stanford.edu/spring2025/)
materials).
