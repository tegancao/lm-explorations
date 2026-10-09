# Tiny chat: talking to two small models I trained from scratch

[`chat.py`](chat.py) is a small interactive front end to two language models I trained from scratch and published on
Hugging Face: [`tegancao/tinystories-22m`](https://huggingface.co/tegancao/tinystories-22m) and
[`tegancao/owt-45m`](https://huggingface.co/tegancao/owt-45m). You type a prompt and get back up to 250 generated
tokens, streamed as they're produced.

From the repo root, with [uv](https://docs.astral.sh/uv/):

```
uv sync
uv run demos/tiny-chat/chat.py                  # TinyStories model (default)
uv run demos/tiny-chat/chat.py --model owt      # OpenWebText model
```

Without uv: `pip install -r demos/tiny-chat/requirements.txt`, then `python demos/tiny-chat/chat.py`. The models
download from Hugging Face on first run (91 MB and 181 MB).

The sessions below are real and unedited. I recorded them with my original from-scratch implementation, which uses the
same weights. `chat.py` samples through Hugging Face `transformers` instead, so the same seed gives different (but
equally typical) text.

| Model (`/model NAME`) | Hugging Face | Parameters | Training data | Final val loss |
|---|---|---|---|---|
| `tinystories` (default) | `tegancao/tinystories-22m` | 22.7M | TinyStories, 327.68M tokens, 10K vocab | 1.382 |
| `owt` | `tegancao/owt-45m` | 45.2M | OpenWebText, 327.68M tokens, 32K vocab | 4.154 |

| Command | What it does |
|---|---|
| *(any text)* | continue that text (each prompt starts fresh) |
| `/more` or an empty line | keep extending the last text |
| `/temp X`, `/top_p X`, `/tokens N` | sampling temperature (default 0.8), nucleus threshold (0.9), max tokens per reply (250) |
| `/model`, `/model tinystories`, `/model owt` | list the two models, or switch |
| `/settings`, `/help`, `/quit` | show settings, list commands and limitations, exit |

---

## 1. A story from a story opening (TinyStories)

`--seed 7`, then `Once upon a time, a little girl named Mia found a tiny door in the garden.`

```text
Once upon a time, a little girl named Mia found a tiny door in the garden. She wanted to open it, but it was locked. Mia felt sad and thought about what to do.
Mia had an idea. She went to her mom and asked for help. Her mom was good at fixing things. Mia's mom fixed the door with a glue and put the door back in. Mia was happy and said thank you.
Mia learned that it's important to help others when they need it. She also learned that even a small act of kindness can make a big difference. Mia's mom was proud of her and they lived happily in the garden.
[117 tokens, 1.9 s, ended with <|endoftext|>]
```

**What works:** this is what the model was trained for. Grammar and spelling are clean, the characters stay
consistent (Mia, her mom), and the text has a story's shape: a problem (the locked door), a turn (mom helps)
and a moral, before the model ends the story on its own with `<|endoftext|>` after 117 tokens.

**What doesn't:** the logic is only loosely connected. Mom "fixed the door with a glue" instead of unlocking it,
the door never gets opened, and the moral ("help others when they need it") doesn't match the story, in which
Mia was the one being helped. The model has learned the *form* of a TinyStories story better than its reasoning.

## 2. Continuing with `/more`

`--seed 4`, `/tokens 60`, then `Tom and his dog went to the beach.`, then `/more`

```text
Tom and his dog went to the beach. They wanted to play in the sand and swim in the water. Tom liked to look at the big blue water and feel the sand between his toes.
As they played, Tom saw a big wave coming. He knew that wave was coming soon. He was scared, but he wanted to be brave
[60 tokens, 0.9 s]

prompt> /more

. So, he took a deep breath and ran to the wave.
Tom jumped in and started to swim. He felt the wind on his fur and the sun on his back. He was so happy! He played in the water and splashed with his dog, Spot. They had a lot of
[60 tokens, 1.0 s]
```

The first reply stops mid-sentence at the 60-token limit; `/more` feeds the whole text back in and the model picks
up exactly where it stopped (". So, he took a deep breath…"). **Limitation visible here:** the model confuses who
is who. "He felt the wind on his fur" describes the dog, but the sentence is about Tom, and the dog only gets a
name (Spot) in the second reply. Small models track entities poorly.

## 3. Asking a question: it continues, it doesn't answer

`--seed 1`, `/tokens 60`, then `What is the capital of France?`

```text
What is the capital of France? Will you be my friend?"
Ben and Lily looked at each other. They did not want to share. They wanted to have fun. They wanted to have fun.
"No, we can't. It is our cap. It belongs to the park," Ben said.
"You are
[60 tokens, 0.9 s]
```

**The most important limitation:** these are *base* language models, not chat models. They were only trained to
predict the next token of their training text, never to follow instructions or answer questions (that takes
instruction tuning / RLHF, which these models never had). So the question just becomes the start of a
story, said by a character. The TinyStories model also has no world knowledge: its training data never mentions
France, so the closest familiar word it latches onto is "cap" ("It is our cap").

## 4. Temperature: greedy vs. too random

`--seed 2`, `/tokens 80`, `/temp 0`, then `The little bird was sad because`; then `/temp 1.5`, `/top_p 1.0` and the same prompt.

**Greedy (`/temp 0`):**

```text
The little bird was sad because it was a gloomy day. The sky was gloomy and the wind was blowing. The little bird wanted to find a friend to play with.
The little bird saw a big tree and thought, "Maybe I can find a friend there." So, the little bird flew to the tree and found a small bug. The bug was also sad because it was a gloomy day.
The little bird said
[80 tokens, 1.1 s]
```

**Temperature 1.5, top-p off:**

```text
The little bird was sad because she needed a picture towalk wash fans. okay after catch- trouble animal, listen and aboard them as it told funny rubbedtQuick celebr skip down hishood.
nder amaz hissed working,'Don robbersanish ghosts instructions towardseverse. led like he expected giving the woman machine “Look pretended laptopabbit straw unusual cheese”. benches countries hadn't attem opportunity well.
Peteranish one last when she fluttering
[80 tokens, 1.1 s]
```

Greedy decoding is always the same, coherent but **repetitive** ("gloomy" three times, "The little bird" over and
over), because it always takes the single most likely token. At temperature 1.5 without top-p, the flattened
distribution lets unlikely tokens through, and the text falls apart into word fragments ("towalk wash fans",
"robbersanish"): these are BPE subword tokens glued together. The defaults (T = 0.8, top-p = 0.9) sit in
between: varied but fluent, as in examples 1 and 2.

## 5. The OpenWebText model

`--model owt --seed 3`, `/tokens 120`, then `Scientists have discovered`

```text
Scientists have discovered the new "sandbox" in a new species of mammals known as "realistic growth" that means the flammable bending can be used to build an extra-sized concomium, but can't be used for a cost of more than $400,000 in annual fundraising efforts.

The scientists say that "sandbox" has been a staple of the species for decades. But the species that have been shared with their rations are still lacking in the few-territorial roles of big-bending theories.

"The new species of mars
[120 tokens, 2.8 s]
```

The OWT model has learned the **style** of a news article (a claim, "the scientists say", quotes, dollar amounts)
but not its **content**: the science is nonsense, it invents words ("concomium", "few-territorial") and the
sentences don't connect. Web text is far more varied than children's stories, and with the same small model and
compute it ends at a much higher loss (4.154 vs. 1.382 validation loss).

## 6. Long texts: repetition, the 256-token window, and the KV cache

`--model owt --seed 1`, `/tokens 150`, then `The stock market`, `/more`, `/more` (excerpts)

```text
The stock market has also gotten its share of interest.

According to the BBC's new series, stocks for 20 years are selling 1.7 billion shares in the UK alone. The stock price rose almost $2.8 billion last year, more than $9 billion last year.

The price was dropped nearly $2 billion last year, and a value of 2.5 billion.

The stock price rose about $3.1 billion in February.
[…]
[150 tokens, 4.4 s]

prompt> /more

The stock price rose 15.2 billion in September, while the price was $16.7 billion.

The stock price rose from just 16.2 billion last year to 14.1 billion last year, and the price was up 8.4 billion over the same period.

The stock price jumped more than $20 billion last year, with an increase of 0.8 billion last year, when the stock price rose to 0.8 billion.
[…]
[150 tokens, 7.3 s]

prompt> /more
(text so far: 304 tokens; the model only sees the last 256)

 the stock price fell 7.7 billion last year.

The stock price rose 15.6 billion last year, with a 7.9 billion last year, and it was only a decade earlier when the stock price dropped from 1.1 billion last year.

The stock price rose $1.1 billion over the same period last year, a year after the stock price dropped 7.9 billion.
[…]
[150 tokens, 8.5 s]
```

Three limitations show up together:

- **Repetition loops:** the model gets stuck on one pattern ("The stock price rose … billion last year") with
  numbers that contradict each other. Once a pattern is in its context, repeating it is the most likely
  continuation, so the loop reinforces itself.
- **The 256-token window:** after the second `/more` the text is 304 tokens long, but the model was trained on
  256-token windows (`context_length`), and RoPE has no positions beyond that, so only the last 256 tokens are fed
  in. Anything earlier, such as names or the topic set up at the start, is invisible to it. `chat.py` generates in
  chunks so the context never exceeds the 256 positions the models were trained on.
- **Replies got slower as the text grew** (in my original implementation): the same 150 tokens took 4.4 s, then
  7.3 s, then 8.5 s, because my first `generate()` re-ran the whole model over the full context for every new token.
  Production systems keep a **KV cache** (the keys and values of earlier tokens), so each new token only needs one
  position's worth of work. `chat.py` uses the KV cache in `transformers`: 300 tokens take about 3 s.

---

## Limitations at a glance

| Limitation | Why | Seen in |
|---|---|---|
| Continues text instead of answering | base LM trained only on next-token prediction; no instruction tuning | §3 |
| No world knowledge (TinyStories) | trained only on simple children's stories | §3 |
| Loose logic, entity confusion | small model (22.7M / 45.2M parameters), limited training (327.68M tokens) | §1, §2 |
| Fluent style, nonsense content (OWT) | web text is far more varied; same small model and budget | §5 |
| Repetition loops | repeating what's in context is often the most likely continuation; worst with greedy decoding | §4, §6 |
| Sensitive to sampling settings | high temperature without top-p lets unlikely subword tokens through | §4 |
| No memory between prompts | each prompt is generated from scratch; only `/more` carries text forward | §2 |
| Sees only the last 256 tokens | `context_length` = 256 in training; RoPE tables end there | §6 |
| Slow generation without a KV cache | the full context is recomputed for every token (fixed in `chat.py`) | §6 |
