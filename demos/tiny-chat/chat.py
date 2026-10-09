"""Tiny chat: type a prompt, get generated text back, from two small models trained from scratch.

    pip install -r requirements.txt
    python chat.py                  # TinyStories model (tegancao/tinystories-22m)
    python chat.py --model owt      # OpenWebText model  (tegancao/owt-45m)

These are base language models, not chat models: they CONTINUE your text rather than answer it.
Commands: /more  /temp X  /top_p X  /tokens N  /model [tinystories|owt]  /settings  /help  /quit
"""

import argparse
import time

try:
    import readline  # noqa: F401  (arrow keys and history in input())
except ImportError:
    pass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

MODELS = {"tinystories": "tegancao/tinystories-22m", "owt": "tegancao/owt-45m"}
EOT = "<|endoftext|>"
CHUNK = 64   # generate in chunks so the context never exceeds the 256 positions the models were trained on


class Streamer(TextStreamer):
    """Prints tokens as they arrive, without the newline TextStreamer adds when each generate() call ends."""

    def on_finalized_text(self, text: str, stream_end: bool = False) -> None:
        print(text, end="", flush=True)

HELP = """commands:
  /more        continue the last text (an empty line does the same)
  /temp X      sampling temperature (0 = greedy, default 0.8)
  /top_p X     nucleus threshold (1.0 = off, default 0.9)
  /tokens N    max tokens per reply (default 250)
  /model       show the two models; /model tinystories or /model owt switches
  /settings    show the current settings
  /quit        exit (or Ctrl-D)

limitations:
  - base language models, not chat models: they CONTINUE your text instead of answering it
    (story openings work best for tinystories, news-style openings for owt)
  - every new prompt starts fresh: no memory of earlier prompts or replies
  - /more only continues a text the token limit cut off; if the model ended it (<|endoftext|>), start anew
  - the models were trained on 256-token windows, so each step sees at most the last {window} tokens:
    in long texts, early names and details drop out of view"""


def load(name: str):
    tok = AutoTokenizer.from_pretrained(MODELS[name])
    model = AutoModelForCausalLM.from_pretrained(MODELS[name]).eval()
    n = sum(p.numel() for p in model.parameters())
    print(f"model {MODELS[name]} ({n / 1e6:.1f}M params, vocab {model.config.vocab_size:,})")
    return tok, model, model.config.max_position_embeddings


@torch.no_grad()
def generate(model, tok, ids: list[int], max_new: int, temp: float, top_p: float, window: int) -> tuple[list[int], bool]:
    """Generate up to max_new tokens after ids, streaming them; returns (new ids, ended with <|endoftext|>)."""
    eot = tok.convert_tokens_to_ids(EOT)
    streamer = Streamer(tok, skip_prompt=True, skip_special_tokens=True)
    new: list[int] = []
    while len(new) < max_new:
        n = min(CHUNK, max_new - len(new))
        ctx = (ids + new)[-(window - n):]                 # keep prompt + output within the trained window
        sampling = dict(do_sample=True, temperature=temp, top_p=top_p) if temp > 0 else dict(do_sample=False)
        out = model.generate(torch.tensor([ctx]), max_new_tokens=n, eos_token_id=eot, pad_token_id=eot,
                             streamer=streamer, **sampling)[0, len(ctx):].tolist()
        new += out
        if out and out[-1] == eot or len(out) < n:
            return new, True
    return new, False


def main() -> None:
    p = argparse.ArgumentParser(description="Type a prompt, get generated text back.")
    p.add_argument("--model", default="tinystories", choices=list(MODELS))
    p.add_argument("--seed", type=int, default=None, help="fix the random seed for reproducible replies")
    args = p.parse_args()
    if args.seed is not None:
        torch.manual_seed(args.seed)

    name = args.model
    tok, model, window = load(name)
    eot = tok.convert_tokens_to_ids(EOT)
    s = {"temp": 0.8, "top_p": 0.9, "tokens": 250}
    print(f"note: it continues your text (not a chat model), each prompt starts fresh, and it sees at most the\n"
          f"last {window} tokens. /help lists commands and limitations\n")

    last: list[int] = []
    ended = False
    while True:
        try:
            line = input("prompt> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if line in ("/quit", "/exit", "/q"):
            break
        if line == "/help":
            print(HELP.format(window=window))
            continue
        if line == "/settings":
            print(f"model {name}, temperature {s['temp']}, top_p {s['top_p']}, max tokens {s['tokens']}")
            continue
        if line == "/model":
            for k, v in MODELS.items():
                print(f" {'*' if k == name else ' '} {k:<12} {v}")
            continue
        if line.startswith("/model "):
            k = line.split(maxsplit=1)[1].lower()
            if k not in MODELS:
                print(f"unknown model {k!r}; choose: {', '.join(MODELS)}")
                continue
            name, (tok, model, window), last, ended = k, load(k), [], False
            eot = tok.convert_tokens_to_ids(EOT)
            continue
        if line.startswith(("/temp ", "/top_p ", "/tokens ")):
            key, _, value = line[1:].partition(" ")
            try:
                s[key] = int(value) if key == "tokens" else float(value)
                print(f"{key} = {s[key]}")
            except ValueError:
                print(f"not a number: {value!r}")
            continue
        if line.startswith("/") and line != "/more":
            print("unknown command; /help lists them")
            continue

        if line in ("", "/more"):
            if not last:
                print("nothing to continue yet; type a prompt")
                continue
            if ended:
                print("the last text already ended with <|endoftext|>; type a new prompt")
                continue
            ids = last
            print()
        else:
            ids = [eot] + tok.encode(line)                # start like a new document
            print("\n" + line, end="", flush=True)
        t0 = time.time()
        new, ended = generate(model, tok, ids, s["tokens"], s["temp"], s["top_p"], window)
        new = [t for t in new if t != eot]
        print(f"\n\033[2m[{len(new)} tokens, {time.time() - t0:.1f} s{', ended' if ended else ''}]\033[0m\n")
        last = ids + new


if __name__ == "__main__":
    main()
