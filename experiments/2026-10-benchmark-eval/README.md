# How do my small models do on standard benchmarks? (October 2026)

Zero-shot evaluation of my two models, [`tegancao/tinystories-22m`](https://huggingface.co/tegancao/tinystories-22m)
and [`tegancao/owt-45m`](https://huggingface.co/tegancao/owt-45m), next to nine public reference models of different
sizes: the original TinyStories models (Eldan & Li), EleutherAI's Pythia suite and OpenAI's GPT-2. Eight benchmarks,
full test sets, run with EleutherAI's [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
(v0.4.13).

> **TL;DR**
> - Small models learn **grammar before knowledge**: my 45M OpenWebText model matches Pythia-70M on the BLiMP grammar
>   test (73.8% vs. 73.3%) while staying near chance on reasoning benchmarks.
> - **The training data sets the ceiling**: TinyStories models are equally near chance on general benchmarks across a
>   70× range of sizes.
> - Compare models by **Transformer body size** and **bits per byte**, not total parameters or perplexity.
> - **Look inside benchmarks**: LAMBADA is mostly copying; a below-chance score can be a surface preference.

![Benchmarks vs. model size](scaling.png)

The x-axis is the size of each model's **Transformer body** (all parameters except the embedding and output layers),
which is the fairer measure of size: my two models have the same 12.5M body, the OpenWebText one is "45M" only because
of its larger vocabulary. Accuracy panels start at the random-chance line, so the height above it is the skill.
ARC-Challenge and Winogrande are left out of this plot because every model is near chance on them; they are in the
heatmap and the table below.

## The models

| Family | Models | Training data | Training tokens |
|---|---|---|---|
| **mine** | tinystories-22m, owt-45m | TinyStories / a sample of OpenWebText | 0.33B each |
| TinyStories (Eldan & Li, 2023) | 1M, 3M, 8M, 33M | TinyStories | not stated on the model cards |
| Pythia (Biderman et al., 2023) | 70M, 160M, 410M | the Pile (web, books, code, papers, **Wikipedia**, ...) | 300B |
| GPT-2 (Radford et al., 2019) | small, medium | WebText (Reddit-linked web pages, **Wikipedia removed**) | ≈ 9B per pass* |

\* OpenAI didn't publish GPT-2's training length; WebText is ~40 GB, about 9B tokens.

## Results

Accuracy in %, except WikiText (bits per byte, lower is better). Random chance: HellaSwag, ARC 25%; PIQA, Winogrande,
BLiMP 50%. Typical standard errors: HellaSwag ±0.4, BLiMP ±0.15, LAMBADA ±0.3–0.7, ARC-Easy ±1, PIQA and ARC-Challenge
±1.2, Winogrande ±1.4.

| Model | Body | Total | HellaSwag | PIQA | ARC-Easy | ARC-Challenge | Winogrande | LAMBADA | BLiMP | WikiText bits/byte |
|---|---|---|---|---|---|---|---|---|---|---|
| TinyStories-1M | 0.4M | 3.7M | 25.7 | 51.6 | 26.5 | 22.0 | 51.4 | 4.0 | 59.9 | 2.99 |
| TinyStories-3M | 1.6M | 8.3M | 26.0 | 52.3 | 27.5 | 21.9 | 51.0 | 8.4 | 59.9 | 3.02 |
| TinyStories-8M | 6.3M | 19.7M | 26.7 | 53.4 | 28.5 | 21.4 | 52.0 | 12.0 | 61.5 | 2.91 |
| **mine: OpenWebText 45M** | 12.5M | 45.2M | 26.6 | 53.0 | 30.7 | 21.8 | 49.6 | 6.1 | 73.8 | 1.45 |
| **mine: TinyStories 22M** | 12.5M | 22.7M | 26.5 | 51.6 | 25.3 | 22.6 | 51.5 | 10.0 | 59.8 | 3.11 |
| pythia-70m | 18.9M | 70.4M | 27.4 | 59.6 | 34.9 | 22.7 | 51.6 | 18.5 | 73.3 | 1.09 |
| TinyStories-33M | 28.3M | 68.5M | 27.1 | 53.0 | 26.8 | 23.4 | 52.3 | 11.7 | 64.3 | 2.98 |
| gpt2-small | 85.1M | 124.4M | 31.1 | 62.5 | 39.5 | 22.7 | 51.6 | 32.6 | 82.1 | 0.98 |
| pythia-160m | 85.1M | 162.3M | 30.3 | 61.4 | 39.6 | 24.1 | 51.5 | 32.7 | 77.5 | 0.95 |
| gpt2-medium | 302.3M | 354.8M | 39.4 | 66.4 | 43.6 | 25.0 | 53.1 | 43.0 | 82.7 | 0.89 |
| pythia-410m | 302.3M | 405.3M | 40.6 | 67.1 | 45.7 | 24.4 | 53.5 | 51.5 | 81.7 | 0.81 |

![Every model on every benchmark](heatmap.png)

The same results at a glance: each cell is a model's accuracy, coloured by how far it is above random guessing. The
BLiMP column lights up for every model; ARC-Challenge and Winogrande stay blank for all of them.

What the benchmarks ask: **HellaSwag**, the most plausible ending of an everyday scene; **PIQA**, the better way to do
a physical task; **ARC**, grade-school science questions (Easy and Challenge sets); **Winogrande**, who a pronoun refers
to; **LAMBADA**, the last word of a passage from a novel; **BLiMP**, which of two nearly identical sentences is
grammatical; **WikiText**, how well the model predicts Wikipedia articles, measured in bits per byte so that models
with different tokenizers can be compared fairly. Full numbers, including standard errors and all 67 BLiMP
sub-tests, are in [`results.json`](results.json).

## What the numbers say

**1. Two benchmarks show almost nothing at this scale.** On ARC-Challenge (21–25%) and Winogrande (49.6–53.5%) every
model up to 400M parameters stays close to chance; only the two largest edge a few points up. Several models are even
slightly *below* chance on ARC-Challenge, which can happen because its questions were selected so that simple methods
get them wrong. (Both plots have zoomed y-axes, so the wiggles look bigger than they are.) These tasks need far larger
models.

**2. The training data sets the ceiling.** The four TinyStories models span a 70× range of body sizes, yet on HellaSwag,
PIQA, ARC and WikiText they are all essentially the same, near chance. A model of children's stories doesn't learn
science or physics however big it gets; only the tasks close to its data (LAMBADA's story passages, grammar) improve
with size.

**3. Grammar comes first.** BLiMP is the benchmark where small models clearly learn something: every model is well
above chance. My OpenWebText model scores **73.8%, the same as Pythia-70M (73.3%)**, which has a larger body and saw about
1,000× more training tokens. Grammar is learned early and cheaply; knowledge is not.

BLiMP is really 67 sub-tests, each a set of 1,000 sentence pairs like *"The cats **are** sleeping"* vs. *"The cats
**is** sleeping"*. Grouped by grammar phenomenon, they show *which* grammar each model has learned:

![BLiMP by grammar phenomenon](blimp_phenomena.png)

- **Some grammar is easy even for tiny models:** anaphor agreement (*he ... himself*) and irregular verb forms already
  reach 83–87% for the 0.4M-body TinyStories model. Determiner–noun agreement (*this dog / these dogs*) is easy for the
  models trained on web text (89–96%) but not for the TinyStories models (60–72%).
- **Some stays hard even at 300M:** on island effects (which questions can be formed from a sentence), NPI licensing
  (where words like *any* and *ever* may appear) and quantifiers, no model gets above 77%.
- **My two models learned different grammar.** The OpenWebText model is far better at subject–verb agreement (77% vs.
  54%) and determiner–noun agreement (90% vs. 68%); web text has much more varied sentences than children's stories.
- **Below chance means a consistent wrong preference, not guessing.** My TinyStories model scores 36% on NPI licensing,
  mostly because of one sub-test where it picks the right sentence only 7% of the time:

  > ✓ *Had Bruce ever played?* &nbsp;&nbsp; ✗ *Bruce had ever played.*

  Every TinyStories model fails this sub-test (4–7%, whatever its size), while my OpenWebText model and Pythia-70M are
  near chance and GPT-2 small reaches 65%. The likely reason is a surface preference rather than a wrong rule about
  *ever*: children's stories are overwhelmingly statements, so the familiar statement shape scores as more likely than
  an inverted question. BLiMP compares whole-sentence probabilities, so it can't tell those two explanations apart.
  More broadly, the TinyStories models are clearly below chance on 13–16 of the 67 sub-tests, the web-trained models
  on only 3–5.

**4. Bits per byte gives the fair language-modeling comparison.** Per byte of Wikipedia text, my OpenWebText model needs
1.45 bits, between the TinyStories models (~3, essentially unable to model encyclopedia text) and Pythia/GPT-2 (0.8–1.1).
At equal size Pythia beats GPT-2 (0.95 vs. 0.98, 0.81 vs. 0.89), plausibly helped by having Wikipedia in its training
data, which GPT-2 deliberately excluded.

**5. Knowledge and reasoning need scale and data together.** HellaSwag, PIQA, ARC-Easy and LAMBADA only climb clearly
for the models with both a large body and lots of training data (Pythia, GPT-2), and fastest at the largest sizes.
At equal size, Pythia (300B tokens) and GPT-2 (≈ 9B+) are close; Pythia-410M pulls ahead on LAMBADA (51.5% vs. 43.0%).

**6. LAMBADA: accuracy and perplexity can disagree.** My TinyStories model beats my OpenWebText model on LAMBADA
accuracy (10.0% vs. 6.1%) while its perplexity is 19× worse (70,408 vs. 3,695). LAMBADA is largely a copying test: in 84%
of passages the missing last word already appears earlier, and 47% of the answers are capitalized, mostly names, which
children's stories repeat constantly. But the TinyStories tokenizer splits 29% of the answers into three or more
tokens (OpenWebText: 7%, GPT-2: 4.5%), and every extra token adds uncertainty, so its perplexity explodes.

## My predictions, checked

Before seeing the results I wrote down what I expected:

| Prediction | Result |
|---|---|
| ARC-Challenge and Winogrande near chance for all models | ✅ |
| BLiMP clearly above chance for every model | ✅ (and my OpenWebText model matches Pythia-70M) |
| WikiText: Pythia and GPT-2 best (Pythia ahead), my OpenWebText model in the middle, TinyStories models worst | ✅ |
| Pythia ≈ GPT-2 at equal body size | ✅ mostly; Pythia ahead on LAMBADA, GPT-2 small ahead on BLiMP |
| My TinyStories model between the official TinyStories-8M and -33M | ❌ |

The miss is the most interesting result for me: with a 2× larger body than TinyStories-8M, my TinyStories model is
slightly *below* it on BLiMP (59.8% vs. 61.5%) and LAMBADA (10.0% vs. 12.0%). The original models were most likely
trained for much longer (their training length isn't on the model cards; mine saw about 61% of the dataset once), and
they use a 50K-token vocabulary instead of my 10K. More training is the obvious next step.

## Model sizes and training budgets (my models vs. GPT-2 small)

| | TinyStories 22M | OpenWebText 45M | GPT-2 small |
|---|---|---|---|
| Parameters, total | 22.7M | 45.2M | 124.4M |
| of which embedding / output layers | 10.2M | 32.8M | 39.4M |
| of which **Transformer body** | **12.5M** | **12.5M** | **85.1M** |
| Vocabulary / context length | 10,000 / 256 | 32,000 / 256 | 50,257 / 1,024 |
| Training tokens | 0.33B | 0.33B | ≈ 9B per pass over WebText |
| Tokens per parameter | 14 | 7 | ≈ 73 |
| Training compute (≈ 6 × parameters × tokens) | 4.5 × 10¹⁶ FLOPs | 8.9 × 10¹⁶ FLOPs | ≈ 6.8 × 10¹⁸ FLOPs (~76× my OWT model) |

For its size my OpenWebText model saw few tokens: 7 per parameter, against the ~20 that the "Chinchilla" study
(Hoffmann et al., 2022) found compute-optimal.

## Takeaways

- **A good loss on your own data is not general ability.** My TinyStories model fits its stories well (validation loss
  1.38) and writes coherent text, yet it is at chance on every reasoning benchmark.
- **Data sets the ceiling, size and data together move it.** Growing a model on narrow data barely helps on general
  benchmarks; Pythia and GPT-2 improve because they combine size with large, broad data.
- **Small models learn grammar before knowledge.** BLiMP shows real progress where the reasoning benchmarks show none.
- **Know what a benchmark really tests,** and use metrics that are fair across tokenizers (bits per byte), not
  perplexity.
- **Small samples mislead.** A pilot on 200 HellaSwag questions put my TinyStories model at 32.5%; all 10,042 gave 26.5%.
  With 200 questions the standard error is about ±3 points.

## Details and caveats

- **Context length.** The harness truncates long inputs to each model's limit: 256 tokens for my models (the TinyStories
  one gets 512, because its tokenizer turns 4 PIQA answers into more than 256 tokens), 2,048 for the TinyStories
  references, 1,024 for GPT-2 and 2,048 for Pythia. Longer contexts help on LAMBADA and WikiText.
- **Data overlap.** Pythia's training data includes Wikipedia, so its WikiText numbers partly reflect having seen that
  kind of text (possibly the same articles).
- **Zero-shot** (no worked examples in the prompt). All scoring is by log-likelihood, so results are deterministic.
- **One training run per model.** My models were each trained once, with one random seed, so differences of a point or
  two between similar models (e.g. my TinyStories model vs. TinyStories-8M) may partly be run-to-run variation; the
  standard errors only cover the finite benchmark size, not training randomness.
- Run on Modal: one NVIDIA L4 per model, up to 8 at a time, about 30 minutes in total and a few dollars. On a laptop
  CPU this would take many hours.

## Reproduce

```
uv run --with modal modal run experiments/2026-10-benchmark-eval/modal_eval.py     # needs a Modal account
uv run experiments/2026-10-benchmark-eval/plot_results.py                          # redraws all three figures
```

Optional: `WANDB_PROJECT=<project>` also logs each model's results to Weights & Biases (via a Modal secret named
`wandb`), and `LIMIT=20` runs a quick smoke test on 20 examples per task.

## References

- Benchmarks: [HellaSwag](https://arxiv.org/abs/1905.07830) (Zellers et al., 2019), [PIQA](https://arxiv.org/abs/1911.11641)
  (Bisk et al., 2020), [ARC](https://arxiv.org/abs/1803.05457) (Clark et al., 2018), [WinoGrande](https://arxiv.org/abs/1907.10641)
  (Sakaguchi et al., 2019), [LAMBADA](https://arxiv.org/abs/1606.06031) (Paperno et al., 2016),
  [BLiMP](https://arxiv.org/abs/1912.00582) (Warstadt et al., 2020), [WikiText](https://arxiv.org/abs/1609.07843)
  (Merity et al., 2016).
- Models: [TinyStories](https://arxiv.org/abs/2305.07759) (Eldan & Li, 2023), [Pythia](https://arxiv.org/abs/2304.01373)
  (Biderman et al., 2023), GPT-2 (Radford et al., 2019, *Language Models are Unsupervised Multitask Learners*).
- [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556) (Hoffmann et al., 2022, "Chinchilla").
