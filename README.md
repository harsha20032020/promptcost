# promptcost

A-priori cost estimation for LLM workloads across **every provider that [LiteLLM](https://github.com/BerriAI/litellm) supports** — OpenAI, Anthropic, Gemini, Bedrock, Cohere, Mistral, DeepSeek, Together, Fireworks, Groq, vLLM-hosted open models, and more. Models the **standard / batch / cached** pricing tiers honestly, lets you supply pre-counted token totals so you don't re-tokenize huge corpora, and asks (once) for batch rates the catalog hasn't filled in.

```
────────────────────── promptcost · 25,000 calls ──────────────────────
┃ Model                       ┃ Provider          ┃ Out/call ┃   Batch ┃  Cached ┃ Standard ┃        Best ┃
│ gpt-5.2                     │ openai            │      100 │ $94.062 │ $89.687 │  $188.12 │ cached ($89.687)
│ anthropic.claude-sonnet-4-6 │ bedrock_converse  │      100 │ $150.00 │ $131.26 │  $300.00 │ cached ($131.26)
│ gemini-2.5-flash            │ vertex_ai         │      100 │ $16.250 │ $15.625 │  $32.500 │ cached ($15.625)
Cheapest: gemini-2.5-flash → $15.625
```

## Install

```bash
pip install promptcostimator
# the import name is still `promptcost`:
python -c "from promptcost import estimate"

# Or, for development:
git clone https://github.com/harsha20032020/promptcost && cd promptcost && pip install -e .
```

## Quick start (Python)

```python
from promptcost import Workload, estimate, fixed_output, print_estimate

workload = Workload(
    system_prompt="You are an entity extractor. Return JSON with people, orgs, locations.",
    inputs=[
        "Apple announces new MacBook Pro with M5 chip in Cupertino.",
        "Federal Reserve raises rates by 25 basis points amid inflation concerns.",
    ],
    output=fixed_output(150),       # or empirical_output([...samples...])
    expected_total=25_000,          # extrapolate cost to a 25K-call run
)

print_estimate(estimate(workload, models=[
    "gpt-5.2",
    "anthropic.claude-sonnet-4-6",
    "gemini-2.5-flash",
]))
```

That's the whole API. `Workload` defines the inputs, `estimate()` runs the pipeline, `print_estimate()` formats the result.

## Inputs: text, pre-counted ints, or a mix

Three of `Workload`'s fields each accept either raw text **or** a pre-counted token integer:

| field           | as text/list                              | as int (skip tokenization)       |
|-----------------|-------------------------------------------|----------------------------------|
| `system_prompt` | `"You are an entity extractor..."`        | `2_500`                          |
| `inputs[i]`     | `"Apple announces new MacBook..."`        | `1_000` (avg tokens per call)    |

Use the int form when you already know your token sizes (from a prior run's `response.usage` or a one-off measurement) and don't want to pay tokenization cost on every estimate. See [`examples/bulk_annotation.py`](examples/bulk_annotation.py) for a 25K-call estimate driven entirely by pre-counted ints.

## Output-length strategies

You have to tell promptcost something about output length — there's no honest way to guess it from the inputs alone.

```python
from promptcost import fixed_output, empirical_output

output=fixed_output(500)                # worst case: every call uses 500 tokens
output=empirical_output([12, 47, 33])   # measured token counts (same for every model)
output=empirical_output([               # actual response strings — re-tokenized per model
    '{"people": ["Tim Cook"], "orgs": ["Apple"]}',
    '{"people": [], "orgs": ["Federal Reserve"]}',
])
```

`empirical_output` summarizes the samples into mean / p50 / p95 / std. The mean drives the cost; the percentiles are reported for situational awareness.

## Pricing tiers

`estimate()` computes three tiers per model and picks the cheapest available one as "Best":

- **standard** — headline rates × tokens.
- **batch** — batch rates × tokens, when the model offers them. If LiteLLM doesn't carry batch rates and you're running interactively, promptcost prompts you once and saves the answer to `~/.promptcost/batch_overrides.json`.
- **cached** — one-time cache write on the *shared* chunk (system prompt) plus (n-1) cache reads, plus per-call tokens at the standard input rate. Assumes 100% hit rate on the shared portion. Skipped automatically when the model doesn't support prompt caching.

Anthropic and Gemini's "above 200K context" step rates are applied automatically when the typical call crosses the threshold.

To restrict to a subset:

```python
estimate(workload, models, tiers=["standard", "batch"])
```

## Missing batch rates

Some LiteLLM entries don't carry batch pricing. promptcost falls back in this order:

1. Catalog batch rates, if present.
2. Saved override in `~/.promptcost/batch_overrides.json`.
3. Interactive prompt (TTY only) — your answer is saved for next time.

Inspect or change saved overrides via the CLI:

```bash
promptcost batch-rate gpt-5.2 --input 1.25 --output 5.00   # save
promptcost batch-rate gpt-5.2                              # show current
promptcost batch-rate gpt-5.2 --clear                      # remove
```

## Caches

| Path                                   | Contents                                                |
|----------------------------------------|---------------------------------------------------------|
| `~/.promptcost/litellm_pricing.json`   | LiteLLM pricing JSON — refreshed every 24h by default.  |
| `~/.promptcost/batch_overrides.json`   | User-supplied batch rates for models the catalog skips. |

Pricing refresh is configurable: `auto` (the default — refresh if older than 24h), `always`, or `never`.

```python
estimate(workload, models, refresh="always")
```

```bash
promptcost refresh   # force-refresh the catalog
promptcost caches    # show where the caches live
```

## Supported models

Any model id present in the LiteLLM pricing catalog works — use the **exact** id (no fuzzy matching). To browse:

```bash
promptcost models --grep claude-sonnet-4
promptcost models --show gpt-5.2,claude-haiku-4-5,gemini-2.5-flash
```

Or in Python:

```python
from promptcost import fetch_pricing, list_model_ids
print([m for m in list_model_ids(fetch_pricing()) if "sonnet" in m])
```

Tokenization runs entirely **locally** via `litellm.token_counter` — no API calls per datapoint, even for Anthropic and Gemini, since LiteLLM ships bundled tokenizers for every model in its catalog.

## CLI

```bash
promptcost estimate \
  --prompt prompt.txt \
  --data data.jsonl \
  --models gpt-5.2,claude-sonnet-4-6,gemini-2.5-flash \
  --tiers standard,batch,cached \
  --output-mode fixed --max-tokens 150 \
  --extrapolate-to 25000 \
  --json report.json
```

Empirical mode is also available:

```bash
promptcost estimate ... --output-mode empirical --samples samples.json
```

`samples.json` can be a JSON list of ints or a file with one int per line.

Other commands:

```bash
promptcost models --grep gpt-5
promptcost refresh
promptcost caches
promptcost batch-rate <model_id>          # show / set / clear (see flags)
```

## What this library deliberately does *not* do

- **It doesn't run your prompts.** It estimates cost — no LLM calls happen.
- **It doesn't guess output length.** Pick `fixed_output` or `empirical_output`. The library won't infer length from input length, schema, or vibes.
- **It doesn't model JSON schema overhead.** If you send structured output envelopes, factor them into `system_prompt` (e.g. add the schema's token count to the int).
- **It doesn't second-guess LiteLLM.** Model ids must match the catalog exactly. No fuzzy match, no aliases.
- **It doesn't model partial cache hits.** The `cached` tier assumes 100% hit on the shared chunk — a useful upper-bound for the savings you'd see with a stable system prompt.

## Project layout

```
promptcost/
├── __init__.py     re-exports
├── pricing.py      LiteLLM JSON fetch+cache, ModelPricing, batch-rate prompt+save
├── estimator.py    Workload + tokenize + fixed/empirical output + tier math + rich table
└── cli.py          typer: estimate / models / refresh / caches / batch-rate

examples/
├── quickstart.py        input/output sentences, empirical output, 25K extrapolation
└── bulk_annotation.py   pre-counted token totals, fixed output, 25K-doc pipeline
```

## License

MIT.

## Contact

Harsha Vardhan Nemani — [nhvardhan2020@gmail.com](mailto:nhvardhan2020@gmail.com)
Issues and feature requests: [github.com/harsha20032020/promptcost/issues](https://github.com/harsha20032020/promptcost/issues)
