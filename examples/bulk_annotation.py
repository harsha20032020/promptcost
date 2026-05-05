"""Bulk annotation: cost estimate using pre-counted token totals.

Useful when you already know your typical token sizes (from a prior
run's `response.usage`) and don't want to re-tokenize a huge corpus
on every estimate.

Scenario
--------
- System prompt:        2,500 tokens
- Avg input per call:   1,000 tokens
- Avg output per call:    100 tokens
- Volume:              25,000 requests

Run:  python -m examples.bulk_annotation
"""

from __future__ import annotations

from promptcost import Workload, estimate, fixed_output, print_estimate


SYSTEM_PROMPT_TOKENS = 2_500
AVG_INPUT_TOKENS = 1_000
AVG_OUTPUT_TOKENS = 100
N_DOCUMENTS = 25_000

MODELS = [
    "gpt-5.2",
    "anthropic.claude-sonnet-4-6",
    "gemini-2.5-flash",
]


def main() -> None:
    workload = Workload(
        system_prompt=SYSTEM_PROMPT_TOKENS,
        inputs=[AVG_INPUT_TOKENS],
        output=fixed_output(AVG_OUTPUT_TOKENS),
        expected_total=N_DOCUMENTS,
    )
    print_estimate(estimate(workload, MODELS))


if __name__ == "__main__":
    main()
