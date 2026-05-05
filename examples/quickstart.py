"""Quickstart: cost estimate from a handful of input/output sentences.

Edit the four knobs below, then run:
    python -m examples.quickstart
"""

from promptcost import Workload, empirical_output, estimate, print_estimate


# --- knobs ---

SYSTEM_PROMPT_TOKENS = 100  # int = pre-counted token count; or pass a string.

INPUT_SENTENCES = [
    "Apple announces new MacBook Pro with M5 chip in Cupertino.",
    "Federal Reserve raises rates by 25 basis points amid inflation concerns.",
    "OpenAI releases GPT-5 to enterprise customers in beta.",
    "European Union fines Google €2.4 billion in antitrust ruling.",
    "SpaceX launches 60 Starlink satellites from Cape Canaveral.",
]

OUTPUT_SENTENCES = [
    "Apple, MacBook Pro M5, Cupertino — tech announcement.",
    "Jerome Powell, Federal Reserve, 25 basis points — finance.",
    "OpenAI, GPT-5, enterprise beta — tech release.",
    "European Union, Google, antitrust fine — politics, finance.",
    "SpaceX, Starlink, Cape Canaveral — tech, science.",
]

N_DATAPOINTS = 25_000

MODELS = [
    "gpt-5.2",
    "anthropic.claude-sonnet-4-6",
    "gemini-2.5-flash",
]


def main() -> None:
    workload = Workload(
        system_prompt=SYSTEM_PROMPT_TOKENS,
        inputs=INPUT_SENTENCES,
        output=empirical_output(OUTPUT_SENTENCES),
        expected_total=N_DATAPOINTS,
    )
    print_estimate(estimate(workload, MODELS))


if __name__ == "__main__":
    main()
