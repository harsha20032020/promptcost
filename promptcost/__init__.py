"""promptcost — a-priori cost estimation for LLM workloads."""

from .estimator import (
    Estimate,           # top-level result: pricing age + one ModelEstimate per model
    ModelEstimate,      # per-model: token totals + cost-per-tier breakdown
    OutputDistribution, # output-length stats: mean / p50 / p95 / std / sample size
    TierCost,           # one row in the cost table: tier name, total $, available?, note
    Workload,           # the inputs to estimate: system prompt + inputs + output strategy + volume
    empirical_output,   # output strategy from measured samples (list[int] tokens or list[str] responses)
    estimate,           # main entrypoint: tokenize + look up pricing + compute tier costs
    fixed_output,       # output strategy: every call uses max_tokens (worst case)
    print_estimate,     # pretty-print the cost matrix (rows = models, cols = tiers)
    print_models,       # pretty-print headline rates for a list of models
)
from .pricing import (
    ModelPricing,       # parsed pricing for one model: rates, batch, cache, above-200K
    fetch_pricing,      # download + cache LiteLLM JSON (24h TTL); refresh: auto/always/never
    get_pricing,        # look up one model's ModelPricing in the catalog (raises KeyError)
    list_model_ids,     # all model ids in the catalog (sorted)
)

__all__ = [
    "Workload",
    "estimate",
    "fixed_output",
    "empirical_output",
    "print_estimate",
    "print_models",
    "fetch_pricing",
    "get_pricing",
    "list_model_ids",
    "ModelPricing",
    "Estimate",
    "ModelEstimate",
    "TierCost",
    "OutputDistribution",
]
__version__ = "0.2.0"
