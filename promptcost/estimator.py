"""Cost estimation: tokenize inputs, compute standard/batch/cached costs, print a table."""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import Any

from rich.console import Console
from rich.table import Table

from .pricing import (
    PRICING_FILE,
    ModelPricing,
    cache_age_seconds,
    fetch_pricing,
    get_pricing,
    resolve_batch_rates,
)


# ============ output-length strategies ============

@dataclass
class OutputDistribution:
    mean: float
    p50: float
    p95: float
    std: float
    n_samples: int
    method: str


def fixed_output(max_tokens: int) -> dict:
    """Worst case: every call uses `max_tokens`."""
    return {"kind": "fixed", "max_tokens": max_tokens}


def empirical_output(samples: list[int] | list[str]) -> dict:
    """From measured token counts (list[int]) or actual response strings (list[str])."""
    if not samples:
        raise ValueError("empirical_output: samples is empty.")
    return {"kind": "empirical", "samples": list(samples)}


def _summarize(counts: list[int], method: str) -> OutputDistribution:
    s = sorted(int(v) for v in counts)
    # ceil(0.95·n)-1 picks the right index regardless of n: e.g. n=10 → idx 9
    # (the top value, since there's no true "95th of 10"), n=20 → idx 18.
    p95_idx = max(0, math.ceil(0.95 * len(s)) - 1)
    return OutputDistribution(
        mean=statistics.fmean(s),
        p50=float(s[len(s) // 2]),
        p95=float(s[p95_idx]),
        std=statistics.pstdev(s) if len(s) > 1 else 0.0,
        n_samples=len(s),
        method=method,
    )


def _resolve_output(strategy: dict, model_id: str) -> OutputDistribution:
    if strategy["kind"] == "fixed":
        n = float(strategy["max_tokens"])
        return OutputDistribution(
            mean=n, p50=n, p95=n, std=0.0,
            n_samples=0, method=f"fixed({int(n)})",
        )
    samples = strategy["samples"]
    # list[int]: pre-counted token totals — same distribution for every model.
    # list[str]: response strings — re-tokenized per model since counts differ
    # across tokenizers (cl100k vs Claude vs Gemini).
    if all(isinstance(v, int) for v in samples):
        return _summarize(samples, "empirical(int)")
    if all(isinstance(v, str) for v in samples):
        return _summarize(
            [count_tokens(model_id, s) for s in samples], "empirical(str)"
        )
    raise TypeError("empirical samples must be uniformly list[int] or list[str].")


# ============ workload ============

@dataclass
class Workload:
    """Inputs for one estimation run.

    `system_prompt` and items in `inputs` accept either raw text (the
    library tokenizes them) or a pre-counted int (skips tokenization —
    useful for huge corpora).
    """
    system_prompt: str | int
    inputs: list[str | int]
    output: dict
    # `expected_total` lets you estimate from a small sample but project the
    # cost to a much larger run. Falls back to len(inputs) when omitted.
    expected_total: int | None = None

    @property
    def n_calls(self) -> int:
        return self.expected_total or len(self.inputs)


# ============ tokenization ============

def count_tokens(model_id: str, text: str | int | None) -> int:
    """Tokenize via litellm; pass-through for ints (already-counted)."""
    if text is None or text == "":
        return 0
    # Pre-counted int = caller already knows the token total. Skip the work
    # entirely so big corpora don't pay tokenization cost on every run.
    if isinstance(text, int):
        return text
    import litellm
    return int(litellm.token_counter(model=model_id, text=str(text)))


# ============ cost math ============

@dataclass
class TierCost:
    name: str
    total: float
    available: bool = True
    note: str = ""


@dataclass
class ModelEstimate:
    model_id: str
    provider: str
    shared_input_tokens: int
    avg_input_per_call: float
    total_input_tokens: int
    output_per_call: float
    total_output_tokens: float
    output_dist: OutputDistribution
    tiers: dict[str, TierCost] = field(default_factory=dict)


@dataclass
class Estimate:
    n_calls: int
    pricing_age_seconds: float | None
    models: list[ModelEstimate]


def _compute_tiers(
    pricing: ModelPricing,
    shared_tokens: int,
    avg_input_per_call: float,
    n_calls: int,
    output_per_call: float,
    requested_tiers: list[str],
    prompt_if_missing: bool,
) -> dict[str, TierCost]:
    # "per-call" = tokens unique to each call (the datapoint).
    # "shared"   = system prompt — identical across calls, what caching exploits.
    per_call_input_total = int(avg_input_per_call * n_calls)
    total_input = shared_tokens * n_calls + per_call_input_total
    total_output = output_per_call * n_calls

    # Some Anthropic / Gemini models step up rates above 200K input tokens
    # per call. We compare the *typical* call's size, not per-call billing.
    in_rate = pricing.input_per_token
    out_rate = pricing.output_per_token
    if (
        pricing.above_200k_input_per_token
        and (shared_tokens + avg_input_per_call) > 200_000
    ):
        in_rate = pricing.above_200k_input_per_token
        out_rate = pricing.above_200k_output_per_token or out_rate

    tiers: dict[str, TierCost] = {}

    if "standard" in requested_tiers:
        tiers["standard"] = TierCost(
            "standard", total_input * in_rate + total_output * out_rate
        )

    if "batch" in requested_tiers:
        rates = resolve_batch_rates(pricing, prompt_if_missing=prompt_if_missing)
        if rates is None:
            tiers["batch"] = TierCost(
                "batch", 0.0, available=False, note="no batch pricing"
            )
        else:
            from_catalog = pricing.batch_input_per_token is not None
            tiers["batch"] = TierCost(
                "batch",
                total_input * rates[0] + total_output * rates[1],
                note="" if from_catalog else "user-supplied batch rates",
            )

    if "cached" in requested_tiers:
        can_cache = (
            pricing.supports_caching
            and pricing.cache_read_per_token is not None
            and n_calls > 1
        )
        if can_cache:
            # Cache economics: the first call pays a one-time write on the
            # shared chunk; the remaining (n-1) calls read it at the (cheap)
            # cache_read rate. The per-call (non-shared) tokens always pay
            # the standard input rate — caching only helps the shared part.
            cache_read_cost = (
                (n_calls - 1) * shared_tokens * pricing.cache_read_per_token
            )
            cache_write_cost = shared_tokens * (pricing.cache_write_per_token or 0.0)
            tiers["cached"] = TierCost(
                "cached",
                per_call_input_total * in_rate
                + total_output * out_rate
                + cache_read_cost
                + cache_write_cost,
                note="100% cache hit on shared portion",
            )
        else:
            tiers["cached"] = TierCost(
                "cached", 0.0, available=False, note="caching not supported"
            )

    return tiers


def estimate(
    workload: Workload,
    models: list[str],
    *,
    tiers: list[str] | None = None,
    refresh: str = "auto",
    prompt_for_missing_batch: bool = True,
) -> Estimate:
    """Run the full pipeline for every model. Skips models missing from the catalog."""
    tiers = tiers or ["standard", "batch", "cached"]
    catalog = fetch_pricing(refresh=refresh)
    results: list[ModelEstimate] = []

    for model_id in models:
        try:
            pricing = get_pricing(catalog, model_id)
        except KeyError as e:
            print(f"[skip] {e}")
            continue

        # Tokenize per-model: same text yields different counts across
        # tokenizers, so each model gets its own shared/avg numbers.
        shared = count_tokens(model_id, workload.system_prompt)

        if workload.inputs:
            counts = [count_tokens(model_id, x) for x in workload.inputs]
            avg_input = sum(counts) / len(counts)
        else:
            avg_input = 0.0

        n = workload.n_calls
        total_input = shared * n + int(avg_input * n)

        dist = _resolve_output(workload.output, model_id)
        out_per_call = dist.mean

        results.append(ModelEstimate(
            model_id=model_id,
            provider=pricing.provider,
            shared_input_tokens=shared,
            avg_input_per_call=avg_input,
            total_input_tokens=total_input,
            output_per_call=out_per_call,
            total_output_tokens=out_per_call * n,
            output_dist=dist,
            tiers=_compute_tiers(
                pricing, shared, avg_input, n, out_per_call,
                tiers, prompt_for_missing_batch,
            ),
        ))

    return Estimate(
        n_calls=workload.n_calls,
        pricing_age_seconds=cache_age_seconds(),
        models=results,
    )


# ============ pretty-printing ============

def _money(amount: float) -> str:
    if amount != amount:  # NaN sentinel for "no available tier"
        return "—"
    if amount >= 100:
        return f"${amount:,.2f}"
    if amount >= 1:
        return f"${amount:,.3f}"
    return f"${amount:,.4f}"


def _rate(rate: float | None) -> str:
    return "—" if rate is None else f"${rate * 1e6:,.2f}/M"


def print_models(catalog: dict[str, Any], model_ids: list[str]) -> None:
    """Pretty-print headline rates for the given models."""
    console = Console()
    console.rule("[bold]Models[/bold]")
    table = Table(show_header=True, header_style="bold")
    table.add_column("Model")
    table.add_column("Provider")
    for col in ("Input", "Output", "Batch in", "Cache read"):
        table.add_column(col, justify="right")
    table.add_column("Caching", justify="center")
    for mid in model_ids:
        try:
            p = get_pricing(catalog, mid)
        except KeyError as e:
            print(f"[skip] {e}")
            continue
        table.add_row(
            p.model_id, p.provider,
            _rate(p.input_per_token), _rate(p.output_per_token),
            _rate(p.batch_input_per_token), _rate(p.cache_read_per_token),
            "✓" if p.supports_caching else "—",
        )
    console.print(table)
    console.print(f"[dim]Pricing cache: {PRICING_FILE}[/dim]")


def print_estimate(est: Estimate) -> None:
    """Pretty-print the cost matrix: rows are models, columns are tiers."""
    console = Console()
    console.rule(f"[bold]promptcost[/bold] · {est.n_calls:,} calls")
    if est.pricing_age_seconds is not None:
        console.print(
            f"Pricing age: {est.pricing_age_seconds / 3600:.1f}h  "
            f"[dim]({PRICING_FILE})[/dim]"
        )

    tier_names = sorted({t for m in est.models for t in m.tiers})
    table = Table(show_header=True, header_style="bold")
    table.add_column("Model")
    table.add_column("Provider")
    table.add_column("Out/call", justify="right")
    for t in tier_names:
        table.add_column(t.capitalize(), justify="right")
    table.add_column("Best", justify="right")

    cheapest: tuple[str | None, float] = (None, float("inf"))
    for m in est.models:
        avail = [(name, c.total) for name, c in m.tiers.items() if c.available]
        best_name, best_total = (
            min(avail, key=lambda r: r[1]) if avail
            else ("standard", float("nan"))
        )
        row = [m.model_id, m.provider, f"{m.output_per_call:.0f}"]
        for t in tier_names:
            c = m.tiers.get(t)
            row.append(_money(c.total) if c and c.available else "—")
        row.append(f"{best_name} ({_money(best_total)})")
        # NaN ≠ NaN — this filters out models with no available tier.
        if best_total == best_total and best_total < cheapest[1]:
            cheapest = (m.model_id, best_total)
        table.add_row(*row)

    console.print(table)
    if cheapest[0]:
        console.print(
            f"\n[bold green]Cheapest:[/bold green] "
            f"{cheapest[0]} → {_money(cheapest[1])}"
        )
    if est.models:
        d = est.models[0].output_dist
        console.print(
            f"\nOutput ({d.method}): "
            f"mean={d.mean:.0f} p50={d.p50:.0f} p95={d.p95:.0f}"
        )
    console.print(
        "\n[dim]Cached tier assumes 100% hit rate on shared portion.[/dim]"
    )
