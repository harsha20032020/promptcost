"""Typer CLI: estimate / models / refresh / caches / batch-rate."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import typer

from .estimator import (
    Workload,
    empirical_output,
    estimate,
    fixed_output,
    print_estimate,
    print_models,
)
from .pricing import (
    BATCH_OVERRIDES_FILE,
    PRICING_FILE,
    clear_batch_override,
    fetch_pricing,
    get_batch_override,
    list_model_ids,
    set_batch_override,
)

app = typer.Typer(add_completion=False, help="promptcost: estimate LLM workload costs.")


def _split(csv: str) -> list[str]:
    return [s.strip() for s in csv.split(",") if s.strip()]


def _load_prompt(value: str) -> str:
    """If `value` is a path that exists, return its contents; otherwise return as-is."""
    if len(value) < 4096:
        p = Path(value)
        if p.exists() and p.is_file():
            return p.read_text(encoding="utf-8")
    return value


def _load_data(value: str) -> list[str]:
    """Path to .jsonl/.txt, or comma-separated literals."""
    p = Path(value)
    if p.exists() and p.is_file():
        if p.suffix.lower() == ".jsonl":
            out: list[str] = []
            for line in p.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                v = json.loads(line)
                out.append(v if isinstance(v, str) else json.dumps(v, ensure_ascii=False))
            return out
        return [
            line for line in p.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    return _split(value)


def _load_samples(path: str) -> list[int]:
    """JSON list of ints, or one int per line."""
    text = Path(path).read_text(encoding="utf-8").strip()
    if text.startswith("["):
        return [int(v) for v in json.loads(text)]
    return [int(line) for line in text.splitlines() if line.strip()]


@app.command("estimate")
def cmd_estimate(
    prompt: str = typer.Option(..., "--prompt", help="Prompt text or path to a prompt file."),
    data: str = typer.Option(..., "--data", help="Path to .jsonl/.txt or comma-separated literals."),
    models: str = typer.Option(..., "--models", help="Comma-separated LiteLLM model ids."),
    tiers: str = typer.Option("standard,batch,cached", "--tiers"),
    output_mode: str = typer.Option("fixed", "--output-mode", help="fixed | empirical"),
    max_tokens: int = typer.Option(500, "--max-tokens"),
    samples: Optional[str] = typer.Option(
        None, "--samples", help="Required for --output-mode=empirical."
    ),
    extrapolate_to: Optional[int] = typer.Option(None, "--extrapolate-to"),
    json_out: Optional[str] = typer.Option(None, "--json", help="Write JSON report."),
    refresh: str = typer.Option("auto", "--pricing-refresh", help="auto | always | never"),
):
    """Estimate cost of a workload across one or more models / tiers."""
    if output_mode == "fixed":
        out = fixed_output(max_tokens)
    elif output_mode == "empirical":
        if not samples:
            typer.echo("--output-mode=empirical requires --samples", err=True)
            raise typer.Exit(2)
        out = empirical_output(_load_samples(samples))
    else:
        typer.echo(f"Unknown --output-mode: {output_mode}", err=True)
        raise typer.Exit(2)

    workload = Workload(
        system_prompt=_load_prompt(prompt),
        inputs=_load_data(data),
        output=out,
        expected_total=extrapolate_to,
    )

    est = estimate(
        workload,
        models=_split(models),
        tiers=_split(tiers),
        refresh=refresh,
    )
    print_estimate(est)
    if json_out:
        Path(json_out).write_text(
            json.dumps(asdict(est), indent=2, default=str),
            encoding="utf-8",
        )
        typer.echo(f"JSON report → {json_out}")


@app.command("models")
def cmd_models(
    grep: Optional[str] = typer.Option(None, "--grep", help="Substring filter."),
    show: Optional[str] = typer.Option(None, "--show", help="Comma-separated ids to pretty-print."),
):
    """List LiteLLM model ids, or pretty-print details for selected ones."""
    catalog = fetch_pricing(refresh="auto")
    if show:
        print_models(catalog, _split(show))
        return
    needle = grep.lower() if grep else None
    for mid in list_model_ids(catalog):
        if needle is None or needle in mid.lower():
            typer.echo(mid)


@app.command("refresh")
def cmd_refresh():
    """Force-refresh the pricing JSON cache."""
    fetch_pricing(refresh="always")
    typer.echo(f"Refreshed → {PRICING_FILE}")


@app.command("caches")
def cmd_caches():
    """Show where promptcost stores its caches."""
    typer.echo(f"Pricing JSON:    {PRICING_FILE}")
    typer.echo(f"Batch overrides: {BATCH_OVERRIDES_FILE}")


@app.command("batch-rate")
def cmd_batch_rate(
    model: str = typer.Argument(..., help="Exact LiteLLM model id."),
    input_per_M: Optional[float] = typer.Option(None, "--input", help="$ per 1M input tokens."),
    output_per_M: Optional[float] = typer.Option(None, "--output", help="$ per 1M output tokens."),
    clear: bool = typer.Option(False, "--clear", help="Remove the saved override."),
):
    """Save or inspect a batch-rate override for one model."""
    if clear:
        clear_batch_override(model)
        typer.echo(f"Cleared override for {model}")
        return
    if input_per_M is not None and output_per_M is not None:
        set_batch_override(model, input_per_M, output_per_M)
        typer.echo(f"Saved {model}: input ${input_per_M}/M  output ${output_per_M}/M")
        return
    saved = get_batch_override(model)
    if saved is None:
        typer.echo(f"No override saved for {model}.")
    else:
        in_per_M = saved[0] * 1e6
        out_per_M = saved[1] * 1e6
        typer.echo(
            f"{model}: input ${in_per_M}/M  output ${out_per_M}/M  "
            f"({BATCH_OVERRIDES_FILE})"
        )


if __name__ == "__main__":
    app()
