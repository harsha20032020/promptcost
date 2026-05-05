"""LiteLLM pricing: fetch+cache the JSON, look up per-model rates, persist user-supplied batch rates."""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

LITELLM_URL = (
    "https://raw.githubusercontent.com/BerriAI/litellm/main/"
    "model_prices_and_context_window.json"
)

CACHE_DIR = Path.home() / ".promptcost"
PRICING_FILE = CACHE_DIR / "litellm_pricing.json"
BATCH_OVERRIDES_FILE = CACHE_DIR / "batch_overrides.json"
TTL_SECONDS = 24 * 60 * 60


@dataclass
class ModelPricing:
    model_id: str
    provider: str
    input_per_token: float
    output_per_token: float
    batch_input_per_token: float | None = None
    batch_output_per_token: float | None = None
    cache_read_per_token: float | None = None
    cache_write_per_token: float | None = None
    supports_caching: bool = False
    above_200k_input_per_token: float | None = None
    above_200k_output_per_token: float | None = None


def fetch_pricing(refresh: str = "auto") -> dict[str, Any]:
    """Return LiteLLM's pricing JSON. refresh: 'auto' | 'always' | 'never'."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    have_cache = PRICING_FILE.exists()
    stale = (
        not have_cache
        or (time.time() - PRICING_FILE.stat().st_mtime) > TTL_SECONDS
    )
    should_fetch = (
        refresh == "always"
        or (refresh == "auto" and stale)
        or (refresh == "never" and not have_cache)
    )
    if should_fetch:
        try:
            r = httpx.get(LITELLM_URL, timeout=20.0, follow_redirects=True)
            r.raise_for_status()
            PRICING_FILE.write_text(json.dumps(r.json()), encoding="utf-8")
        except Exception as e:
            if not PRICING_FILE.exists():
                raise RuntimeError(f"Failed to fetch pricing JSON: {e}") from e
    return json.loads(PRICING_FILE.read_text(encoding="utf-8"))


def cache_age_seconds() -> float | None:
    if not PRICING_FILE.exists():
        return None
    return time.time() - PRICING_FILE.stat().st_mtime


def list_model_ids(catalog: dict[str, Any]) -> list[str]:
    return sorted(k for k, v in catalog.items() if isinstance(v, dict))


def get_pricing(catalog: dict[str, Any], model_id: str) -> ModelPricing:
    """Parse one entry. Raises KeyError if the model is missing or has no pricing."""
    entry = catalog.get(model_id)
    if not isinstance(entry, dict):
        raise KeyError(f"Model {model_id!r} not in LiteLLM catalog (use exact id).")
    if "input_cost_per_token" not in entry or "output_cost_per_token" not in entry:
        raise KeyError(f"Model {model_id!r} has no usable pricing.")

    provider = entry.get("litellm_provider") or (
        model_id.split("/", 1)[0] if "/" in model_id else "openai"
    )

    return ModelPricing(
        model_id=model_id,
        provider=provider,
        input_per_token=entry["input_cost_per_token"],
        output_per_token=entry["output_cost_per_token"],
        batch_input_per_token=entry.get("input_cost_per_token_batches"),
        batch_output_per_token=entry.get("output_cost_per_token_batches"),
        cache_read_per_token=entry.get("cache_read_input_token_cost"),
        cache_write_per_token=entry.get("cache_creation_input_token_cost"),
        supports_caching=bool(entry.get("supports_prompt_caching", False)),
        above_200k_input_per_token=entry.get("input_cost_per_token_above_200k_tokens"),
        above_200k_output_per_token=entry.get("output_cost_per_token_above_200k_tokens"),
    )


# ---------- batch-rate overrides ----------

def _load_overrides() -> dict[str, dict[str, float]]:
    if not BATCH_OVERRIDES_FILE.exists():
        return {}
    try:
        return json.loads(BATCH_OVERRIDES_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_overrides(overrides: dict[str, dict[str, float]]) -> None:
    BATCH_OVERRIDES_FILE.parent.mkdir(parents=True, exist_ok=True)
    BATCH_OVERRIDES_FILE.write_text(
        json.dumps(overrides, indent=2), encoding="utf-8"
    )


def get_batch_override(model_id: str) -> tuple[float, float] | None:
    """Return (input_per_token, output_per_token) — converted from the stored $/M units."""
    entry = _load_overrides().get(model_id)
    if not entry:
        return None
    return entry["input_per_M"] / 1e6, entry["output_per_M"] / 1e6


def set_batch_override(model_id: str, input_per_M: float, output_per_M: float) -> None:
    overrides = _load_overrides()
    overrides[model_id] = {
        "input_per_M": float(input_per_M),
        "output_per_M": float(output_per_M),
    }
    _save_overrides(overrides)


def clear_batch_override(model_id: str) -> None:
    overrides = _load_overrides()
    overrides.pop(model_id, None)
    _save_overrides(overrides)


def ask_for_batch_rates(pricing: ModelPricing) -> tuple[float, float] | None:
    """Interactive prompt — only if stdin is a TTY. Saves the answer for future runs."""
    if not sys.stdin.isatty():
        return None
    std_in = pricing.input_per_token * 1e6
    std_out = pricing.output_per_token * 1e6
    print(
        f"\nBatch pricing missing for {pricing.model_id!r}.\n"
        f"  Standard rates: input ${std_in:.4f}/M  output ${std_out:.4f}/M\n"
        f"  Enter batch rates ($ per 1M tokens), or blank to skip:"
    )
    try:
        in_str = input("    Input  $/M: ").strip()
        if not in_str:
            return None
        out_str = input("    Output $/M: ").strip()
        if not out_str:
            return None
        in_per_M = float(in_str)
        out_per_M = float(out_str)
    except (EOFError, ValueError):
        print("  Skipped.")
        return None
    set_batch_override(pricing.model_id, in_per_M, out_per_M)
    print(f"  Saved → {BATCH_OVERRIDES_FILE}")
    return in_per_M / 1e6, out_per_M / 1e6


def resolve_batch_rates(
    pricing: ModelPricing, prompt_if_missing: bool = True
) -> tuple[float, float] | None:
    """1) catalog rates → 2) saved override → 3) interactive prompt (if TTY)."""
    if (
        pricing.batch_input_per_token is not None
        and pricing.batch_output_per_token is not None
    ):
        return pricing.batch_input_per_token, pricing.batch_output_per_token
    saved = get_batch_override(pricing.model_id)
    if saved is not None:
        return saved
    if prompt_if_missing:
        return ask_for_batch_rates(pricing)
    return None
