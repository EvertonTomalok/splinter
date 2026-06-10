"""Pre-run cost estimator for ladder tiers.

Values are relative cost proxies suitable for *comparing* tiers — not billing.
opencode-go/* models are subscription-metered; the numbers below reflect observed
relative credit consumption. Claude models use real USD/MTok.
"""

from __future__ import annotations

from splinter.models.roster import Tier

# Token estimates (input, output) by effort level.
_EFFORT_TOKENS: dict[str, tuple[int, int]] = {
    "trivial": (4_000, 500),
    "normal": (8_000, 1_500),
    "hard": (16_000, 3_000),
    "critical": (32_000, 6_000),
}
_DEFAULT_TOKENS: tuple[int, int] = (8_000, 1_500)

# (input_price_per_mtok, output_price_per_mtok) by model id.
_MODEL_PRICES: dict[str, tuple[float, float]] = {
    "opencode/deepseek-v4-flash-free": (0.27, 1.10),
    "opencode-go/deepseek-v4-flash": (0.27, 1.10),
    "opencode-go/deepseek-v4-pro": (0.27, 1.10),
    "opencode-go/minimax-m3": (0.80, 4.50),
    "opencode-go/kimi-k2.6": (0.55, 2.50),
    "opencode-go/qwen3.7-plus": (1.50, 6.00),
    "sonnet": (3.00, 15.00),
    "opus": (15.00, 75.00),
    "haiku": (0.25, 1.25),
}
_DEFAULT_PRICE: tuple[float, float] = (1.0, 5.0)


def _price_for(model_id: str) -> tuple[float, float]:
    if model_id in _MODEL_PRICES:
        return _MODEL_PRICES[model_id]
    for prefix, price in _MODEL_PRICES.items():
        if model_id.startswith(prefix):
            return price
    return _DEFAULT_PRICE


def estimate_tier_cost(tier: Tier, effort: str) -> float:
    """Estimated relative cost for running *effort* workload on *tier*.

    Result is a float in the same unit as _MODEL_PRICES (USD-equivalent per MTok
    for Claude, credit-proxy for opencode). Useful only for *ranking* tiers, not
    as a billing forecast.
    """
    inp_tok, out_tok = _EFFORT_TOKENS.get(effort, _DEFAULT_TOKENS)
    p_in, p_out = _price_for(tier.models[0])
    return (inp_tok * p_in + out_tok * p_out) / 1_000_000
