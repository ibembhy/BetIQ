"""
Deterministic betting math helpers shared across live trading and backtests.
"""

from __future__ import annotations

from typing import Iterable


def _validate_american_odds(odds: float) -> float:
    if odds == 0:
        raise ValueError("American odds cannot be 0.")
    return float(odds)


def american_odds_to_implied_probability(odds: float) -> float:
    """
    Convert American odds to implied win probability in the range [0, 1].
    """
    odds = _validate_american_odds(odds)
    if odds > 0:
        return 100.0 / (odds + 100.0)
    return abs(odds) / (abs(odds) + 100.0)


def american_odds_to_decimal(odds: float) -> float:
    """
    Convert American odds to decimal odds, inclusive of stake.
    """
    odds = _validate_american_odds(odds)
    if odds > 0:
        return 1.0 + odds / 100.0
    return 1.0 + 100.0 / abs(odds)


def no_vig_probabilities(implied_probabilities: Iterable[float]) -> list[float]:
    """
    Normalize one or more implied probabilities so they sum to 1.0.
    """
    probs = [float(p) for p in implied_probabilities]
    if not probs:
        raise ValueError("At least one implied probability is required.")
    if any(p <= 0 for p in probs):
        raise ValueError("Implied probabilities must be positive.")

    total = sum(probs)
    if total <= 0:
        raise ValueError("Probability total must be positive.")
    return [p / total for p in probs]


def no_vig_probabilities_from_odds(*odds_values: float) -> list[float]:
    """
    Convert American odds to no-vig fair probabilities.
    """
    if not odds_values:
        raise ValueError("At least one odds value is required.")
    implied = [american_odds_to_implied_probability(odds) for odds in odds_values]
    return no_vig_probabilities(implied)


def expected_value(win_probability: float, odds: float, stake: float = 1.0) -> float:
    """
    Return expected net profit in dollars for the provided stake.
    """
    if not 0.0 <= win_probability <= 1.0:
        raise ValueError("Win probability must be between 0 and 1.")
    if stake < 0:
        raise ValueError("Stake must be non-negative.")

    decimal_odds = american_odds_to_decimal(odds)
    profit_if_win = stake * (decimal_odds - 1.0)
    loss_if_lose = stake
    return (win_probability * profit_if_win) - ((1.0 - win_probability) * loss_if_lose)


def kelly_fraction(
    win_probability: float,
    odds: float,
    fractional_kelly: float = 0.5,
    max_fraction: float | None = None,
) -> float:
    """
    Return the bankroll fraction recommended by Kelly sizing.
    """
    if not 0.0 <= win_probability <= 1.0:
        raise ValueError("Win probability must be between 0 and 1.")
    if fractional_kelly < 0:
        raise ValueError("fractional_kelly must be non-negative.")
    if max_fraction is not None and max_fraction < 0:
        raise ValueError("max_fraction must be non-negative.")

    net_decimal_odds = american_odds_to_decimal(odds) - 1.0
    raw_fraction = ((win_probability * net_decimal_odds) - (1.0 - win_probability)) / net_decimal_odds
    sized_fraction = max(raw_fraction, 0.0) * fractional_kelly
    if max_fraction is not None:
        sized_fraction = min(sized_fraction, max_fraction)
    return sized_fraction


def kelly_stake(
    bankroll: float,
    win_probability: float,
    odds: float,
    fractional_kelly: float = 0.5,
    max_fraction: float | None = None,
) -> tuple[float, float]:
    """
    Return (stake_amount, bankroll_fraction) from Kelly sizing.
    """
    if bankroll < 0:
        raise ValueError("Bankroll must be non-negative.")

    fraction = kelly_fraction(
        win_probability=win_probability,
        odds=odds,
        fractional_kelly=fractional_kelly,
        max_fraction=max_fraction,
    )
    return bankroll * fraction, fraction
