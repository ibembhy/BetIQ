"""
Deterministic recommendation support for the live betting flow.
"""

from __future__ import annotations

from typing import Iterable

from betting_math import (
    american_odds_to_implied_probability,
    expected_value,
    kelly_fraction,
)


SUPPORTED_PROBABILITY_MARKETS = {"moneyline"}
BET_EDGE_THRESHOLD = 5.0
LEAN_EDGE_THRESHOLD = 2.0
BET_DATA_QUALITY_MIN = 65
LEAN_DATA_QUALITY_MIN = 45


def clamp_probability(probability: float) -> float:
    return min(max(probability, 0.0), 1.0)


def _status_weight(status: str) -> float:
    status_l = (status or "").lower()
    if "out" in status_l:
        return 1.0
    if "doubtful" in status_l:
        return 0.7
    if "questionable" in status_l:
        return 0.4
    if "probable" in status_l:
        return 0.15
    return 0.25 if status_l else 0.0


def weighted_injury_count(report: dict | None) -> float:
    if not isinstance(report, dict):
        return 0.0
    return round(sum(_status_weight(i.get("status", "")) for i in report.get("injuries", [])), 2)


def rest_adjustment(selected_rest_days: int | None, opponent_rest_days: int | None) -> float:
    if selected_rest_days is None or opponent_rest_days is None:
        return 0.0
    diff = selected_rest_days - opponent_rest_days
    return max(min(diff * 0.01, 0.03), -0.03)


def injury_adjustment(selected_weighted_injuries: float, opponent_weighted_injuries: float) -> float:
    diff = opponent_weighted_injuries - selected_weighted_injuries
    return max(min(diff * 0.0125, 0.05), -0.05)


def public_money_adjustment(
    selected_ticket_pct: float | None,
    selected_money_pct: float | None,
    opponent_ticket_pct: float | None,
    opponent_money_pct: float | None,
) -> float:
    if None in (selected_ticket_pct, selected_money_pct, opponent_ticket_pct, opponent_money_pct):
        return 0.0

    selected_diff = selected_money_pct - selected_ticket_pct
    opponent_diff = opponent_money_pct - opponent_ticket_pct
    adjustment = 0.0

    if selected_diff >= 15:
        adjustment += 0.015
    elif selected_diff >= 8:
        adjustment += 0.008

    if opponent_diff >= 15:
        adjustment -= 0.015
    elif opponent_diff >= 8:
        adjustment -= 0.008

    if selected_ticket_pct >= 70 and selected_money_pct <= selected_ticket_pct:
        adjustment -= 0.015
    if opponent_ticket_pct >= 70 and opponent_money_pct <= opponent_ticket_pct:
        adjustment += 0.015

    return max(min(adjustment, 0.03), -0.03)


def moneyline_movement_adjustment(
    selected_ml_move: float | None,
    opponent_ml_move: float | None,
) -> float:
    if selected_ml_move is None and opponent_ml_move is None:
        return 0.0

    selected_signal = 0.0
    if selected_ml_move is not None:
        if selected_ml_move <= -20:
            selected_signal += 0.015
        elif selected_ml_move <= -10:
            selected_signal += 0.008
        elif selected_ml_move >= 20:
            selected_signal -= 0.015
        elif selected_ml_move >= 10:
            selected_signal -= 0.008

    opponent_signal = 0.0
    if opponent_ml_move is not None:
        if opponent_ml_move <= -20:
            opponent_signal -= 0.015
        elif opponent_ml_move <= -10:
            opponent_signal -= 0.008
        elif opponent_ml_move >= 20:
            opponent_signal += 0.015
        elif opponent_ml_move >= 10:
            opponent_signal += 0.008

    return max(min(selected_signal + opponent_signal, 0.03), -0.03)


def compute_moneyline_probability(
    base_probability: float,
    selected_rest_days: int | None,
    opponent_rest_days: int | None,
    selected_weighted_injuries: float,
    opponent_weighted_injuries: float,
    selected_ticket_pct: float | None,
    selected_money_pct: float | None,
    opponent_ticket_pct: float | None,
    opponent_money_pct: float | None,
    selected_ml_move: float | None,
    opponent_ml_move: float | None,
) -> dict:
    adjustments = {
        "rest": rest_adjustment(selected_rest_days, opponent_rest_days),
        "injuries": injury_adjustment(selected_weighted_injuries, opponent_weighted_injuries),
        "public_money": public_money_adjustment(
            selected_ticket_pct,
            selected_money_pct,
            opponent_ticket_pct,
            opponent_money_pct,
        ),
        "line_movement": moneyline_movement_adjustment(selected_ml_move, opponent_ml_move),
    }
    adjusted_probability = clamp_probability(base_probability + sum(adjustments.values()))
    return {
        "base_probability": base_probability,
        "adjusted_probability": adjusted_probability,
        "adjustments": adjustments,
    }


def compute_data_quality_score(
    *,
    market_supported: bool,
    current_odds_found: bool,
    selected_side_found: bool,
    opposite_side_found: bool,
    elo_source: str | None,
    rest_data_complete: bool,
    injury_data_complete: bool,
    roster_data_complete: bool,
    public_data_complete: bool,
    line_snapshots: int,
    submitted_vs_market_delta: float | None = None,
    conflicting_signals: bool = False,
) -> dict:
    score = 100
    penalties: list[str] = []

    if not market_supported:
        score -= 25
        penalties.append("No deterministic probability model for this market type.")
    if not current_odds_found:
        score -= 25
        penalties.append("Live market odds for the matchup could not be confirmed.")
    if not selected_side_found:
        score -= 20
        penalties.append("Submitted pick could not be matched cleanly to the current market.")
    if not opposite_side_found:
        score -= 8
        penalties.append("Opposite-side price unavailable, so no-vig fair probability is weaker.")
    if elo_source == "unavailable":
        score -= 18
        penalties.append("Elo baseline unavailable.")
    elif elo_source == "not_initialized":
        score -= 10
        penalties.append("Elo baseline exists only as a default 50/50 fallback.")
    if not rest_data_complete:
        score -= 8
        penalties.append("Rest-day data incomplete.")
    if not injury_data_complete:
        score -= 12
        penalties.append("Injury data incomplete or stale.")
    if not roster_data_complete:
        score -= 10
        penalties.append("Current roster validation incomplete.")
    if not public_data_complete:
        score -= 6
        penalties.append("Public betting data unavailable.")
    if line_snapshots < 2:
        score -= 6
        penalties.append("Line movement has too few local snapshots.")
    if submitted_vs_market_delta is not None and submitted_vs_market_delta >= 20:
        score -= 8
        penalties.append("Submitted odds differ materially from the current best market price.")
    if conflicting_signals:
        score -= 10
        penalties.append("Signals conflict with each other.")

    return {"score": max(score, 0), "penalties": penalties}


def infer_decision(
    *,
    market_supported: bool,
    model_probability: float | None,
    fair_probability_no_vig: float | None,
    edge_pct: float | None,
    expected_value_per_unit: float | None,
    data_quality_score: float,
    llm_edge_pct: float | None = None,
) -> str:
    if market_supported and None not in (model_probability, fair_probability_no_vig, edge_pct, expected_value_per_unit):
        if data_quality_score >= BET_DATA_QUALITY_MIN and edge_pct >= BET_EDGE_THRESHOLD and expected_value_per_unit > 0:
            return "BET"
        if data_quality_score >= LEAN_DATA_QUALITY_MIN and edge_pct >= LEAN_EDGE_THRESHOLD and expected_value_per_unit > 0:
            return "LEAN"
        return "PASS"

    if llm_edge_pct is not None and data_quality_score >= LEAN_DATA_QUALITY_MIN and llm_edge_pct >= BET_EDGE_THRESHOLD:
        return "LEAN"
    return "PASS"


def recommended_stake(
    bankroll: float,
    model_probability: float | None,
    odds: float,
    decision: str,
    *,
    fractional_kelly: float = 0.5,
    max_fraction: float = 0.12,
    min_fraction_for_bet: float = 0.01,
) -> tuple[float, float]:
    if decision != "BET" or model_probability is None:
        return 0.0, 0.0

    stake_fraction = kelly_fraction(
        win_probability=model_probability,
        odds=odds,
        fractional_kelly=fractional_kelly,
        max_fraction=max_fraction,
    )
    stake_fraction = max(stake_fraction, min_fraction_for_bet)
    return bankroll * stake_fraction, stake_fraction


def recommendation_snapshot(
    *,
    odds: float,
    bankroll: float,
    market_supported: bool,
    model_probability: float | None,
    market_implied_probability: float | None,
    fair_probability_no_vig: float | None,
    data_quality_score: float,
    llm_edge_pct: float | None = None,
) -> dict:
    edge_pct = None
    ev_per_unit = None
    if market_supported and model_probability is not None:
        ev_per_unit = expected_value(model_probability, odds, 1.0)
        if fair_probability_no_vig is not None:
            edge_pct = (model_probability - fair_probability_no_vig) * 100.0
        elif market_implied_probability is not None:
            edge_pct = (model_probability - market_implied_probability) * 100.0

    decision = infer_decision(
        market_supported=market_supported,
        model_probability=model_probability,
        fair_probability_no_vig=fair_probability_no_vig,
        edge_pct=edge_pct,
        expected_value_per_unit=ev_per_unit,
        data_quality_score=data_quality_score,
        llm_edge_pct=llm_edge_pct,
    )
    stake_amount, stake_fraction = recommended_stake(
        bankroll=bankroll,
        model_probability=model_probability,
        odds=odds,
        decision=decision,
    )
    ev = expected_value(model_probability, odds, stake_amount) if model_probability is not None and stake_amount > 0 else 0.0
    return {
        "market_implied_prob": market_implied_probability,
        "fair_prob_no_vig": fair_probability_no_vig,
        "model_prob": model_probability,
        "edge_pct": edge_pct,
        "ev_per_unit": ev_per_unit,
        "ev": ev,
        "decision": decision,
        "stake_pct": stake_fraction * 100.0,
        "stake_amount": stake_amount,
        "data_quality_score": data_quality_score,
        "llm_edge_pct": llm_edge_pct,
        "offered_implied_prob": american_odds_to_implied_probability(odds),
    }
