"""
BetIQ — Betfair Exchange integration.
Handles authentication, market discovery, and bet placement for NBA games.

Set BETFAIR_LIVE_MODE=true in .env to enable real betting.
Set BETFAIR_MAX_STAKE to cap the maximum stake per bet.
"""

import os
import logging
from datetime import datetime, timedelta

import betfairlightweight
from betfairlightweight import filters as bf_filters
from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger("betiq.betfair")

BETFAIR_USERNAME  = os.getenv("BETFAIR_USERNAME", "")
BETFAIR_PASSWORD  = os.getenv("BETFAIR_PASSWORD", "")
BETFAIR_APP_KEY   = os.getenv("BETFAIR_APP_KEY", "")
BETFAIR_LIVE_MODE = os.getenv("BETFAIR_LIVE_MODE", "false").lower() == "true"
BETFAIR_MAX_STAKE = float(os.getenv("BETFAIR_MAX_STAKE", "20"))

BASKETBALL_EVENT_TYPE_ID = "7522"


# ── Helpers ───────────────────────────────────────────────────────────────────

def is_configured() -> bool:
    return bool(BETFAIR_USERNAME and BETFAIR_PASSWORD and BETFAIR_APP_KEY)


def is_live() -> bool:
    return BETFAIR_LIVE_MODE and is_configured()


def american_to_decimal(american_odds: int) -> float:
    """Convert American odds to Betfair decimal odds."""
    if american_odds > 0:
        return round(1 + american_odds / 100, 2)
    return round(1 + 100 / abs(american_odds), 2)


def round_betfair_price(price: float) -> float:
    """Round to the nearest valid Betfair price increment."""
    if price < 1.01:
        return 1.01
    elif price < 2.0:
        return round(round(price / 0.01) * 0.01, 2)
    elif price < 3.0:
        return round(round(price / 0.02) * 0.02, 2)
    elif price < 4.0:
        return round(round(price / 0.05) * 0.05, 2)
    elif price < 6.0:
        return round(round(price / 0.1) * 0.1, 2)
    elif price < 10.0:
        return round(round(price / 0.2) * 0.2, 2)
    elif price < 20.0:
        return round(round(price / 0.5) * 0.5, 2)
    elif price < 30.0:
        return float(round(price))
    elif price < 50.0:
        return round(round(price / 2.0) * 2.0, 2)
    elif price < 100.0:
        return round(round(price / 5.0) * 5.0, 2)
    return round(round(price / 10.0) * 10.0, 2)


def _get_client() -> betfairlightweight.APIClient:
    client = betfairlightweight.APIClient(
        username=BETFAIR_USERNAME,
        password=BETFAIR_PASSWORD,
        app_key=BETFAIR_APP_KEY,
    )
    client.login()
    return client


def _team_matches(runner_name: str, team_name: str) -> bool:
    """Fuzzy match a Betfair runner name against a BetIQ team name."""
    runner = runner_name.lower()
    team   = team_name.lower()
    # Match on any word longer than 3 chars (catches "Lakers", "Celtics", etc.)
    return any(word in runner for word in team.split() if len(word) > 3)


# ── Market discovery ──────────────────────────────────────────────────────────

def find_nba_market(home_team: str, away_team: str) -> dict:
    """
    Find the Betfair MATCH_ODDS market for a specific NBA game today.

    Returns:
        {market_id, home_runner_id, away_runner_id, home_name, away_name}
        or {error: "..."}
    """
    if not is_configured():
        return {"error": "Betfair credentials not set in .env"}

    try:
        client = _get_client()
    except Exception as exc:
        return {"error": f"Betfair login failed: {exc}"}

    now      = datetime.utcnow()
    tomorrow = now + timedelta(hours=30)

    try:
        markets = client.betting.list_market_catalogue(
            filter=bf_filters.market_filter(
                event_type_ids=[BASKETBALL_EVENT_TYPE_ID],
                market_types=["MATCH_ODDS"],
                market_start_time=bf_filters.time_range(
                    from_=now.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    to=tomorrow.strftime("%Y-%m-%dT%H:%M:%SZ"),
                ),
            ),
            market_projection=["RUNNER_DESCRIPTION", "EVENT"],
            max_results=100,
        )
    except Exception as exc:
        return {"error": f"list_market_catalogue failed: {exc}"}

    for market in markets:
        runners = market.runners
        if len(runners) != 2:
            continue

        home_runner = next(
            (r for r in runners if _team_matches(r.runner_name, home_team)), None
        )
        away_runner = next(
            (r for r in runners if _team_matches(r.runner_name, away_team)), None
        )

        if home_runner and away_runner and home_runner.selection_id != away_runner.selection_id:
            log.info(
                f"Market found: {market.market_id} | "
                f"{away_runner.runner_name} @ {home_runner.runner_name}"
            )
            return {
                "market_id":       market.market_id,
                "home_runner_id":  home_runner.selection_id,
                "away_runner_id":  away_runner.selection_id,
                "home_name":       home_runner.runner_name,
                "away_name":       away_runner.runner_name,
            }

    return {"error": f"No Betfair market found for {away_team} @ {home_team}"}


# ── Bet placement ─────────────────────────────────────────────────────────────

def place_bet(
    market_id: str,
    selection_id: int,
    american_odds: int,
    stake: float,
) -> dict:
    """
    Place a BACK bet on Betfair Exchange.

    Args:
        market_id:     Betfair market ID
        selection_id:  Runner (team) selection ID
        american_odds: American odds for the pick (used to compute decimal price)
        stake:         Stake in account currency (capped at BETFAIR_MAX_STAKE)

    Returns:
        {bet_id, size_matched, price, stake} or {error: "..."}
    """
    if not is_configured():
        return {"error": "Betfair credentials not set"}

    stake = min(round(stake, 2), BETFAIR_MAX_STAKE)
    stake = max(stake, 2.0)  # Betfair minimum bet

    decimal_price = american_to_decimal(american_odds)
    # Slightly reduce price to improve matching chance without losing much value
    decimal_price = round_betfair_price(decimal_price - 0.02)
    decimal_price = max(decimal_price, 1.01)

    try:
        client = _get_client()
        result = client.betting.place_orders(
            market_id=market_id,
            instructions=[
                bf_filters.place_instruction(
                    order_type="LIMIT",
                    selection_id=selection_id,
                    side="BACK",
                    limit_order=bf_filters.limit_order(
                        size=stake,
                        price=decimal_price,
                        persistence_type="LAPSE",  # Cancel if unmatched at game start
                    ),
                )
            ],
        )
    except Exception as exc:
        return {"error": f"place_orders failed: {exc}"}

    if result.status != "SUCCESS":
        return {"error": f"Order status: {result.status}"}

    report = result.instruction_reports[0]
    if report.status != "SUCCESS":
        return {"error": f"Instruction status: {report.status} — {getattr(report, 'error_code', '')}"}

    log.info(
        f"Betfair bet placed: ID={report.bet_id} | "
        f"market={market_id} | selection={selection_id} | "
        f"stake={stake} @ {decimal_price}"
    )
    return {
        "bet_id":       report.bet_id,
        "size_matched": getattr(report, "size_matched", 0),
        "price":        decimal_price,
        "stake":        stake,
    }


# ── Bet cancellation ──────────────────────────────────────────────────────────

def cancel_bet(betfair_bet_id: str, market_id: str) -> dict:
    """Cancel an unmatched Betfair bet."""
    if not is_configured():
        return {"error": "Betfair credentials not set"}

    try:
        client = _get_client()
        result = client.betting.cancel_orders(
            market_id=market_id,
            instructions=[
                bf_filters.cancel_instruction(bet_id=betfair_bet_id)
            ],
        )
    except Exception as exc:
        return {"error": f"cancel_orders failed: {exc}"}

    if result.status == "SUCCESS":
        log.info(f"Betfair bet cancelled: {betfair_bet_id}")
        return {"cancelled": True, "bet_id": betfair_bet_id}
    return {"error": f"Cancel failed: {result.status}"}


# ── Full place flow ───────────────────────────────────────────────────────────

def place_live_bet(
    matchup: str,
    pick: str,
    american_odds: int,
    stake: float,
) -> dict:
    """
    High-level: find the right market and place the bet.

    pick should be the team name exactly as it appears in BetIQ
    (e.g. "Boston Celtics ML" or "Los Angeles Lakers ML").
    matchup should be "Away Team vs Home Team".

    Returns {bet_id, market_id, ...} or {error: "..."}
    """
    if not is_live():
        return {"skipped": "Live mode is off — paper bet only"}

    # Parse teams from matchup ("Away @ Home" or "Away vs Home")
    parts = matchup.replace(" @ ", " vs ").split(" vs ")
    if len(parts) != 2:
        return {"error": f"Cannot parse matchup: {matchup}"}
    away_team, home_team = parts[0].strip(), parts[1].strip()

    # Find which team is being backed
    pick_lower = pick.lower()
    if any(word in pick_lower for word in home_team.lower().split() if len(word) > 3):
        backing_home = True
    elif any(word in pick_lower for word in away_team.lower().split() if len(word) > 3):
        backing_home = False
    else:
        return {"error": f"Cannot determine which team to back from pick: {pick}"}

    market = find_nba_market(home_team, away_team)
    if "error" in market:
        return market

    selection_id = market["home_runner_id"] if backing_home else market["away_runner_id"]

    result = place_bet(
        market_id=market["market_id"],
        selection_id=selection_id,
        american_odds=american_odds,
        stake=stake,
    )

    if "error" in result:
        return result

    return {
        **result,
        "market_id": market["market_id"],
        "pick":      pick,
        "matchup":   matchup,
    }
