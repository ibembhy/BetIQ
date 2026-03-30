"""
BetIQ — Kalshi API client.
Handles auth, market discovery, pricing, and order placement.
Completely separate from the BetIQ paper trading system.
"""

import logging
import os
import asyncio
import json
from datetime import datetime, timedelta
from typing import Optional

import pytz
import requests as _requests
from dotenv import load_dotenv
from kalshi_python_sync import Configuration, KalshiClient as _KalshiSDK
import websockets

load_dotenv()

log = logging.getLogger("betiq.kalshi")

KALSHI_HOST = "https://api.elections.kalshi.com/trade-api/v2"
KALSHI_WS_URL = os.getenv("KALSHI_WS_URL", "wss://api.elections.kalshi.com/trade-api/ws/v2")
KALSHI_WS_PATH = "/trade-api/ws/v2"
NBA_SERIES  = "KXNBAGAME"
EST         = pytz.timezone("America/New_York")

# ── Team abbreviation maps ─────────────────────────────────────────────────────

ABBREV_TO_TEAM: dict[str, str] = {
    "ATL": "Atlanta Hawks",
    "BKN": "Brooklyn Nets",
    "BOS": "Boston Celtics",
    "CHA": "Charlotte Hornets",
    "CHI": "Chicago Bulls",
    "CLE": "Cleveland Cavaliers",
    "DAL": "Dallas Mavericks",
    "DEN": "Denver Nuggets",
    "DET": "Detroit Pistons",
    "GSW": "Golden State Warriors",
    "HOU": "Houston Rockets",
    "IND": "Indiana Pacers",
    "LAC": "Los Angeles Clippers",
    "LAL": "Los Angeles Lakers",
    "MEM": "Memphis Grizzlies",
    "MIA": "Miami Heat",
    "MIL": "Milwaukee Bucks",
    "MIN": "Minnesota Timberwolves",
    "NOP": "New Orleans Pelicans",
    "NYK": "New York Knicks",
    "OKC": "Oklahoma City Thunder",
    "ORL": "Orlando Magic",
    "PHI": "Philadelphia 76ers",
    "PHX": "Phoenix Suns",
    "POR": "Portland Trail Blazers",
    "SAC": "Sacramento Kings",
    "SAS": "San Antonio Spurs",
    "TOR": "Toronto Raptors",
    "UTA": "Utah Jazz",
    "WAS": "Washington Wizards",
}

TEAM_TO_ABBREV: dict[str, str] = {v: k for k, v in ABBREV_TO_TEAM.items()}
# Extra aliases for fuzzy matching
TEAM_TO_ABBREV.update({
    "Golden State":          "GSW",
    "Los Angeles Lakers":    "LAL",
    "Los Angeles Clippers":  "LAC",
    "LA Lakers":             "LAL",
    "LA Clippers":           "LAC",
    "New Orleans":           "NOP",
    "Oklahoma City":         "OKC",
    "San Antonio":           "SAS",
    "New York":              "NYK",
    "Portland":              "POR",
    "Minnesota":             "MIN",
})


def resolve_abbrev(team_name: str) -> str | None:
    """Map a full team name to Kalshi 3-letter abbreviation."""
    if team_name in TEAM_TO_ABBREV:
        return TEAM_TO_ABBREV[team_name]
    # Partial match on city/nickname
    tl = team_name.lower()
    for full, abbrev in TEAM_TO_ABBREV.items():
        if tl in full.lower() or full.lower() in tl:
            return abbrev
    return None


# ── Client singleton ───────────────────────────────────────────────────────────

_client: Optional[_KalshiSDK] = None
_auth = None


def _http_session() -> _requests.Session:
    session = _requests.Session()
    session.trust_env = False
    return session


def _get_auth():
    """Return a cached KalshiAuth instance for raw HTTP requests."""
    global _auth
    if _auth is None:
        from kalshi_python_sync.auth import KalshiAuth
        key_path = os.getenv("KALSHI_PRIVATE_KEY_PATH", "kalshi_private_key.pem")
        with open(key_path, "r") as f:
            pem = f.read()
        _auth = KalshiAuth(os.getenv("KALSHI_API_KEY_ID"), pem)
    return _auth


def _raw_get(path: str) -> dict:
    """Raw authenticated GET to Kalshi API, bypassing SDK pydantic validation."""
    url = f"{KALSHI_HOST}{path}"
    headers = _get_auth().create_auth_headers("GET", url)
    r = _http_session().get(url, headers=headers, timeout=10)
    r.raise_for_status()
    return r.json()


def create_websocket_headers() -> dict:
    """Create authenticated WebSocket handshake headers."""
    return _get_auth().create_auth_headers("GET", KALSHI_WS_PATH)


def _get_client() -> _KalshiSDK:
    global _client
    if _client is None:
        config = Configuration(host=KALSHI_HOST)
        key_path = os.getenv("KALSHI_PRIVATE_KEY_PATH", "kalshi_private_key.pem")
        with open(key_path, "r") as f:
            config.private_key_pem = f.read()
        config.api_key_id = os.getenv("KALSHI_API_KEY_ID")
        _client = _KalshiSDK(config)
    return _client


# ── Public API ─────────────────────────────────────────────────────────────────

def get_balance() -> dict:
    """Returns account balance in dollars."""
    try:
        b = _get_client().get_balance()
        return {
            "balance_dollars":        round((b.balance or 0) / 100, 2),
            "portfolio_value_dollars": round((b.portfolio_value or 0) / 100, 2),
        }
    except Exception as e:
        log.error(f"get_balance failed: {e}")
        return {"error": str(e)}


_MONTHS = {1:"JAN",2:"FEB",3:"MAR",4:"APR",5:"MAY",6:"JUN",
           7:"JUL",8:"AUG",9:"SEP",10:"OCT",11:"NOV",12:"DEC"}


def _ticker_date(dt: datetime) -> str:
    """Convert datetime to Kalshi date string e.g. 26MAR29 — locale-independent."""
    return f"{dt.strftime('%y')}{_MONTHS[dt.month]}{dt.strftime('%d')}"


def get_today_games(include_tomorrow: bool = True) -> list[dict]:
    """
    Returns all open NBA game events for today (and tomorrow if include_tomorrow).
    Each dict: event_ticker, away_abbrev, home_abbrev, away_team, home_team, date_str
    """
    try:
        events = _get_client().get_events(series_ticker=NBA_SERIES, status="open", limit=50)
        now = datetime.now(EST)
        valid_dates = {_ticker_date(now)}
        if include_tomorrow:
            valid_dates.add(_ticker_date(now + timedelta(days=1)))

        results = []
        for e in (events.events or []):
            ticker = e.event_ticker          # KXNBAGAME-26MAR29ORLTOR
            tail   = ticker.replace("KXNBAGAME-", "")  # 26MAR29ORLTOR
            date_str   = tail[:7]            # 26MAR29
            teams_str  = tail[7:]            # ORLTOR

            if date_str not in valid_dates or len(teams_str) != 6:
                continue

            away_abbrev = teams_str[:3]
            home_abbrev = teams_str[3:]
            results.append({
                "event_ticker": ticker,
                "away_abbrev":  away_abbrev,
                "home_abbrev":  home_abbrev,
                "away_team":    ABBREV_TO_TEAM.get(away_abbrev, away_abbrev),
                "home_team":    ABBREV_TO_TEAM.get(home_abbrev, home_abbrev),
                "date_str":     date_str,
                "sub_title":    e.sub_title or "",
            })
        return results
    except Exception as e:
        log.error(f"get_today_games failed: {e}")
        return []


def get_market_tickers_for_games(games: list[dict]) -> list[str]:
    """Expand Kalshi NBA game events into per-team market tickers."""
    market_tickers: list[str] = []
    for game in games:
        tail = game["event_ticker"].replace("KXNBAGAME-", "")
        away_abbrev = tail[7:10]
        home_abbrev = tail[10:13]
        market_tickers.append(f"{game['event_ticker']}-{away_abbrev}")
        market_tickers.append(f"{game['event_ticker']}-{home_abbrev}")
    return market_tickers


def get_game_prices(event_ticker: str) -> dict:
    """
    Returns YES bid/ask prices for both teams in a game.
    Keys are team abbreviations (e.g. "MIA", "PHI").
    Each value: {ticker, yes_bid, yes_ask, implied_prob, volume}
    implied_prob = yes_ask (cost to buy 1 YES contract = probability you're paying for)
    """
    tail       = event_ticker.replace("KXNBAGAME-", "")
    away_abbrev = tail[7:10]
    home_abbrev = tail[10:13]

    result = {}
    for abbrev in (away_abbrev, home_abbrev):
        market_ticker = f"{event_ticker}-{abbrev}"
        try:
            # Use raw HTTP to avoid SDK pydantic validation failures on null fields
            data   = _raw_get(f"/markets/{market_ticker}")
            market = data.get("market", {})
            yes_ask = _safe_float(market.get("yes_ask_dollars"))
            yes_bid = _safe_float(market.get("yes_bid_dollars"))
            result[abbrev] = {
                "ticker":       market_ticker,
                "yes_bid":      yes_bid,
                "yes_ask":      yes_ask,
                "implied_prob": yes_ask,   # 0.51 = 51% implied
                "volume":       _safe_float(market.get("volume_fp")),
            }
        except Exception as e:
            log.warning(f"Could not get price for {market_ticker}: {e}")
            result[abbrev] = {
                "ticker":       market_ticker,
                "yes_bid":      None,
                "yes_ask":      None,
                "implied_prob": None,
                "volume":       None,
            }
    return result


def place_order(
    ticker: str,
    side: str,
    contracts: int,
    limit_price_cents: int,
) -> dict:
    """
    Place a limit order.
    side: "yes" or "no"
    contracts: number of $1-face-value contracts
    limit_price_cents: 1–99

    When KALSHI_LIVE=False (default), logs the order without sending it.
    """
    live = os.getenv("KALSHI_LIVE", "False").strip().lower() == "true"
    if not live:
        log.info(f"[PAPER] {ticker} {side} x{contracts} @ {limit_price_cents}c")
        return {
            "status":             "paper",
            "ticker":             ticker,
            "side":               side,
            "contracts":          contracts,
            "limit_price_cents":  limit_price_cents,
            "note":               "KALSHI_LIVE=False — order not sent to Kalshi",
        }
    try:
        from kalshi_python_sync.models import CreateOrderRequest
        req = CreateOrderRequest(
            ticker=ticker,
            side=side,
            action="buy",
            count=contracts,
            yes_price=limit_price_cents if side == "yes" else None,
            no_price=limit_price_cents  if side == "no"  else None,
            time_in_force="good_till_canceled",
        )
        resp  = _get_client().create_order(req)
        order = resp.order if hasattr(resp, "order") else resp
        d     = order.to_dict()
        log.info(f"Order placed: {ticker} {side} x{contracts} @ {limit_price_cents}c → id={d.get('order_id')}")
        return {
            "status":            "placed",
            "order_id":          d.get("order_id"),
            "ticker":            ticker,
            "side":              side,
            "contracts":         contracts,
            "limit_price_cents": limit_price_cents,
        }
    except Exception as e:
        log.error(f"place_order failed: {e}")
        return {"status": "error", "error": str(e)}


def get_positions() -> list[dict]:
    """Returns current open positions on Kalshi."""
    try:
        resp = _get_client().get_positions(count_filter="position")
        positions = getattr(resp, "market_positions", None) or []
        return [p.to_dict() for p in positions]
    except Exception as e:
        log.error(f"get_positions failed: {e}")
        return []


# ── Helpers ────────────────────────────────────────────────────────────────────

def _safe_float(val) -> float | None:
    try:
        return float(val) if val is not None else None
    except (TypeError, ValueError):
        return None


def build_subscribe_message(
    message_id: int,
    channels: list[str],
    market_tickers: list[str] | None = None,
) -> dict:
    params: dict = {"channels": channels}
    if market_tickers:
        params["market_tickers"] = market_tickers
    return {"id": message_id, "cmd": "subscribe", "params": params}


def parse_ws_message(message: str) -> dict:
    return json.loads(message)


def extract_ticker_update(payload: dict) -> dict | None:
    if payload.get("type") != "ticker":
        return None
    msg = payload.get("msg", {})
    market_ticker = msg.get("market_ticker")
    if not market_ticker:
        return None
    event_ticker = market_ticker.rsplit("-", 1)[0]
    side = market_ticker.rsplit("-", 1)[-1]
    return {
        "event_ticker": event_ticker,
        "market_ticker": market_ticker,
        "side": side,
        "yes_bid": _safe_float(msg.get("yes_bid_dollars")),
        "yes_ask": _safe_float(msg.get("yes_ask_dollars")),
        "yes_bid_cents": msg.get("yes_bid"),
        "yes_ask_cents": msg.get("yes_ask"),
        "volume": _safe_float(msg.get("volume")),
        "raw": payload,
    }


async def stream_market_updates(
    market_tickers: list[str],
    on_message,
    channels: list[str] | None = None,
) -> None:
    """
    Connect to Kalshi WebSocket and stream updates forever.

    on_message may be sync or async and receives the parsed JSON payload.
    """
    channels = channels or ["ticker"]
    reconnect_delay = 1

    while True:
        try:
            headers = create_websocket_headers()
            async with websockets.connect(
                KALSHI_WS_URL,
                additional_headers=headers,
                proxy=None,
            ) as websocket:
                log.info(
                    f"Kalshi WebSocket connected. Subscribing to {len(market_tickers)} markets on {channels}."
                )
                await websocket.send(
                    json.dumps(build_subscribe_message(1, channels=channels, market_tickers=market_tickers))
                )

                async for raw_message in websocket:
                    payload = parse_ws_message(raw_message)
                    maybe_coro = on_message(payload)
                    if asyncio.iscoroutine(maybe_coro):
                        await maybe_coro

            reconnect_delay = 1
        except Exception as exc:
            log.warning(f"Kalshi WebSocket disconnected: {exc}. Reconnecting in {reconnect_delay}s...")
            await asyncio.sleep(reconnect_delay)
            reconnect_delay = min(reconnect_delay * 2, 30)
