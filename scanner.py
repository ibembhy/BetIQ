"""
BetIQ Scanner — Pure math betting scanner using Kalshi prediction markets.
No Claude, no LLM. Statistical model + deterministic rules only.

Architecture:
  - Slow data (stats, form, splits, rest, injuries, H2H) fetched once and cached per day
  - Fast data (Kalshi prices) fetched every POLL_INTERVAL_MINUTES
  - Model edge recomputed each poll cycle with fresh prices
  - Orders placed automatically when edge qualifies

Runs completely independently from BetIQ (runner.py / app.py).
Uses its own database: kalshi.db
Real orders only when KALSHI_LIVE=True in .env (default: False = paper log only).

Usage:
    venv/Scripts/python scanner.py
"""

import logging
import os
import sqlite3
import time
import asyncio
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import pytz
from apscheduler.schedulers.blocking import BlockingScheduler
from dotenv import load_dotenv

import kalshi
import model as _model
import tools as t

load_dotenv()

# ── Config ─────────────────────────────────────────────────────────────────────

POLL_INTERVAL_MINUTES = 5    # Re-check prices every N minutes
MIN_EDGE_PCT          = 5.0  # Minimum edge % to place order
MIN_IMPLIED_PROB      = 0.25 # Never bet a team the market prices below 25%
MIN_DQ_SCORE          = 65   # Minimum data quality score (0–100)
MAX_POSITIONS         = 5    # Max simultaneous open positions
KELLY_FRACTION        = 0.5  # Half-Kelly
MAX_KELLY_PCT         = 0.12 # Cap at 12% of bankroll per bet
MIN_CONTRACTS         = 1    # Minimum contracts to place
USE_KALSHI_WEBSOCKET  = os.getenv("KALSHI_USE_WEBSOCKET", "False").strip().lower() == "true"

EST     = pytz.timezone("America/New_York")
DB_PATH = os.path.join(os.path.dirname(__file__), "kalshi.db")
SLOW_CACHE_PATH = os.path.join(os.path.dirname(__file__), "data", "slow_context_cache.json")

# ── Logging ────────────────────────────────────────────────────────────────────

def _setup_logging() -> logging.Logger:
    """Configure scanner logging exactly once, idempotently."""
    # Wipe root handlers so basicConfig calls from imported libs don't double-print
    root = logging.getLogger()
    for h in root.handlers[:]:
        root.removeHandler(h)
    root.setLevel(logging.WARNING)

    logger = logging.getLogger("betiq.scanner")
    logger.propagate = False
    for h in logger.handlers[:]:
        logger.removeHandler(h)

    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s")

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    try:
        fh = logging.FileHandler("scanner.log", encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    except OSError:
        pass

    for name in ("betiq.kalshi", "betiq.tools", "betiq.injury"):
        logging.getLogger(name).propagate = False

    return logger


log = _setup_logging()


# ── Database ───────────────────────────────────────────────────────────────────

def _init_db() -> None:
    conn = sqlite3.connect(DB_PATH)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS orders (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            placed_at           TEXT    NOT NULL,
            event_ticker        TEXT    NOT NULL,
            market_ticker       TEXT    NOT NULL,
            matchup             TEXT    NOT NULL,
            pick                TEXT    NOT NULL,
            side                TEXT    NOT NULL,
            contracts           INTEGER NOT NULL,
            limit_price_cents   INTEGER NOT NULL,
            implied_prob        REAL,
            model_prob          REAL,
            edge_pct            REAL,
            dq_score            REAL,
            kelly_pct           REAL,
            dollar_stake        REAL,
            order_id            TEXT,
            status              TEXT    NOT NULL,
            result              TEXT    DEFAULT NULL,
            pnl                 REAL    DEFAULT NULL
        );

        CREATE TABLE IF NOT EXISTS scan_log (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            scanned_at  TEXT    NOT NULL,
            matchup     TEXT    NOT NULL,
            home_team   TEXT    NOT NULL,
            away_team   TEXT    NOT NULL,
            model_prob  REAL,
            implied_prob REAL,
            edge_pct    REAL,
            dq_score    REAL,
            decision    TEXT    NOT NULL,
            reason      TEXT
        );
    """)
    conn.commit()
    conn.close()


def _log_scan(matchup, home, away, model_prob, implied_prob, edge_pct, dq_score, decision, reason=""):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        INSERT INTO scan_log
            (scanned_at, matchup, home_team, away_team, model_prob, implied_prob, edge_pct, dq_score, decision, reason)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (datetime.now(EST).isoformat(), matchup, home, away,
          model_prob, implied_prob, edge_pct, dq_score, decision, reason))
    conn.commit()
    conn.close()


def _log_order(event_ticker, market_ticker, matchup, pick, side, contracts,
               limit_price_cents, implied_prob, model_prob, edge_pct, dq_score,
               kelly_pct, dollar_stake, order_id, status):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        INSERT INTO orders
            (placed_at, event_ticker, market_ticker, matchup, pick, side, contracts,
             limit_price_cents, implied_prob, model_prob, edge_pct, dq_score,
             kelly_pct, dollar_stake, order_id, status)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (datetime.now(EST).isoformat(), event_ticker, market_ticker, matchup, pick, side,
          contracts, limit_price_cents, implied_prob, model_prob, edge_pct, dq_score,
          kelly_pct, dollar_stake, order_id, status))
    conn.commit()
    conn.close()


def _already_bet(event_ticker: str) -> bool:
    """True if we already have a non-cancelled order for this game today."""
    conn   = sqlite3.connect(DB_PATH)
    row    = conn.execute(
        "SELECT 1 FROM orders WHERE event_ticker=? AND status NOT IN ('cancelled','error')",
        (event_ticker,),
    ).fetchone()
    conn.close()
    return row is not None


def _open_position_count() -> int:
    conn  = sqlite3.connect(DB_PATH)
    count = conn.execute(
        "SELECT COUNT(*) FROM orders WHERE status IN ('placed','paper') AND result IS NULL"
    ).fetchone()[0]
    conn.close()
    return count


# ── Result resolution ──────────────────────────────────────────────────────────

def resolve_results() -> None:
    """
    Check open orders against BDL game results and mark wins/losses.
    Runs nightly after games finish. Also sends a daily P&L Telegram summary.
    """
    conn = sqlite3.connect(DB_PATH)
    open_orders = conn.execute(
        "SELECT id, placed_at, matchup, pick, limit_price_cents, contracts, status "
        "FROM orders WHERE result IS NULL AND status IN ('placed','paper')"
    ).fetchall()
    conn.close()

    if not open_orders:
        log.info("resolve_results: no open orders to resolve.")
        return

    log.info(f"resolve_results: checking {len(open_orders)} open order(s)...")

    # Fetch BDL results for the last 3 days to cover any late updates
    from datetime import date, timedelta
    resolved = []
    errors   = []

    for order in open_orders:
        order_id, placed_at, matchup, pick, limit_price_cents, contracts, status = order
        # Parse game date from placed_at
        try:
            game_date = placed_at[:10]
        except Exception:
            continue

        # BDL stores late games (tipoff > midnight UTC) under the next calendar day.
        # Check both the game date and +1 day to handle all tip-off times.
        from datetime import date as _date, timedelta as _td
        next_date = (_date.fromisoformat(game_date) + _td(days=1)).isoformat()
        all_games_data: list = []
        for query_date in (game_date, next_date):
            data = t._bdl_get("/games", {"dates[]": query_date, "per_page": 100})
            if "error" not in data:
                all_games_data.extend(data.get("data", []))

        if not all_games_data:
            log.warning(f"resolve_results: BDL error or no data for {matchup}")
            errors.append(matchup)
            continue

        # Find the matching game by team names in matchup ("Away @ Home")
        parts = matchup.split(" @ ")
        if len(parts) != 2:
            continue
        away_name, home_name = parts[0].lower(), parts[1].lower()

        matched_game = None
        for g in all_games_data:
            if g.get("status") != "Final":
                continue
            bdl_home = g["home_team"]["full_name"].lower()
            bdl_away = g["visitor_team"]["full_name"].lower()
            # Fuzzy match: check if any word in the team name is present
            home_match = any(w in bdl_home for w in home_name.split() if len(w) > 3)
            away_match = any(w in bdl_away for w in away_name.split() if len(w) > 3)
            if home_match and away_match:
                matched_game = g
                break

        if not matched_game:
            log.info(f"resolve_results: no Final result yet for {matchup} on {game_date}")
            continue

        home_score    = matched_game["home_team_score"]
        visitor_score = matched_game["visitor_team_score"]
        bdl_home_name = matched_game["home_team"]["full_name"].lower()
        pick_lower    = pick.lower()

        # Determine if our pick won
        pick_is_home = any(w in bdl_home_name for w in pick_lower.split() if len(w) > 3)
        if pick_is_home:
            won = home_score > visitor_score
        else:
            won = visitor_score > home_score

        # P&L: YES contract at limit_price_cents.
        # Win: profit = contracts * (100 - limit_price_cents) / 100
        # Loss: loss = contracts * limit_price_cents / 100
        cost_per = limit_price_cents / 100.0
        if won:
            pnl    = round(contracts * (1.0 - cost_per), 2)
            result = "win"
        else:
            pnl    = round(-contracts * cost_per, 2)
            result = "loss"

        conn = sqlite3.connect(DB_PATH)
        conn.execute(
            "UPDATE orders SET result=?, pnl=? WHERE id=?",
            (result, pnl, order_id)
        )
        conn.commit()
        conn.close()

        score_str = f"{matched_game['visitor_team']['abbreviation']} {visitor_score}–{home_score} {matched_game['home_team']['abbreviation']}"
        log.info(f"resolve_results: {matchup} → {result.upper()} (picked {pick}, {score_str}, P&L ${pnl:+.2f})")
        resolved.append((matchup, pick, result, pnl, score_str))

    _send_daily_pnl_report(resolved, errors)


def _send_daily_pnl_report(resolved: list, errors: list) -> None:
    """Send Telegram summary of resolved bets."""
    if not resolved and not errors:
        return

    total_pnl  = sum(r[3] for r in resolved)
    wins       = sum(1 for r in resolved if r[2] == "win")
    losses     = len(resolved) - wins

    lines = [f"📊 BetIQ Daily Results — {datetime.now(EST).strftime('%b %d')}"]
    lines.append(f"Record: {wins}W–{losses}L | P&L: ${total_pnl:+.2f}")
    lines.append("")
    for matchup, pick, result, pnl, score in resolved:
        icon = "✅" if result == "win" else "❌"
        lines.append(f"{icon} {pick} → {result.upper()} ${pnl:+.2f}  ({score})")
    if errors:
        lines.append(f"\n⚠️ Could not resolve: {', '.join(errors)}")

    t._send_notification(
        title="BetIQ Daily Results",
        message="\n".join(lines),
    )
    log.info(f"Daily P&L: {wins}W–{losses}L, ${total_pnl:+.2f}")


# ── Data fetching ──────────────────────────────────────────────────────────────

def _fetch_slow_data(home: str, away: str) -> dict:
    """Fetch all slow (non-price) data for a game in parallel."""
    fetches = {
        "team_stats_home":  (t.get_team_stats,       (home,)),
        "team_stats_away":  (t.get_team_stats,        (away,)),
        "recent_form_home": (t.get_recent_form,       (home,)),
        "recent_form_away": (t.get_recent_form,       (away,)),
        "splits_home":      (t.get_home_away_splits,  (home,)),
        "splits_away":      (t.get_home_away_splits,  (away,)),
        "rest_home":        (t.get_rest_days,         (home,)),
        "rest_away":        (t.get_rest_days,         (away,)),
        "injuries_home":    (t.get_injury_report,     (home,)),
        "injuries_away":    (t.get_injury_report,     (away,)),
        "h2h":              (t.get_head_to_head,      (home, away)),
        "elo_prob":         (t.get_elo_probability,   (home, away)),
    }
    results: dict = {}
    with ThreadPoolExecutor(max_workers=4) as ex:
        future_map = {ex.submit(fn, *args): key for key, (fn, args) in fetches.items()}
        for future in as_completed(future_map):
            key = future_map[future]
            try:
                results[key] = future.result()
            except Exception as e:
                results[key] = {"error": str(e)}
    return results


# ── Slow data cache (refreshed once per day) ───────────────────────────────────

_slow_cache: dict[str, dict] = {}
_cache_date: str = ""
_slow_cache_loaded = False


def _ensure_today_cache() -> None:
    global _cache_date
    today = datetime.now(EST).strftime("%Y-%m-%d")
    if _cache_date != today:
        _slow_cache.clear()
        _cache_date = today


def _persist_slow_cache() -> None:
    os.makedirs(os.path.dirname(SLOW_CACHE_PATH), exist_ok=True)
    payload = {
        "cache_date": _cache_date,
        "games": _slow_cache,
    }
    with open(SLOW_CACHE_PATH, "w", encoding="utf-8") as fh:
        json.dump(payload, fh)


def _load_slow_cache() -> None:
    global _slow_cache_loaded, _cache_date
    if _slow_cache_loaded:
        return

    _slow_cache_loaded = True
    _ensure_today_cache()

    if not os.path.exists(SLOW_CACHE_PATH):
        return

    try:
        with open(SLOW_CACHE_PATH, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except (OSError, ValueError):
        return

    today = datetime.now(EST).strftime("%Y-%m-%d")
    if payload.get("cache_date") != today:
        return

    cached_games = payload.get("games")
    if isinstance(cached_games, dict):
        _slow_cache.clear()
        _slow_cache.update(cached_games)
        _cache_date = today


def preload_slow_context(games: list[dict], force_refresh: bool = False) -> dict[str, dict]:
    """
    Preload slow context for the provided games and persist it for reuse later.
    """
    _load_slow_cache()
    _ensure_today_cache()

    for game in games:
        event_ticker = game["event_ticker"]
        if force_refresh or event_ticker not in _slow_cache:
            home = game["home_team"]
            away = game["away_team"]
            log.info(f"Fetching slow data for {home} vs {away}...")
            _slow_cache[event_ticker] = _fetch_slow_data(home, away)
            log.info(f"Slow data ready for {home} vs {away}.")

    _persist_slow_cache()
    return dict(_slow_cache)


def _get_slow_data(event_ticker: str, home: str, away: str) -> dict:
    _load_slow_cache()
    _ensure_today_cache()
    if event_ticker not in _slow_cache:
        log.info(f"Fetching slow data for {home} vs {away}...")
        _slow_cache[event_ticker] = _fetch_slow_data(home, away)
        log.info(f"Slow data ready for {home} vs {away}.")
        _persist_slow_cache()
    return _slow_cache[event_ticker]


# ── Data quality score ─────────────────────────────────────────────────────────

def _dq_score(game_data: dict, has_kalshi_price: bool) -> tuple[float, list[str]]:
    """Returns (score 0–100, list of penalty reasons)."""
    score    = 100.0
    penalties: list[str] = []

    def missing(key):
        d = game_data.get(key) or {}
        return not d or "error" in d

    if missing("team_stats_home") or missing("team_stats_away"):
        score -= 20;  penalties.append("Team stats unavailable.")
    if missing("recent_form_home") or missing("recent_form_away"):
        score -= 10;  penalties.append("Recent form unavailable.")
    if missing("rest_home") or missing("rest_away"):
        score -= 10;  penalties.append("Rest days unavailable.")
    if missing("injuries_home") or missing("injuries_away"):
        score -= 15;  penalties.append("Injury data unavailable.")
    if missing("elo_prob"):
        score -= 20;  penalties.append("Elo probability unavailable.")
    if not has_kalshi_price:
        score -= 25;  penalties.append("Kalshi price unavailable.")

    return max(score, 0.0), penalties


# ── Kelly stake sizing ─────────────────────────────────────────────────────────

def _kelly_contracts(
    balance_dollars: float,
    model_prob: float,
    yes_ask: float,
) -> tuple[int, float, float]:
    """
    Returns (contracts, dollar_stake, kelly_pct).
    yes_ask: probability in 0.0–1.0 (e.g. 0.54)
    Each contract pays $1 if YES wins. Cost = yes_ask dollars per contract.
    """
    if yes_ask <= 0 or yes_ask >= 1:
        return 0, 0.0, 0.0

    b = (1.0 - yes_ask) / yes_ask   # net decimal odds per dollar risked
    kelly_raw = (model_prob * b - (1 - model_prob)) / b
    kelly     = max(0.0, min(kelly_raw * KELLY_FRACTION, MAX_KELLY_PCT))

    dollar_stake = round(balance_dollars * kelly, 2)
    dollar_stake = max(dollar_stake, yes_ask)        # at least 1 contract
    contracts    = max(MIN_CONTRACTS, int(dollar_stake / yes_ask))
    actual_stake = round(contracts * yes_ask, 2)
    kelly_pct    = round(kelly * 100, 2)

    return contracts, actual_stake, kelly_pct


def _build_price_snapshot(game: dict, prices: dict) -> tuple[dict, bool]:
    home_abbrev = game["home_abbrev"]
    away_abbrev = game["away_abbrev"]
    home_price = prices.get(home_abbrev, {})
    away_price = prices.get(away_abbrev, {})
    home_ask = home_price.get("yes_ask")
    away_ask = away_price.get("yes_ask")
    has_price = home_ask is not None and away_ask is not None
    return {
        "prices": prices,
        "home_ask": home_ask,
        "away_ask": away_ask,
    }, has_price


def evaluate_game_signal(
    game: dict,
    prices_override: dict | None = None,
) -> dict:
    """
    Evaluate the current market for a game without placing any order.

    Returns a dict with the current signal state so observer tools can log
    WOULD BET / WOULD PASS decisions without touching execution.
    """
    event_ticker = game["event_ticker"]
    home = game["home_team"]
    away = game["away_team"]
    home_abbrev = game["home_abbrev"]
    away_abbrev = game["away_abbrev"]
    matchup = f"{away} @ {home}"

    game_data = _get_slow_data(event_ticker, home, away)

    prices = prices_override or kalshi.get_game_prices(event_ticker)
    price_snapshot, has_price = _build_price_snapshot(game, prices)
    home_ask = price_snapshot["home_ask"]
    away_ask = price_snapshot["away_ask"]

    dq, penalties = _dq_score(game_data, has_price)
    if dq < MIN_DQ_SCORE:
        return {
            "event_ticker": event_ticker,
            "matchup": matchup,
            "decision": "PASS",
            "reason": f"DQ {dq:.0f} < {MIN_DQ_SCORE} — {'; '.join(penalties)}",
            "dq_score": dq,
            "penalties": penalties,
            "prices": prices,
        }

    if not has_price:
        return {
            "event_ticker": event_ticker,
            "matchup": matchup,
            "decision": "PASS",
            "reason": "No Kalshi price",
            "dq_score": dq,
            "penalties": penalties,
            "prices": prices,
        }

    features = _model.extract_features_from_prefetch(home, away, game_data)
    home_prob = _model.predict_win_prob(home, away, features)
    away_prob = 1.0 - home_prob

    home_edge = (home_prob - home_ask) * 100.0
    away_edge = (away_prob - away_ask) * 100.0

    if home_edge >= away_edge:
        pick_abbrev = home_abbrev
        pick_team = home
        model_prob = home_prob
        implied_prob = home_ask
        edge_pct = home_edge
    else:
        pick_abbrev = away_abbrev
        pick_team = away
        model_prob = away_prob
        implied_prob = away_ask
        edge_pct = away_edge

    market_ticker = prices[pick_abbrev]["ticker"]

    if implied_prob < MIN_IMPLIED_PROB:
        return {
            "event_ticker": event_ticker,
            "matchup": matchup,
            "decision": "PASS",
            "reason": f"implied prob {implied_prob:.1%} below minimum {MIN_IMPLIED_PROB:.0%} — heavy underdog",
            "dq_score": dq,
            "penalties": penalties,
            "prices": prices,
        }

    decision = "BET" if edge_pct >= MIN_EDGE_PCT else "PASS"
    reason = (
        f"best edge {edge_pct:.1f}% on {pick_team} "
        f"(model {model_prob:.1%} vs implied {implied_prob:.1%}) | DQ {dq:.0f}"
    )

    return {
        "event_ticker": event_ticker,
        "matchup": matchup,
        "decision": decision,
        "reason": reason,
        "dq_score": dq,
        "penalties": penalties,
        "pick_abbrev": pick_abbrev,
        "pick_team": pick_team,
        "market_ticker": market_ticker,
        "model_prob": model_prob,
        "implied_prob": implied_prob,
        "edge_pct": edge_pct,
        "prices": prices,
    }


# ── Core analysis for one game ─────────────────────────────────────────────────

def _analyze_game(
    game: dict,
    balance_dollars: float,
    prices_override: dict | None = None,
    trigger_label: str = "scan",
) -> str:
    event_ticker = game["event_ticker"]
    home = game["home_team"]
    away = game["away_team"]
    matchup = f"{away} @ {home}"

    # Skip if already bet this game today
    if _already_bet(event_ticker):
        return f"{matchup}: already bet — skipping"

    # Skip games that have already started or finished
    sub_title = game.get("sub_title", "")
    if sub_title in ("Final", "In Progress") or (sub_title and not sub_title[0].isdigit() and sub_title not in ("", "scheduled")):
        return f"{matchup}: skipping — game already live ({sub_title or 'started'})"

    signal = evaluate_game_signal(game, prices_override=prices_override)
    dq = signal["dq_score"]
    decision = signal["decision"]
    reason = signal["reason"]
    model_prob = signal.get("model_prob")
    implied_prob = signal.get("implied_prob")
    edge_pct = signal.get("edge_pct")
    pick_team = signal.get("pick_team")
    market_ticker = signal.get("market_ticker")

    _log_scan(matchup, home, away, model_prob, implied_prob, edge_pct, dq,
              decision,
              "" if decision == "BET" else reason)

    if decision != "BET":
        return f"{matchup}: PASS — {reason}"

    # Position cap
    if _open_position_count() >= MAX_POSITIONS:
        return f"{matchup}: PASS — max {MAX_POSITIONS} positions already open"

    # Size bet
    contracts, dollar_stake, kelly_pct = _kelly_contracts(balance_dollars, model_prob, implied_prob)
    if contracts < MIN_CONTRACTS:
        return f"{matchup}: PASS — stake too small (${dollar_stake:.2f})"

    limit_price_cents = max(1, min(99, int(implied_prob * 100)))

    # Place order
    order_result = kalshi.place_order(
        ticker=market_ticker,
        side="yes",
        contracts=contracts,
        limit_price_cents=limit_price_cents,
    )

    _log_order(
        event_ticker=event_ticker,
        market_ticker=market_ticker,
        matchup=matchup,
        pick=pick_team,
        side="yes",
        contracts=contracts,
        limit_price_cents=limit_price_cents,
        implied_prob=implied_prob,
        model_prob=model_prob,
        edge_pct=edge_pct,
        dq_score=dq,
        kelly_pct=kelly_pct,
        dollar_stake=dollar_stake,
        order_id=order_result.get("order_id"),
        status=order_result.get("status", "error"),
    )

    action = (
        "ORDER PLACED" if order_result.get("status") == "placed"
        else order_result.get("status", "?").upper()
    )
    summary = (
        f"{matchup}: {action} — {pick_team} YES @ {limit_price_cents}c | "
        f"Edge: {edge_pct:.1f}% | Model: {model_prob:.1%} vs Implied: {implied_prob:.1%} | "
        f"{contracts} contracts (${dollar_stake:.2f}) | DQ: {dq:.0f} | Trigger: {trigger_label}"
    )
    log.info(summary)
    t._send_notification(title=f"BetIQ Scanner — {action}", message=summary)
    return summary


def _prepare_websocket_state(games: list[dict]) -> tuple[dict[str, dict], dict[str, dict], dict[str, tuple]]:
    games_by_event = {game["event_ticker"]: game for game in games}
    prices_by_event: dict[str, dict] = {}
    ask_pairs_by_event: dict[str, tuple] = {}

    for game in games:
        prices = kalshi.get_game_prices(game["event_ticker"])
        prices_by_event[game["event_ticker"]] = prices
        snapshot, _ = _build_price_snapshot(game, prices)
        ask_pairs_by_event[game["event_ticker"]] = (snapshot["home_ask"], snapshot["away_ask"])

    return games_by_event, prices_by_event, ask_pairs_by_event


async def run_websocket_listener() -> None:
    """
    Main WebSocket loop. Runs forever, reconnecting each day at midnight
    to subscribe to the new day's markets.
    Kalshi pushes every price change — we react instantly rather than polling.
    """
    while True:
        today = datetime.now(EST).strftime("%Y-%m-%d")
        games = kalshi.get_today_games()

        if not games:
            log.info("No NBA games on Kalshi today — waiting 60s before retry.")
            await asyncio.sleep(60)
            continue

        _paper_balance = float(os.getenv("PAPER_BALANCE", "0"))
        _is_paper = os.getenv("KALSHI_LIVE", "False").strip().lower() != "true"

        balance_ref = {"dollars": _paper_balance if (_is_paper and _paper_balance > 0) else 0.0}
        def _refresh_balance():
            if _is_paper and _paper_balance > 0:
                balance_ref["dollars"] = _paper_balance
                return
            bal = kalshi.get_balance()
            if "error" not in bal:
                balance_ref["dollars"] = bal["balance_dollars"]

        _refresh_balance()
        log.info(f"WebSocket: {len(games)} game(s) today | Balance: ${balance_ref['dollars']:.2f}")

        # Initial startup scan before subscribing
        for game in games:
            try:
                summary = _analyze_game(game, balance_ref["dollars"], trigger_label="startup")
                log.info(summary)
            except Exception as exc:
                log.error(f"{game.get('away_team')} @ {game.get('home_team')}: startup error — {exc}")

        games_by_event, prices_by_event, ask_pairs_by_event = _prepare_websocket_state(games)
        market_tickers = kalshi.get_market_tickers_for_games(games)

        async def _handle_message(payload: dict) -> None:
            ticker_update = kalshi.extract_ticker_update(payload)
            if not ticker_update:
                if payload.get("type") == "error":
                    log.warning(f"Kalshi WS error msg: {payload}")
                return

            event_ticker = ticker_update["event_ticker"]
            game = games_by_event.get(event_ticker)
            if not game:
                return

            # Skip live games — no point analyzing
            sub_title = game.get("sub_title", "")
            if sub_title and sub_title not in ("", "scheduled") and not sub_title[0].isdigit():
                return

            event_prices = prices_by_event.setdefault(
                event_ticker, kalshi.get_game_prices(event_ticker)
            )
            event_prices[ticker_update["side"]] = {
                "ticker":       ticker_update["market_ticker"],
                "yes_bid":      ticker_update["yes_bid"],
                "yes_ask":      ticker_update["yes_ask"],
                "implied_prob": ticker_update["yes_ask"],
                "volume":       ticker_update["volume"],
            }

            snapshot, has_price = _build_price_snapshot(game, event_prices)
            if not has_price:
                return

            # Only re-analyze when price actually changed
            ask_pair = (snapshot["home_ask"], snapshot["away_ask"])
            if ask_pair == ask_pairs_by_event.get(event_ticker):
                return
            ask_pairs_by_event[event_ticker] = ask_pair

            # Refresh balance before each potential bet
            _refresh_balance()

            try:
                summary = _analyze_game(
                    game,
                    balance_ref["dollars"],
                    prices_override=event_prices,
                    trigger_label="websocket",
                )
                log.info(summary)
            except Exception as exc:
                log.error(f"{game['away_team']} @ {game['home_team']}: ws error — {exc}")

        # Stream until midnight, then reconnect with next day's games
        midnight = datetime.now(EST).replace(hour=0, minute=0, second=0, microsecond=0)
        from datetime import timedelta
        next_midnight = midnight + timedelta(days=1)
        seconds_until_midnight = (next_midnight - datetime.now(EST)).total_seconds()

        try:
            await asyncio.wait_for(
                kalshi.stream_market_updates(market_tickers, _handle_message, channels=["ticker"]),
                timeout=seconds_until_midnight,
            )
        except asyncio.TimeoutError:
            log.info("Midnight reached — reconnecting WebSocket for new day's games.")
        except Exception as exc:
            log.warning(f"WebSocket outer error: {exc} — restarting in 10s.")
            await asyncio.sleep(10)


# ── Main scan ──────────────────────────────────────────────────────────────────

def run_scan() -> None:
    now = datetime.now(EST).strftime("%Y-%m-%d %H:%M EST")
    log.info(f"Scan — {now}")

    games = kalshi.get_today_games()
    if not games:
        log.info("No NBA games found on Kalshi today.")
        return

    log.info(f"{len(games)} game(s) on Kalshi today.")

    _paper_balance = float(os.getenv("PAPER_BALANCE", "0"))
    _is_paper = os.getenv("KALSHI_LIVE", "False").strip().lower() != "true"
    if _is_paper and _paper_balance > 0:
        balance_dollars = _paper_balance
    else:
        bal = kalshi.get_balance()
        if "error" in bal:
            log.error(f"Cannot get balance: {bal['error']}")
            return
        balance_dollars = bal["balance_dollars"]
    log.info(f"Balance: ${balance_dollars:.2f}")

    summaries = []
    for game in games:
        try:
            summary = _analyze_game(game, balance_dollars)
            summaries.append(f"• {summary}")
            log.info(summary)
        except Exception as e:
            msg = f"• {game.get('home_team')} vs {game.get('away_team')}: ERROR — {e}"
            summaries.append(msg)
            log.error(msg)

    log.info("Scan complete.\n" + "\n".join(summaries))


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    _init_db()

    live_mode = os.getenv("KALSHI_LIVE", "False").strip().lower() == "true"
    log.info(f"BetIQ Scanner starting — mode: {'LIVE' if live_mode else 'PAPER (set KALSHI_LIVE=True to go live)'}")

    # Pre-load slow data for all today's games once at startup.
    # This runs sequentially per game (BDL 60 req/min plan).
    # After this, all scans and WebSocket callbacks read from cache instantly.
    _startup_games = kalshi.get_today_games()
    if _startup_games:
        log.info(f"Pre-loading slow data for {len(_startup_games)} game(s)...")
        preload_slow_context(_startup_games)
        log.info("Slow data pre-load complete.")
    else:
        log.info("No games today — skipping pre-load.")

    # Shared background scheduler for daily jobs (runs in both modes)
    scheduler = BlockingScheduler(timezone=EST)

    # 2 AM — resolve yesterday's results and send P&L report
    scheduler.add_job(resolve_results, "cron", hour=2, minute=0)

    # 10 AM — preload today's slow data before games start
    def _daily_preload():
        games = kalshi.get_today_games()
        if games:
            log.info(f"Daily preload: {len(games)} game(s)...")
            preload_slow_context(games, force_refresh=True)
            log.info("Daily preload complete.")

    scheduler.add_job(_daily_preload, "cron", hour=10, minute=0)

    if USE_KALSHI_WEBSOCKET:
        log.info(f"Edge threshold: {MIN_EDGE_PCT}% | DQ threshold: {MIN_DQ_SCORE} | Mode: WebSocket (real-time)")

        # Run scheduler in a background thread so asyncio can take the main thread
        import threading
        scheduler_thread = threading.Thread(target=scheduler.start, daemon=True)
        scheduler_thread.start()

        try:
            asyncio.run(run_websocket_listener())
        except KeyboardInterrupt:
            log.info("Scanner stopped.")
    else:
        log.info(f"Edge threshold: {MIN_EDGE_PCT}% | DQ threshold: {MIN_DQ_SCORE} | Mode: Polling every {POLL_INTERVAL_MINUTES} min")

        # Add polling scans noon–midnight
        for hour in range(12, 24):
            for minute in range(0, 60, POLL_INTERVAL_MINUTES):
                scheduler.add_job(run_scan, "cron", hour=hour, minute=minute)

        # Run immediately on startup
        run_scan()

        log.info(f"Scheduled: scans every {POLL_INTERVAL_MINUTES} min (12 PM–midnight), preload at 10 AM, results at 2 AM.")
        log.info("Press Ctrl+C to stop.")

        try:
            scheduler.start()
        except KeyboardInterrupt:
            log.info("Scanner stopped.")
