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
MIN_DQ_SCORE          = 65   # Minimum data quality score (0–100)
MAX_POSITIONS         = 5    # Max simultaneous open positions
KELLY_FRACTION        = 0.5  # Half-Kelly
MAX_KELLY_PCT         = 0.12 # Cap at 12% of bankroll per bet
MIN_CONTRACTS         = 1    # Minimum contracts to place

EST     = pytz.timezone("America/New_York")
DB_PATH = os.path.join(os.path.dirname(__file__), "kalshi.db")

# ── Logging ────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.FileHandler("scanner.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("betiq.scanner")


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
    with ThreadPoolExecutor(max_workers=6) as ex:
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


def _get_slow_data(event_ticker: str, home: str, away: str) -> dict:
    global _cache_date
    today = datetime.now(EST).strftime("%Y-%m-%d")
    if _cache_date != today:
        _slow_cache.clear()
        _cache_date = today
    if event_ticker not in _slow_cache:
        log.info(f"Fetching slow data for {home} vs {away}...")
        _slow_cache[event_ticker] = _fetch_slow_data(home, away)
        log.info(f"Slow data ready for {home} vs {away}.")
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


# ── Core analysis for one game ─────────────────────────────────────────────────

def _analyze_game(game: dict, balance_dollars: float) -> str:
    event_ticker = game["event_ticker"]
    home         = game["home_team"]
    away         = game["away_team"]
    home_abbrev  = game["home_abbrev"]
    away_abbrev  = game["away_abbrev"]
    matchup      = f"{away} @ {home}"

    # Skip if already bet this game today
    if _already_bet(event_ticker):
        return f"{matchup}: already bet — skipping"

    # Slow data (cached per day)
    game_data = _get_slow_data(event_ticker, home, away)

    # Fresh Kalshi prices
    prices     = kalshi.get_game_prices(event_ticker)
    home_price = prices.get(home_abbrev, {})
    away_price = prices.get(away_abbrev, {})
    home_ask   = home_price.get("yes_ask")
    away_ask   = away_price.get("yes_ask")
    has_price  = bool(home_ask and away_ask)

    # DQ gate
    dq, penalties = _dq_score(game_data, has_price)
    if dq < MIN_DQ_SCORE:
        reason = f"DQ {dq:.0f} < {MIN_DQ_SCORE} — {'; '.join(penalties)}"
        _log_scan(matchup, home, away, None, None, None, dq, "PASS", reason)
        return f"{matchup}: PASS — {reason}"

    if not has_price:
        _log_scan(matchup, home, away, None, None, None, dq, "PASS", "No Kalshi price")
        return f"{matchup}: PASS — no Kalshi price"

    # Run model
    features = _model.extract_features_from_prefetch(home, away, game_data)
    home_prob = _model.predict_win_prob(home, away, features)
    away_prob = 1.0 - home_prob

    # Evaluate both sides — pick the better edge
    home_edge = (home_prob - home_ask) * 100.0
    away_edge = (away_prob - away_ask) * 100.0

    if home_edge >= away_edge:
        pick_abbrev  = home_abbrev
        pick_team    = home
        model_prob   = home_prob
        implied_prob = home_ask
        edge_pct     = home_edge
    else:
        pick_abbrev  = away_abbrev
        pick_team    = away
        model_prob   = away_prob
        implied_prob = away_ask
        edge_pct     = away_edge

    market_ticker = prices[pick_abbrev]["ticker"]

    _log_scan(matchup, home, away, model_prob, implied_prob, edge_pct, dq,
              "BET" if edge_pct >= MIN_EDGE_PCT else "PASS",
              "" if edge_pct >= MIN_EDGE_PCT else f"edge {edge_pct:.1f}% < {MIN_EDGE_PCT}%")

    if edge_pct < MIN_EDGE_PCT:
        return (
            f"{matchup}: PASS — best edge {edge_pct:.1f}% on {pick_team} "
            f"(model {model_prob:.1%} vs implied {implied_prob:.1%}) | DQ {dq:.0f}"
        )

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
        f"{contracts} contracts (${dollar_stake:.2f}) | DQ: {dq:.0f}"
    )
    log.info(summary)
    t._send_notification(title=f"BetIQ Scanner — {action}", message=summary)
    return summary


# ── Main scan ──────────────────────────────────────────────────────────────────

def run_scan() -> None:
    now = datetime.now(EST).strftime("%Y-%m-%d %H:%M EST")
    log.info(f"Scan — {now}")

    games = kalshi.get_today_games()
    if not games:
        log.info("No NBA games found on Kalshi today.")
        return

    log.info(f"{len(games)} game(s) on Kalshi today.")

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
    log.info(f"Edge threshold: {MIN_EDGE_PCT}% | DQ threshold: {MIN_DQ_SCORE} | Poll: every {POLL_INTERVAL_MINUTES} min")

    # Run immediately on startup
    run_scan()

    # Then every POLL_INTERVAL_MINUTES from noon to midnight EST
    scheduler = BlockingScheduler(timezone=EST)
    for hour in range(12, 24):
        for minute in range(0, 60, POLL_INTERVAL_MINUTES):
            scheduler.add_job(run_scan, "cron", hour=hour, minute=minute)

    log.info(f"Scheduled every {POLL_INTERVAL_MINUTES} min, 12 PM–midnight EST.")
    log.info("Press Ctrl+C to stop.")

    try:
        scheduler.start()
    except KeyboardInterrupt:
        log.info("Scanner stopped.")
