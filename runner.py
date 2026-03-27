"""
BetIQ — Autonomous background runner.
Scans for bets 2x per day at strategic times (EST):
  - 2:00 PM  : Lines posted, initial scan
  - 6:00 PM  : Injury reports in, best pre-game window

Run with:
    venv\\Scripts\\activate
    python runner.py
"""

import json
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import pytz
from apscheduler.schedulers.blocking import BlockingScheduler
from dotenv import load_dotenv

load_dotenv()

import tools as t
from agent import run_agent, run_agent_prefetch

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.FileHandler("runner.log", encoding="utf-8"),
        logging.StreamHandler(open(sys.stdout.fileno(), mode='w', encoding='utf-8', closefd=False)),
    ],
)
log = logging.getLogger("betiq")

EST = pytz.timezone("America/New_York")

scheduler = None

# ── Pre-fetch helpers ──────────────────────────────────────────────────────────

def _prefetch_shared() -> dict:
    """Fetch scan-level context once (notes, history, bankroll). Free — no Claude call."""
    with ThreadPoolExecutor(max_workers=3) as ex:
        f_notes   = ex.submit(t.get_notes)
        f_history = ex.submit(t.get_bet_history)
        f_bankroll= ex.submit(t.get_bankroll)
    return {
        "notes":       f_notes.result(),
        "bet_history": f_history.result(),
        "bankroll":    f_bankroll.result(),
    }


def _prefetch_game(home: str, away: str, shared: dict) -> str:
    """
    Fetch all read-only data for one game in parallel, then bundle into a
    single context string for Claude. Zero Anthropic API calls.
    """
    fetches = {
        "team_stats_home":    (t.get_team_stats,            (home,), {}),
        "team_stats_away":    (t.get_team_stats,            (away,), {}),
        "season_stats_home":  (t.get_season_stats,          (home,), {}),
        "season_stats_away":  (t.get_season_stats,          (away,), {}),
        "recent_form_home":   (t.get_recent_form,           (home,), {}),
        "recent_form_away":   (t.get_recent_form,           (away,), {}),
        "splits_home":        (t.get_home_away_splits,      (home,), {}),
        "splits_away":        (t.get_home_away_splits,      (away,), {}),
        "rest_home":          (t.get_rest_days,             (home,), {}),
        "rest_away":          (t.get_rest_days,             (away,), {}),
        "injuries_home":      (t.get_injury_report,         (home,), {}),
        "injuries_away":      (t.get_injury_report,         (away,), {}),
        "h2h":                (t.get_head_to_head,          (home, away), {}),
        "odds":               (t.get_current_odds,          (home,), {}),
        "discrepancies_home": (t.get_book_discrepancies,    (home,), {}),
        "discrepancies_away": (t.get_book_discrepancies,    (away,), {}),
        "public_pct":         (t.get_public_betting_percentages, (home,), {}),
        "line_move_home":     (t.get_line_movement,         (home,), {}),
        "line_move_away":     (t.get_line_movement,         (away,), {}),
    }

    results = {}
    with ThreadPoolExecutor(max_workers=6) as ex:
        future_map = {
            ex.submit(fn, *args, **kwargs): key
            for key, (fn, args, kwargs) in fetches.items()
        }
        for future in as_completed(future_map):
            key = future_map[future]
            try:
                results[key] = future.result()
            except Exception as exc:
                results[key] = {"error": str(exc)}

    def j(v):
        return json.dumps(v, default=str)

    return f"""## Game: {away} @ {home}

### Scan Context
**Past lessons / notes:** {j(shared['notes'])}
**Bet history & performance:** {j(shared['bet_history'])}
**Current bankroll:** {j(shared['bankroll'])}

### {home} (Home)
**Season stats:** {j(results['team_stats_home'])}
**3-season trend:** {j(results['season_stats_home'])}
**Recent form (last 10):** {j(results['recent_form_home'])}
**Home/Away splits:** {j(results['splits_home'])}
**Rest days:** {j(results['rest_home'])}
**Injuries:** {j(results['injuries_home'])}

### {away} (Away)
**Season stats:** {j(results['team_stats_away'])}
**3-season trend:** {j(results['season_stats_away'])}
**Recent form (last 10):** {j(results['recent_form_away'])}
**Home/Away splits:** {j(results['splits_away'])}
**Rest days:** {j(results['rest_away'])}
**Injuries:** {j(results['injuries_away'])}

### Matchup Data
**Head-to-head:** {j(results['h2h'])}
**Current odds:** {j(results['odds'])}
**Book discrepancies ({home}):** {j(results['discrepancies_home'])}
**Book discrepancies ({away}):** {j(results['discrepancies_away'])}
**Public betting %:** {j(results['public_pct'])}
**Line movement ({home}):** {j(results['line_move_home'])}
**Line movement ({away}):** {j(results['line_move_away'])}

Analyse this game. If edge ≥ 5% and bankroll rules allow, place a bet. Otherwise log a candidate bet with the reason. Save a note with any pattern or lesson observed."""

# ── Core scan function ─────────────────────────────────────────────────────────

def run_scan(label: str) -> None:
    now = datetime.now(EST).strftime("%Y-%m-%d %H:%M EST")
    log.info(f"Starting {label} scan — {now}")

    # Check if there are any NBA games today before burning API calls
    games = t.get_todays_games()
    game_count = games.get("count", 0)

    if "error" in games:
        log.error(f"Could not fetch games: {games['error']}")
        t._send_notification(
            title="BetIQ Scan Error",
            message=f"{label} scan ({now}): Could not fetch games — {games['error']}",
        )
        return

    if game_count == 0:
        log.info("No NBA games today — skipping scan.")
        t._send_notification(
            title="BetIQ Scan Skipped",
            message=f"{label} scan ({now}): No NBA games today.",
        )
        return

    log.info(f"{game_count} game(s) today — running agent analysis.")
    t._send_notification(
        title=f"BetIQ {label} Scan",
        message=f"Starting analysis of {game_count} game(s)...",
    )

    # Resolve open bets and snapshot closing odds once before game loop
    try:
        t.resolve_bets()
    except Exception as exc:
        log.warning(f"Pre-scan resolve failed: {exc}")
    try:
        t.snapshot_closing_odds()
    except Exception as exc:
        log.warning(f"Pre-scan snapshot failed: {exc}")

    # Pre-fetch scan-level context once (free — no Claude call)
    try:
        shared = _prefetch_shared()
        log.info("Shared context pre-fetched (notes, history, bankroll).")
    except Exception as exc:
        log.warning(f"Shared prefetch failed: {exc}")
        shared = {"notes": {}, "bet_history": {}, "bankroll": {}}

    summaries = []
    for game in games.get("games", []):
        home = game.get("home_team", "Home")
        away = game.get("visitor_team", "Away")
        matchup = f"{away} @ {home}"
        log.info(f"Pre-fetching data for {matchup}...")
        try:
            context = _prefetch_game(home, away, shared)
            log.info(f"Data ready. Sending to Claude for analysis...")
            response, _, _ = run_agent_prefetch(context, conversation_history=[])
            log.info(f"{matchup} done.\n{response}")
            summaries.append(f"• {matchup}: {response[:300]}")
        except Exception as exc:
            log.error(f"{matchup} failed: {exc}")
            summaries.append(f"• {matchup}: ERROR — {exc}")

    summary = f"{label} scan complete. {game_count} game(s) analysed.\n\n" + "\n\n".join(summaries)
    log.info(summary)
    t._send_notification(
        title=f"BetIQ {label} Scan Complete",
        message=summary[:3000],
    )


# ── Late resolve (resolve-only, no agent call) ─────────────────────────────────

def resolve_only() -> None:
    """Settle any remaining open bets after a late-finishing game slate."""
    now = datetime.now(EST).strftime("%Y-%m-%d %H:%M EST")
    log.info(f"Late resolve scan — {now}")
    try:
        result = t.resolve_bets()
        log.info(f"Late resolve complete: {result}")
        t._send_notification(
            title="BetIQ Late Resolve Complete",
            message=str(result)[:3000],
        )
    except Exception as exc:
        log.error(f"Late resolve failed: {exc}")
        t._send_notification(
            title="BetIQ Late Resolve Error",
            message=str(exc),
        )


# ── Scheduled jobs ─────────────────────────────────────────────────────────────

def afternoon_scan():
    run_scan("Afternoon (2PM)")

def evening_scan():
    run_scan("Evening (6PM)")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Prevent duplicate instances
    LOCK_FILE = os.path.join(os.path.dirname(__file__), "runner.lock")
    if os.path.exists(LOCK_FILE):
        with open(LOCK_FILE) as f:
            pid = f.read().strip()
        print(f"Runner already running (PID {pid}). Exiting.")
        sys.exit(1)
    with open(LOCK_FILE, "w") as f:
        f.write(str(os.getpid()))

    scheduler = BlockingScheduler(timezone=EST)

    # 2:00 PM EST — lines posted
    scheduler.add_job(afternoon_scan, "cron", hour=14, minute=0)
    # 6:00 PM EST — injury reports in, best pre-game window
    scheduler.add_job(evening_scan,   "cron", hour=18, minute=0)

    log.info("BetIQ runner started. Scans scheduled at 2:00 PM, 6:00 PM EST.")
    log.info("Press Ctrl+C to stop.")

    t._send_notification(
        title="BetIQ Runner Started",
        message="Autonomous scanner is running.\nScans at 2:00 PM, 6:00 PM EST.\nYou'll be notified of every bet placed and resolved.",
    )

    try:
        scheduler.start()
    except KeyboardInterrupt:
        log.info("BetIQ runner stopped.")
        t._send_notification(
            title="BetIQ Runner Stopped",
            message="The autonomous scanner has been stopped.",
        )
    finally:
        if os.path.exists(LOCK_FILE):
            os.remove(LOCK_FILE)
