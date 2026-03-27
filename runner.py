"""
BetIQ — Autonomous background runner.
Scans for bets 3x per day at strategic times (EST):
  - 2:00 PM  : Lines posted, initial scan
  - 6:00 PM  : Injury reports in, best pre-game window
  - 9:30 PM  : West coast games, resolve completed games

After the 9:30 PM scan, if any game tips off late enough to finish past
midnight EST, a one-time resolve-only job is dynamically scheduled.

Run with:
    venv\\Scripts\\activate
    python runner.py
"""

import logging
import os
import sys
from datetime import datetime, timedelta

import pytz
from apscheduler.schedulers.blocking import BlockingScheduler
from dotenv import load_dotenv

load_dotenv()

import tools as t
from agent import run_agent

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

# Global scheduler reference (set in __main__, used by late_scan)
scheduler = None

# ── Per-game prompt ────────────────────────────────────────────────────────────

def _game_prompt(home: str, away: str) -> str:
    return (
        f"Analyse the game: {away} @ {home}. "
        "Before analysing, call get_notes and get_bet_history to load context, "
        "then call get_bankroll and resolve_bets. "
        "Gather the full analysis: team stats, season stats, recent form, home/away splits, "
        "rest days, injuries, head-to-head, current odds, book discrepancies, "
        "public betting percentages, and line movement for both teams. "
        "If you find a bet with 5%+ edge and bankroll rules allow, place it. "
        "Otherwise log a candidate bet with the reason. "
        "Report what you found and what action you took."
    )

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

    # Resolve open bets once before game loop
    try:
        t.resolve_bets()
    except Exception as exc:
        log.warning(f"Pre-scan resolve failed: {exc}")

    summaries = []
    for game in games.get("games", []):
        home = game.get("home_team", "Home")
        away = game.get("visitor_team", "Away")
        matchup = f"{away} @ {home}"
        log.info(f"Analysing {matchup}...")
        try:
            response, _, _ = run_agent(_game_prompt(home, away), conversation_history=[])
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


def _maybe_schedule_late_resolve() -> None:
    """
    Called after the 9:30 PM scan.  Checks if any of today's games tip off
    late enough that they will finish past midnight EST.  If so — and only
    then — schedules a single resolve-only job for that time.
    """
    if scheduler is None:
        return

    # Only worth doing if we actually have open bets
    bankroll = t.get_bankroll()
    if not bankroll.get("open_bets"):
        log.info("No open bets — skipping late resolve check.")
        return

    games = t.get_todays_games()
    if "error" in games or games.get("count", 0) == 0:
        return

    now_est = datetime.now(EST)
    # Midnight that ends tonight (start of tomorrow)
    midnight_est = now_est.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)

    latest_resolve: datetime | None = None

    for game in games.get("games", []):
        time_str = game.get("time", "")
        if not time_str:
            continue
        try:
            # BallDontLie returns UTC ISO-8601, e.g. "2025-03-27T02:30:00.000Z"
            tip_utc = datetime.fromisoformat(time_str.replace("Z", "+00:00"))
            tip_est = tip_utc.astimezone(EST)
            # Games average ~2 h 15 min; add 2 h 30 min buffer to be safe
            resolve_est = tip_est + timedelta(hours=2, minutes=30)
            if resolve_est > midnight_est:
                if latest_resolve is None or resolve_est > latest_resolve:
                    latest_resolve = resolve_est
        except (ValueError, TypeError, AttributeError):
            continue

    if latest_resolve is None:
        log.info("All games finish before midnight — no late resolve needed.")
        return

    log.info(f"Late game detected. Scheduling resolve at {latest_resolve.strftime('%I:%M %p EST')}.")
    scheduler.add_job(
        resolve_only,
        trigger="date",
        run_date=latest_resolve,
        id="late_resolve",
        replace_existing=True,
    )
    t._send_notification(
        title="BetIQ Late Resolve Scheduled",
        message=f"Last game finishes after midnight.\nResolve scan scheduled for {latest_resolve.strftime('%I:%M %p EST')}.",
    )


# ── Scheduled jobs ─────────────────────────────────────────────────────────────

def afternoon_scan():
    run_scan("Afternoon (2PM)")

def evening_scan():
    run_scan("Evening (6PM)")

def late_scan():
    run_scan("Late (9:30PM)")
    # Dynamically schedule a resolve-only job if games end past midnight
    _maybe_schedule_late_resolve()


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
