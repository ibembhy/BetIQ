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
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("betiq")

EST = pytz.timezone("America/New_York")

# Global scheduler reference (set in __main__, used by late_scan)
scheduler = None

# ── Scan prompt ───────────────────────────────────────────────────────────────

SCAN_PROMPT = (
    "Scan all of today's NBA games. For each game, gather the full analysis "
    "(team stats, recent form, home/away splits, rest days, injuries, head-to-head, "
    "and current odds). Identify any bets with a 5%+ edge and place them if bankroll "
    "rules allow. Also resolve any open bets from previous games. "
    "Report what you found and what actions you took."
)

# ── Core scan function ─────────────────────────────────────────────────────────

def run_scan(label: str) -> None:
    now = datetime.now(EST).strftime("%Y-%m-%d %H:%M EST")
    log.info(f"Starting {label} scan — {now}")

    # Check if there are any NBA games today before burning API calls
    games = t.get_todays_games()
    game_count = games.get("count", 0)

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

    try:
        response, _, _ = run_agent(SCAN_PROMPT, conversation_history=[])
        log.info(f"{label} scan complete.\n{response}")
        t._send_notification(
            title=f"BetIQ {label} Scan Complete",
            message=response[:3000],  # Telegram max ~4096 chars
        )
    except Exception as exc:
        log.error(f"{label} scan failed: {exc}")
        t._send_notification(
            title="BetIQ Scan Error",
            message=f"{label} scan failed: {exc}",
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
    scheduler = BlockingScheduler(timezone=EST)

    # 2:00 PM EST — lines posted
    scheduler.add_job(afternoon_scan, "cron", hour=14, minute=0)
    # 6:00 PM EST — injury reports in, best pre-game window
    scheduler.add_job(evening_scan,   "cron", hour=18, minute=0)
    # 9:30 PM EST — west coast games, resolve early games
    scheduler.add_job(late_scan,      "cron", hour=21, minute=30)

    log.info("BetIQ runner started. Scans scheduled at 2:00 PM, 6:00 PM, 9:30 PM EST.")
    log.info("Press Ctrl+C to stop.")

    t._send_notification(
        title="BetIQ Runner Started",
        message="Autonomous scanner is running.\nScans at 2:00 PM, 6:00 PM, 9:30 PM EST.\nYou'll be notified of every bet placed and resolved.",
    )

    try:
        scheduler.start()
    except KeyboardInterrupt:
        log.info("BetIQ runner stopped.")
        t._send_notification(
            title="BetIQ Runner Stopped",
            message="The autonomous scanner has been stopped.",
        )
