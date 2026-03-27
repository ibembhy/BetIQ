"""
BetIQ — Autonomous background runner.
Scans for bets 3x per day at strategic times (EST):
  - 2:00 PM  : Lines posted, initial scan
  - 6:00 PM  : Injury reports in, best pre-game window
  - 9:30 PM  : West coast games, resolve completed games

Run with:
    venv\\Scripts\\activate
    python runner.py
"""

import logging
import sys
from datetime import datetime

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


# ── Scheduled jobs ─────────────────────────────────────────────────────────────

def afternoon_scan():
    run_scan("Afternoon (2PM)")

def evening_scan():
    run_scan("Evening (6PM)")

def late_scan():
    run_scan("Late (9:30PM)")


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
