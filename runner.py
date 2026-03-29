"""
BetIQ — Autonomous background runner.
Scans for bets 2x per day at strategic times (EST):
  - 2:00 PM  : Lines posted, initial scan
  - 6:00 PM  : Injury reports in, best pre-game window

Run with:
    venv\\Scripts\\activate
    python runner.py
"""

import logging
import os
import sys
import time
from datetime import datetime

import pytz
from apscheduler.schedulers.blocking import BlockingScheduler
from dotenv import load_dotenv
from scan_context import build_prefetch_context, fetch_game_data, prefetch_shared_context

load_dotenv()

import tools as t
import model as _betiq_model
from agent import run_agent, run_agent_prefetch
from injury_monitor import poll_injuries

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
    return prefetch_shared_context(t)


def _prefetch_game(home: str, away: str, shared: dict) -> tuple[str, dict | None]:
    """Returns (context_string, model_edge_info). model_edge_info is None if model unavailable."""
    game_results = fetch_game_data(home, away, t)
    model_edge_info = None
    try:
        features = _betiq_model.extract_features_from_prefetch(home, away, game_results)
        # Extract home moneyline odds from the fetched odds
        home_odds = None
        odds_games = game_results.get("odds", {}).get("games", [])
        if odds_games:
            ml = odds_games[0].get("best_lines", {}).get("moneyline", {})
            for team_name, price in ml.items():
                if home.lower() in team_name.lower() or team_name.lower() in home.lower():
                    home_odds = int(price)
                    break
        if home_odds is not None:
            model_edge_info = _betiq_model.get_edge(home, away, features, home_odds)
    except Exception as exc:
        log.warning(f"Model edge computation failed for {home} vs {away}: {exc}")

    context = build_prefetch_context(home, away, shared, game_results)
    # Append model edge to context so Claude can validate and optionally override
    if model_edge_info:
        import json
        context += f"\n\n### Statistical Model Edge (Logistic Regression — 36K game calibrated)\n{json.dumps(model_edge_info, default=str)}\n\nThe model's `edge_pct` is your **default edge to report** in `submit_analysis`. Override it only if you see a concrete reason (injury, trade, lineup change) the model cannot know about, and explain the override in `save_note`."
    return context, model_edge_info


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
        # Re-check live status right before analysis — scan can run 20-30 min and games may tip off
        live_status = game.get("status", "")
        game_id = game.get("id")
        if game_id:
            fresh = t._bdl_get(f"/games/{game_id}", {})
            if "id" in fresh:
                live_status = fresh.get("status", live_status)
        if live_status in ("Final", "In Progress") or (live_status and not live_status[0].isdigit() and live_status not in ("", "scheduled")):
            log.info(f"Skipping {matchup} — game already started (status: {live_status})")
            summaries.append(f"• {matchup}: SKIPPED (already in progress)")
            continue
        log.info(f"Pre-fetching data for {matchup}...")
        try:
            context, model_edge_info = _prefetch_game(home, away, shared)
            log.info(f"Data ready. Model edge: {model_edge_info.get('edge_pct') if model_edge_info else 'N/A'}%. Sending to Claude for analysis...")
            response, _, _ = run_agent_prefetch(context, conversation_history=[], model_edge_info=model_edge_info)
            log.info(f"{matchup} done.\n{response}")
            summaries.append(f"• {matchup}: {response[:300]}")
        except Exception as exc:
            log.error(f"{matchup} failed: {exc}")
            summaries.append(f"• {matchup}: ERROR — {exc}")
        time.sleep(15)  # Pause between games to avoid Haiku rate limit bursts

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


# ── Morning injury snapshot ────────────────────────────────────────────────────

def morning_injury_check() -> None:
    """
    Fetch injury reports for all teams playing today and save as a note.
    No Claude call, no cost — just ESPN API (free). Runs at 10 AM so the
    2 PM scan already has injury context baked into agent notes.
    """
    now = datetime.now(EST).strftime("%Y-%m-%d %H:%M EST")
    log.info(f"Morning injury check — {now}")

    games = t.get_todays_games()
    if "error" in games or games.get("count", 0) == 0:
        log.info("Morning injury check: no games today.")
        return

    lines = []
    for game in games.get("games", []):
        home = game.get("home_team", "")
        away = game.get("visitor_team", "")
        for team in (home, away):
            report = t.get_injury_report(team)
            injuries = report.get("injuries", [])
            if injuries:
                names = ", ".join(
                    f"{i['player']} ({i['status']})" for i in injuries
                )
                lines.append(f"{team}: {names}")

    if lines:
        note = "INJURY REPORT (" + now + "):\n" + "\n".join(lines)
        t.save_note(note_type="injury_snapshot", content=note)
        log.info(f"Morning injury check saved: {len(lines)} team(s) with injuries.")
    else:
        log.info("Morning injury check: no injuries reported today.")


# ── Scheduled jobs ─────────────────────────────────────────────────────────────

def afternoon_scan():
    run_scan("Afternoon (2:30PM)")

def evening_scan():
    run_scan("Evening (6:35PM)")

def closing_snapshot():
    """Capture closing odds only — no Claude call, no cost."""
    now = datetime.now(EST).strftime("%Y-%m-%d %H:%M EST")
    log.info(f"Closing odds snapshot — {now}")
    try:
        t.snapshot_closing_odds()
        log.info("Closing odds snapshot complete.")
    except Exception as exc:
        log.error(f"Closing odds snapshot failed: {exc}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Prevent duplicate instances — verify the locked PID is actually alive
    LOCK_FILE = os.path.join(os.path.dirname(__file__), "runner.lock")
    if os.path.exists(LOCK_FILE):
        with open(LOCK_FILE) as f:
            pid = f.read().strip()
        try:
            os.kill(int(pid), 0)  # signal 0 = just check if process exists
            print(f"Runner already running (PID {pid}). Exiting.")
            sys.exit(1)
        except (OSError, ValueError):
            # PID is not running — stale lock, safe to overwrite
            pass
    with open(LOCK_FILE, "w") as f:
        f.write(str(os.getpid()))

    scheduler = BlockingScheduler(timezone=EST)

    # 10:00 AM EST — injury snapshot (free, ESPN API only)
    scheduler.add_job(morning_injury_check, "cron", hour=10, minute=0)
    # 2:30 PM EST — lines posted, catches early tip-offs (3:30 PM games)
    scheduler.add_job(afternoon_scan,    "cron", hour=14, minute=30)
    # 6:35 PM EST — late games only (6 PM games already started, analyzes 7 PM+ tip-offs)
    scheduler.add_job(evening_scan,      "cron", hour=18, minute=35)
    # 7:00 PM EST — closing snapshot for early tip-offs
    scheduler.add_job(closing_snapshot,  "cron", hour=19, minute=0)
    # 8:30 PM EST — closing snapshot for 8 PM tip-offs
    scheduler.add_job(closing_snapshot,  "cron", hour=20, minute=30)
    # 10:00 PM EST — closing snapshot for 9:30 PM tip-offs
    scheduler.add_job(closing_snapshot,  "cron", hour=22, minute=0)
    # 11:30 PM EST — final sweep for west coast late games
    scheduler.add_job(closing_snapshot,  "cron", hour=23, minute=30)

    # Injury monitor — every 30 min from 12:00 PM to 7:00 PM EST
    # Polls ESPN (free), only hits Odds API if a new injury is detected
    for _hour in range(12, 19):
        scheduler.add_job(poll_injuries, "cron", hour=_hour, minute=0)
        scheduler.add_job(poll_injuries, "cron", hour=_hour, minute=30)

    log.info("BetIQ runner started. Injury check 10:00 AM. Scans at 2:30 PM, 6:35 PM EST. Closing snapshots at 7:00 PM, 8:30 PM, 10:00 PM, 11:30 PM EST. Injury monitor every 30 min 12–7 PM.")
    log.info("Press Ctrl+C to stop.")

    t._send_notification(
        title="BetIQ Runner Started",
        message="Autonomous scanner is running.\nScans at 2:30 PM, 6:35 PM EST.\nInjury monitor every 30 min 12–7 PM.\nClosing snapshots at 7:00 PM, 8:30 PM, 10:00 PM, 11:30 PM EST.\nYou'll be notified of every bet placed and resolved.",
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
