"""
BetIQ — Injury monitor.

Polls ESPN injury reports every 30 minutes on game days (12 PM–7 PM EST).
When a star player is newly listed as OUT or DOUBTFUL and the line hasn't
moved yet, triggers an immediate single-game Claude scan.
"""

import logging
from datetime import date, datetime

import pytz

import database as db
import tools as t
from agent import run_agent_prefetch
from scan_context import build_prefetch_context, fetch_game_data, prefetch_shared_context

log = logging.getLogger("betiq.injury")
EST = pytz.timezone("America/New_York")

# Statuses that indicate a player is likely to miss the game
TRIGGER_STATUSES = {"out", "doubtful"}

# If the ML has already moved this many points since our baseline snapshot,
# books have priced in the news — don't trigger.
LINE_MOVED_THRESHOLD = 8


def _normalize_status(status: str) -> str:
    return (status or "").lower().strip()


def _game_is_live(status: str) -> bool:
    """Return True if the game has started or finished."""
    if not status:
        return False
    if status in ("Final", "In Progress"):
        return True
    # BDL uses time strings like "7:30 pm ET" for scheduled games
    if status and not status[0].isdigit() and status not in ("", "scheduled"):
        return True
    return False


def _detect_new_injuries(team: str, game_date: str) -> list[str]:
    """
    Fetch current injury report, compare to stored snapshot.
    Returns list of human-readable strings for newly OUT/DOUBTFUL players.
    """
    current = t.get_injury_report(team)
    current_injuries = current.get("injuries", [])

    previous = db.get_injury_snapshot(game_date, team)
    db.save_injury_snapshot(game_date, team, current_injuries)

    if not previous:
        # First poll of the day — save baseline, no comparison yet
        return []

    prev_map = {p["player"]: _normalize_status(p["status"]) for p in previous}
    newly_critical = []

    for inj in current_injuries:
        player = inj.get("player", "")
        status = _normalize_status(inj.get("status", ""))
        if status in TRIGGER_STATUSES:
            prev_status = prev_map.get(player, "")
            if prev_status not in TRIGGER_STATUSES:
                newly_critical.append(f"{player} ({inj.get('status', '')})")

    return newly_critical


def _line_already_moved(home_team: str, game_date: str) -> bool:
    """
    Compare current ML to the earliest snapshot we have for today.
    If it moved LINE_MOVED_THRESHOLD+ points, books already know.
    """
    baseline = db.get_first_odds_snapshot_today(home_team, game_date)
    if not baseline:
        # No baseline stored — assume not moved, trigger anyway
        return False

    current_odds = t.get_current_odds(home_team)
    for game in current_odds.get("games", []):
        if home_team.lower() in game.get("home_team", "").lower():
            current_ml = game.get("home_ml")
            baseline_ml = baseline.get("home_ml")
            if current_ml and baseline_ml:
                moved = abs(int(current_ml) - int(baseline_ml))
                log.info(
                    f"Line check {home_team}: baseline ML={baseline_ml}, "
                    f"current ML={current_ml}, moved={moved}pts"
                )
                return moved >= LINE_MOVED_THRESHOLD
    return False


def _run_injury_scan(
    home: str, away: str, matchup: str, new_injuries: list[str]
) -> None:
    """Trigger an immediate single-game scan with an urgency header."""
    injury_str = ", ".join(new_injuries)

    t._send_notification(
        title=f"⚡ BetIQ Injury Alert — {matchup}",
        message=f"New: {injury_str}\nLine hasn't moved yet. Scanning for edge...",
    )

    try:
        shared = prefetch_shared_context(t)
        game_data = fetch_game_data(home, away, t)
        context = build_prefetch_context(home, away, shared, game_data)

        urgency_header = (
            f"## ⚡ URGENT — Injury-Triggered Scan\n"
            f"**Newly reported:** {injury_str}\n"
            f"**The line has NOT yet moved to reflect this.** "
            f"This is a time-sensitive opportunity — act before books adjust.\n"
            f"Assess how much this injury shifts win probability. "
            f"If real edge exists, submit immediately.\n\n"
        )
        context = urgency_header + context

        response, _, _ = run_agent_prefetch(context, conversation_history=[])
        log.info(f"Injury scan complete — {matchup}:\n{response}")
        t._send_notification(
            title=f"BetIQ Injury Scan Complete — {matchup}",
            message=response[:600],
        )
    except Exception as exc:
        log.error(f"Injury scan failed for {matchup}: {exc}")
        t._send_notification(
            title=f"BetIQ Injury Scan Error — {matchup}",
            message=str(exc)[:300],
        )


def poll_injuries() -> None:
    """
    Main entry point — called by the scheduler every 30 minutes.
    Checks all teams playing today for new OUT/DOUBTFUL players.
    """
    now_str = datetime.now(EST).strftime("%Y-%m-%d %H:%M EST")
    log.info(f"Injury poll — {now_str}")

    games = t.get_todays_games()
    if "error" in games or games.get("count", 0) == 0:
        log.info("No games today — skipping injury poll.")
        return

    today = date.today().isoformat()
    triggers = []

    for game in games.get("games", []):
        home = game.get("home_team", "")
        away = game.get("visitor_team", "")
        matchup = f"{away} @ {home}"

        if _game_is_live(game.get("status", "")):
            log.info(f"Skipping {matchup} — already live.")
            continue

        if db.is_injury_triggered(today, matchup):
            log.info(f"Skipping {matchup} — already triggered today.")
            continue

        triggered = False
        for team in (home, away):
            new_injuries = _detect_new_injuries(team, today)
            if new_injuries:
                log.info(f"New injury — {team}: {new_injuries}")
                if _line_already_moved(home, today):
                    log.info(f"{matchup}: line already moved — books priced it in.")
                    t._send_notification(
                        title="BetIQ Injury Alert (Too Late)",
                        message=f"{matchup}: {', '.join(new_injuries)} ruled out but line already moved.",
                    )
                else:
                    triggers.append((game, home, away, matchup, new_injuries))
                triggered = True
                break  # One trigger per game per poll

        if not triggered:
            log.info(f"{matchup}: no new injuries.")

    for game, home, away, matchup, new_injuries in triggers:
        db.mark_injury_triggered(today, matchup)
        _run_injury_scan(home, away, matchup, new_injuries)
