"""
Standalone manual scan — run directly, does not import runner.py.
Usage: venv/bin/python manual_trigger.py
"""
import os
import sys

# Must load env BEFORE any other local imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env'))

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import pytz

import tools as t
from agent import run_agent_prefetch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.FileHandler(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "manual_scan.log"),
            encoding="utf-8"
        ),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("betiq.manual")
EST = pytz.timezone("America/New_York")


def _prefetch_shared():
    with ThreadPoolExecutor(max_workers=3) as ex:
        f_notes    = ex.submit(t.get_notes)
        f_history  = ex.submit(t.get_bet_history)
        f_bankroll = ex.submit(t.get_bankroll)
    return {
        "notes":       f_notes.result(),
        "bet_history": f_history.result(),
        "bankroll":    f_bankroll.result(),
    }


def _prefetch_game(home, away, shared):
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
        "advanced_home":      (t.get_advanced_stats,        (home,), {}),
        "advanced_away":      (t.get_advanced_stats,        (away,), {}),
        "roster_home":        (t.get_current_roster,        (home,), {}),
        "roster_away":        (t.get_current_roster,        (away,), {}),
        "elo_prob":           (t.get_elo_probability,       (home, away), {}),
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

### CURRENT ROSTERS — Only reference players listed here.
**{home} roster:** {j(results['roster_home'])}
**{away} roster:** {j(results['roster_away'])}

### {home} (Home)
**Season stats:** {j(results['team_stats_home'])}
**Advanced metrics:** {j(results['advanced_home'])}
**3-season trend:** {j(results['season_stats_home'])}
**Recent form (last 10):** {j(results['recent_form_home'])}
**Home/Away splits:** {j(results['splits_home'])}
**Rest days:** {j(results['rest_home'])}
**Injuries:** {j(results['injuries_home'])}

### {away} (Away)
**Season stats:** {j(results['team_stats_away'])}
**Advanced metrics:** {j(results['advanced_away'])}
**3-season trend:** {j(results['season_stats_away'])}
**Recent form (last 10):** {j(results['recent_form_away'])}
**Home/Away splits:** {j(results['splits_away'])}
**Rest days:** {j(results['rest_away'])}
**Injuries:** {j(results['injuries_away'])}

### Matchup Data
**Elo model probability (use as baseline win probability):** {j(results['elo_prob'])}
**Head-to-head:** {j(results['h2h'])}
**Current odds:** {j(results['odds'])}
**Book discrepancies ({home}):** {j(results['discrepancies_home'])}
**Book discrepancies ({away}):** {j(results['discrepancies_away'])}
**Public betting %:** {j(results['public_pct'])}
**Line movement ({home}):** {j(results['line_move_home'])}
**Line movement ({away}):** {j(results['line_move_away'])}

Analyse this game. If edge >= 5% and bankroll rules allow, place a bet. Otherwise log a candidate bet with the reason. Save a note with any pattern or lesson observed."""


def run():
    now = datetime.now(EST).strftime("%Y-%m-%d %H:%M EST")
    log.info(f"Starting Manual scan -- {now}")

    games = t.get_todays_games()
    game_count = games.get("count", 0)
    log.info(f"get_todays_games returned: {game_count} games")

    if "error" in games:
        log.error(f"Could not fetch games: {games['error']}")
        return

    if game_count == 0:
        log.info("No NBA games today -- skipping scan.")
        return

    try:
        t.resolve_bets()
    except Exception as exc:
        log.warning(f"Pre-scan resolve failed: {exc}")
    try:
        t.snapshot_closing_odds()
    except Exception as exc:
        log.warning(f"Pre-scan snapshot failed: {exc}")

    shared = _prefetch_shared()
    log.info("Shared context pre-fetched.")

    for game in games.get("games", []):
        home = game.get("home_team", "Home")
        away = game.get("visitor_team", "Away")
        matchup = f"{away} @ {home}"
        live_status = game.get("status", "")
        game_id = game.get("id")
        if game_id:
            fresh = t._bdl_get(f"/games/{game_id}", {})
            if "id" in fresh:
                live_status = fresh.get("status", live_status)
        if live_status in ("Final", "In Progress") or (
            live_status
            and not live_status[0].isdigit()
            and live_status not in ("", "scheduled")
        ):
            log.info(f"Skipping {matchup} -- already started (status: {live_status})")
            continue
        log.info(f"Pre-fetching data for {matchup}...")
        try:
            context = _prefetch_game(home, away, shared)
            log.info("Data ready. Sending to Claude...")
            response, _, _ = run_agent_prefetch(context, conversation_history=[])
            log.info(f"{matchup} done.\n{response}")
        except Exception as exc:
            log.error(f"{matchup} failed: {exc}")
        time.sleep(15)

    log.info("Manual scan complete.")
    t._send_notification(title="BetIQ Manual Scan Complete", message="Manual scan finished.")


if __name__ == "__main__":
    run()
