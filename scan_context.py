"""
Shared prefetch/context-building utilities for scheduled and manual scans.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed


def prefetch_shared_context(tools_module) -> dict:
    with ThreadPoolExecutor(max_workers=3) as ex:
        f_notes = ex.submit(tools_module.get_notes)
        f_history = ex.submit(tools_module.get_bet_history)
        f_bankroll = ex.submit(tools_module.get_bankroll)
    return {
        "notes": f_notes.result(),
        "bet_history": f_history.result(),
        "bankroll": f_bankroll.result(),
    }


def fetch_game_data(home: str, away: str, tools_module) -> dict:
    fetches = {
        "team_stats_home": (tools_module.get_team_stats, (home,), {}),
        "team_stats_away": (tools_module.get_team_stats, (away,), {}),
        "season_stats_home": (tools_module.get_season_stats, (home,), {}),
        "season_stats_away": (tools_module.get_season_stats, (away,), {}),
        "recent_form_home": (tools_module.get_recent_form, (home,), {}),
        "recent_form_away": (tools_module.get_recent_form, (away,), {}),
        "splits_home": (tools_module.get_home_away_splits, (home,), {}),
        "splits_away": (tools_module.get_home_away_splits, (away,), {}),
        "rest_home": (tools_module.get_rest_days, (home,), {}),
        "rest_away": (tools_module.get_rest_days, (away,), {}),
        "injuries_home": (tools_module.get_injury_report, (home,), {}),
        "injuries_away": (tools_module.get_injury_report, (away,), {}),
        "h2h": (tools_module.get_head_to_head, (home, away), {}),
        "odds": (tools_module.get_current_odds, (home,), {}),
        "discrepancies_home": (tools_module.get_book_discrepancies, (home,), {}),
        "discrepancies_away": (tools_module.get_book_discrepancies, (away,), {}),
        "public_pct": (tools_module.get_public_betting_percentages, (home,), {}),
        "line_move_home": (tools_module.get_line_movement, (home,), {}),
        "line_move_away": (tools_module.get_line_movement, (away,), {}),
        "advanced_home": (tools_module.get_advanced_stats, (home,), {}),
        "advanced_away": (tools_module.get_advanced_stats, (away,), {}),
        "roster_home": (tools_module.get_current_roster, (home,), {}),
        "roster_away": (tools_module.get_current_roster, (away,), {}),
        "elo_prob": (tools_module.get_elo_probability, (home, away), {}),
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
    return results


def build_prefetch_context(home: str, away: str, shared: dict, results: dict) -> str:
    def j(value):
        return json.dumps(value, default=str)

    return f"""## Game: {away} @ {home}

### Scan Context
**Past lessons / notes:** {j(shared['notes'])}
**Bet history & performance:** {j(shared['bet_history'])}
**Current bankroll:** {j(shared['bankroll'])}

### ⚠️ CURRENT ROSTERS — Only reference players listed here. Any player not on this list no longer plays for this team.
**{home} roster:** {j(results['roster_home'])}
**{away} roster:** {j(results['roster_away'])}

### {home} (Home)
**Season stats:** {j(results['team_stats_home'])}
**Advanced metrics (OffRtg/DefRtg/Pace/TS%):** {j(results['advanced_home'])}
**3-season trend:** {j(results['season_stats_home'])}
**Recent form (last 10):** {j(results['recent_form_home'])}
**Home/Away splits:** {j(results['splits_home'])}
**Rest days:** {j(results['rest_home'])}
**Injuries:** {j(results['injuries_home'])}

### {away} (Away)
**Season stats:** {j(results['team_stats_away'])}
**Advanced metrics (OffRtg/DefRtg/Pace/TS%):** {j(results['advanced_away'])}
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

### Market Discipline
**Deterministic support:** moneyline probabilities are supported via coded Elo + signal adjustments.
**Not yet supported deterministically:** spread cover probabilities and totals probabilities. If you choose those markets, explain them qualitatively but expect the system to downgrade them to LEAN or PASS unless a coded model exists.

Analyse this game. If edge ≥ 5% and bankroll rules allow, place a bet. Otherwise log a candidate bet with the reason. Save a note with any pattern or lesson observed."""
