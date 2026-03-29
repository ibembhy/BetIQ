"""
BetIQ — All tool implementations.
Covers: Balldontlie API, The Odds API, SQLite bet management.
"""

import logging
import os
import re
import time
import requests

log = logging.getLogger("betiq.tools")
from datetime import datetime, date
from typing import Optional
from functools import lru_cache

try:
    from nba_api.stats.static import teams as _nba_static_teams
    from nba_api.stats.endpoints import (
        CommonTeamRoster,
        TeamGameLog,
        LeagueDashTeamStats,
        TeamDashboardByGeneralSplits,
    )
    _NBA_API_AVAILABLE = True
except ImportError:
    _NBA_API_AVAILABLE = False

import database as db
import data_loader as dl
from betting_math import (
    american_odds_to_decimal,
    american_odds_to_implied_probability,
    expected_value,
    kelly_stake,
    no_vig_probabilities_from_odds,
)
from decision_support import (
    SUPPORTED_PROBABILITY_MARKETS,
    clamp_probability,
    compute_data_quality_score,
    compute_moneyline_probability,
    infer_decision,
    recommendation_snapshot,
    weighted_injury_count,
)

# ── Config ────────────────────────────────────────────────────────────────────

ODDS_API_KEY       = os.getenv("ODDS_API_KEY", "")
BDL_API_KEY        = os.getenv("BALLDONTLIE_API_KEY", "")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID", "")

ODDS_BASE = "https://api.the-odds-api.com/v4"
BDL_BASE  = "https://api.balldontlie.io/v1"

# Simple in-memory cache {key: (data, timestamp)}
_cache: dict = {}
_CACHE_TTL = 300  # 5 minutes


def _send_notification(title: str, message: str, tags: str = "") -> None:
    """Send a Telegram message. Silently skips if credentials are not set."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return
    try:
        text = f"*{title}*\n{message}"
        requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
            json={"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "Markdown"},
            timeout=5,
        )
    except Exception:
        pass  # Never block the main flow on a notification failure


def _cached(key: str):
    now = time.time()
    if key in _cache:
        data, ts = _cache[key]
        if now - ts < _CACHE_TTL:
            return data
    return None


def _set_cache(key: str, data):
    _cache[key] = (data, time.time())


# ── HTTP helpers ──────────────────────────────────────────────────────────────

def _bdl_headers() -> dict:
    h = {"Accept": "application/json"}
    if BDL_API_KEY:
        h["Authorization"] = BDL_API_KEY
    return h


def _bdl_get(endpoint: str, params: dict = None) -> dict:
    """GET from Balldontlie v1. Handles array params like dates[]."""
    param_list = []
    for key, value in (params or {}).items():
        if isinstance(value, list):
            for v in value:
                param_list.append((key, v))
        else:
            param_list.append((key, value))
    try:
        r = requests.get(
            f"{BDL_BASE}{endpoint}",
            params=param_list or None,
            headers=_bdl_headers(),
            timeout=15,
        )
        if r.status_code == 401:
            return {"error": "Balldontlie API key required. Add BALLDONTLIE_API_KEY to .env"}
        r.raise_for_status()
        db.log_api_call("BallDontLie", endpoint)
        return r.json()
    except requests.HTTPError as e:
        db.log_api_call("BallDontLie", endpoint)
        return {"error": f"HTTP {r.status_code}: {r.text[:300]}"}
    except Exception as e:
        return {"error": str(e)}


def _odds_get(endpoint: str, params: dict = None) -> dict | list:
    if not ODDS_API_KEY:
        return {"error": "ODDS_API_KEY not set in .env"}
    p = dict(params or {})
    cache_key = f"odds_{endpoint}_{sorted((k, v) for k, v in p.items())}"
    if cached := _cached(cache_key):
        db.log_api_call("Odds API", endpoint, cached=True)
        return cached
    p["apiKey"] = ODDS_API_KEY
    try:
        r = requests.get(f"{ODDS_BASE}{endpoint}", params=p, timeout=15)
        r.raise_for_status()
        result = r.json()
        _cache[cache_key] = (result, time.time())
        db.log_api_call("Odds API", endpoint, cached=False)
        return result
    except requests.HTTPError as e:
        db.log_api_call("Odds API", endpoint, cached=False)
        return {"error": f"Odds API HTTP {r.status_code}: {r.text[:300]}"}
    except Exception as e:
        return {"error": str(e)}


# ── Team resolution ───────────────────────────────────────────────────────────

_teams_cache: dict = {}

def _get_all_teams() -> dict:
    global _teams_cache
    if _teams_cache:
        return _teams_cache
    data = _bdl_get("/teams", {"per_page": 100})
    if "error" in data or not data.get("data"):
        return {}  # Don't cache failures — retry next call
    result: dict = {}
    for team in data.get("data", []):
        for key in (
            team["full_name"].lower(),
            team["name"].lower(),
            team["abbreviation"].lower(),
            team["city"].lower(),
        ):
            result[key] = team
    _teams_cache = result
    return _teams_cache


def _resolve_team(name: str) -> Optional[dict]:
    teams = _get_all_teams()
    nl = name.lower().strip()
    if nl in teams:
        return teams[nl]
    for key, team in teams.items():
        if nl in key or key in nl:
            return team
    return None


_NBA_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Referer": "https://www.nba.com/",
    "Origin": "https://www.nba.com",
    "x-nba-stats-origin": "stats",
    "x-nba-stats-token": "true",
}
_nba_id_cache: dict = {}

def _nba_team_id(team_name: str) -> Optional[int]:
    if not _NBA_API_AVAILABLE:
        return None
    if team_name in _nba_id_cache:
        return _nba_id_cache[team_name]
    nl = team_name.lower().strip()
    for t in _nba_static_teams.get_teams():
        if nl == t["full_name"].lower() or nl in t["full_name"].lower() or t["nickname"].lower() in nl:
            _nba_id_cache[team_name] = t["id"]
            return t["id"]
    return None

def _nba_season_str() -> str:
    s = _current_season()
    return f"{s}-{str(s + 1)[-2:]}"

def _current_season() -> int:
    now = datetime.now()
    return now.year if now.month >= 10 else now.year - 1


# ── Tool implementations ──────────────────────────────────────────────────────

def get_todays_games() -> dict:
    """
    Fetch today's NBA games. Queries both today AND tomorrow from BallDontLie
    because late games (9:30 PM EST+) tip off after midnight UTC and get classified
    as the next day in BDL. All games are normalized to today's EST date.
    """
    from datetime import timedelta
    today     = date.today()
    tomorrow  = today + timedelta(days=1)
    today_str = today.isoformat()
    cache_key = f"today_games_{today_str}"
    if cached := _cached(cache_key):
        return cached

    seen_ids = set()
    games = []

    for query_date in (today_str, tomorrow.isoformat()):
        data = _bdl_get("/games", {"dates[]": query_date, "per_page": 100})
        if "error" in data:
            continue
        for g in data.get("data", []):
            if g["id"] in seen_ids:
                continue
            if query_date == tomorrow.isoformat():
                status = g.get("status", "")
                # Only include tomorrow-date games that are already in progress/Final
                # (those are tonight's late games that crossed midnight UTC),
                # OR games that tip off before 06:00 UTC (= before 2 AM EST = tonight).
                # Skip daytime games on tomorrow's date — they are genuinely tomorrow.
                is_started = status in ("Final", "In Progress") or (status and not status[0].isdigit())
                is_late_tonight = status and status[0].isdigit() and status[11:16] < "06:00"
                if not is_started and not is_late_tonight:
                    continue
            seen_ids.add(g["id"])
            games.append({
                "id":                 g["id"],
                "date":               today_str,  # normalize all to EST today
                "bdl_date":           g.get("date", today_str)[:10],  # keep original for resolution
                "home_team":          g["home_team"]["full_name"],
                "visitor_team":       g["visitor_team"]["full_name"],
                "home_team_score":    g.get("home_team_score", 0),
                "visitor_team_score": g.get("visitor_team_score", 0),
                "status":             g.get("status", "scheduled"),
                "time":               g.get("time", ""),
            })

    result = {"games": games, "count": len(games), "date": today_str}
    _set_cache(cache_key, result)
    return result


def get_team_stats(team_name: str, season: int = None) -> dict:
    season = season or _current_season()
    local = dl.get_team_stats_local(team_name, season)
    if local is not None:
        return local

    cache_key = f"team_stats_{team_name}_{season}"
    if cached := _cached(cache_key):
        return cached

    if _NBA_API_AVAILABLE:
        team_id = _nba_team_id(team_name)
        if team_id:
            try:
                season_str = f"{season}-{str(season + 1)[-2:]}"
                time.sleep(0.6)
                ep = LeagueDashTeamStats(season=season_str, timeout=15, headers=_NBA_HEADERS)
                df = ep.league_dash_team_stats.get_data_frame()
                nl = team_name.lower()
                match = df[df["TEAM_NAME"].str.lower().str.contains(nl.split()[-1], na=False)]
                if match.empty:
                    return {"error": f"Team not found: {team_name}"}
                row = match.iloc[0]
                result = {
                    "team": row["TEAM_NAME"], "season": season, "source": "nba_api",
                    "record": {"wins": int(row["W"]), "losses": int(row["L"]), "win_pct": round(float(row["W_PCT"]), 3)},
                    "averages": {
                        "pts": round(float(row["PTS"]), 1), "reb": round(float(row["REB"]), 1),
                        "ast": round(float(row["AST"]), 1), "stl": round(float(row["STL"]), 1),
                        "blk": round(float(row["BLK"]), 1), "tov": round(float(row["TOV"]), 1),
                        "fg_pct": round(float(row["FG_PCT"]), 3), "fg3_pct": round(float(row["FG3_PCT"]), 3),
                    },
                }
                _set_cache(cache_key, result)
                return result
            except Exception:
                pass

    # BDL fallback
    team = _resolve_team(team_name)
    if not team:
        return {"error": f"Team not found: {team_name}"}
    avg_data = _bdl_get("/season_averages", {"season": season, "team_ids[]": team["id"]})
    games_data = _bdl_get("/games", {"seasons[]": season, "team_ids[]": team["id"], "per_page": 100})
    wins = losses = 0
    if "data" in games_data:
        for g in games_data["data"]:
            if g.get("status") != "Final":
                continue
            if g["home_team"]["id"] == team["id"]:
                if g["home_team_score"] > g["visitor_team_score"]: wins += 1
                else: losses += 1
            else:
                if g["visitor_team_score"] > g["home_team_score"]: wins += 1
                else: losses += 1
    result = {
        "team": team["full_name"], "season": season, "source": "bdl",
        "record": {"wins": wins, "losses": losses, "win_pct": round(wins/max(wins+losses,1), 3)},
        "averages": avg_data.get("data", []),
    }
    _set_cache(cache_key, result)
    return result


def get_season_stats(team_name: str, seasons: list = None) -> dict:
    cur     = _current_season()
    seasons = seasons or [cur - 2, cur - 1, cur]

    # Pull local data for all seasons in the Kaggle dataset
    local = dl.get_season_stats_local(team_name, seasons)
    multi = dict(local["multi_season_stats"]) if local else {}

    # Fill in any seasons not covered locally (recent seasons) via API
    for s in seasons:
        if str(s) not in multi:
            multi[str(s)] = get_team_stats(team_name, s)

    team_name_resolved = local["team"] if local else team_name
    return {"team": team_name_resolved, "multi_season_stats": multi}


def get_recent_form(team_name: str, last_n: int = 10) -> dict:
    cache_key = f"form_{team_name}_{last_n}"
    if cached := _cached(cache_key):
        return cached

    if _NBA_API_AVAILABLE:
        team_id = _nba_team_id(team_name)
        if team_id:
            try:
                time.sleep(0.6)
                ep = TeamGameLog(team_id=team_id, season=_nba_season_str(), timeout=15, headers=_NBA_HEADERS)
                df = ep.team_game_log.get_data_frame().head(last_n)
                form = []
                for _, row in df.iterrows():
                    matchup = str(row.get("MATCHUP", ""))
                    home = " vs. " in matchup
                    opponent = matchup.split(" vs. ")[-1] if home else matchup.split(" @ ")[-1]
                    pts_for = int(row.get("PTS", 0))
                    margin = int(row.get("PLUS_MINUS", 0))
                    pts_against = pts_for - margin
                    form.append({
                        "date": str(row.get("GAME_DATE", ""))[:10],
                        "opponent": opponent,
                        "location": "Home" if home else "Away",
                        "score": f"{pts_for}-{pts_against}",
                        "result": str(row.get("WL", "")),
                        "margin": margin,
                        "source": "nba_api",
                    })
                wins = sum(1 for g in form if g["result"] == "W")
                result = {
                    "team": team_name, "games_back": len(form),
                    "record": f"{wins}-{len(form) - wins}",
                    "win_pct": round(wins / max(len(form), 1), 3),
                    "avg_margin": round(sum(g["margin"] for g in form) / max(len(form), 1), 1),
                    "form": form,
                }
                _set_cache(cache_key, result)
                return result
            except Exception:
                pass

    # BDL fallback
    team = _resolve_team(team_name)
    if not team:
        return {"error": f"Team not found: {team_name}"}
    season = _current_season()
    data = _bdl_get("/games", {"seasons[]": season, "team_ids[]": team["id"], "per_page": 100})
    form = []
    if "error" not in data:
        finished = [g for g in data.get("data", []) if g.get("status") == "Final"]
        finished.sort(key=lambda g: g.get("date", ""), reverse=True)
        for g in finished[:last_n]:
            home        = g["home_team"]["id"] == team["id"]
            pts_for     = g["home_team_score"]    if home else g["visitor_team_score"]
            pts_against = g["visitor_team_score"] if home else g["home_team_score"]
            opponent    = g["visitor_team"]["full_name"] if home else g["home_team"]["full_name"]
            won         = pts_for > pts_against
            form.append({
                "date": g.get("date", "")[:10], "opponent": opponent,
                "location": "Home" if home else "Away",
                "score": f"{pts_for}-{pts_against}",
                "result": "W" if won else "L", "margin": pts_for - pts_against, "source": "bdl",
            })
    if len(form) < last_n:
        needed = last_n - len(form)
        local = dl.get_recent_form_local(team_name, dl.KAGGLE_MAX_SEASON, needed)
        if local:
            for g in local["form"]:
                g["source"] = "local_kaggle"
            form.extend(local["form"])
    wins = sum(1 for g in form if g["result"] == "W")
    result = {
        "team": team["full_name"], "games_back": len(form),
        "record": f"{wins}-{len(form) - wins}",
        "win_pct": round(wins / max(len(form), 1), 3),
        "avg_margin": round(sum(g["margin"] for g in form) / max(len(form), 1), 1),
        "form": form,
    }
    _set_cache(cache_key, result)
    return result


def get_home_away_splits(team_name: str, season: int = None) -> dict:
    season = season or _current_season()
    local = dl.get_home_away_splits_local(team_name, season)
    if local is not None:
        return local

    season_str = f"{season}-{str(season + 1)[-2:]}"
    cache_key = f"splits_{team_name}_{season}"
    if cached := _cached(cache_key):
        return cached

    if _NBA_API_AVAILABLE:
        team_id = _nba_team_id(team_name)
        if team_id:
            try:
                time.sleep(0.6)
                ep = TeamDashboardByGeneralSplits(team_id=team_id, season=season_str, timeout=15, headers=_NBA_HEADERS)
                loc_df = ep.location_team_dashboard.get_data_frame()
                def _split(label):
                    rows = loc_df[loc_df["GROUP_VALUE"] == label]
                    if rows.empty:
                        return {"record": "0-0", "win_pct": 0, "avg_pts_for": 0, "avg_pts_against": 0, "avg_margin": 0}
                    r = rows.iloc[0]
                    pts = float(r["PTS"]); pm = float(r["PLUS_MINUS"])
                    return {
                        "record": f"{int(r['W'])}-{int(r['L'])}",
                        "win_pct": round(float(r["W_PCT"]), 3),
                        "avg_pts_for": round(pts, 1),
                        "avg_pts_against": round(pts - pm, 1),
                        "avg_margin": round(pm, 1),
                    }
                result = {"team": team_name, "season": season, "source": "nba_api",
                          "home": _split("Home"), "away": _split("Road")}
                _set_cache(cache_key, result)
                return result
            except Exception:
                pass

    # BDL fallback
    team = _resolve_team(team_name)
    if not team:
        return {"error": f"Team not found: {team_name}"}
    data = _bdl_get("/games", {"seasons[]": season, "team_ids[]": team["id"], "per_page": 100})
    if "error" in data:
        return data
    hw = hl = aw = al = 0
    hpf = hpa = apf = apa = 0
    for g in data.get("data", []):
        if g.get("status") != "Final":
            continue
        if g["home_team"]["id"] == team["id"]:
            pf, pa = g["home_team_score"], g["visitor_team_score"]
            hpf += pf; hpa += pa
            if pf > pa: hw += 1
            else: hl += 1
        else:
            pf, pa = g["visitor_team_score"], g["home_team_score"]
            apf += pf; apa += pa
            if pf > pa: aw += 1
            else: al += 1
    hg = hw + hl; ag = aw + al
    result = {
        "team": team["full_name"], "season": season, "source": "bdl",
        "home": {"record": f"{hw}-{hl}", "win_pct": round(hw/max(hg,1),3), "avg_pts_for": round(hpf/max(hg,1),1), "avg_pts_against": round(hpa/max(hg,1),1), "avg_margin": round((hpf-hpa)/max(hg,1),1)},
        "away": {"record": f"{aw}-{al}", "win_pct": round(aw/max(ag,1),3), "avg_pts_for": round(apf/max(ag,1),1), "avg_pts_against": round(apa/max(ag,1),1), "avg_margin": round((apf-apa)/max(ag,1),1)},
    }
    _set_cache(cache_key, result)
    return result


def get_rest_days(team_name: str) -> dict:
    today = date.today()
    cache_key = f"rest_days_{team_name}_{today}"
    if cached := _cached(cache_key):
        return cached

    if _NBA_API_AVAILABLE:
        team_id = _nba_team_id(team_name)
        if team_id:
            try:
                time.sleep(0.6)
                ep = TeamGameLog(team_id=team_id, season=_nba_season_str(), timeout=15, headers=_NBA_HEADERS)
                df = ep.team_game_log.get_data_frame()
                last_game_date = None
                last_matchup = ""
                for _, row in df.iterrows():
                    try:
                        from datetime import datetime as _dt
                        gd = _dt.strptime(str(row.get("GAME_DATE", "")), "%b %d, %Y").date()
                        if gd < today:
                            last_game_date = gd
                            last_matchup = str(row.get("MATCHUP", ""))
                            break
                    except Exception:
                        continue
                result: dict = {"team": team_name}
                if last_game_date:
                    rest = (today - last_game_date).days
                    result.update({
                        "last_game_date": last_game_date.isoformat(),
                        "last_game_matchup": last_matchup,
                        "days_rest": rest,
                        "fatigue_level": "High" if rest <= 1 else ("Medium" if rest <= 2 else "Low"),
                        "source": "nba_api",
                    })
                else:
                    result.update({"days_rest": None, "fatigue_level": "Unknown"})
                _set_cache(cache_key, result)
                return result
            except Exception:
                pass

    # BDL fallback
    team = _resolve_team(team_name)
    if not team:
        return {"error": f"Team not found: {team_name}"}
    season = _current_season()
    data = _bdl_get("/games", {"seasons[]": season, "team_ids[]": team["id"], "per_page": 100})
    if "error" in data:
        return data
    games = [g for g in data.get("data", []) if g.get("date")]
    games.sort(key=lambda g: g["date"], reverse=True)
    last_game = next((g for g in games if g.get("status") == "Final" and date.fromisoformat(g["date"][:10]) < today), None)
    next_game = next((g for g in reversed(games) if date.fromisoformat(g["date"][:10]) >= today), None)
    result = {"team": team["full_name"]}
    if last_game:
        last_date = date.fromisoformat(last_game["date"][:10])
        rest = (today - last_date).days
        result.update({"last_game_date": last_game["date"][:10], "days_rest": rest, "fatigue_level": "High" if rest <= 1 else ("Medium" if rest <= 2 else "Low")})
    else:
        result.update({"days_rest": None, "fatigue_level": "Unknown"})
    if next_game:
        opp = (next_game["visitor_team"]["full_name"] if next_game["home_team"]["id"] == team["id"] else next_game["home_team"]["full_name"])
        result.update({"next_game_date": next_game["date"][:10], "next_opponent": opp, "home_or_away": "Home" if next_game["home_team"]["id"] == team["id"] else "Away"})
    _set_cache(cache_key, result)
    return result


def get_injury_report(team_name: str) -> dict:
    """Fetch live injury report from ESPN's free API."""
    today = date.today().isoformat()
    cache_key = f"injuries_{team_name}_{today}"
    if cached := _cached(cache_key):
        return cached

    try:
        r = requests.get(
            "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries",
            timeout=10,
        )
        r.raise_for_status()
        data = r.json()
        db.log_api_call("ESPN", "/injuries")
    except Exception as e:
        return {"error": f"ESPN injury API failed: {e}"}

    nl = team_name.lower().strip()
    injuries = []

    for team_entry in data.get("injuries", []):
        team_display = team_entry.get("displayName", "")
        # Match by team name
        if nl not in team_display.lower() and team_display.lower() not in nl:
            # Try nickname/city match
            parts = team_display.lower().split()
            if not any(p in nl or nl in p for p in parts):
                continue

        for inj in team_entry.get("injuries", []):
            athlete = inj.get("athlete", {})
            injuries.append({
                "player":       athlete.get("displayName", "Unknown"),
                "status":       inj.get("status", "Unknown"),
                "short_note":   inj.get("shortComment", ""),
                "detail":       inj.get("longComment", ""),
                "updated":      inj.get("date", "")[:10] if inj.get("date") else "",
            })
        break  # found the team

    result = {
        "team":     team_name,
        "source":   "espn_live",
        "injuries": injuries,
        "count":    len(injuries),
        "note":     "Live data from ESPN." if injuries else "No injuries reported for this team.",
    }
    _set_cache(cache_key, result)
    return result


def get_head_to_head(team1_name: str, team2_name: str, num_seasons: int = 3) -> dict:
    # Pull full H2H history from local Kaggle data (all available seasons)
    local = dl.get_head_to_head_local(team1_name, team2_name, num_seasons, include_all_history=True)
    if local is None:
        return {"error": f"Team not found: {team1_name} or {team2_name}"}

    # Also fetch recent seasons (post-dataset) from the API and merge
    team1 = _resolve_team(team1_name)
    team2 = _resolve_team(team2_name)

    cur            = _current_season()
    recent_seasons = list(range(dl.KAGGLE_MAX_SEASON + 1, cur + 1))
    api_matchups   = []

    if team1 and team2:
        for season in recent_seasons:
            data = _bdl_get("/games", {
                "seasons[]":  season,
                "team_ids[]": team1["id"],
                "per_page":   100,
            })
            if "error" in data:
                continue
            for g in data.get("data", []):
                if g.get("status") != "Final":
                    continue
                home_id    = g["home_team"]["id"]
                visitor_id = g["visitor_team"]["id"]
                if not (
                    (home_id == team1["id"] and visitor_id == team2["id"]) or
                    (home_id == team2["id"] and visitor_id == team1["id"])
                ):
                    continue
                t1_home  = home_id == team1["id"]
                t1_score = g["home_team_score"]    if t1_home else g["visitor_team_score"]
                t2_score = g["visitor_team_score"] if t1_home else g["home_team_score"]
                api_matchups.append({
                    "date":       g.get("date", "")[:10],
                    "season":     season,
                    "home_team":  g["home_team"]["full_name"],
                    "home_score": g["home_team_score"],
                    "away_score": g["visitor_team_score"],
                    "winner":     team1["full_name"] if t1_score > t2_score else team2["full_name"],
                    "t1_score":   t1_score,
                    "t2_score":   t2_score,
                    "margin":     t1_score - t2_score,
                    "source":     "api",
                })

    all_matchups = api_matchups + local["matchups"]
    all_matchups.sort(key=lambda x: x["date"], reverse=True)

    t1_name = local["teams"][0]
    t2_name = local["teams"][1]
    t1_wins   = sum(1 for m in all_matchups if m["winner"] == t1_name)
    avg_total = sum(m["t1_score"] + m["t2_score"] for m in all_matchups) / max(len(all_matchups), 1)

    recent_10      = all_matchups[:10]
    recent_t1_wins = sum(1 for m in recent_10 if m["winner"] == t1_name)

    return {
        "teams":            [t1_name, t2_name],
        "total_h2h_games":  len(all_matchups),
        "h2h_record":       {t1_name: t1_wins, t2_name: len(all_matchups) - t1_wins},
        "recent_10_h2h":    {t1_name: recent_t1_wins, t2_name: len(recent_10) - recent_t1_wins},
        "avg_total_points": round(avg_total, 1),
        "matchups":         all_matchups[:50],
    }


def get_current_roster(team_name: str) -> dict:
    today = date.today().isoformat()
    cache_key = f"roster_{team_name}_{today}"
    if cached := _cached(cache_key):
        return cached

    if _NBA_API_AVAILABLE:
        team_id = _nba_team_id(team_name)
        if team_id:
            try:
                time.sleep(0.6)
                ep = CommonTeamRoster(team_id=team_id, season=_nba_season_str(), timeout=15, headers=_NBA_HEADERS)
                df = ep.common_team_roster.get_data_frame()
                players = [
                    {"name": row["PLAYER"], "position": row["POSITION"], "jersey": str(row["NUM"])}
                    for _, row in df.iterrows()
                ]
                result = {
                    "team": team_name, "players": players, "count": len(players),
                    "source": "nba_api",
                    "note": "These are the ONLY players currently on this roster. Do not reference any player not on this list.",
                }
                _set_cache(cache_key, result)
                return result
            except Exception:
                pass

    # BDL fallback
    team = _resolve_team(team_name)
    if not team:
        return {"error": f"Team not found: {team_name}"}
    data = _bdl_get("/players/active", {"team_ids[]": team["id"], "per_page": 100})
    if "error" in data:
        return data
    players = [
        {"name": f"{p.get('first_name','')} {p.get('last_name','')}".strip(),
         "position": p.get("position", ""), "jersey": p.get("jersey_number", "")}
        for p in data.get("data", [])
    ]
    result = {
        "team": team["full_name"], "players": players, "count": len(players), "source": "bdl",
        "note": "These are the ONLY players currently on this roster. Do not reference any player not on this list.",
    }
    _set_cache(cache_key, result)
    return result


def get_advanced_stats(team_name: str) -> dict:
    """
    Fetch advanced team metrics from NBA Stats API (free, unofficial).
    Returns OffRtg, DefRtg, Pace, TS%, AST%, TOV% for the current season.
    """
    today = date.today().isoformat()
    cache_key = f"advanced_stats_{team_name}_{today}"
    if cached := _cached(cache_key):
        return cached

    season = _current_season()
    season_str = f"{season}-{str(season + 1)[-2:]}"

    headers = {
        "User-Agent":           "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Referer":              "https://www.nba.com/",
        "Origin":               "https://www.nba.com",
        "x-nba-stats-origin":  "stats",
        "x-nba-stats-token":   "true",
        "Accept":               "application/json",
    }

    try:
        r = requests.get(
            "https://stats.nba.com/stats/leaguedashteamstats",
            params={
                "MeasureType":  "Advanced",
                "PerMode":      "PerGame",
                "Season":       season_str,
                "SeasonType":   "Regular Season",
                "LeagueID":     "00",
                "LastNGames":   0,
                "Month":        0,
                "OpponentTeamID": 0,
                "PaceAdjust":   "N",
                "PlusMinus":    "N",
                "Rank":         "N",
            },
            headers=headers,
            timeout=15,
        )
        r.raise_for_status()
        data = r.json()
        db.log_api_call("NBA Stats", "/leaguedashteamstats")
    except Exception as e:
        return {"available": False, "note": "Advanced metrics not available (server network restriction)", "team": team_name}

    result_set = data.get("resultSets", [{}])[0]
    headers_list = result_set.get("headers", [])
    rows = result_set.get("rowSet", [])

    nl = team_name.lower()
    for row in rows:
        row_dict = dict(zip(headers_list, row))
        if nl in row_dict.get("TEAM_NAME", "").lower():
            result = {
                "team":     row_dict.get("TEAM_NAME"),
                "season":   season_str,
                "off_rtg":  row_dict.get("OFF_RATING"),
                "def_rtg":  row_dict.get("DEF_RATING"),
                "net_rtg":  row_dict.get("NET_RATING"),
                "pace":     row_dict.get("PACE"),
                "ts_pct":   row_dict.get("TS_PCT"),
                "ast_pct":  row_dict.get("AST_PCT"),
                "tov_pct":  row_dict.get("TM_TOV_PCT"),
                "oreb_pct": row_dict.get("OREB_PCT"),
                "dreb_pct": row_dict.get("DREB_PCT"),
                "source":   "nba_stats_api",
            }
            _set_cache(cache_key, result)
            return result

    return {"available": False, "note": "Advanced metrics not available (server network restriction)", "team": team_name}


def get_public_betting_percentages(team_name: str = None) -> dict:
    """
    Fetch public betting ticket % and money % from Action Network (free, unofficial).
    High public ticket % + line moving the other way = sharp money on the other side.
    Money % >> ticket % on a team = sharp/professional money driving that side.
    """
    today = date.today().isoformat()
    cache_key = f"public_pct_{today}_{team_name or 'all'}"
    if cached := _cached(cache_key):
        return cached

    try:
        r = requests.get(
            "https://api.actionnetwork.com/web/v1/games",
            params={"sport": "nba", "date": today.replace("-", "")},
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=10,
        )
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        return {"error": f"Action Network unavailable: {e}", "source": "action_network"}

    games = []
    for game in data.get("games", []):
        teams = game.get("teams", [])
        if len(teams) < 2:
            continue

        home = next((t for t in teams if t.get("side") == "home"), teams[0])
        away = next((t for t in teams if t.get("side") == "away"), teams[1])
        home_name = home.get("full_name", home.get("name", ""))
        away_name  = away.get("full_name", away.get("name", ""))

        if team_name:
            tl = team_name.lower()
            if tl not in home_name.lower() and tl not in away_name.lower():
                continue

        def _pct(team: dict, key: str):
            val = team.get(key)
            return round(float(val), 1) if val is not None else None

        home_t = _pct(home, "tickets")
        away_t = _pct(away, "tickets")
        home_m = _pct(home, "money")
        away_m = _pct(away, "money")

        signal = _public_signal(home_name, away_name, home_t, away_t, home_m, away_m)

        games.append({
            "home_team":       home_name,
            "away_team":       away_name,
            "home_ticket_pct": home_t,
            "away_ticket_pct": away_t,
            "home_money_pct":  home_m,
            "away_money_pct":  away_m,
            "signal":          signal,
        })

    result = {
        "date":   today,
        "source": "action_network",
        "games":  games,
        "count":  len(games),
        "note": (
            "ticket_pct = share of individual bets placed. money_pct = share of total dollars. "
            "When money_pct >> ticket_pct on a team, sharp/professional money is on that side. "
            "When 70%+ tickets are on one team but the line moves the other way, fade the public."
        ),
    }
    _set_cache(cache_key, result)
    return result


def _public_signal(home: str, away: str,
                   home_t, away_t, home_m, away_m) -> str:
    if None in (home_t, away_t, home_m, away_m):
        return "Insufficient data for signal."
    signals = []
    if home_t >= 70:
        signals.append(f"Public heavily on {home} ({home_t:.0f}% tickets) — potential sharp fade toward {away}.")
    elif away_t >= 70:
        signals.append(f"Public heavily on {away} ({away_t:.0f}% tickets) — potential sharp fade toward {home}.")
    home_diff = home_m - home_t
    away_diff = away_m - away_t
    if home_diff >= 15:
        signals.append(f"Sharp money on {home}: money% exceeds ticket% by {home_diff:.0f}pts.")
    elif away_diff >= 15:
        signals.append(f"Sharp money on {away}: money% exceeds ticket% by {away_diff:.0f}pts.")
    return " ".join(signals) if signals else "No strong public/sharp divergence detected."


def get_current_odds(team_name: str = None) -> dict:
    data = _odds_get("/sports/basketball_nba/odds/", {
        "regions":    "us",
        "markets":    "h2h,spreads,totals",
        "oddsFormat": "american",
        "dateFormat": "iso",
    })
    if isinstance(data, dict) and "error" in data:
        return data

    games = []
    for event in (data if isinstance(data, list) else []):
        if team_name:
            tl = team_name.lower()
            if tl not in event.get("home_team", "").lower() and tl not in event.get("away_team", "").lower():
                continue
        best  = _extract_best_lines(event)
        home  = event.get("home_team", "")
        away  = event.get("away_team", "")

        # Auto-snapshot for line movement tracking (no extra API call)
        _save_odds_snapshot(home, away, best)

        game = {
            "id":            event.get("id"),
            "commence_time": event.get("commence_time"),
            "home_team":     home,
            "away_team":     away,
            "best_lines":    best,
            "books":         best.get("books", {}),
        }
        games.append(game)

    return {"games": games, "count": len(games)}


def get_book_discrepancies(team_name: str) -> dict:
    """
    Scan all bookmakers for a team's game and flag significant line discrepancies.
    A large spread between books signals a stale line, sharp action, or unpriced news.
    Returns the best available line per market, which book offers it, and the spread vs worst book.
    """
    data = _odds_get("/sports/basketball_nba/odds/", {
        "regions":    "us",
        "markets":    "h2h,spreads,totals",
        "oddsFormat": "american",
        "dateFormat": "iso",
    })
    if isinstance(data, dict) and "error" in data:
        return data

    event = None
    for e in (data if isinstance(data, list) else []):
        tl = team_name.lower()
        if tl in e.get("home_team", "").lower() or tl in e.get("away_team", "").lower():
            event = e
            break

    if not event:
        return {"error": f"No game found today for {team_name}."}

    home = event["home_team"]
    away = event["away_team"]

    # Collect per-book ML prices
    ml_prices: dict = {home: {}, away: {}}
    spread_prices: dict = {home: {}, away: {}}
    total_over: dict = {}

    for bm in event.get("bookmakers", []):
        bm_name = bm.get("title", bm.get("key", "Unknown"))
        for market in bm.get("markets", []):
            key = market["key"]
            for outcome in market.get("outcomes", []):
                name  = outcome["name"]
                price = outcome.get("price")
                point = outcome.get("point")
                if key == "h2h" and name in ml_prices:
                    ml_prices[name][bm_name] = price
                elif key == "spreads" and name in spread_prices and point is not None:
                    spread_prices[name][bm_name] = {"point": point, "price": price}
                elif key == "totals" and name == "Over" and point is not None:
                    total_over[bm_name] = {"point": point, "price": price}

    def _ml_analysis(team: str) -> dict:
        prices = {k: v for k, v in ml_prices[team].items() if v is not None}
        if len(prices) < 2:
            return {}
        best_book  = max(prices, key=lambda b: prices[b])
        worst_book = min(prices, key=lambda b: prices[b])
        best_price  = prices[best_book]
        worst_price = prices[worst_book]
        spread = best_price - worst_price
        return {
            "best_price":  best_price,
            "best_book":   best_book,
            "worst_price": worst_price,
            "worst_book":  worst_book,
            "spread_pts":  spread,
            "all_books":   prices,
            "signal": (
                "LARGE DISCREPANCY — possible stale line or unpriced news" if spread >= 15
                else "Moderate discrepancy — worth noting" if spread >= 8
                else "Books in agreement"
            ),
        }

    def _total_analysis() -> dict:
        if len(total_over) < 2:
            return {}
        points = {k: v["point"] for k, v in total_over.items() if v.get("point") is not None}
        if not points:
            return {}
        high_book = max(points, key=lambda b: points[b])
        low_book  = min(points, key=lambda b: points[b])
        spread = points[high_book] - points[low_book]
        return {
            "highest_total": {"book": high_book, "line": points[high_book]},
            "lowest_total":  {"book": low_book,  "line": points[low_book]},
            "spread_pts":    spread,
            "all_books":     points,
            "signal": (
                "LARGE DISCREPANCY — total line unsettled" if spread >= 2.5
                else "Moderate discrepancy" if spread >= 1.0
                else "Books in agreement"
            ),
        }

    home_ml = _ml_analysis(home)
    away_ml = _ml_analysis(away)
    total   = _total_analysis()

    # Surface top-level signals for easy agent parsing
    signals = []
    for team, analysis in [(home, home_ml), (away, away_ml)]:
        if analysis.get("spread_pts", 0) >= 8:
            signals.append(
                f"{team} ML: {analysis['spread_pts']:+d} pt spread — "
                f"best {analysis['best_price']:+d} at {analysis['best_book']}, "
                f"worst {analysis['worst_price']:+d} at {analysis['worst_book']}. "
                f"{analysis['signal']}."
            )
    if total.get("spread_pts", 0) >= 1.0:
        signals.append(
            f"Total line: {total['spread_pts']:.1f} pt spread — "
            f"high {total['highest_total']['line']} at {total['highest_total']['book']}, "
            f"low {total['lowest_total']['line']} at {total['lowest_total']['book']}. "
            f"{total['signal']}."
        )

    return {
        "matchup":    f"{away} @ {home}",
        "books_seen": len(event.get("bookmakers", [])),
        "signals":    signals if signals else ["No significant discrepancies — books are aligned."],
        "home_ml":    home_ml,
        "away_ml":    away_ml,
        "totals":     total,
    }


def _save_odds_snapshot(home: str, away: str, best_lines: dict):
    """Persist current odds to DB for line movement tracking. Called inside get_current_odds."""
    import json
    try:
        ml      = best_lines.get("moneyline", {})
        spread  = best_lines.get("spread", {})
        total   = best_lines.get("total", {})

        home_spread_info = spread.get(home, {})
        away_spread_info = spread.get(away, {})
        over_info  = total.get("Over", {})
        under_info = total.get("Under", {})

        db.save_odds_snapshot(home, away, {
            "home_ml":            ml.get(home),
            "away_ml":            ml.get(away),
            "home_spread":        home_spread_info.get("point"),
            "home_spread_price":  home_spread_info.get("price"),
            "away_spread_price":  away_spread_info.get("price"),
            "total_line":         over_info.get("point"),
            "over_price":         over_info.get("price"),
            "under_price":        under_info.get("price"),
            "bookmaker_odds":     json.dumps(best_lines.get("books", {})),
        })
    except Exception:
        pass  # Never block odds fetching on a snapshot failure


def get_line_movement(team_name: str) -> dict:
    """
    Show how the moneyline, spread, and total have moved throughout the day
    for a team's game. Reads from locally stored snapshots — no API call.
    Call this during matchup analysis to detect sharp line movement.
    """
    all_bets_teams = [team_name]

    # Try to find the opponent from today's games (cached)
    games_data = _cached(f"today_games_{date.today().isoformat()}")
    opponent = None
    if games_data:
        for g in games_data.get("games", []):
            tl = team_name.lower()
            if tl in g["home_team"].lower():
                opponent = g["visitor_team"]
                break
            if tl in g["visitor_team"].lower():
                opponent = g["home_team"]
                break

    snaps = db.get_odds_snapshots(team_name, opponent or "")
    if not snaps:
        return {
            "team":    team_name,
            "message": "No snapshots yet — line movement data builds up across the day's scans.",
            "snapshots": [],
        }

    # Summarise movement: first vs latest snapshot
    first = snaps[0]
    last  = snaps[-1]

    def _move(key):
        f, l = first.get(key), last.get(key)
        if f is None or l is None:
            return None
        return l - f

    home = last["home_team"]
    away = last["away_team"]

    movement = {
        "home_ml_move":     _move("home_ml"),
        "away_ml_move":     _move("away_ml"),
        "spread_move":      _move("home_spread"),
        "total_move":       _move("total_line"),
    }

    signals = []
    if movement["home_ml_move"] is not None:
        if movement["home_ml_move"] < -10:
            signals.append(f"{home} ML shortened (money coming in on {home}).")
        elif movement["home_ml_move"] > 10:
            signals.append(f"{home} ML lengthened (money moving toward {away}).")
    if movement["spread_move"] is not None:
        if movement["spread_move"] > 0.5:
            signals.append(f"Spread moved {movement['spread_move']:+.1f} pts against {home} (sharp action on {away}).")
        elif movement["spread_move"] < -0.5:
            signals.append(f"Spread moved {movement['spread_move']:+.1f} pts for {home} (sharp action on {home}).")
    if movement["total_move"] is not None:
        if abs(movement["total_move"]) >= 0.5:
            direction = "UP" if movement["total_move"] > 0 else "DOWN"
            signals.append(f"Total moved {direction} {abs(movement['total_move']):.1f} pts.")

    return {
        "matchup":    f"{away} @ {home}",
        "snapshots":  len(snaps),
        "first_seen": first["captured_at"][:16],
        "latest":     last["captured_at"][:16],
        "opening":    {k: first.get(k) for k in ("home_ml", "away_ml", "home_spread", "total_line")},
        "current":    {k: last.get(k)  for k in ("home_ml", "away_ml", "home_spread", "total_line")},
        "movement":   movement,
        "signals":    signals if signals else ["No significant line movement detected yet."],
    }


def _extract_best_lines(event: dict) -> dict:
    h2h: dict    = {}
    spread: dict = {}
    total: dict  = {}
    books: dict  = {}

    for bm in event.get("bookmakers", []):
        bm_title = bm.get("title", bm.get("key", "Unknown"))
        bm_data: dict = {}
        for market in bm.get("markets", []):
            key = market["key"]
            for outcome in market.get("outcomes", []):
                name  = outcome["name"]
                price = outcome.get("price")
                point = outcome.get("point")
                if key == "h2h":
                    if name not in h2h or (price is not None and price > h2h[name]):
                        h2h[name] = price
                    bm_data[f"ml_{name}"] = price
                elif key == "spreads":
                    if name not in spread:
                        spread[name] = {"price": price, "point": point}
                    bm_data[f"spread_{name}"] = {"point": point, "price": price}
                elif key == "totals":
                    if name not in total:
                        total[name] = {"price": price, "point": point}
                    bm_data[f"total_{name}"] = {"point": point, "price": price}
        if bm_data:
            books[bm_title] = bm_data

    return {"moneyline": h2h, "spread": spread, "total": total, "books": books}


def get_advanced_stats(team_name: str, season: int = None) -> dict:
    """Advanced stats (paint pts, fast-break, lead changes) from local Kaggle data."""
    season = season or _current_season()
    result = dl.get_advanced_stats_local(team_name, season)
    if result is None:
        return {"error": f"No advanced stats for {team_name} season {season} in local dataset."}
    return result


def get_historical_odds(team_name: str = None, days_back: int = 3) -> dict:
    data = _odds_get("/sports/basketball_nba/scores/", {
        "daysFrom":   min(days_back, 3),
        "dateFormat": "iso",
    })
    if isinstance(data, dict) and "error" in data:
        return data

    results = []
    for event in (data if isinstance(data, list) else []):
        if team_name:
            tl = team_name.lower()
            if tl not in event.get("home_team", "").lower() and tl not in event.get("away_team", "").lower():
                continue
        scores_raw = event.get("scores") or []
        results.append({
            "commence_time": event.get("commence_time"),
            "home_team":     event.get("home_team"),
            "away_team":     event.get("away_team"),
            "completed":     event.get("completed", False),
            "scores":        {s["name"]: s["score"] for s in scores_raw},
        })
    return {"results": results, "count": len(results)}


def calculate_implied_probability(odds: int) -> dict:
    implied = american_odds_to_implied_probability(odds)
    return {
        "american_odds":          odds,
        "implied_probability":    round(implied, 4),
        "implied_probability_pct": round(implied * 100, 2),
        "decimal_odds":           round(american_odds_to_decimal(odds), 3),
        "note": (
            "Vig-adjusted fair probability is slightly higher. "
            "Compare to your estimated win probability to find edge."
        ),
    }


def get_elo_probability(home_team: str, away_team: str) -> dict:
    """Elo-model win probability and implied spread for a matchup."""
    try:
        import elo
        return elo.win_probability(home_team, away_team)
    except Exception as exc:
        log.warning(f"Elo probability lookup failed: {exc}")
        return {
            "home_team": home_team, "away_team": away_team,
            "home_rating": 1500, "away_rating": 1500,
            "home_prob_pct": 50.0, "away_prob_pct": 50.0,
            "implied_spread": 0.0, "source": "unavailable",
        }


def _split_matchup(matchup: str) -> tuple[str | None, str | None]:
    if " @ " in matchup:
        away, home = matchup.split(" @ ", 1)
        return away.strip(), home.strip()
    if " vs " in matchup:
        away, home = matchup.split(" vs ", 1)
        return away.strip(), home.strip()
    if " vs. " in matchup:
        away, home = matchup.split(" vs. ", 1)
        return away.strip(), home.strip()
    return None, None


def _team_matches_pick(team_name: str, pick: str) -> bool:
    team_tokens = [t for t in re.findall(r"[A-Za-z]+", team_name.lower()) if len(t) > 2]
    pick_l = pick.lower()
    return any(token in pick_l for token in team_tokens)


def _pick_team_from_matchup(matchup: str, pick: str) -> tuple[str | None, str | None]:
    away, home = _split_matchup(matchup)
    if not away or not home:
        return None, None
    if _team_matches_pick(home, pick):
        return home, away
    if _team_matches_pick(away, pick):
        return away, home
    return None, None


def _find_current_market(matchup: str) -> tuple[dict | None, str | None, str | None]:
    away, home = _split_matchup(matchup)
    if not away or not home:
        return None, None, None

    odds_result = get_current_odds(home)
    for game in odds_result.get("games", []):
        if _matchup_matches(matchup, game.get("home_team", ""), game.get("away_team", "")):
            return game, game.get("home_team"), game.get("away_team")
    return None, home, away


def _extract_public_game(public_pct: dict, home_team: str, away_team: str) -> dict:
    if not isinstance(public_pct, dict):
        return {}
    for game in public_pct.get("games", []):
        if _matchup_matches(f"{away_team} @ {home_team}", game.get("home_team", ""), game.get("away_team", "")):
            return game
    return {}


def _safe_rest_days(rest: dict) -> int | None:
    if not isinstance(rest, dict):
        return None
    return rest.get("days_rest")


def _line_snapshot_count(line_move: dict) -> int:
    if not isinstance(line_move, dict):
        return 0
    return int(line_move.get("snapshots", 0) or 0)


def _market_price_bundle(matchup: str, pick: str, bet_type: str, submitted_odds: int) -> dict:
    game, home_team, away_team = _find_current_market(matchup)
    if not game:
        return {
            "matchup": matchup,
            "current_odds_found": False,
            "selected_side_found": False,
            "opposite_side_found": False,
            "home_team": home_team,
            "away_team": away_team,
            "submitted_odds": submitted_odds,
        }

    best = game.get("best_lines", {})
    bundle = {
        "matchup": matchup,
        "current_odds_found": True,
        "selected_side_found": False,
        "opposite_side_found": False,
        "home_team": home_team,
        "away_team": away_team,
        "submitted_odds": submitted_odds,
    }

    if bet_type == "moneyline":
        selected_team, other_team = _pick_team_from_matchup(matchup, pick)
        if not selected_team:
            return bundle
        selected_best = best.get("moneyline", {}).get(selected_team)
        other_best = best.get("moneyline", {}).get(other_team)
        bundle.update({
            "selected_team": selected_team,
            "other_team": other_team,
            "selected_best_odds": selected_best,
            "opposite_best_odds": other_best,
            "selected_side_found": selected_best is not None,
            "opposite_side_found": other_best is not None,
            "market_implied_prob": american_odds_to_implied_probability(submitted_odds),
        })
        if other_best is not None:
            submitted_side_fair, _ = no_vig_probabilities_from_odds(submitted_odds, other_best)
            bundle["fair_prob_no_vig"] = submitted_side_fair
        return bundle

    if bet_type == "spread":
        selected_team, other_team = _pick_team_from_matchup(matchup, pick)
        if not selected_team:
            return bundle
        selected_info = best.get("spread", {}).get(selected_team, {})
        other_info = best.get("spread", {}).get(other_team, {})
        selected_best = selected_info.get("price")
        other_best = other_info.get("price")
        bundle.update({
            "selected_team": selected_team,
            "other_team": other_team,
            "selected_best_odds": selected_best,
            "opposite_best_odds": other_best,
            "selected_side_found": selected_best is not None,
            "opposite_side_found": other_best is not None,
            "market_implied_prob": american_odds_to_implied_probability(submitted_odds),
        })
        if other_best is not None:
            submitted_side_fair, _ = no_vig_probabilities_from_odds(submitted_odds, other_best)
            bundle["fair_prob_no_vig"] = submitted_side_fair
        return bundle

    if bet_type == "total":
        selected_side = "Over" if "over" in pick.lower() else "Under" if "under" in pick.lower() else None
        other_side = "Under" if selected_side == "Over" else "Over" if selected_side == "Under" else None
        selected_info = best.get("total", {}).get(selected_side, {}) if selected_side else {}
        other_info = best.get("total", {}).get(other_side, {}) if other_side else {}
        selected_best = selected_info.get("price")
        other_best = other_info.get("price")
        bundle.update({
            "selected_team": selected_side,
            "other_team": other_side,
            "selected_best_odds": selected_best,
            "opposite_best_odds": other_best,
            "selected_side_found": selected_best is not None,
            "opposite_side_found": other_best is not None,
            "market_implied_prob": american_odds_to_implied_probability(submitted_odds),
        })
        if other_best is not None:
            submitted_side_fair, _ = no_vig_probabilities_from_odds(submitted_odds, other_best)
            bundle["fair_prob_no_vig"] = submitted_side_fair
        return bundle

    return bundle


def evaluate_recommendation(
    matchup: str,
    pick: str,
    bet_type: str,
    odds: int,
    confidence: str = "Medium",
    reasoning: str = "",
    llm_edge_pct: float = None,
) -> dict:
    bankroll = db.get_balance()
    market_supported = bet_type in SUPPORTED_PROBABILITY_MARKETS
    price_bundle = _market_price_bundle(matchup, pick, bet_type, odds)

    away_team, home_team = _split_matchup(matchup)
    home_team = price_bundle.get("home_team") or home_team
    away_team = price_bundle.get("away_team") or away_team

    rest_home = get_rest_days(home_team) if home_team else {}
    rest_away = get_rest_days(away_team) if away_team else {}
    injury_home = get_injury_report(home_team) if home_team else {}
    injury_away = get_injury_report(away_team) if away_team else {}
    roster_home = get_current_roster(home_team) if home_team else {}
    roster_away = get_current_roster(away_team) if away_team else {}
    public_pct = get_public_betting_percentages(home_team) if home_team else {}
    line_home = get_line_movement(home_team) if home_team else {}
    line_away = get_line_movement(away_team) if away_team else {}
    elo_prob = get_elo_probability(home_team, away_team) if home_team and away_team else {"source": "unavailable"}

    model_prob = None
    model_detail = {}
    conflicting_signals = False

    if market_supported and home_team and away_team:
        selected_team = price_bundle.get("selected_team")
        other_team = price_bundle.get("other_team")
        selected_is_home = selected_team == home_team
        base_prob = (elo_prob.get("home_prob_pct", 50.0) / 100.0) if selected_is_home else (elo_prob.get("away_prob_pct", 50.0) / 100.0)

        public_game = _extract_public_game(public_pct, home_team, away_team)
        home_ticket = public_game.get("home_ticket_pct")
        away_ticket = public_game.get("away_ticket_pct")
        home_money = public_game.get("home_money_pct")
        away_money = public_game.get("away_money_pct")

        selected_ticket = home_ticket if selected_is_home else away_ticket
        selected_money = home_money if selected_is_home else away_money
        opponent_ticket = away_ticket if selected_is_home else home_ticket
        opponent_money = away_money if selected_is_home else home_money

        selected_rest = _safe_rest_days(rest_home if selected_is_home else rest_away)
        opponent_rest = _safe_rest_days(rest_away if selected_is_home else rest_home)
        selected_injury_weight = weighted_injury_count(injury_home if selected_is_home else injury_away)
        opponent_injury_weight = weighted_injury_count(injury_away if selected_is_home else injury_home)
        selected_ml_move = line_home.get("movement", {}).get("home_ml_move") if selected_is_home else line_away.get("movement", {}).get("away_ml_move")
        opponent_ml_move = line_away.get("movement", {}).get("away_ml_move") if selected_is_home else line_home.get("movement", {}).get("home_ml_move")

        model_detail = compute_moneyline_probability(
            base_probability=base_prob,
            selected_rest_days=selected_rest,
            opponent_rest_days=opponent_rest,
            selected_weighted_injuries=selected_injury_weight,
            opponent_weighted_injuries=opponent_injury_weight,
            selected_ticket_pct=selected_ticket,
            selected_money_pct=selected_money,
            opponent_ticket_pct=opponent_ticket,
            opponent_money_pct=opponent_money,
            selected_ml_move=selected_ml_move,
            opponent_ml_move=opponent_ml_move,
        )
        model_prob = model_detail["adjusted_probability"]

        line_adj = model_detail["adjustments"].get("line_movement", 0.0)
        public_adj = model_detail["adjustments"].get("public_money", 0.0)
        conflicting_signals = (line_adj < 0 and public_adj > 0) or (line_adj > 0 and public_adj < 0)

    submitted_vs_market_delta = None
    if price_bundle.get("selected_best_odds") is not None:
        submitted_vs_market_delta = abs(int(price_bundle["selected_best_odds"]) - int(odds))

    quality = compute_data_quality_score(
        market_supported=market_supported,
        current_odds_found=price_bundle.get("current_odds_found", False),
        selected_side_found=price_bundle.get("selected_side_found", False),
        opposite_side_found=price_bundle.get("opposite_side_found", False),
        elo_source=elo_prob.get("source"),
        rest_data_complete=_safe_rest_days(rest_home) is not None and _safe_rest_days(rest_away) is not None,
        injury_data_complete="error" not in injury_home and "error" not in injury_away,
        roster_data_complete=roster_home.get("count", 0) > 0 and roster_away.get("count", 0) > 0,
        public_data_complete=bool(_extract_public_game(public_pct, home_team, away_team)),
        line_snapshots=min(_line_snapshot_count(line_home), _line_snapshot_count(line_away)),
        submitted_vs_market_delta=submitted_vs_market_delta,
        conflicting_signals=conflicting_signals,
    )

    snapshot = recommendation_snapshot(
        odds=odds,
        bankroll=bankroll,
        market_supported=market_supported,
        model_probability=model_prob,
        market_implied_probability=price_bundle.get("market_implied_prob"),
        fair_probability_no_vig=price_bundle.get("fair_prob_no_vig"),
        data_quality_score=quality["score"],
        llm_edge_pct=llm_edge_pct,
    )

    if not market_supported and llm_edge_pct is not None:
        snapshot["edge_pct"] = llm_edge_pct

    signal_notes = quality["penalties"][:]
    if not market_supported:
        signal_notes.append("Market downgraded because no deterministic probability model is implemented for this market.")

    return {
        "matchup": matchup,
        "pick": pick,
        "bet_type": bet_type,
        "odds": odds,
        "confidence": confidence,
        "reasoning": reasoning,
        "market_supported": market_supported,
        "home_team": home_team,
        "away_team": away_team,
        "model_detail": model_detail,
        "signal_notes": signal_notes,
        **snapshot,
    }


def _skip_reason_from_evaluation(evaluation: dict, open_count: int) -> str:
    if open_count >= 5 and evaluation.get("decision") == "BET":
        return "slots_full"
    if not evaluation.get("market_supported"):
        return "unsupported_market_model"
    if (evaluation.get("data_quality_score") or 0) < 45:
        return "low_confidence"
    if evaluation.get("decision") == "LEAN":
        return "edge_below_threshold"
    return "other"


def submit_recommendation(
    matchup: str,
    pick: str,
    bet_type: str,
    odds: int,
    confidence: str,
    reasoning: str,
    game_date: str = None,
    replaces_bet_id: int = None,
    edge_type: str = None,
    llm_edge_pct: float = None,
) -> dict:
    evaluation = evaluate_recommendation(
        matchup=matchup,
        pick=pick,
        bet_type=bet_type,
        odds=odds,
        confidence=confidence,
        reasoning=reasoning,
        llm_edge_pct=llm_edge_pct,
    )
    open_count = len(db.get_open_bets())

    if evaluation.get("decision") == "BET" and open_count < 5:
        return _place_paper_bet_from_evaluation(
            evaluation=evaluation,
            game_date=game_date,
            replaces_bet_id=replaces_bet_id,
            edge_type=edge_type,
        )

    skip_reason = _skip_reason_from_evaluation(evaluation, open_count)
    candidate = log_candidate_bet(
        matchup=matchup,
        pick=pick,
        bet_type=bet_type,
        odds=odds,
        edge_pct=evaluation.get("edge_pct", llm_edge_pct or 0.0) or 0.0,
        confidence=confidence,
        skip_reason=skip_reason,
        reasoning=reasoning,
        model_prob=evaluation.get("model_prob"),
        market_implied_prob=evaluation.get("market_implied_prob"),
        fair_prob_no_vig=evaluation.get("fair_prob_no_vig"),
        ev=evaluation.get("ev"),
        stake_pct=evaluation.get("stake_pct"),
        stake_amount=evaluation.get("stake_amount"),
        data_quality_score=evaluation.get("data_quality_score"),
        decision=evaluation.get("decision"),
        llm_edge_pct=llm_edge_pct,
    )
    return {**candidate, **evaluation}


def _kelly_stake(balance: float, edge_pct: float, odds: int) -> tuple[float, float]:
    """
    Half-Kelly stake sizing. Returns (dollar_stake, kelly_fraction).
    edge_pct: percentage edge (e.g. 8.0 for 8%).
    odds: American odds (e.g. -110, +150).
    """
    implied = american_odds_to_implied_probability(odds)
    win_probability = min(max(implied + edge_pct / 100, 0.0), 1.0)
    _, fraction = kelly_stake(
        bankroll=balance,
        win_probability=win_probability,
        odds=odds,
        fractional_kelly=0.5,
        max_fraction=0.12,
    )

    # Preserve current live behavior: accepted bets always stake at least 1% of bankroll.
    fraction = max(fraction, 0.01)

    return round(balance * fraction, 2), round(fraction * 100, 2)


def _place_paper_bet_from_evaluation(
    evaluation: dict,
    game_date: str = None,
    replaces_bet_id: int = None,
    edge_type: str = None,
) -> dict:
    balance    = db.get_balance()
    open_bets  = db.get_open_bets()
    matchup = evaluation["matchup"]
    pick = evaluation["pick"]
    bet_type = evaluation["bet_type"]
    odds = evaluation["odds"]
    confidence = evaluation.get("confidence", "Medium")
    reasoning = evaluation.get("reasoning", "")
    edge = evaluation.get("edge_pct")
    if edge is None:
        edge = evaluation.get("llm_edge_pct", 0.0) or 0.0

    if len(open_bets) >= 5:
        return {"error": "Max 5 open bets already active. Wait for resolution.", "open_count": 5}

    if evaluation.get("decision") != "BET":
        return {"error": f"Recommendation decision is {evaluation.get('decision')}, not BET.", "decision": evaluation.get("decision")}

    if confidence not in ("High", "Medium"):
        confidence = "High" if edge >= 10.0 else "Medium"

    stake = round(evaluation.get("stake_amount", 0.0) or 0.0, 2)
    kelly_pct = round(evaluation.get("stake_pct", 0.0) or 0.0, 2)
    stake = max(stake, 1.0)
    if kelly_pct <= 0:
        stake, kelly_pct = _kelly_stake(balance, edge, odds)
        stake = max(stake, 1.0)

    if odds > 0:
        potential_payout = round(stake * (odds / 100) + stake, 2)
    else:
        potential_payout = round(stake * (100 / abs(odds)) + stake, 2)
    model_probability = evaluation.get("model_prob")
    ev_dollars = round(evaluation.get("ev", 0.0) or 0.0, 2)
    ev_pct = round((ev_dollars / stake) * 100, 2) if stake else 0.0

    bet = {
        "game_date":       game_date or date.today().isoformat(),
        "matchup":         matchup,
        "pick":            pick,
        "bet_type":        bet_type,
        "odds":            odds,
        "stake":           stake,
        "potential_payout": potential_payout,
        "confidence":      confidence,
        "edge":            edge,
        "reasoning":       reasoning,
        "model_prob":      model_probability,
        "market_implied_prob": evaluation.get("market_implied_prob"),
        "fair_prob_no_vig": evaluation.get("fair_prob_no_vig"),
        "edge_pct":        edge,
        "ev":              ev_dollars,
        "stake_pct":       kelly_pct,
        "stake_amount":    stake,
        "data_quality_score": evaluation.get("data_quality_score"),
        "decision":        evaluation.get("decision"),
        "llm_edge_pct":    evaluation.get("llm_edge_pct"),
    }

    bet_id = db.insert_bet(bet)
    db.update_balance(balance - stake)

    if edge_type:
        import sqlite3 as _sqlite3
        conn = _sqlite3.connect(db.DB_PATH)
        conn.execute("UPDATE bets SET edge_type=? WHERE id=?", (edge_type, bet_id))
        conn.commit()
        conn.close()

    if replaces_bet_id:
        db.link_replacement(bet_id, replaces_bet_id)

    # ── Betfair live bet (only when BETFAIR_LIVE_MODE=true) ───────────────────
    betfair_result = {}
    if bet_type == "moneyline":
        try:
            import betfair as bf
            if bf.is_live():
                betfair_result = bf.place_live_bet(
                    matchup=matchup,
                    pick=pick,
                    american_odds=odds,
                    stake=stake,
                )
                if "bet_id" in betfair_result:
                    db.set_betfair_ids(bet_id, betfair_result["bet_id"], betfair_result["market_id"])
        except Exception as exc:
            betfair_result = {"error": str(exc)}

    bf_msg = ""
    if betfair_result.get("bet_id"):
        bf_msg = f"\nBetfair: ✅ Bet placed (ID {betfair_result['bet_id']})"
    elif betfair_result.get("error"):
        bf_msg = f"\nBetfair: ⚠️ {betfair_result['error']}"
    elif betfair_result.get("skipped"):
        bf_msg = "\nBetfair: Paper mode only"

    _send_notification(
        title=f"BetIQ Bet Placed",
        message=(
            f"{pick} ({bet_type})\n{matchup}\nOdds: {odds:+d} | Decision: {evaluation.get('decision')} | "
            f"Edge: {edge:.1f}% | EV: ${ev_dollars:+.2f} | Confidence: {confidence}\n"
            f"Kelly: {kelly_pct:.1f}% | Stake: ${stake:.2f} | DQ: {evaluation.get('data_quality_score', 0):.0f} | "
            f"Balance: ${round(balance - stake, 2):.2f}{bf_msg}"
        ),
    )

    # Validate player names in reasoning against actual rosters
    try:
        parts = matchup.replace(" @ ", " vs ").split(" vs ")
        if len(parts) == 2:
            validate_reasoning_players(reasoning, parts[1].strip(), parts[0].strip())
    except Exception:
        pass

    return {
        "success":            True,
        "bet_id":             bet_id,
        "matchup":            matchup,
        "pick":               pick,
        "odds":               odds,
        "stake":              stake,
        "kelly_fraction_pct": kelly_pct,
        "potential_payout":   potential_payout,
        "confidence":         confidence,
        "edge_pct":           edge,
        "model_prob":         model_probability,
        "market_implied_prob": evaluation.get("market_implied_prob"),
        "fair_prob_no_vig":   evaluation.get("fair_prob_no_vig"),
        "expected_value":     ev_dollars,
        "expected_value_pct": ev_pct,
        "stake_pct":          kelly_pct,
        "stake_amount":       stake,
        "data_quality_score": evaluation.get("data_quality_score"),
        "decision":           evaluation.get("decision"),
        "signal_notes":       evaluation.get("signal_notes", []),
        "new_balance":        round(balance - stake, 2),
        "betfair":            betfair_result,
        "message":            f"Bet #{bet_id} placed: ${stake:.2f} ({kelly_pct:.1f}% Kelly) on {pick} at {odds:+d}{bf_msg}",
    }


def place_paper_bet(
    matchup: str,
    pick: str,
    bet_type: str,
    odds: int,
    confidence: str,
    edge: float,
    reasoning: str,
    game_date: str = None,
    replaces_bet_id: int = None,
    edge_type: str = None,
) -> dict:
    evaluation = evaluate_recommendation(
        matchup=matchup,
        pick=pick,
        bet_type=bet_type,
        odds=odds,
        confidence=confidence,
        reasoning=reasoning,
        llm_edge_pct=edge,
    )
    if evaluation.get("decision") != "BET":
        candidate = log_candidate_bet(
            matchup=matchup,
            pick=pick,
            bet_type=bet_type,
            odds=odds,
            edge_pct=evaluation.get("edge_pct", edge) or edge,
            confidence=confidence,
            skip_reason=_skip_reason_from_evaluation(evaluation, len(db.get_open_bets())),
            reasoning=reasoning,
            model_prob=evaluation.get("model_prob"),
            market_implied_prob=evaluation.get("market_implied_prob"),
            fair_prob_no_vig=evaluation.get("fair_prob_no_vig"),
            ev=evaluation.get("ev"),
            stake_pct=evaluation.get("stake_pct"),
            stake_amount=evaluation.get("stake_amount"),
            data_quality_score=evaluation.get("data_quality_score"),
            decision=evaluation.get("decision"),
            llm_edge_pct=edge,
        )
        return {
            "error": f"Recommendation downgraded to {evaluation.get('decision')} by deterministic evaluation.",
            **candidate,
            **evaluation,
        }
    return _place_paper_bet_from_evaluation(
        evaluation=evaluation,
        game_date=game_date,
        replaces_bet_id=replaces_bet_id,
        edge_type=edge_type,
    )


def cancel_bet(bet_id: int, reason: str = "") -> dict:
    """
    Cancel an open bet and refund the stake to the bankroll.
    Use this to swap a weaker bet for a stronger one found later.
    """
    bet = db.cancel_bet(bet_id, reason)
    if not bet:
        return {"error": f"Bet #{bet_id} not found or already resolved."}

    # Refund stake
    db.update_balance(db.get_balance() + bet["stake"])

    _send_notification(
        title=f"BetIQ Bet Cancelled",
        message=f"Bet #{bet_id} cancelled: {bet['pick']}\nStake refunded: ${bet['stake']:.2f}\nReason: {reason}",
    )

    return {
        "success":         True,
        "cancelled_bet_id": bet_id,
        "pick":            bet["pick"],
        "stake_refunded":  bet["stake"],
        "new_balance":     round(db.get_balance(), 2),
        "reason":          reason,
    }


def get_bankroll() -> dict:
    balance   = db.get_balance()
    open_bets = db.get_open_bets()
    staked    = sum(b["stake"] for b in open_bets)
    return {
        "current_balance":          round(balance, 2),
        "starting_balance":         1000.0,
        "total_staked_open_bets":   round(staked, 2),
        "total_return_pct":         round((balance - 1000.0) / 1000.0 * 100, 2),
        "open_bets":                open_bets,
        "open_bets_count":          len(open_bets),
        "slots_remaining":          5 - len(open_bets),
    }


def get_bet_history() -> dict:
    all_bets = db.get_all_bets()
    resolved = [b for b in all_bets if b["status"] in ("won", "lost", "push")]
    open_bets = [b for b in all_bets if b["status"] == "open"]

    wins   = [b for b in resolved if b["status"] == "won"]
    losses = [b for b in resolved if b["status"] == "lost"]
    total_pnl  = sum(b["pnl"] for b in resolved)
    win_rate   = len(wins) / max(len(resolved), 1)

    high = [b for b in resolved if b["confidence"] == "High"]
    med  = [b for b in resolved if b["confidence"] == "Medium"]

    recent = sorted(resolved, key=lambda x: x.get("resolved_at") or "", reverse=True)[:10]
    recent_wins = sum(1 for b in recent if b["status"] == "won")

    # CLV stats: bets where closing_odds was captured
    clv_bets = [b for b in all_bets if b.get("clv") is not None]
    avg_clv  = sum(b["clv"] for b in clv_bets) / max(len(clv_bets), 1)
    pos_clv  = sum(1 for b in clv_bets if b["clv"] > 0)
    scored_bets = [b for b in all_bets if b.get("data_quality_score") is not None]
    avg_data_quality = sum(b["data_quality_score"] for b in scored_bets) / max(len(scored_bets), 1)

    return {
        "total_bets":              len(all_bets),
        "resolved":                len(resolved),
        "open":                    len(open_bets),
        "wins":                    len(wins),
        "losses":                  len(losses),
        "win_rate_pct":            round(win_rate * 100, 1),
        "total_pnl":               round(total_pnl, 2),
        "recent_10_win_rate_pct":  round(recent_wins / max(len(recent), 1) * 100, 1),
        "high_confidence_record":  f"{sum(1 for b in high if b['status']=='won')}-{sum(1 for b in high if b['status']=='lost')}",
        "medium_confidence_record": f"{sum(1 for b in med if b['status']=='won')}-{sum(1 for b in med if b['status']=='lost')}",
        "clv_tracked_bets":        len(clv_bets),
        "avg_clv_pct":             round(avg_clv * 100, 2),
        "positive_clv_rate_pct":   round(pos_clv / max(len(clv_bets), 1) * 100, 1),
        "avg_data_quality_score":  round(avg_data_quality, 1) if scored_bets else None,
        "clv_note": (
            "Positive avg CLV = consistently beating the closing line = long-term edge. "
            "Negative CLV = entering too late or fading sharp money. "
            f"You have beaten the closing line on {pos_clv}/{len(clv_bets)} tracked bets."
        ),
        "all_bets":                all_bets,
        "weekly":                  db.get_weekly_pnl(),
        "performance_note": (
            "Evaluate recent win rate vs. long-term win rate. "
            "If recent performance is declining, reduce confidence levels."
        ),
    }


def validate_reasoning_players(reasoning: str, home_team: str, away_team: str) -> dict:
    """
    Cross-check player names mentioned in Claude's reasoning against actual rosters.
    Returns a list of hallucinated players (names mentioned but not on current roster).
    Called automatically after place_paper_bet or log_candidate_bet.
    """
    if not reasoning:
        return {"hallucinations": [], "checked": False}

    home_roster = get_current_roster(home_team)
    away_roster = get_current_roster(away_team)

    all_players = set()
    for roster in (home_roster, away_roster):
        for p in roster.get("players", []):
            name = p["name"].lower()
            parts = name.split()
            all_players.add(name)
            if parts:
                all_players.add(parts[-1])  # last name alone
                if len(parts) > 1:
                    all_players.add(parts[0])  # first name alone

    reasoning_lower = reasoning.lower()
    words = set(reasoning_lower.split())

    # Common NBA last names that appear in reasoning but aren't on either team
    # We check multi-word names (first + last) not found in roster
    import re as _re
    # Extract capitalized word pairs (likely player names)
    name_pattern = _re.findall(r'\b([A-Z][a-z]+(?:\s[A-Z][a-z]+)+)\b', reasoning)
    hallucinated = []
    for name in name_pattern:
        nl = name.lower()
        last = nl.split()[-1]
        # Check if neither the full name nor last name is in any roster
        if nl not in all_players and last not in all_players:
            # Filter out common non-player proper nouns
            skip = {"nba", "golden", "state", "los", "angeles", "san", "new", "york",
                    "oklahoma", "city", "new", "orleans", "half", "kelly", "monday",
                    "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"}
            if last not in skip and len(last) > 3:
                hallucinated.append(name)

    if hallucinated:
        warning = f"HALLUCINATION ALERT: Claude mentioned {hallucinated} in reasoning but they are NOT on {home_team} or {away_team} current rosters. Review this bet."
        db.add_agent_note("hallucination_warning", warning)
        log.warning(warning)

    return {
        "hallucinations": hallucinated,
        "checked": True,
        "roster_size_home": len(home_roster.get("players", [])),
        "roster_size_away": len(away_roster.get("players", [])),
    }


def save_note(note_type: str, content: str) -> dict:
    """
    Save a lesson, pattern, or hypothesis to persistent memory.
    Call this after every scan to record what you predicted, what happened, and what to adjust.
    note_type options: 'lesson', 'pattern', 'hypothesis', 'model_update'
    """
    db.add_agent_note(note_type, content)
    return {"saved": True, "note_type": note_type, "content": content}


def get_notes(note_type: str = None, limit: int = 30) -> dict:
    """
    Retrieve past lessons and patterns saved from previous scans.
    Always call this at the start of a scan to recall what you've learned.
    """
    notes = db.get_agent_notes(note_type, limit)
    return {
        "count": len(notes),
        "notes": notes,
        "instruction": (
            "Review these notes before making any picks. "
            "Apply lessons learned and avoid repeating past mistakes."
        ),
    }


def log_candidate_bet(
    matchup: str,
    pick: str,
    bet_type: str,
    odds: int,
    edge_pct: float,
    confidence: str,
    skip_reason: str,
    reasoning: str = "",
    model_prob: float = None,
    market_implied_prob: float = None,
    fair_prob_no_vig: float = None,
    ev: float = None,
    stake_pct: float = None,
    stake_amount: float = None,
    data_quality_score: float = None,
    decision: str = None,
    llm_edge_pct: float = None,
) -> dict:
    """
    Log a near-miss bet — a pick you analysed and liked but decided NOT to place.
    Use this whenever you find a potential edge but can't bet it (slots full,
    edge below threshold, uncertainty too high, sharp money opposing, etc.).
    These are shown in the UI as 'Runner-Up Bets' for the user to review.

    skip_reason options (use the most accurate):
      'edge_below_threshold' — edge found but < 5%
      'slots_full'           — all 5 slots already occupied
      'sharp_money_opposing' — public/sharp signals contradict the pick
      'injury_uncertainty'   — key injury status unknown
      'line_moved_against'   — line moved unfavourably since opening
      'low_confidence'       — data too thin or conflicting to bet confidently
      'unsupported_market_model' — no deterministic probability model exists for the selected market
      'other'                — explain in reasoning
    """
    game_date = date.today().isoformat()
    db.save_candidate_bet(
        game_date=game_date,
        matchup=matchup,
        pick=pick,
        bet_type=bet_type,
        odds=odds,
        edge_pct=edge_pct,
        confidence=confidence,
        skip_reason=skip_reason,
        reasoning=reasoning,
        model_prob=model_prob,
        market_implied_prob=market_implied_prob,
        fair_prob_no_vig=fair_prob_no_vig,
        ev=ev,
        stake_pct=stake_pct,
        stake_amount=stake_amount,
        data_quality_score=data_quality_score,
        decision=decision,
        llm_edge_pct=llm_edge_pct,
    )
    return {
        "logged": True,
        "pick": pick,
        "skip_reason": skip_reason,
        "decision": decision,
        "data_quality_score": data_quality_score,
    }


def snapshot_closing_odds() -> dict:
    """
    For each open bet, look up current market odds and store them as the closing line.
    Call this during every scan — the pre-game odds captured closest to tip-off are the
    most accurate CLV benchmark.

    CLV = closing_implied_prob - our_implied_prob
    Positive CLV means we got better odds than the market closed at (good long-term signal).
    """
    open_bets = db.get_open_bets()
    if not open_bets:
        return {"message": "No open bets.", "updated": []}

    # Fetch all current live odds in one call
    odds_data = _odds_get("/sports/basketball_nba/odds/", {
        "regions":    "us",
        "markets":    "h2h,spreads,totals",
        "oddsFormat": "american",
        "dateFormat": "iso",
    })
    if isinstance(odds_data, dict) and "error" in odds_data:
        return {"message": "Odds API unavailable — CLV not captured.", "error": odds_data["error"]}

    events = odds_data if isinstance(odds_data, list) else []
    updated = []

    for bet in open_bets:
        if bet.get("closing_odds") is not None:
            continue  # already captured

        # Find the matching event
        closing_odds_val = None
        for event in events:
            home = event.get("home_team", "")
            away = event.get("away_team", "")
            if not _matchup_matches(bet["matchup"], home, away):
                continue

            best = _extract_best_lines(event)
            pick_lower = bet["pick"].lower()
            bet_type   = bet.get("bet_type", "moneyline").lower()

            if bet_type == "moneyline":
                for team_name, ml_odds in best["moneyline"].items():
                    if any(w in team_name.lower() for w in pick_lower.split()):
                        closing_odds_val = ml_odds
                        break
            elif bet_type == "spread":
                for team_name, info in best["spread"].items():
                    if any(w in team_name.lower() for w in pick_lower.split()):
                        closing_odds_val = info.get("price")
                        break
            elif bet_type == "total":
                side = "Over" if "over" in pick_lower else "Under"
                info = best["total"].get(side)
                if info:
                    closing_odds_val = info.get("price")
            break

        if closing_odds_val is None:
            continue

        # Calculate CLV: closing_implied - our_implied (positive = we beat the line)
        def _imp(o):
            return abs(o) / (abs(o) + 100) if o < 0 else 100 / (o + 100)

        clv = _imp(closing_odds_val) - _imp(bet["odds"])
        db.update_bet_clv(bet["id"], closing_odds_val, clv)
        updated.append({
            "bet_id":        bet["id"],
            "pick":          bet["pick"],
            "our_odds":      bet["odds"],
            "closing_odds":  closing_odds_val,
            "clv_pct":       round(clv * 100, 2),
            "clv_signal":    "positive (beat the line)" if clv > 0 else "negative (line moved against us)",
        })

    return {
        "updated_count": len(updated),
        "updated":       updated,
        "note":          "Positive CLV means you got better odds than the closing market price.",
    }


def resolve_bets() -> dict:
    open_bets = db.get_open_bets()
    if not open_bets:
        return {"message": "No open bets to resolve.", "resolved": []}

    resolved_list = []

    # Try The Odds API scores first
    scores_data = _odds_get("/sports/basketball_nba/scores/", {"daysFrom": 3, "dateFormat": "iso"})
    if not (isinstance(scores_data, dict) and "error" in scores_data):
        completed = [e for e in (scores_data if isinstance(scores_data, list) else []) if e.get("completed")]
        for bet in open_bets:
            for game in completed:
                home  = game.get("home_team", "")
                away  = game.get("away_team", "")
                if not _matchup_matches(bet["matchup"], home, away):
                    continue
                raw_scores = {s["name"]: int(float(s.get("score", 0))) for s in (game.get("scores") or [])}
                home_score = raw_scores.get(home, 0)
                away_score = raw_scores.get(away, 0)
                _finalize_bet(bet, home, away, home_score, away_score, resolved_list)
                break

    # Fallback: Balldontlie for bets not yet resolved
    already = {r["bet_id"] for r in resolved_list}
    remaining = [b for b in open_bets if b["id"] not in already]
    if remaining:
        _resolve_via_balldontlie(remaining, resolved_list)

    # Trigger daily report when the last open bet for a game_date is now settled
    resolved_dates = {r["game_date"] for r in resolved_list if r.get("game_date")}
    for gd in resolved_dates:
        all_for_date = [b for b in db.get_all_bets() if b["game_date"] == gd]
        if all_for_date and all(b["status"] != "open" for b in all_for_date):
            try:
                import reporter as _reporter
                generated = _reporter.maybe_generate_report(gd)
                if generated:
                    _send_notification(
                        title="BetIQ Daily Report Ready",
                        message=f"Daily report for {gd} is ready. View it in the Reports tab.",
                    )
            except Exception:
                pass

    return {
        "resolved_count": len(resolved_list),
        "resolved":       resolved_list,
        "still_open":     len(open_bets) - len(resolved_list),
    }


def _matchup_matches(matchup: str, home: str, away: str) -> bool:
    parts = re.split(r" vs\.? | @ ", matchup.lower())
    if len(parts) < 2:
        return False
    t1, t2 = parts[0].strip(), parts[1].strip()
    t1_kw = t1.split()[-1]  # last word (city name or team name)
    t2_kw = t2.split()[-1]
    hl, al = home.lower(), away.lower()
    return (t1_kw in hl or t1_kw in al) and (t2_kw in hl or t2_kw in al)


def _finalize_bet(bet: dict, home: str, away: str, home_score: int, away_score: int, resolved_list: list):
    won = _evaluate_pick(bet, home, away, home_score, away_score)
    if won is None:
        return
    if won:
        odds = bet["odds"]
        profit = round(bet["stake"] * (odds / 100) if odds > 0 else bet["stake"] * (100 / abs(odds)), 2)
        pnl, status = profit, "won"
        db.update_balance(db.get_balance() + bet["stake"] + profit)
        _send_notification(
            title=f"BetIQ WON +${profit:.2f}",
            message=f"{bet['pick']}\n{home} {home_score} - {away} {away_score}\nStake: ${bet['stake']:.2f} | Profit: +${profit:.2f}",
            tags="white_check_mark,moneybag",
        )
    else:
        pnl, status = -bet["stake"], "lost"
        _send_notification(
            title=f"BetIQ LOST -${bet['stake']:.2f}",
            message=f"{bet['pick']}\n{home} {home_score} - {away} {away_score}\nStake: ${bet['stake']:.2f}",
            tags="x,chart_with_downwards_trend",
        )

    score_str = f"{home} {home_score} - {away} {away_score}"
    db.resolve_bet(bet["id"], status, pnl)

    # Update Elo ratings after game resolution
    try:
        import elo as _elo
        _elo.process_game_result(home, away, home_score, away_score, _current_season())
    except Exception as exc:
        log.warning(f"Elo update failed for {home} vs {away}: {exc}")

    # Generate post-game analysis report (Haiku, single call)
    try:
        from agent import generate_bet_report
        resolved_bet = {**bet, "status": status, "pnl": pnl}
        report = generate_bet_report(resolved_bet, score_str)
        db.set_bet_report(bet["id"], report)
    except Exception as exc:
        log.warning(f"Bet report generation failed for bet {bet['id']}: {exc}")

    resolved_list.append({
        "bet_id":    bet["id"],
        "game_date": bet.get("game_date", ""),
        "matchup":   bet["matchup"],
        "pick":      bet["pick"],
        "status":    status,
        "pnl":       pnl,
        "score":     score_str,
    })


def _resolve_via_balldontlie(bets: list, resolved_list: list):
    from datetime import timedelta
    today = date.today()
    for bet in bets:
        gd = bet.get("game_date", "")
        if not gd:
            continue
        try:
            # Allow tomorrow's date — late games (9 PM EST) tip off after midnight UTC
            if date.fromisoformat(gd) > today + timedelta(days=1):
                continue
        except ValueError:
            continue

        # Query both the stored game_date AND the next day — late EST games
        # (9:30 PM+) tip off after midnight UTC and BDL stores them as tomorrow.
        next_day = (date.fromisoformat(gd) + timedelta(days=1)).isoformat()
        found = False
        for query_date in (gd, next_day):
            data = _bdl_get("/games", {"dates[]": query_date, "per_page": 50})
            if "error" in data:
                continue
            for g in data.get("data", []):
                if g.get("status") != "Final":
                    continue
                home  = g["home_team"]["full_name"]
                away  = g["visitor_team"]["full_name"]
                if not _matchup_matches(bet["matchup"], home, away):
                    continue
                _finalize_bet(bet, home, away, g["home_team_score"], g["visitor_team_score"], resolved_list)
                found = True
                break
            if found:
                break


def _evaluate_pick(bet: dict, home: str, away: str, home_score: int, away_score: int) -> Optional[bool]:
    bet_type = bet.get("bet_type", "moneyline").lower()
    pick     = bet["pick"].lower()
    hl, al   = home.lower(), away.lower()

    if bet_type == "moneyline":
        if any(w in hl for w in pick.split()):
            return home_score > away_score
        if any(w in al for w in pick.split()):
            return away_score > home_score
        return None

    if bet_type == "spread":
        m = re.search(r"([+-]?\d+\.?\d*)\s*$", pick)
        if not m:
            return None
        spread = float(m.group(1))
        if any(w in hl for w in pick.split()):
            return (home_score - away_score) > -spread
        if any(w in al for w in pick.split()):
            return (away_score - home_score) > -spread
        return None

    if bet_type == "total":
        total = home_score + away_score
        m = re.search(r"(\d+\.?\d*)", pick)
        if not m:
            return None
        line = float(m.group(1))
        if "over" in pick:
            return total > line
        if "under" in pick:
            return total < line
        return None

    return None
