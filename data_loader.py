"""
BetIQ — Local Kaggle dataset loader.
Provides fast, offline access to NBA game history (1946–2023).
All DataFrames are loaded once and cached for the process lifetime.
"""

import os
import pandas as pd
from functools import lru_cache

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "csv")

# Dataset covers through the 2022-23 regular season
KAGGLE_MAX_SEASON = 2022


# ── Loaders (cached) ──────────────────────────────────────────────────────────

@lru_cache(maxsize=None)
def _games() -> pd.DataFrame:
    df = pd.read_csv(
        os.path.join(DATA_DIR, "game.csv"),
        parse_dates=["game_date"],
        low_memory=False,
    )
    df = df[df["season_type"] == "Regular Season"].copy()
    df["season_year"] = (df["season_id"] % 10000).astype(int)
    return df


@lru_cache(maxsize=None)
def _teams() -> pd.DataFrame:
    return pd.read_csv(os.path.join(DATA_DIR, "team.csv"))


@lru_cache(maxsize=None)
def _other_stats() -> pd.DataFrame:
    return pd.read_csv(os.path.join(DATA_DIR, "other_stats.csv"))


# ── Team resolution ───────────────────────────────────────────────────────────

@lru_cache(maxsize=None)
def _team_index() -> tuple:
    """
    Build lookup structures once. Returns (exact_map, rows) where:
    - exact_map: {lowercase key → row index} for O(1) exact lookups
    - rows: tuple of row dicts for fallback substring matching
    Cached for the process lifetime since team names don't change.
    """
    teams = _teams()
    exact: dict = {}
    rows = []
    for _, row in teams.iterrows():
        d = row.to_dict()
        rows.append(d)
        for field in ("full_name", "abbreviation", "nickname", "city"):
            val = str(d.get(field, "")).lower()
            if val:
                exact.setdefault(val, d)
    return exact, tuple(rows)


def resolve_team(name: str):
    """Match a name string to a team row, prioritising exact matches. Returns None if not found."""
    exact, rows = _team_index()
    nl = name.lower().strip()

    # O(1): exact full name, abbreviation, nickname, or city
    if nl in exact:
        return exact[nl]

    # Substring fallback: full name contains query or vice-versa
    for row in rows:
        full = str(row.get("full_name", "")).lower()
        if nl in full or full in nl:
            return row

    return None


# ── Internal helper ───────────────────────────────────────────────────────────

def _team_games(team_id: int, season_year: int) -> pd.DataFrame:
    """
    Return all regular-season games for a team in a given season,
    normalised so every row has the team's perspective (pts_for, pts_against, etc.)
    """
    df = _games()
    season_df = df[df["season_year"] == season_year]

    shared_cols = ["game_id", "game_date", "season_year"]

    home = season_df[season_df["team_id_home"] == team_id].copy()
    home["is_home"]      = True
    home["pts_for"]      = home["pts_home"]
    home["pts_against"]  = home["pts_away"]
    home["result"]       = home["wl_home"]
    home["opponent"]     = home["team_name_away"]
    home["fg_pct"]       = home["fg_pct_home"]
    home["fg3_pct"]      = home["fg3_pct_home"]
    home["ft_pct"]       = home["ft_pct_home"]
    home["reb"]          = home["reb_home"]
    home["ast"]          = home["ast_home"]
    home["stl"]          = home["stl_home"]
    home["blk"]          = home["blk_home"]
    home["tov"]          = home["tov_home"]
    home["plus_minus"]   = home["plus_minus_home"]

    away = season_df[season_df["team_id_away"] == team_id].copy()
    away["is_home"]      = False
    away["pts_for"]      = away["pts_away"]
    away["pts_against"]  = away["pts_home"]
    away["result"]       = away["wl_away"]
    away["opponent"]     = away["team_name_home"]
    away["fg_pct"]       = away["fg_pct_away"]
    away["fg3_pct"]      = away["fg3_pct_away"]
    away["ft_pct"]       = away["ft_pct_away"]
    away["reb"]          = away["reb_away"]
    away["ast"]          = away["ast_away"]
    away["stl"]          = away["stl_away"]
    away["blk"]          = away["blk_away"]
    away["tov"]          = away["tov_away"]
    away["plus_minus"]   = away["plus_minus_away"]

    keep = shared_cols + [
        "is_home", "pts_for", "pts_against", "result", "opponent",
        "fg_pct", "fg3_pct", "ft_pct", "reb", "ast", "stl", "blk", "tov", "plus_minus",
    ]
    combined = pd.concat([home[keep], away[keep]], ignore_index=True)
    return combined.sort_values("game_date").reset_index(drop=True)


def _avg(series) -> float:
    return round(float(series.mean()), 3) if len(series) > 0 else 0.0


# ── Public query functions ────────────────────────────────────────────────────

def get_team_stats_local(team_name: str, season: int) -> dict | None:
    """Season averages + record from local CSV. Returns None if season > KAGGLE_MAX_SEASON."""
    if season > KAGGLE_MAX_SEASON:
        return None
    team = resolve_team(team_name)
    if team is None:
        return None

    tg = _team_games(int(team["id"]), season)
    if tg.empty:
        return None

    wins   = int((tg["result"] == "W").sum())
    losses = int((tg["result"] == "L").sum())
    n      = len(tg)

    return {
        "team":   team["full_name"],
        "season": season,
        "source": "local_kaggle",
        "record": {
            "wins":    wins,
            "losses":  losses,
            "win_pct": round(wins / max(wins + losses, 1), 3),
        },
        "averages": [{
            "gp":          n,
            "pts":         _avg(tg["pts_for"]),
            "pts_allowed": _avg(tg["pts_against"]),
            "net_rtg":     _avg(tg["pts_for"] - tg["pts_against"]),
            "reb":         _avg(tg["reb"]),
            "ast":         _avg(tg["ast"]),
            "stl":         _avg(tg["stl"]),
            "blk":         _avg(tg["blk"]),
            "tov":         _avg(tg["tov"]),
            "fg_pct":      _avg(tg["fg_pct"]),
            "fg3_pct":     _avg(tg["fg3_pct"]),
            "ft_pct":      _avg(tg["ft_pct"]),
            "plus_minus":  _avg(tg["plus_minus"]),
        }],
    }


def get_season_stats_local(team_name: str, seasons: list) -> dict | None:
    """Multi-season stats for seasons available in the local dataset."""
    team = resolve_team(team_name)
    if team is None:
        return None

    multi = {}
    for s in seasons:
        if s <= KAGGLE_MAX_SEASON:
            result = get_team_stats_local(team_name, s)
            if result:
                multi[str(s)] = result

    if not multi:
        return None

    return {
        "team":              team["full_name"],
        "source":            "local_kaggle",
        "multi_season_stats": multi,
    }


def get_recent_form_local(team_name: str, season: int, last_n: int = 10) -> dict | None:
    """Last N games for a team in a given season from local CSV."""
    if season > KAGGLE_MAX_SEASON:
        return None
    team = resolve_team(team_name)
    if team is None:
        return None

    tg = _team_games(int(team["id"]), season).tail(last_n)
    if tg.empty:
        return None

    form = []
    for _, row in tg.iterrows():
        margin = int(row["pts_for"] - row["pts_against"])
        form.append({
            "date":     str(row["game_date"])[:10],
            "opponent": row["opponent"],
            "location": "Home" if row["is_home"] else "Away",
            "score":    f"{int(row['pts_for'])}-{int(row['pts_against'])}",
            "result":   row["result"],
            "margin":   margin,
        })

    wins = sum(1 for g in form if g["result"] == "W")
    return {
        "team":       team["full_name"],
        "source":     "local_kaggle",
        "games_back": len(form),
        "record":     f"{wins}-{len(form) - wins}",
        "win_pct":    round(wins / max(len(form), 1), 3),
        "avg_margin": round(sum(g["margin"] for g in form) / max(len(form), 1), 1),
        "form":       form,
    }


def get_home_away_splits_local(team_name: str, season: int) -> dict | None:
    """Home/away splits from local CSV."""
    if season > KAGGLE_MAX_SEASON:
        return None
    team = resolve_team(team_name)
    if team is None:
        return None

    tg = _team_games(int(team["id"]), season)
    if tg.empty:
        return None

    def _splits(df: pd.DataFrame) -> dict:
        if df.empty:
            return {}
        wins = int((df["result"] == "W").sum())
        g    = len(df)
        return {
            "record":          f"{wins}-{g - wins}",
            "win_pct":         round(wins / max(g, 1), 3),
            "avg_pts_for":     _avg(df["pts_for"]),
            "avg_pts_against": _avg(df["pts_against"]),
            "avg_margin":      _avg(df["pts_for"] - df["pts_against"]),
            "fg_pct":          _avg(df["fg_pct"]),
            "fg3_pct":         _avg(df["fg3_pct"]),
        }

    return {
        "team":   team["full_name"],
        "season": season,
        "source": "local_kaggle",
        "home":   _splits(tg[tg["is_home"]]),
        "away":   _splits(tg[~tg["is_home"]]),
    }


def get_head_to_head_local(
    team1_name: str,
    team2_name: str,
    num_seasons: int = 3,
    include_all_history: bool = True,
) -> dict | None:
    """
    Head-to-head history from local CSV.
    If include_all_history=True, returns all available H2H games (not just last N seasons).
    """
    team1 = resolve_team(team1_name)
    team2 = resolve_team(team2_name)
    if team1 is None or team2 is None:
        return None

    df   = _games()
    t1id = int(team1["id"])
    t2id = int(team2["id"])

    mask = (
        ((df["team_id_home"] == t1id) & (df["team_id_away"] == t2id)) |
        ((df["team_id_home"] == t2id) & (df["team_id_away"] == t1id))
    )

    if not include_all_history:
        seasons = list(range(KAGGLE_MAX_SEASON - num_seasons + 1, KAGGLE_MAX_SEASON + 1))
        mask = mask & df["season_year"].isin(seasons)

    h2h = df[mask].copy()
    if h2h.empty:
        return {
            "teams":           [team1["full_name"], team2["full_name"]],
            "total_h2h_games": 0,
            "matchups":        [],
        }

    matchups = []
    for _, row in h2h.iterrows():
        t1_home  = row["team_id_home"] == t1id
        t1_score = int(row["pts_home"]) if t1_home else int(row["pts_away"])
        t2_score = int(row["pts_away"]) if t1_home else int(row["pts_home"])
        matchups.append({
            "date":       str(row["game_date"])[:10],
            "season":     int(row["season_year"]),
            "home_team":  row["team_name_home"],
            "home_score": int(row["pts_home"]),
            "away_score": int(row["pts_away"]),
            "winner":     team1["full_name"] if t1_score > t2_score else team2["full_name"],
            "t1_score":   t1_score,
            "t2_score":   t2_score,
            "margin":     t1_score - t2_score,
        })

    matchups.sort(key=lambda x: x["date"], reverse=True)
    t1_wins   = sum(1 for m in matchups if m["winner"] == team1["full_name"])
    avg_total = sum(m["t1_score"] + m["t2_score"] for m in matchups) / max(len(matchups), 1)

    # Recent form in H2H (last 10)
    recent_10     = matchups[:10]
    recent_t1_wins = sum(1 for m in recent_10 if m["winner"] == team1["full_name"])

    return {
        "teams":           [team1["full_name"], team2["full_name"]],
        "source":          "local_kaggle",
        "total_h2h_games": len(matchups),
        "h2h_record": {
            team1["full_name"]: t1_wins,
            team2["full_name"]: len(matchups) - t1_wins,
        },
        "recent_10_h2h": {
            team1["full_name"]: recent_t1_wins,
            team2["full_name"]: len(recent_10) - recent_t1_wins,
        },
        "avg_total_points": round(avg_total, 1),
        "matchups":         matchups[:50],  # cap at 50 to keep response size sane
    }


def get_advanced_stats_local(team_name: str, season: int) -> dict | None:
    """Paint points, fast-break pts, lead changes, turnovers from other_stats.csv."""
    if season > KAGGLE_MAX_SEASON:
        return None
    team = resolve_team(team_name)
    if team is None:
        return None

    df   = _games()
    t_id = int(team["id"])
    season_game_ids = set(
        df[df["season_year"] == season]["game_id"].tolist()
    )

    os_df = _other_stats()
    home = os_df[
        (os_df["team_id_home"] == t_id) & (os_df["game_id"].isin(season_game_ids))
    ].copy()
    home["pts_paint"]    = home["pts_paint_home"]
    home["pts_fb"]       = home["pts_fb_home"]
    home["pts_2nd"]      = home["pts_2nd_chance_home"]
    home["team_tov"]     = home["team_turnovers_home"]
    home["pts_off_to"]   = home["pts_off_to_home"]
    home["lead_changes"] = home["lead_changes"]

    away = os_df[
        (os_df["team_id_away"] == t_id) & (os_df["game_id"].isin(season_game_ids))
    ].copy()
    away["pts_paint"]    = away["pts_paint_away"]
    away["pts_fb"]       = away["pts_fb_away"]
    away["pts_2nd"]      = away["pts_2nd_chance_away"]
    away["team_tov"]     = away["team_turnovers_away"]
    away["pts_off_to"]   = away["pts_off_to_away"]
    away["lead_changes"] = away["lead_changes"]

    keep = ["pts_paint", "pts_fb", "pts_2nd", "team_tov", "pts_off_to", "lead_changes"]
    combined = pd.concat([home[keep], away[keep]], ignore_index=True)

    if combined.empty:
        return None

    return {
        "team":   team["full_name"],
        "season": season,
        "source": "local_kaggle_advanced",
        "avg_pts_in_paint":      _avg(combined["pts_paint"]),
        "avg_pts_fast_break":    _avg(combined["pts_fb"]),
        "avg_pts_2nd_chance":    _avg(combined["pts_2nd"]),
        "avg_team_turnovers":    _avg(combined["team_tov"]),
        "avg_pts_off_turnovers": _avg(combined["pts_off_to"]),
        "avg_lead_changes":      _avg(combined["lead_changes"]),
    }
