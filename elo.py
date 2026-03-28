"""
BetIQ — NBA Elo rating engine.

Parameters:
  K = 20                (standard for NBA)
  HOME_ADVANTAGE = 100  (adds 100 Elo pts to home team's expected score calc)
  DEFAULT_RATING = 1500 (starting rating for all teams)
  REGRESSION = 0.25     (pull 25% toward 1500 at each new season)

Margin of Victory multiplier (538-style):
  MOV = log(|margin| + 1) * (2.2 / (winner_elo_diff * 0.001 + 2.2))

Spread conversion:
  implied_spread = (home_win_prob - 0.5) * 28
"""

import logging
import math
import time

import database as db

log = logging.getLogger("betiq.elo")

K                 = 20
HOME_ADVANTAGE    = 100
DEFAULT_RATING    = 1500.0
REGRESSION_FACTOR = 0.25
SPREAD_SCALE      = 28.0


# ── Core math ─────────────────────────────────────────────────────────────────

def expected_score(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))


def mov_multiplier(margin: int, winner_elo_diff: float) -> float:
    return math.log(abs(margin) + 1) * (2.2 / (winner_elo_diff * 0.001 + 2.2))


def update_ratings(home_r: float, away_r: float, home_score: int, away_score: int) -> tuple[float, float]:
    home_adj = home_r + HOME_ADVANTAGE
    e_home = expected_score(home_adj, away_r)
    s_home = 1.0 if home_score > away_score else 0.0
    s_away = 1.0 - s_home
    margin = abs(home_score - away_score)
    winner_diff = (home_adj - away_r) if home_score > away_score else (away_r - home_adj)
    mult = mov_multiplier(margin, winner_diff)
    return (
        home_r + K * mult * (s_home - e_home),
        away_r + K * mult * (s_away - (1.0 - e_home)),
    )


def apply_season_regression(rating: float) -> float:
    return rating + REGRESSION_FACTOR * (DEFAULT_RATING - rating)


# ── Public API ────────────────────────────────────────────────────────────────

def win_probability(home_team: str, away_team: str) -> dict:
    """
    Return Elo-based win probabilities and implied spread for a matchup.
    Falls back to 50/50 if ratings not yet initialized.
    """
    # Lazy import to avoid circular imports (tools imports elo, elo imports tools)
    import tools as t

    home_info = t._resolve_team(home_team)
    away_info = t._resolve_team(away_team)

    home_id = home_info["id"] if home_info else None
    away_id = away_info["id"] if away_info else None
    home_name = home_info["full_name"] if home_info else home_team
    away_name = away_info["full_name"] if away_info else away_team

    home_row = db.get_elo_rating(home_id) if home_id else None
    away_row = db.get_elo_rating(away_id) if away_id else None

    home_r = home_row["rating"] if home_row else DEFAULT_RATING
    away_r = away_row["rating"] if away_row else DEFAULT_RATING
    source = "elo_db" if (home_row and away_row) else "not_initialized"

    home_prob = expected_score(home_r + HOME_ADVANTAGE, away_r)
    away_prob = 1.0 - home_prob
    implied_spread = round((home_prob - 0.5) * SPREAD_SCALE, 1)

    return {
        "home_team":      home_name,
        "away_team":      away_name,
        "home_rating":    round(home_r, 1),
        "away_rating":    round(away_r, 1),
        "home_prob_pct":  round(home_prob * 100, 1),
        "away_prob_pct":  round(away_prob * 100, 1),
        "implied_spread": implied_spread,
        "note": "positive spread = home team favoured by N points",
        "source":         source,
    }


def process_game_result(home_team: str, away_team: str, home_score: int, away_score: int, season: int) -> dict:
    """Update Elo ratings after a completed game. Called from tools._finalize_bet."""
    import tools as t

    home_info = t._resolve_team(home_team)
    away_info = t._resolve_team(away_team)
    if not home_info or not away_info:
        return {"error": f"Could not resolve: {home_team} / {away_team}"}

    home_id, away_id = home_info["id"], away_info["id"]
    home_name, away_name = home_info["full_name"], away_info["full_name"]

    home_row = db.get_elo_rating(home_id)
    away_row = db.get_elo_rating(away_id)

    home_r  = home_row["rating"]       if home_row else DEFAULT_RATING
    away_r  = away_row["rating"]       if away_row else DEFAULT_RATING
    home_gp = home_row["games_played"] if home_row else 0
    away_gp = away_row["games_played"] if away_row else 0

    # Season regression if needed
    if home_row and home_row["season"] < season:
        home_r = apply_season_regression(home_r)
        home_gp = 0
    if away_row and away_row["season"] < season:
        away_r = apply_season_regression(away_r)
        away_gp = 0

    new_home_r, new_away_r = update_ratings(home_r, away_r, home_score, away_score)

    db.upsert_elo_rating(home_id, home_name, new_home_r, season, home_gp + 1)
    db.upsert_elo_rating(away_id, away_name, new_away_r, season, away_gp + 1)

    return {
        "home_team": home_name, "away_team": away_name,
        "home_delta": round(new_home_r - home_r, 1),
        "away_delta": round(new_away_r - away_r, 1),
        "home_rating_new": round(new_home_r, 1),
        "away_rating_new": round(new_away_r, 1),
    }


# ── Historical build ──────────────────────────────────────────────────────────

def build_from_history(seasons: list = None) -> dict:
    """
    Fetch all historical games from BallDontLie and compute Elo ratings from scratch.
    Takes ~4 minutes due to API rate limit pacing.
    """
    if seasons is None:
        seasons = [2022, 2023, 2024, 2025]

    import tools as t

    db.init_db()
    ratings = {}  # {team_id: {"rating", "season", "gp", "name"}}
    total_games = 0

    for season in seasons:
        log.info(f"Fetching season {season}...")
        games = _fetch_season(season, t)
        log.info(f"  {len(games)} games fetched for {season}.")

        # Apply season regression to all teams at season boundary
        for tid in ratings:
            if ratings[tid]["season"] < season:
                ratings[tid]["rating"] = apply_season_regression(ratings[tid]["rating"])
                ratings[tid]["gp"] = 0
                ratings[tid]["season"] = season

        games.sort(key=lambda g: g.get("date", ""))

        for game in games:
            if game.get("status") != "Final":
                continue
            hs = game.get("home_team_score") or 0
            vs = game.get("visitor_team_score") or 0
            if hs == 0 and vs == 0:
                continue

            ht = game["home_team"]
            at = game["visitor_team"]
            hid, aid = ht["id"], at["id"]

            hr = ratings.get(hid, {}).get("rating", DEFAULT_RATING)
            ar = ratings.get(aid, {}).get("rating", DEFAULT_RATING)
            new_hr, new_ar = update_ratings(hr, ar, hs, vs)

            ratings[hid] = {"rating": new_hr, "season": season, "gp": ratings.get(hid, {}).get("gp", 0) + 1, "name": ht["full_name"]}
            ratings[aid] = {"rating": new_ar, "season": season, "gp": ratings.get(aid, {}).get("gp", 0) + 1, "name": at["full_name"]}
            total_games += 1

        time.sleep(3)

    current_season = t._current_season()
    for tid, data in ratings.items():
        db.upsert_elo_rating(tid, data["name"], data["rating"], current_season, data["gp"])

    log.info(f"Elo build complete: {len(ratings)} teams, {total_games} games.")
    return {
        "teams": len(ratings),
        "games_processed": total_games,
        "seasons": seasons,
        "ratings": {d["name"]: round(d["rating"], 1) for d in sorted(ratings.values(), key=lambda x: x["rating"], reverse=True)},
    }


def _fetch_season(season: int, t) -> list:
    """Paginate BDL /games for a full season."""
    games = []
    cursor = None
    while True:
        params = {"seasons[]": season, "per_page": 100}
        if cursor is not None:
            params["cursor"] = cursor
        data = t._bdl_get("/games", params)
        if "error" in data:
            log.warning(f"BDL error season {season}: {data['error']}")
            break
        page = data.get("data", [])
        games.extend(page)
        cursor = data.get("meta", {}).get("next_cursor")
        if not cursor or not page:
            break
        time.sleep(0.6)
    return games
