"""
BetIQ — Calibrated logistic regression model for NBA win probability.

Training source: data/csv/archive (1)/TeamStatistics.csv
  - 72K+ regular-season game rows from 2000 to present
  - One row per team per game

Features (all computed as home − away differentials, no lookahead):
  1. elo_diff            — Elo rating differential (with home advantage)
  2. rest_diff           — Rest days differential
  3. season_win_pct_diff — Season win % differential
  4. home_away_win_pct_diff — Home team's home win% minus away team's away win%
  5. form_diff           — Recent form differential (last 10 games win%)
  6. net_rtg_diff        — Scoring margin differential (pts − pts_allowed avg)
  7. h2h_win_pct         — Home team's historical win% vs this opponent

Usage:
    from model import predict_win_prob, get_edge, train
    from model import extract_features_from_prefetch

    # Training (run once, then model is saved to disk):
    python model.py

    # Prediction:
    feature_dict = extract_features_from_prefetch(home, away, prefetch_results)
    prob = predict_win_prob(home, away, feature_dict)
    edge = get_edge(home, away, feature_dict, market_odds_home=-115)
"""

from __future__ import annotations

import logging
import math
import os
from collections import defaultdict

import numpy as np
import pandas as pd

log = logging.getLogger("betiq.model")

# ── Paths ──────────────────────────────────────────────────────────────────────

_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(_DIR, "betiq_model.joblib")
TS_PATH = os.path.join(_DIR, "data", "csv", "archive (1)", "TeamStatistics.csv")

# ── Constants ──────────────────────────────────────────────────────────────────

MIN_PRIOR_GAMES = 10       # ignore team rows with fewer prior games
MIN_TRAIN_SEASON = 2000    # only modern NBA
ELO_K = 20
ELO_HOME_ADV = 100
ELO_DEFAULT = 1500.0

FEATURE_NAMES = [
    "elo_diff",
    "rest_diff",
    "season_win_pct_diff",
    "home_away_win_pct_diff",
    "form_diff",
    "net_rtg_diff",
    "h2h_win_pct",
]

# ── Model storage ──────────────────────────────────────────────────────────────

_model = None


def _load_model():
    global _model
    if _model is not None:
        return _model
    if os.path.exists(MODEL_PATH):
        try:
            import joblib
            _model = joblib.load(MODEL_PATH)
            log.info("BetIQ logistic regression model loaded.")
        except Exception as exc:
            log.warning(f"Could not load model file: {exc}")
            _model = None
    return _model


# Load on import — zero training delay if model file exists
_load_model()


# ── Training: Elo computation ──────────────────────────────────────────────────

def _compute_elo_series(games_df: pd.DataFrame):
    """
    Walk through games chronologically, compute pre-game Elo for each.
    Uses MOV multiplier (538-style) to update ratings after each game.
    Returns two lists: home_elos, away_elos.
    """
    ratings: dict[int, float] = {}
    home_elos, away_elos = [], []

    for _, row in games_df.iterrows():
        h_id = int(row["teamId_home"])
        a_id = int(row["teamId_away"])

        h_base = ratings.get(h_id, ELO_DEFAULT)
        a_base = ratings.get(a_id, ELO_DEFAULT)
        home_elos.append(h_base)
        away_elos.append(a_base)

        h_elo = h_base + ELO_HOME_ADV
        a_elo = a_base
        exp_h = 1.0 / (1.0 + 10.0 ** ((a_elo - h_elo) / 400.0))
        actual_h = float(row["win_home"])

        margin = abs(float(row.get("margin_home", 1)))
        elo_diff = (h_elo - a_elo) if actual_h else (a_elo - h_elo)
        mov_mult = math.log(margin + 1) * (2.2 / (abs(elo_diff) * 0.001 + 2.2))

        delta = ELO_K * mov_mult * (actual_h - exp_h)
        ratings[h_id] = h_base + delta
        ratings[a_id] = a_base - delta

    return home_elos, away_elos


# ── Training: H2H feature ─────────────────────────────────────────────────────

def _compute_h2h_series(games_df: pd.DataFrame) -> list[float]:
    """
    For each game (in chronological order), compute the home team's win rate
    in all prior matchups between these two teams. Returns 0.5 if < 3 prior games.
    """
    # (home_id, away_id) -> list of int (1=home won, 0=home lost)
    history: dict[tuple, list] = defaultdict(list)
    result = []

    for _, row in games_df.iterrows():
        h_id = int(row["teamId_home"])
        a_id = int(row["teamId_away"])

        # Games where these teams met: home_id was home OR away_id was home
        as_home = history[(h_id, a_id)]       # home team was home: 1=won
        as_away = history[(a_id, h_id)]       # home team was away: 1=opponent won

        total = len(as_home) + len(as_away)
        if total >= 3:
            home_wins = sum(as_home) + sum(1 for x in as_away if x == 0)
            result.append(home_wins / total)
        else:
            result.append(0.5)

        history[(h_id, a_id)].append(int(row["win_home"]))

    return result


# ── Training: full pipeline ────────────────────────────────────────────────────

def _build_training_data() -> tuple[np.ndarray, np.ndarray]:
    log.info("Loading TeamStatistics.csv …")
    df = pd.read_csv(TS_PATH, low_memory=False)

    df["game_date"] = pd.to_datetime(df["gameDateTimeEst"], errors="coerce")
    df = df.dropna(subset=["game_date", "win", "home"])
    df["win"] = df["win"].astype(float)
    df["home"] = df["home"].astype(int)

    # NBA season year: October start means season belongs to next calendar year
    df["season_year"] = df["game_date"].dt.year
    df.loc[df["game_date"].dt.month >= 10, "season_year"] += 1

    df = df[df["season_year"] >= MIN_TRAIN_SEASON].copy()
    df["margin"] = df["teamScore"] - df["opponentScore"]
    df = df.sort_values(["teamId", "game_date"]).reset_index(drop=True)

    log.info(f"Building rolling features for {len(df):,} team-game rows …")

    # Season win pct (prior games in same season only)
    df["s_wins"] = df.groupby(["teamId", "season_year"])["win"].transform(
        lambda x: x.shift(1).expanding().sum().fillna(0)
    )
    df["s_games"] = df.groupby(["teamId", "season_year"])["win"].transform(
        lambda x: x.shift(1).expanding().count().fillna(0)
    )
    df["season_win_pct"] = (df["s_wins"] / df["s_games"].clip(lower=1)).fillna(0.5)

    # Rolling form: last 10 games win pct
    df["form"] = (
        df.groupby("teamId")["win"]
        .transform(lambda x: x.shift(1).rolling(10, min_periods=5).mean())
        .fillna(0.5)
    )

    # Net scoring margin (all-time rolling avg)
    df["net_rtg"] = (
        df.groupby("teamId")["margin"]
        .transform(lambda x: x.shift(1).expanding().mean())
        .fillna(0.0)
    )

    # Home win pct
    df["is_home_win"] = ((df["home"] == 1) & (df["win"] == 1)).astype(float)
    df["home_games_prior"] = df.groupby("teamId")["home"].transform(
        lambda x: x.shift(1).expanding().sum().fillna(0)
    )
    df["home_wins_prior"] = df.groupby("teamId")["is_home_win"].transform(
        lambda x: x.shift(1).expanding().sum().fillna(0)
    )
    df["home_win_pct"] = (
        df["home_wins_prior"] / df["home_games_prior"].clip(lower=1)
    ).fillna(0.5)

    # Away win pct
    df["is_away_win"] = ((df["home"] == 0) & (df["win"] == 1)).astype(float)
    df["away_games_prior"] = df.groupby("teamId")["home"].transform(
        lambda x: (1 - x).shift(1).expanding().sum().fillna(0)
    )
    df["away_wins_prior"] = df.groupby("teamId")["is_away_win"].transform(
        lambda x: x.shift(1).expanding().sum().fillna(0)
    )
    df["away_win_pct"] = (
        df["away_wins_prior"] / df["away_games_prior"].clip(lower=1)
    ).fillna(0.5)

    # Games played (for minimum filter)
    df["games_played"] = df.groupby("teamId")["win"].transform(
        lambda x: x.shift(1).expanding().count().fillna(0)
    )

    # Rest days
    df["prev_date"] = df.groupby("teamId")["game_date"].shift(1)
    df["rest_days"] = (
        (df["game_date"] - df["prev_date"]).dt.days.clip(upper=7).fillna(3)
    )

    # ── Merge home and away rows into one row per game ──
    log.info("Merging home/away into game rows …")
    df = df.sort_values("game_date").reset_index(drop=True)

    home_df = df[df["home"] == 1].copy()
    away_df = df[df["home"] == 0].copy()

    keep_away = [
        "gameId", "teamId", "margin", "win",
        "season_win_pct", "form", "net_rtg",
        "away_win_pct", "rest_days", "games_played",
    ]
    away_renamed = away_df[keep_away].rename(columns={
        "teamId":         "teamId_away",
        "margin":         "margin_away",
        "win":            "win_away",
        "season_win_pct": "season_win_pct_away",
        "form":           "form_away",
        "net_rtg":        "net_rtg_away",
        "away_win_pct":   "away_win_pct_away",
        "rest_days":      "rest_days_away",
        "games_played":   "games_played_away",
    })

    games = home_df.merge(away_renamed, on="gameId").rename(columns={
        "teamId":         "teamId_home",
        "margin":         "margin_home",
        "win":            "win_home",
        "season_win_pct": "season_win_pct_home",
        "form":           "form_home",
        "net_rtg":        "net_rtg_home",
        "home_win_pct":   "home_win_pct_home",
        "rest_days":      "rest_days_home",
        "games_played":   "games_played_home",
    })

    # Both teams need enough prior games
    games = games[
        (games["games_played_home"] >= MIN_PRIOR_GAMES) &
        (games["games_played_away"] >= MIN_PRIOR_GAMES)
    ].copy()
    games = games.sort_values("game_date").reset_index(drop=True)

    log.info(f"Game rows after filter: {len(games):,}")

    # ── Elo ──
    log.info("Computing Elo series …")
    home_elos, away_elos = _compute_elo_series(games)
    games["elo_home"] = home_elos
    games["elo_away"] = away_elos

    # ── H2H ──
    log.info("Computing H2H series …")
    games["h2h_win_pct"] = _compute_h2h_series(games)

    # ── Assemble feature matrix ──
    games["elo_diff"] = games["elo_home"] - games["elo_away"]
    games["rest_diff"] = games["rest_days_home"] - games["rest_days_away"]
    games["season_win_pct_diff"] = (
        games["season_win_pct_home"] - games["season_win_pct_away"]
    )
    games["home_away_win_pct_diff"] = (
        games["home_win_pct_home"] - games["away_win_pct_away"]
    )
    games["form_diff"] = games["form_home"] - games["form_away"]
    games["net_rtg_diff"] = games["net_rtg_home"] - games["net_rtg_away"]

    X = games[FEATURE_NAMES].values.astype(np.float32)
    y = games["win_home"].values.astype(np.float32)

    log.info(f"Training matrix: {X.shape[0]:,} games × {X.shape[1]} features")
    return X, y


# ── Public: train ──────────────────────────────────────────────────────────────

def train() -> object:
    """
    Train a calibrated logistic regression on all historical game data.
    Saves the model to betiq_model.joblib. Returns the fitted model.
    Run once: python model.py
    """
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    import joblib

    X, y = _build_training_data()

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(C=1.0, max_iter=1000, random_state=42)),
    ])
    model = CalibratedClassifierCV(pipe, method="isotonic", cv=5)

    log.info("Training calibrated logistic regression …")
    model.fit(X, y)

    joblib.dump(model, MODEL_PATH)
    log.info(f"Model saved → {MODEL_PATH}")

    global _model
    _model = model

    # Quick accuracy check
    preds = model.predict(X)
    acc = (preds == y).mean()
    log.info(f"In-sample accuracy: {acc:.3f}")
    print(f"Trained on {len(X):,} games. In-sample accuracy: {acc:.3f}")
    print(f"Model saved to {MODEL_PATH}")
    return model


# ── Public: predict ────────────────────────────────────────────────────────────

def _build_feature_vector(feature_dict: dict) -> list[float]:
    """Convert a feature dict to the ordered vector the model expects."""
    return [
        feature_dict.get("elo_home", ELO_DEFAULT) - feature_dict.get("elo_away", ELO_DEFAULT),
        feature_dict.get("rest_home", 2) - feature_dict.get("rest_away", 2),
        feature_dict.get("season_win_pct_home", 0.5) - feature_dict.get("season_win_pct_away", 0.5),
        feature_dict.get("home_win_pct_home", 0.5) - feature_dict.get("away_win_pct_away", 0.5),
        feature_dict.get("form_home", 0.5) - feature_dict.get("form_away", 0.5),
        feature_dict.get("net_rtg_home", 0.0) - feature_dict.get("net_rtg_away", 0.0),
        feature_dict.get("h2h_win_pct", 0.5),
    ]


def predict_win_prob(home_team: str, away_team: str, feature_dict: dict) -> float:
    """
    Return calibrated probability of home team winning (0.0–1.0).
    Falls back to Elo-only if model file not found.
    """
    model = _load_model()
    features = _build_feature_vector(feature_dict)

    if model is None:
        # Elo fallback: use elo_diff feature with home advantage baked in
        elo_diff = features[0] + ELO_HOME_ADV
        return round(1.0 / (1.0 + 10.0 ** (-elo_diff / 400.0)), 4)

    X = np.array([features], dtype=np.float32)
    prob = float(model.predict_proba(X)[0][1])
    return round(prob, 4)


def get_edge(
    home_team: str,
    away_team: str,
    feature_dict: dict,
    market_odds_home: int,
) -> dict:
    """
    Full edge assessment for a home-team moneyline bet.

    Args:
        home_team:        e.g. "Boston Celtics"
        away_team:        e.g. "Miami Heat"
        feature_dict:     output of extract_features_from_prefetch()
        market_odds_home: American odds for home team (e.g. -115, +130)

    Returns dict with: model_prob, implied_prob, edge_pct, kelly_fraction, model_source
    """
    model_prob = predict_win_prob(home_team, away_team, feature_dict)

    # Implied probability from American odds (no vig not removed here — edge is raw)
    if market_odds_home > 0:
        implied_prob = 100.0 / (market_odds_home + 100.0)
        decimal_odds = market_odds_home / 100.0
    else:
        abs_odds = abs(market_odds_home)
        implied_prob = abs_odds / (abs_odds + 100.0)
        decimal_odds = 100.0 / abs_odds

    edge_pct = (model_prob - implied_prob) * 100.0

    # Half-Kelly, capped at 12%
    p, q, b = model_prob, 1 - model_prob, decimal_odds
    kelly_raw = (p * b - q) / b
    half_kelly = max(0.0, min(kelly_raw / 2, 0.12))

    return {
        "model_prob":     round(model_prob, 4),
        "implied_prob":   round(implied_prob, 4),
        "edge_pct":       round(edge_pct, 2),
        "kelly_fraction": round(half_kelly, 4),
        "model_source":   "logistic_regression" if _model is not None else "elo_fallback",
    }


# ── Public: feature extraction from prefetch results ──────────────────────────

def extract_features_from_prefetch(
    home_team: str,
    away_team: str,
    results: dict,
) -> dict:
    """
    Extract a feature_dict from the already-fetched prefetch results dict.
    results: the dict returned by scan_context.fetch_game_data()

    Returns a feature_dict ready for predict_win_prob() / get_edge().
    """
    def _safe(d, *keys, default=None):
        for k in keys:
            if isinstance(d, dict):
                d = d.get(k)
            else:
                return default
        return d if d is not None else default

    # ── Elo ──
    elo = results.get("elo_prob", {})
    elo_home = _safe(elo, "home_elo", default=ELO_DEFAULT)
    elo_away = _safe(elo, "away_elo", default=ELO_DEFAULT)

    # ── Rest days ──
    def _rest(rest_result):
        return _safe(rest_result, "days_rest", default=2)

    rest_home = _rest(results.get("rest_home", {}))
    rest_away = _rest(results.get("rest_away", {}))

    # ── Season win pct ──
    def _win_pct(stats_result):
        avgs = _safe(stats_result, "averages", default=[])
        if avgs:
            return _safe(avgs[0], "win_pct", default=0.5)
        record = _safe(stats_result, "record", default={})
        return _safe(record, "win_pct", default=0.5)

    season_win_pct_home = _win_pct(results.get("team_stats_home", {}))
    season_win_pct_away = _win_pct(results.get("team_stats_away", {}))

    # ── Home/away win pct ──
    def _home_win_pct(splits_result):
        return _safe(splits_result, "home", "win_pct", default=0.5)

    def _away_win_pct(splits_result):
        return _safe(splits_result, "away", "win_pct", default=0.5)

    home_win_pct_home = _home_win_pct(results.get("splits_home", {}))
    away_win_pct_away = _away_win_pct(results.get("splits_away", {}))

    # ── Recent form (last 10 win pct) ──
    def _form(form_result):
        return _safe(form_result, "win_pct", default=0.5)

    form_home = _form(results.get("recent_form_home", {}))
    form_away = _form(results.get("recent_form_away", {}))

    # ── Net scoring margin ──
    def _net_rtg(stats_result):
        avgs = _safe(stats_result, "averages", default=[])
        if avgs:
            pts = _safe(avgs[0], "pts", default=None)
            pts_allowed = _safe(avgs[0], "pts_allowed", default=None)
            if pts is not None and pts_allowed is not None:
                return float(pts) - float(pts_allowed)
        return 0.0

    net_rtg_home = _net_rtg(results.get("team_stats_home", {}))
    net_rtg_away = _net_rtg(results.get("team_stats_away", {}))

    # ── H2H win pct ──
    def _h2h_win_pct(h2h_result, home_team_name):
        record = _safe(h2h_result, "h2h_record", default={})
        if not record:
            return 0.5
        # Find the key that matches home team
        home_wins = None
        total = 0
        for team_name, wins in record.items():
            total += wins
            if home_team_name.lower() in team_name.lower() or team_name.lower() in home_team_name.lower():
                home_wins = wins
        if home_wins is not None and total > 0:
            return home_wins / total
        return 0.5

    h2h_win_pct = _h2h_win_pct(results.get("h2h", {}), home_team)

    return {
        "elo_home":            float(elo_home),
        "elo_away":            float(elo_away),
        "rest_home":           float(rest_home),
        "rest_away":           float(rest_away),
        "season_win_pct_home": float(season_win_pct_home),
        "season_win_pct_away": float(season_win_pct_away),
        "home_win_pct_home":   float(home_win_pct_home),
        "away_win_pct_away":   float(away_win_pct_away),
        "form_home":           float(form_home),
        "form_away":           float(form_away),
        "net_rtg_home":        float(net_rtg_home),
        "net_rtg_away":        float(net_rtg_away),
        "h2h_win_pct":         float(h2h_win_pct),
    }


# ── Entry point: train from command line ───────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    train()
