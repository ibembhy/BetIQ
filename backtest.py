"""
BetIQ — Historical backtester.

Strategy:
  For each game (in chronological order), use each team's rolling
  point differential over their last N games to estimate a win
  probability.  Compare to the market-implied probability from the
  moneyline.  If the edge ≥ threshold, simulate a Half-Kelly bet.

No Claude API calls — pure Python/pandas, runs in seconds.
"""

import math
import numpy as np
import pandas as pd

# ── Constants ─────────────────────────────────────────────────────────────────

ODDS_FILE         = "data/archive/nba_2008-2025.csv"
STARTING_BANKROLL = 1000.0
EDGE_THRESHOLD    = 5.0   # percent — default, overridable
KELLY_CAP         = 0.12  # max 12% of bankroll per bet
ROLLING_WINDOW    = 10    # games of recent form
HOME_ADVANTAGE    = 3.0   # points — league-wide historical average
GAME_STD_DEV      = 12.0  # std dev of NBA game margins (historical ~12 pts)

# ── Team name mapping ─────────────────────────────────────────────────────────

ABBREV_TO_NAME = {
    "atl":  "Atlanta Hawks",
    "bkn":  "Brooklyn Nets",
    "bos":  "Boston Celtics",
    "cha":  "Charlotte Hornets",
    "chi":  "Chicago Bulls",
    "cle":  "Cleveland Cavaliers",
    "dal":  "Dallas Mavericks",
    "den":  "Denver Nuggets",
    "det":  "Detroit Pistons",
    "gs":   "Golden State Warriors",
    "hou":  "Houston Rockets",
    "ind":  "Indiana Pacers",
    "lac":  "LA Clippers",
    "lal":  "Los Angeles Lakers",
    "mem":  "Memphis Grizzlies",
    "mia":  "Miami Heat",
    "mil":  "Milwaukee Bucks",
    "min":  "Minnesota Timberwolves",
    "no":   "New Orleans Pelicans",
    "ny":   "New York Knicks",
    "okc":  "Oklahoma City Thunder",
    "orl":  "Orlando Magic",
    "phi":  "Philadelphia 76ers",
    "phx":  "Phoenix Suns",
    "por":  "Portland Trail Blazers",
    "sa":   "San Antonio Spurs",
    "sac":  "Sacramento Kings",
    "tor":  "Toronto Raptors",
    "utah": "Utah Jazz",
    "wsh":  "Washington Wizards",
}

# ── Math helpers ──────────────────────────────────────────────────────────────

def _norm_cdf(x: float) -> float:
    """Standard normal CDF using math.erf (no scipy needed)."""
    return (1 + math.erf(x / math.sqrt(2))) / 2


def _implied_prob(american_odds: float) -> float:
    """American odds → raw implied probability (includes vig)."""
    if american_odds > 0:
        return 100 / (american_odds + 100)
    return abs(american_odds) / (abs(american_odds) + 100)


def _win_prob(home_avg_diff: float, away_avg_diff: float) -> float:
    """
    Estimate home team win probability from rolling point differentials.
    Uses a normal distribution over expected margin.
      expected_margin = home_avg_diff - away_avg_diff + HOME_ADVANTAGE
    """
    expected = (home_avg_diff - away_avg_diff) + HOME_ADVANTAGE
    return _norm_cdf(expected / GAME_STD_DEV)


def _kelly_stake(
    bankroll: float, edge_pct: float, american_odds: float
) -> tuple[float, float]:
    """Half-Kelly stake. Returns (dollar_stake, fraction_pct)."""
    if american_odds > 0:
        b = american_odds / 100
        impl = 100 / (american_odds + 100)
    else:
        b = 100 / abs(american_odds)
        impl = abs(american_odds) / (abs(american_odds) + 100)
    p = impl + edge_pct / 100
    kelly = max((p * b - (1 - p)) / b, 0)
    fraction = min(kelly * 0.5, KELLY_CAP)
    fraction = max(fraction, 0.01)
    return round(bankroll * fraction, 2), round(fraction * 100, 2)


def _pnl(stake: float, american_odds: float, won: bool) -> float:
    if not won:
        return -stake
    if american_odds > 0:
        return round(stake * american_odds / 100, 2)
    return round(stake * 100 / abs(american_odds), 2)

# ── Core backtest ─────────────────────────────────────────────────────────────

def run_backtest(
    edge_threshold: float = EDGE_THRESHOLD,
    rolling_window: int   = ROLLING_WINDOW,
    regular_only:   bool  = True,
    seasons:        list  = None,      # e.g. [2020, 2021, 2022]
) -> tuple[pd.DataFrame, float]:
    """
    Run the full backtest.

    Returns:
        bets_df      — DataFrame of every simulated bet
        final_bankroll — ending balance after all bets
    """
    df = pd.read_csv(ODDS_FILE)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Filters
    if regular_only:
        df = df[df["regular"] == True]
    if seasons:
        df = df[df["season"].isin(seasons)]

    # Drop rows with no moneyline data
    df = df.dropna(subset=["moneyline_away", "moneyline_home"])

    # Rolling point differential tracker: team_abbrev → list of margins
    team_diffs: dict[str, list[float]] = {}

    bets      = []
    bankroll  = STARTING_BANKROLL

    for _, game in df.iterrows():
        home       = game["home"]
        away       = game["away"]
        home_score = game["score_home"]
        away_score = game["score_away"]
        home_margin = home_score - away_score  # positive = home won

        home_recent = team_diffs.get(home, [])
        away_recent = team_diffs.get(away, [])

        # Only bet once both teams have enough history
        if len(home_recent) >= rolling_window and len(away_recent) >= rolling_window:
            home_avg = float(np.mean(home_recent[-rolling_window:]))
            away_avg = float(np.mean(away_recent[-rolling_window:]))

            home_win_prob = _win_prob(home_avg, away_avg)
            away_win_prob = 1 - home_win_prob

            # Remove vig: normalise implied probs to sum to 1
            raw_home_impl = _implied_prob(game["moneyline_home"])
            raw_away_impl = _implied_prob(game["moneyline_away"])
            total_impl    = raw_home_impl + raw_away_impl
            home_impl     = raw_home_impl / total_impl
            away_impl     = raw_away_impl / total_impl

            home_edge = (home_win_prob - home_impl) * 100
            away_edge = (away_win_prob - away_impl) * 100

            actual_home_win = home_score > away_score

            for side, edge, ml, won in [
                ("home", home_edge, game["moneyline_home"], actual_home_win),
                ("away", away_edge, game["moneyline_away"], not actual_home_win),
            ]:
                if edge >= edge_threshold and bankroll >= 10:
                    stake, kelly_pct = _kelly_stake(bankroll, edge, ml)
                    stake = min(stake, bankroll)   # never bet more than remaining bankroll
                    profit = _pnl(stake, ml, won)
                    bankroll = round(bankroll + profit, 2)

                    bets.append({
                        "date":        game["date"].strftime("%Y-%m-%d"),
                        "season":      int(game["season"]),
                        "home":        ABBREV_TO_NAME.get(home, home),
                        "away":        ABBREV_TO_NAME.get(away, away),
                        "pick":        f"{ABBREV_TO_NAME.get(home if side == 'home' else away, side)} ML",
                        "side":        side,
                        "edge":        round(edge, 2),
                        "odds":        int(ml),
                        "stake":       stake,
                        "kelly_pct":   kelly_pct,
                        "won":         won,
                        "pnl":         profit,
                        "bankroll":    bankroll,
                    })

        # Update rolling history AFTER recording the bet (no lookahead)
        team_diffs.setdefault(home, []).append(float(home_margin))
        team_diffs.setdefault(away, []).append(float(-home_margin))

    return pd.DataFrame(bets), bankroll


# ── Summary helpers ───────────────────────────────────────────────────────────

def summary(bets_df: pd.DataFrame, final_bankroll: float) -> dict:
    if bets_df.empty:
        return {}
    wins        = int(bets_df["won"].sum())
    total       = len(bets_df)
    total_staked = bets_df["stake"].sum()
    total_pnl   = bets_df["pnl"].sum()
    return {
        "total_bets":     total,
        "wins":           wins,
        "losses":         total - wins,
        "win_rate":       round(wins / total * 100, 1),
        "total_pnl":      round(total_pnl, 2),
        "roi":            round(total_pnl / total_staked * 100, 2),
        "final_bankroll": round(final_bankroll, 2),
        "return_pct":     round((final_bankroll - STARTING_BANKROLL) / STARTING_BANKROLL * 100, 1),
        "avg_edge":       round(bets_df["edge"].mean(), 2),
        "avg_odds":       round(bets_df["odds"].mean(), 1),
    }


def season_breakdown(bets_df: pd.DataFrame) -> pd.DataFrame:
    if bets_df.empty:
        return pd.DataFrame()
    grp = bets_df.groupby("season").apply(lambda g: pd.Series({
        "bets":     len(g),
        "wins":     int(g["won"].sum()),
        "win_rate": round(g["won"].mean() * 100, 1),
        "pnl":      round(g["pnl"].sum(), 2),
        "roi":      round(g["pnl"].sum() / g["stake"].sum() * 100, 2),
    }), include_groups=False).reset_index()
    return grp


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Running BetIQ backtest (2008–2025, regular season)…")
    bets_df, final_bankroll = run_backtest()
    s = summary(bets_df, final_bankroll)

    print(f"\n{'='*50}")
    print(f"  Total bets   : {s['total_bets']}")
    print(f"  Win rate     : {s['win_rate']}%")
    print(f"  ROI          : {s['roi']}%")
    print(f"  Net P&L      : ${s['total_pnl']:+,.2f}")
    print(f"  Final bankroll: ${s['final_bankroll']:,.2f}  ({s['return_pct']:+.1f}%)")
    print(f"  Avg edge     : {s['avg_edge']}%")
    print(f"{'='*50}\n")

    print("Season breakdown:")
    print(season_breakdown(bets_df).to_string(index=False))
