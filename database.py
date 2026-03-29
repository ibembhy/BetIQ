"""
BetIQ — SQLite database layer.
Tables: bankroll, bets, agent_notes
"""

import sqlite3
import os
from datetime import datetime, timezone

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "betiq.db")


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_connection()
    c = conn.cursor()

    c.execute("""
        CREATE TABLE IF NOT EXISTS bankroll (
            id         INTEGER PRIMARY KEY,
            balance    REAL    NOT NULL DEFAULT 1000.0,
            updated_at TEXT    NOT NULL
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS bets (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            game_date        TEXT    NOT NULL,
            matchup          TEXT    NOT NULL,
            pick             TEXT    NOT NULL,
            bet_type         TEXT    NOT NULL,
            odds             INTEGER NOT NULL,
            stake            REAL    NOT NULL,
            potential_payout REAL    NOT NULL,
            confidence       TEXT    NOT NULL,
            edge             REAL    NOT NULL,
            reasoning        TEXT,
            status           TEXT    NOT NULL DEFAULT 'open',
            pnl              REAL    DEFAULT 0.0,
            placed_at        TEXT    NOT NULL,
            resolved_at      TEXT,
            model_prob       REAL,
            market_implied_prob REAL,
            fair_prob_no_vig REAL,
            edge_pct         REAL,
            ev               REAL,
            stake_pct        REAL,
            stake_amount     REAL,
            data_quality_score REAL,
            decision         TEXT,
            llm_edge_pct     REAL
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS agent_notes (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            note_type  TEXT NOT NULL,
            content    TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS odds_snapshots (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            captured_at     TEXT NOT NULL,
            home_team       TEXT NOT NULL,
            away_team       TEXT NOT NULL,
            home_ml         INTEGER,
            away_ml         INTEGER,
            home_spread     REAL,
            home_spread_price INTEGER,
            away_spread_price INTEGER,
            total_line      REAL,
            over_price      INTEGER,
            under_price     INTEGER
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS api_usage (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            api        TEXT NOT NULL,
            endpoint   TEXT NOT NULL,
            cached     INTEGER NOT NULL DEFAULT 0,
            called_at  TEXT NOT NULL
        )
    """)

    # Add columns to existing databases that predate these features
    for col, definition in [
        ("closing_odds",      "INTEGER"),
        ("clv",               "REAL"),
        ("cancel_reason",     "TEXT"),
        ("replaces_bet_id",   "INTEGER"),
        ("betfair_bet_id",    "TEXT"),
        ("betfair_market_id", "TEXT"),
        ("bet_report",        "TEXT"),
        ("edge_type",         "TEXT"),
        ("model_prob",        "REAL"),
        ("market_implied_prob", "REAL"),
        ("fair_prob_no_vig",  "REAL"),
        ("edge_pct",          "REAL"),
        ("ev",                "REAL"),
        ("stake_pct",         "REAL"),
        ("stake_amount",      "REAL"),
        ("data_quality_score", "REAL"),
        ("decision",          "TEXT"),
        ("llm_edge_pct",      "REAL"),
    ]:
        try:
            c.execute(f"ALTER TABLE bets ADD COLUMN {col} {definition}")
        except Exception:
            pass

    try:
        c.execute("ALTER TABLE odds_snapshots ADD COLUMN bookmaker_odds TEXT")
    except Exception:
        pass  # Column already exists

    for col, definition in [
        ("input_tokens",  "INTEGER DEFAULT 0"),
        ("output_tokens", "INTEGER DEFAULT 0"),
    ]:
        try:
            c.execute(f"ALTER TABLE api_usage ADD COLUMN {col} {definition}")
        except Exception:
            pass

    c.execute("""
        CREATE TABLE IF NOT EXISTS candidate_bets (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            game_date  TEXT    NOT NULL,
            matchup    TEXT    NOT NULL,
            pick       TEXT    NOT NULL,
            bet_type   TEXT    NOT NULL,
            odds       INTEGER NOT NULL,
            edge_pct   REAL    NOT NULL,
            confidence TEXT    NOT NULL,
            skip_reason TEXT   NOT NULL,
            reasoning  TEXT,
            logged_at  TEXT    NOT NULL,
            model_prob REAL,
            market_implied_prob REAL,
            fair_prob_no_vig REAL,
            ev         REAL,
            stake_pct  REAL,
            stake_amount REAL,
            data_quality_score REAL,
            decision   TEXT,
            llm_edge_pct REAL
        )
    """)

    for col, definition in [
        ("model_prob", "REAL"),
        ("market_implied_prob", "REAL"),
        ("fair_prob_no_vig", "REAL"),
        ("ev", "REAL"),
        ("stake_pct", "REAL"),
        ("stake_amount", "REAL"),
        ("data_quality_score", "REAL"),
        ("decision", "TEXT"),
        ("llm_edge_pct", "REAL"),
    ]:
        try:
            c.execute(f"ALTER TABLE candidate_bets ADD COLUMN {col} {definition}")
        except Exception:
            pass

    c.execute("""
        CREATE TABLE IF NOT EXISTS elo_ratings (
            team_id      INTEGER PRIMARY KEY,
            team_name    TEXT    NOT NULL,
            rating       REAL    NOT NULL DEFAULT 1500.0,
            season       INTEGER NOT NULL,
            games_played INTEGER NOT NULL DEFAULT 0,
            updated_at   TEXT    NOT NULL
        )
    """)

    # Seed bankroll on first run
    c.execute("SELECT COUNT(*) FROM bankroll")
    if c.fetchone()[0] == 0:
        c.execute(
            "INSERT INTO bankroll (balance, updated_at) VALUES (?, ?)",
            (1000.0, datetime.now(timezone.utc).isoformat()),
        )

    conn.commit()
    conn.close()


# ── Bankroll ──────────────────────────────────────────────────────────────────

def get_balance() -> float:
    conn = get_connection()
    row = conn.execute("SELECT balance FROM bankroll ORDER BY id DESC LIMIT 1").fetchone()
    conn.close()
    return row["balance"] if row else 1000.0


def update_balance(new_balance: float):
    conn = get_connection()
    conn.execute(
        "UPDATE bankroll SET balance=?, updated_at=? WHERE id=(SELECT MAX(id) FROM bankroll)",
        (round(new_balance, 2), datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()
    conn.close()


# ── Bets ──────────────────────────────────────────────────────────────────────

def insert_bet(bet: dict) -> int:
    conn = get_connection()
    c = conn.cursor()
    c.execute(
        """
        INSERT INTO bets
            (game_date, matchup, pick, bet_type, odds, stake, potential_payout,
             confidence, edge, reasoning, status, placed_at,
             model_prob, market_implied_prob, fair_prob_no_vig, edge_pct, ev,
             stake_pct, stake_amount, data_quality_score, decision, llm_edge_pct)
        VALUES (?,?,?,?,?,?,?,?,?,?,'open',?,?,?,?,?,?,?,?,?,?)
        """,
        (
            bet["game_date"], bet["matchup"], bet["pick"], bet["bet_type"],
            bet["odds"], bet["stake"], bet["potential_payout"],
            bet["confidence"], bet["edge"], bet["reasoning"],
            datetime.now(timezone.utc).isoformat(),
            bet.get("model_prob"),
            bet.get("market_implied_prob"),
            bet.get("fair_prob_no_vig"),
            bet.get("edge_pct", bet.get("edge")),
            bet.get("ev"),
            bet.get("stake_pct"),
            bet.get("stake_amount", bet.get("stake")),
            bet.get("data_quality_score"),
            bet.get("decision"),
            bet.get("llm_edge_pct"),
        ),
    )
    bet_id = c.lastrowid
    conn.commit()
    conn.close()
    return bet_id


def set_betfair_ids(bet_id: int, betfair_bet_id: str, betfair_market_id: str):
    conn = get_connection()
    conn.execute(
        "UPDATE bets SET betfair_bet_id=?, betfair_market_id=? WHERE id=?",
        (betfair_bet_id, betfair_market_id, bet_id),
    )
    conn.commit()
    conn.close()


def set_bet_report(bet_id: int, report: str):
    conn = get_connection()
    conn.execute("UPDATE bets SET bet_report=? WHERE id=?", (report, bet_id))
    conn.commit()
    conn.close()


def log_api_call(api: str, endpoint: str, cached: bool = False,
                 input_tokens: int = 0, output_tokens: int = 0):
    try:
        conn = get_connection()
        conn.execute(
            "INSERT INTO api_usage (api, endpoint, cached, called_at, input_tokens, output_tokens) VALUES (?, ?, ?, ?, ?, ?)",
            (api, endpoint, 1 if cached else 0, datetime.now(timezone.utc).isoformat(),
             input_tokens, output_tokens),
        )
        conn.commit()
        conn.close()
    except Exception:
        pass  # Never block main flow on logging


def get_api_usage() -> list:
    conn = get_connection()
    rows = [dict(r) for r in conn.execute(
        "SELECT * FROM api_usage ORDER BY called_at DESC"
    ).fetchall()]
    conn.close()
    return rows


def get_open_bets() -> list:
    conn = get_connection()
    rows = [_normalize_bet_row(dict(r)) for r in conn.execute(
        "SELECT * FROM bets WHERE status='open' ORDER BY placed_at DESC"
    ).fetchall()]
    conn.close()
    return rows


def get_all_bets() -> list:
    conn = get_connection()
    rows = [_normalize_bet_row(dict(r)) for r in conn.execute(
        "SELECT * FROM bets ORDER BY placed_at DESC"
    ).fetchall()]
    conn.close()
    return rows


def cancel_bet(bet_id: int, reason: str = "") -> dict | None:
    """Cancel an open bet and refund the stake. Returns the cancelled bet or None if not found."""
    conn = get_connection()
    row = conn.execute("SELECT * FROM bets WHERE id=? AND status='open'", (bet_id,)).fetchone()
    if not row:
        conn.close()
        return None
    bet = dict(row)
    conn.execute(
        "UPDATE bets SET status='cancelled', resolved_at=?, cancel_reason=? WHERE id=?",
        (datetime.now(timezone.utc).isoformat(), reason, bet_id),
    )
    conn.commit()
    conn.close()
    return bet


def link_replacement(new_bet_id: int, cancelled_bet_id: int):
    """Mark a bet as the replacement for a previously cancelled bet."""
    conn = get_connection()
    conn.execute(
        "UPDATE bets SET replaces_bet_id=? WHERE id=?",
        (cancelled_bet_id, new_bet_id),
    )
    conn.commit()
    conn.close()


def get_replaced_bets() -> list:
    """Return pairs of (cancelled bet, replacement bet) ordered by most recent."""
    conn = get_connection()
    rows = conn.execute(
        """
        SELECT
            old.id            AS old_id,
            old.matchup       AS old_matchup,
            old.pick          AS old_pick,
            old.bet_type      AS old_bet_type,
            old.odds          AS old_odds,
            old.edge          AS old_edge,
            old.cancel_reason AS reason,
            old.resolved_at   AS cancelled_at,
            new.id            AS new_id,
            new.matchup       AS new_matchup,
            new.pick          AS new_pick,
            new.bet_type      AS new_bet_type,
            new.odds          AS new_odds,
            new.edge          AS new_edge,
            new.status        AS new_status,
            new.pnl           AS new_pnl,
            new.placed_at     AS replaced_at
        FROM bets old
        JOIN bets new ON new.replaces_bet_id = old.id
        ORDER BY new.placed_at DESC
        """
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def update_bet_clv(bet_id: int, closing_odds: int, clv: float):
    """Store the closing line odds and computed CLV for a bet."""
    conn = get_connection()
    conn.execute(
        "UPDATE bets SET closing_odds=?, clv=? WHERE id=?",
        (closing_odds, round(clv, 4), bet_id),
    )
    conn.commit()
    conn.close()


def resolve_bet(bet_id: int, status: str, pnl: float):
    conn = get_connection()
    conn.execute(
        "UPDATE bets SET status=?, pnl=?, resolved_at=? WHERE id=?",
        (status, round(pnl, 2), datetime.now(timezone.utc).isoformat(), bet_id),
    )
    conn.commit()
    conn.close()


def get_weekly_pnl() -> dict:
    conn = get_connection()
    row = conn.execute("""
        SELECT
            COALESCE(SUM(CASE WHEN status='won'  THEN pnl ELSE 0 END),0) AS total_won,
            COALESCE(SUM(CASE WHEN status='lost' THEN pnl ELSE 0 END),0) AS total_lost,
            COUNT(CASE WHEN status='won'  THEN 1 END) AS wins,
            COUNT(CASE WHEN status='lost' THEN 1 END) AS losses,
            COALESCE(SUM(pnl),0) AS net_pnl
        FROM bets
        WHERE resolved_at >= datetime('now','-7 days')
          AND status IN ('won','lost')
    """).fetchone()
    conn.close()
    return dict(row) if row else {"wins": 0, "losses": 0, "net_pnl": 0}


def save_odds_snapshot(home_team: str, away_team: str, snapshot: dict):
    conn = get_connection()
    conn.execute(
        """
        INSERT INTO odds_snapshots
            (captured_at, home_team, away_team,
             home_ml, away_ml,
             home_spread, home_spread_price, away_spread_price,
             total_line, over_price, under_price, bookmaker_odds)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        (
            datetime.now(timezone.utc).isoformat(),
            home_team, away_team,
            snapshot.get("home_ml"), snapshot.get("away_ml"),
            snapshot.get("home_spread"), snapshot.get("home_spread_price"),
            snapshot.get("away_spread_price"),
            snapshot.get("total_line"), snapshot.get("over_price"),
            snapshot.get("under_price"), snapshot.get("bookmaker_odds"),
        ),
    )
    conn.commit()
    conn.close()


def get_odds_snapshots(home_team: str, away_team: str, limit: int = 10) -> list:
    conn = get_connection()
    rows = conn.execute(
        """
        SELECT * FROM odds_snapshots
        WHERE (home_team LIKE ? AND away_team LIKE ?)
           OR (home_team LIKE ? AND away_team LIKE ?)
        ORDER BY captured_at ASC
        LIMIT ?
        """,
        (f"%{home_team}%", f"%{away_team}%",
         f"%{away_team}%", f"%{home_team}%", limit),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def add_agent_note(note_type: str, content: str):
    conn = get_connection()
    conn.execute(
        "INSERT INTO agent_notes (note_type, content, created_at) VALUES (?,?,?)",
        (note_type, content, datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()
    conn.close()


def get_agent_notes(note_type: str = None, limit: int = 30) -> list:
    conn = get_connection()
    if note_type:
        rows = conn.execute(
            "SELECT * FROM agent_notes WHERE note_type=? ORDER BY created_at DESC LIMIT ?",
            (note_type, limit),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM agent_notes ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def save_candidate_bet(
    game_date: str,
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
):
    conn = get_connection()
    conn.execute(
        """
        INSERT INTO candidate_bets
            (game_date, matchup, pick, bet_type, odds, edge_pct,
             confidence, skip_reason, reasoning, logged_at,
             model_prob, market_implied_prob, fair_prob_no_vig, ev,
             stake_pct, stake_amount, data_quality_score, decision, llm_edge_pct)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        (
            game_date, matchup, pick, bet_type, odds,
            round(edge_pct, 2), confidence, skip_reason,
            reasoning, datetime.now(timezone.utc).isoformat(),
            model_prob, market_implied_prob, fair_prob_no_vig, ev,
            stake_pct, stake_amount, data_quality_score, decision, llm_edge_pct,
        ),
    )
    conn.commit()
    conn.close()


def get_candidate_bets(limit: int = 50) -> list:
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM candidate_bets ORDER BY logged_at DESC LIMIT ?",
        (limit,),
    ).fetchall()
    conn.close()
    return [_normalize_candidate_row(dict(r)) for r in rows]


def _normalize_bet_row(row: dict) -> dict:
    row["edge_pct"] = row.get("edge_pct", row.get("edge"))
    if row["edge_pct"] is None:
        row["edge_pct"] = row.get("edge")
    row["stake_amount"] = row.get("stake_amount", row.get("stake"))
    if row["stake_amount"] is None:
        row["stake_amount"] = row.get("stake")
    if row.get("stake_pct") is None and row.get("stake") and row.get("placed_at"):
        row["stake_pct"] = None
    if row.get("decision") is None:
        row["decision"] = "BET" if row.get("status") != "cancelled" else "PASS"
    return row


def _normalize_candidate_row(row: dict) -> dict:
    row["stake_amount"] = row.get("stake_amount", 0.0) or 0.0
    row["stake_pct"] = row.get("stake_pct", 0.0) or 0.0
    if row.get("decision") is None:
        row["decision"] = "LEAN" if (row.get("edge_pct") or 0) >= 2.0 else "PASS"
    return row


# ── Elo ratings ────────────────────────────────────────────────────────────────

def get_elo_rating(team_id: int) -> dict | None:
    conn = get_connection()
    row = conn.execute("SELECT * FROM elo_ratings WHERE team_id=?", (team_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def upsert_elo_rating(team_id: int, team_name: str, rating: float, season: int, games_played: int):
    conn = get_connection()
    conn.execute(
        """
        INSERT INTO elo_ratings (team_id, team_name, rating, season, games_played, updated_at)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(team_id) DO UPDATE SET
            team_name=excluded.team_name,
            rating=excluded.rating,
            season=excluded.season,
            games_played=excluded.games_played,
            updated_at=excluded.updated_at
        """,
        (team_id, team_name, round(rating, 2), season, games_played,
         datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()
    conn.close()


def get_all_elo_ratings() -> list:
    conn = get_connection()
    rows = conn.execute("SELECT * FROM elo_ratings ORDER BY rating DESC").fetchall()
    conn.close()
    return [dict(r) for r in rows]
