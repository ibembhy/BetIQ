"""
BetIQ — Streamlit frontend.
Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from datetime import datetime

import database as db
import tools as t
import reporter
from agent import run_agent

load_dotenv()
db.init_db()

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="BetIQ · NBA Betting Analyst",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ────────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ── Header ── */
.betiq-header {
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 18px 0 10px 0;
    border-bottom: 1px solid #252540;
    margin-bottom: 20px;
}
.betiq-logo {
    font-size: 2rem;
    line-height: 1;
}
.betiq-title {
    font-size: 1.6rem;
    font-weight: 700;
    color: #f1f5f9;
    letter-spacing: -0.5px;
}
.betiq-subtitle {
    font-size: 0.8rem;
    color: #64748b;
    margin-top: 2px;
}

/* ── Stat cards ── */
.stat-row {
    display: flex;
    gap: 10px;
    margin-bottom: 12px;
}
.stat-card {
    flex: 1;
    background: #1a1a2e;
    border: 1px solid #252540;
    border-radius: 10px;
    padding: 12px 14px;
    text-align: center;
}
.stat-value {
    font-size: 1.3rem;
    font-weight: 700;
    color: #f1f5f9;
}
.stat-label {
    font-size: 0.7rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-top: 2px;
}
.stat-value.green { color: #22c55e; }
.stat-value.red   { color: #ef4444; }
.stat-value.orange{ color: #f97316; }

/* ── Bet cards ── */
.bet-card {
    background: #1a1a2e;
    border: 1px solid #252540;
    border-radius: 12px;
    padding: 14px 16px;
    margin-bottom: 10px;
    line-height: 1.6;
    transition: border-color 0.2s;
}
.bet-card:hover { border-color: #3b3b6b; }
.bet-card.open   { border-left: 3px solid #3b82f6; }
.bet-card.won    { border-left: 3px solid #22c55e; }
.bet-card.lost   { border-left: 3px solid #ef4444; }
.bet-card.cancelled { border-left: 3px solid #64748b; }

.bet-pick   { font-size: 0.95rem; font-weight: 600; color: #f1f5f9; }
.bet-meta   { font-size: 0.78rem; color: #94a3b8; margin-top: 4px; }
.bet-badge  {
    display: inline-block;
    font-size: 0.68rem;
    font-weight: 600;
    padding: 2px 8px;
    border-radius: 20px;
    margin-right: 4px;
}
.badge-open  { background: #1e3a5f; color: #60a5fa; }
.badge-won   { background: #14532d; color: #4ade80; }
.badge-lost  { background: #450a0a; color: #f87171; }
.badge-high  { background: #431407; color: #fb923c; }
.badge-med   { background: #1e3a5f; color: #60a5fa; }
.badge-edge  { background: #1e1b4b; color: #a78bfa; }

/* ── Game cards ── */
.game-card {
    background: #1a1a2e;
    border: 1px solid #252540;
    border-radius: 10px;
    padding: 10px 14px;
    margin-bottom: 8px;
    font-size: 0.82rem;
}
.game-teams { font-weight: 600; color: #e2e8f0; }
.game-time  { color: #64748b; font-size: 0.75rem; }

/* ── Quick prompt chips ── */
.chip-row {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-bottom: 16px;
}

/* ── Scanner status ── */
.scanner-status {
    display: flex;
    align-items: center;
    gap: 8px;
    background: #1a1a2e;
    border: 1px solid #252540;
    border-radius: 8px;
    padding: 8px 12px;
    font-size: 0.78rem;
    color: #94a3b8;
    margin-bottom: 10px;
}
.dot-green { color: #22c55e; font-size: 0.6rem; }
.dot-grey  { color: #475569; font-size: 0.6rem; }

/* ── Section headers ── */
.section-header {
    font-size: 0.75rem;
    font-weight: 600;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin: 16px 0 8px 0;
}

/* Hide streamlit branding ── */
#MainMenu, footer { visibility: hidden; }

/* ── Today dashboard ── */
.dash-card {
    background: #1a1a2e;
    border: 1px solid #252540;
    border-radius: 16px;
    padding: 20px 24px 16px 24px;
    margin-bottom: 14px;
    transition: border-color 0.2s;
}
.dash-card:hover { border-color: #3b3b6b; }
.dash-card.has-bet { border-left: 3px solid #3b82f6; }

.dash-team-block {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 6px;
}
.dash-logo {
    width: 72px;
    height: 72px;
    object-fit: contain;
    filter: drop-shadow(0 2px 6px rgba(0,0,0,0.5));
}
.dash-team-name {
    font-size: 0.95rem;
    font-weight: 700;
    color: #f1f5f9;
    text-align: center;
    line-height: 1.2;
}
.dash-record {
    font-size: 0.78rem;
    color: #64748b;
    text-align: center;
}
.dash-vs {
    font-size: 1.1rem;
    font-weight: 700;
    color: #334155;
    text-align: center;
}
.dash-score-block {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 4px;
}
.dash-score {
    font-size: 1.8rem;
    font-weight: 800;
    color: #f1f5f9;
    line-height: 1;
}
.dash-score-sep {
    font-size: 0.9rem;
    color: #475569;
}
.live-pill {
    display: inline-block;
    background: #dc2626;
    color: #fff;
    font-size: 0.65rem;
    font-weight: 700;
    padding: 2px 8px;
    border-radius: 20px;
    letter-spacing: 0.5px;
    animation: pulse 1.5s infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0.6; }
}
.final-pill {
    display: inline-block;
    background: #1e293b;
    color: #94a3b8;
    font-size: 0.65rem;
    font-weight: 600;
    padding: 2px 8px;
    border-radius: 20px;
}
.game-time {
    font-size: 0.78rem;
    color: #64748b;
    text-align: center;
    margin-bottom: 14px;
}
.odds-strip {
    display: flex;
    justify-content: space-around;
    background: #0f0f1e;
    border-radius: 10px;
    padding: 10px 0;
    margin-top: 16px;
}
.odds-cell {
    text-align: center;
    flex: 1;
}
.odds-cell + .odds-cell {
    border-left: 1px solid #1e1e3a;
}
.odds-label {
    font-size: 0.65rem;
    color: #475569;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 3px;
}
.odds-value {
    font-size: 0.82rem;
    font-weight: 600;
    color: #cbd5e1;
}
.agent-bet-row {
    margin-top: 10px;
    display: flex;
    gap: 6px;
    flex-wrap: wrap;
}
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def pnl_color(val: float) -> str:
    return "green" if val > 0 else ("red" if val < 0 else "")


def _to_decimal(odds: int) -> float:
    """American odds → decimal (e.g. -110 → 1.909, +150 → 2.5)."""
    if odds > 0:
        return round(1 + odds / 100, 3)
    return round(1 + 100 / abs(odds), 3)


def _to_fractional(odds: int) -> str:
    """American odds → simplified fractional string (e.g. +150 → '3/2', -110 → '10/11')."""
    from math import gcd
    if odds > 0:
        num, den = odds, 100
    else:
        num, den = 100, abs(odds)
    g = gcd(num, den)
    return f"{num // g}/{den // g}"


def format_odds(odds: int, fmt: str = "American") -> str:
    if fmt == "Decimal":
        return f"{_to_decimal(odds):.2f}x"
    if fmt == "Fractional":
        return _to_fractional(odds)
    return f"+{odds}" if odds > 0 else str(odds)


_ESPN = {
    "Atlanta Hawks": "atl", "Boston Celtics": "bos", "Brooklyn Nets": "bkn",
    "Charlotte Hornets": "cha", "Chicago Bulls": "chi", "Cleveland Cavaliers": "cle",
    "Dallas Mavericks": "dal", "Denver Nuggets": "den", "Detroit Pistons": "det",
    "Golden State Warriors": "gs", "Houston Rockets": "hou", "Indiana Pacers": "ind",
    "LA Clippers": "lac", "Los Angeles Lakers": "lal", "Memphis Grizzlies": "mem",
    "Miami Heat": "mia", "Milwaukee Bucks": "mil", "Minnesota Timberwolves": "min",
    "New Orleans Pelicans": "no", "New York Knicks": "ny", "Oklahoma City Thunder": "okc",
    "Orlando Magic": "orl", "Philadelphia 76ers": "phi", "Phoenix Suns": "phx",
    "Portland Trail Blazers": "por", "Sacramento Kings": "sac", "San Antonio Spurs": "sa",
    "Toronto Raptors": "tor", "Utah Jazz": "utah", "Washington Wizards": "wsh",
}

def team_logo(name: str) -> str:
    abbrev = _ESPN.get(name)
    if not abbrev:
        # fuzzy fallback — match on last word of team name
        last = name.split()[-1].lower()
        abbrev = next((v for k, v in _ESPN.items() if k.split()[-1].lower() == last), None)
    if not abbrev:
        return ""
    return f"https://a.espncdn.com/i/teamlogos/nba/500/{abbrev}.png"


def fo(odds: int) -> str:
    """format_odds using the session-state-selected format."""
    label = st.session_state.get("odds_fmt", "American (+/-)")
    if "Decimal" in label:
        return format_odds(odds, "Decimal")
    if "Fractional" in label:
        return format_odds(odds, "Fractional")
    return format_odds(odds, "American")

def confidence_badge(conf: str) -> str:
    cls = "badge-high" if conf == "High" else "badge-med"
    return f'<span class="bet-badge {cls}">{conf}</span>'

def edge_badge(edge: float) -> str:
    return f'<span class="bet-badge badge-edge">Edge {edge:.1f}%</span>'

def status_badge(status: str) -> str:
    icons = {"open": "🔵 Open", "won": "✅ Won", "lost": "❌ Lost", "cancelled": "⚪ Cancelled", "push": "➡️ Push"}
    cls   = {"open": "badge-open", "won": "badge-won", "lost": "badge-lost"}.get(status, "badge-open")
    return f'<span class="bet-badge {cls}">{icons.get(status, status)}</span>'

def bet_stats(bets: list) -> dict:
    """Compute win rate, P&L, ROI from a list of resolved bets."""
    wins      = sum(1 for b in bets if b["status"] == "won")
    total_pnl = sum(b["pnl"] for b in bets)
    total_stk = sum(b["stake"] for b in bets)
    return {
        "wins":      wins,
        "losses":    len(bets) - wins,
        "total_pnl": total_pnl,
        "roi":       total_pnl / total_stk * 100 if total_stk else 0,
        "win_rate":  wins / len(bets) * 100 if bets else 0,
    }


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div class="betiq-header">
        <div class="betiq-logo">🏀</div>
        <div>
            <div class="betiq-title">BetIQ</div>
            <div class="betiq-subtitle">NBA Autonomous Betting Analyst</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Bankroll ──
    br       = t.get_bankroll()
    balance  = br["current_balance"]
    start    = br["starting_balance"]
    pnl_pct  = br["total_return_pct"]
    pnl_abs  = round(balance - start, 2)
    color    = pnl_color(pnl_abs)

    st.markdown('<div class="section-header">Bankroll</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="stat-row">
        <div class="stat-card">
            <div class="stat-value">${balance:,.2f}</div>
            <div class="stat-label">Balance</div>
        </div>
        <div class="stat-card">
            <div class="stat-value {color}">${pnl_abs:+.2f}</div>
            <div class="stat-label">P&L ({pnl_pct:+.1f}%)</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── This week ──
    weekly = db.get_weekly_pnl()
    wins   = weekly.get("wins", 0) or 0
    losses = weekly.get("losses", 0) or 0
    net    = weekly.get("net_pnl", 0) or 0

    st.markdown('<div class="section-header">This Week</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="stat-row">
        <div class="stat-card">
            <div class="stat-value green">{wins}</div>
            <div class="stat-label">Wins</div>
        </div>
        <div class="stat-card">
            <div class="stat-value red">{losses}</div>
            <div class="stat-label">Losses</div>
        </div>
        <div class="stat-card">
            <div class="stat-value {pnl_color(net)}">${net:+.2f}</div>
            <div class="stat-label">Net P&L</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Open bets ──
    open_bets = br["open_bets"]
    slots     = br["slots_remaining"]
    st.markdown(f'<div class="section-header">Open Bets &nbsp; {len(open_bets)}/5 slots used</div>', unsafe_allow_html=True)

    if not open_bets:
        st.caption("No open bets — scanner will place bets at next scan.")
    else:
        for bet in open_bets:
            st.markdown(f"""
            <div class="bet-card open">
                <div class="bet-pick">{bet['pick']}</div>
                <div class="bet-meta">{bet['matchup']}</div>
                <div style="margin-top:6px;">
                    {confidence_badge(bet['confidence'])}
                    {edge_badge(bet['edge'])}
                    <span class="bet-badge badge-open">{fo(bet['odds'])}</span>
                </div>
                <div class="bet-meta" style="margin-top:4px;">Stake: ${bet['stake']:.2f}</div>
            </div>""", unsafe_allow_html=True)

    if st.button("Check Results", use_container_width=True):
        with st.spinner("Checking…"):
            result = t.resolve_bets()
        if result.get("resolved_count", 0) > 0:
            st.success(f"Resolved {result['resolved_count']} bet(s)")
            for r in result.get("resolved", []):
                icon = "✅" if r["status"] == "won" else "❌"
                st.write(f"{icon} {r['pick']} — **{r['status'].upper()}** (${r['pnl']:+.2f})")
            st.rerun()
        else:
            st.caption("No completed games found yet.")

    # ── Next scan countdown ──
    st.markdown('<div class="section-header">Next Scan</div>', unsafe_allow_html=True)
    import streamlit.components.v1 as _components
    _components.html("""
<style>
  #countdown-wrap {
    background: #1a1a2e;
    border: 1px solid #252540;
    border-radius: 10px;
    padding: 10px 14px;
    text-align: center;
    font-family: 'Inter', sans-serif;
  }
  #scan-label {
    font-size: 0.68rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 4px;
  }
  #scan-name {
    font-size: 0.75rem;
    color: #94a3b8;
    margin-bottom: 6px;
  }
  #countdown {
    font-size: 1.5rem;
    font-weight: 700;
    color: #f1f5f9;
    letter-spacing: 2px;
    font-variant-numeric: tabular-nums;
  }
  #dot { color: #22c55e; font-size: 0.55rem; vertical-align: middle; margin-right: 4px; }
</style>
<div id="countdown-wrap">
  <div id="scan-label"><span id="dot">&#9679;</span> Scanner running</div>
  <div id="scan-name"></div>
  <div id="countdown">--:--:--</div>
</div>
<script>
  const SCANS = [
    { hour: 14, minute: 0,  label: "Afternoon (2:00 PM)" },
    { hour: 18, minute: 0,  label: "Evening (6:00 PM)"   },
    { hour: 21, minute: 30, label: "Late (9:30 PM)"      },
  ];

  function nextScan() {
    const now = new Date();
    // Get current time in EST
    const estStr = now.toLocaleString("en-US", { timeZone: "America/New_York" });
    const est    = new Date(estStr);
    const h = est.getHours(), m = est.getMinutes(), s = est.getSeconds();
    const nowSec = h * 3600 + m * 60 + s;

    for (const scan of SCANS) {
      const scanSec = scan.hour * 3600 + scan.minute * 60;
      if (scanSec > nowSec) {
        return { label: scan.label, diffSec: scanSec - nowSec };
      }
    }
    // Past 9:30 PM — next is tomorrow's 2 PM
    return { label: "Afternoon (2:00 PM)", diffSec: (14 * 3600) + (86400 - nowSec) };
  }

  function pad(n) { return String(n).padStart(2, "0"); }

  function tick() {
    const { label, diffSec } = nextScan();
    const h = Math.floor(diffSec / 3600);
    const m = Math.floor((diffSec % 3600) / 60);
    const s = diffSec % 60;
    document.getElementById("countdown").textContent = pad(h) + ":" + pad(m) + ":" + pad(s);
    document.getElementById("scan-name").textContent = label;
  }

  tick();
  setInterval(tick, 1000);
</script>
""", height=110)

    st.divider()
    odds_fmt = st.selectbox(
        "Odds format",
        ["American (+/-)", "Decimal (x)", "Fractional"],
        index=0,
        key="odds_fmt",
    )
    st.caption("Paper trading only · Not real money")


# ── Shared data (fetched once for all tabs) ───────────────────────────────────

all_bets = db.get_all_bets()
resolved = sorted(
    [b for b in all_bets if b["status"] in ("won", "lost")],
    key=lambda x: x.get("resolved_at") or "",
)
stats = bet_stats(resolved)

# ── Main tabs ─────────────────────────────────────────────────────────────────

tab_today, tab_chat, tab_history, tab_perf, tab_reports, tab_runners, tab_replaced = st.tabs(["🏀 Today", "💬 Chat", "📋 Bet History", "📈 Performance", "📄 Daily Reports", "👀 Runner-Up Bets", "🔄 Replaced Bets"])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Today
# ═══════════════════════════════════════════════════════════════════════════════

with tab_today:
    from datetime import date as _date_cls
    today_str = _date_cls.today().strftime("%A, %B %d")
    st.markdown(f"### {today_str}")

    with st.spinner("Loading today's games…"):
        games_today = t.get_todays_games().get("games", [])
        odds_today  = t.get_current_odds().get("games", [])

    # Build odds lookup: lower(home_team) → best_lines + per-book data
    odds_map  = {}
    books_map = {}
    for og in odds_today:
        key = og.get("home_team", "").lower()
        odds_map[key]  = og.get("best_lines", {})
        books_map[key] = og.get("books", {})

    # Build agent-bet lookup: games with open bets
    open_picks = {}  # lower matchup keyword → list of bet dicts
    for b in br["open_bets"]:
        for kw in b["matchup"].lower().split():
            open_picks.setdefault(kw, []).append(b)

    if not games_today:
        st.info("No NBA games scheduled today.")
    else:
        # Fetch team records (cached after first call)
        all_teams_today = {g["home_team"] for g in games_today} | {g["visitor_team"] for g in games_today}
        records = {}
        with st.spinner("Fetching team records…"):
            for team in all_teams_today:
                ts = t.get_team_stats(team)
                rec = ts.get("record", {})
                if rec:
                    records[team] = f"{rec.get('wins', 0)}-{rec.get('losses', 0)}"
                else:
                    records[team] = "—"

        for g in games_today:
            home    = g["home_team"]
            away    = g["visitor_team"]
            status  = g.get("status", "Scheduled")
            h_score = g.get("home_team_score", 0)
            a_score = g.get("visitor_team_score", 0)
            gtime   = g.get("time", "")

            is_live  = bool(status) and status not in ("Final", "Scheduled", "")
            is_final = status == "Final"

            # ── Odds ──
            lines = odds_map.get(home.lower(), {})
            ml    = lines.get("moneyline", {})
            sp    = lines.get("spread", {})
            tot   = lines.get("total", {})

            def _ml(team):
                v = ml.get(team)
                return fo(v) if v else "-"

            def _sp(team):
                info = sp.get(team, {})
                pt, pr = info.get("point"), info.get("price")
                if pt is None: return "-"
                return f"{pt:+.1f} ({fo(pr)})" if pr else f"{pt:+.1f}"

            over_pt  = tot.get("Over",  {}).get("point")
            over_pr  = tot.get("Over",  {}).get("price")
            under_pr = tot.get("Under", {}).get("price")
            total_str = f"O {over_pt} ({fo(over_pr)}) / U ({fo(under_pr)})" if over_pt else "-"

            away_last = away.split()[-1]
            home_last = home.split()[-1]
            ml_str    = f"{away_last} {_ml(away)} / {home_last} {_ml(home)}"
            sp_str    = f"{away_last} {_sp(away)} / {home_last} {_sp(home)}"

            # ── Agent bets ──
            game_bets, seen = [], set()
            for kw in (home_last.lower(), away_last.lower()):
                game_bets.extend(open_picks.get(kw, []))
            unique_bets = [b for b in game_bets
                           if not (b["id"] in seen or seen.add(b["id"]))]
            has_bet = bool(unique_bets)

            # ── Pre-build HTML pieces (no nested expressions in main card) ──
            border = "border-left:3px solid #3b82f6;" if has_bet else ""

            if is_live:
                status_pill = (
                    f'<span style="display:inline-block;background:#dc2626;color:#fff;'
                    f'font-size:0.65rem;font-weight:700;padding:2px 8px;border-radius:20px;">'
                    f'LIVE {gtime}</span>'
                )
            elif is_final:
                status_pill = (
                    '<span style="display:inline-block;background:#1e293b;color:#94a3b8;'
                    'font-size:0.65rem;font-weight:600;padding:2px 8px;border-radius:20px;">Final</span>'
                )
            else:
                status_pill = (
                    f'<span style="font-size:0.78rem;color:#64748b;">{status} {gtime}</span>'
                )

            if is_live or is_final:
                center_block = (
                    f'<div style="text-align:center;">'
                    f'<div style="font-size:1.8rem;font-weight:800;color:#f1f5f9;line-height:1;">{a_score}</div>'
                    f'<div style="font-size:0.75rem;color:#475569;margin:4px 0;">-</div>'
                    f'<div style="font-size:1.8rem;font-weight:800;color:#f1f5f9;line-height:1;">{h_score}</div>'
                    f'</div>'
                )
            else:
                center_block = (
                    '<div style="text-align:center;padding:20px 0;'
                    'font-size:1.1rem;font-weight:700;color:#334155;">VS</div>'
                )

            away_logo_url = team_logo(away)
            home_logo_url = team_logo(home)
            logo_s = "width:68px;height:68px;object-fit:contain;"
            away_img = (f'<img src="{away_logo_url}" style="{logo_s}">'
                        if away_logo_url else "")
            home_img = (f'<img src="{home_logo_url}" style="{logo_s}">'
                        if home_logo_url else "")

            away_rec = records.get(away, "-")
            home_rec = records.get(home, "-")

            bet_strip = ""
            for b in unique_bets:
                bet_strip += (
                    f'<span style="display:inline-block;font-size:0.68rem;font-weight:600;'
                    f'padding:2px 8px;border-radius:20px;background:#1e3a5f;color:#60a5fa;margin-right:4px;">'
                    f'BetIQ: {b["pick"]} {fo(b["odds"])} &middot; ${b["stake"]:.0f}</span>'
                )
            if bet_strip:
                bet_strip = f'<div style="margin-top:10px;">{bet_strip}</div>'

            # ── Compose final card (all variables, no logic) ──
            card = (
                f'<div style="background:#1a1a2e;border:1px solid #252540;border-radius:16px;'
                f'padding:20px 24px 16px;margin-bottom:14px;{border}">'

                f'<div style="text-align:center;margin-bottom:14px;">{status_pill}</div>'

                f'<div style="display:flex;align-items:center;justify-content:space-between;">'

                f'<div style="flex:1;display:flex;flex-direction:column;align-items:center;gap:6px;">'
                f'{away_img}'
                f'<div style="font-size:0.95rem;font-weight:700;color:#f1f5f9;text-align:center;">{away}</div>'
                f'<div style="font-size:0.78rem;color:#64748b;">{away_rec}</div>'
                f'</div>'

                f'<div style="flex:0 0 90px;">{center_block}</div>'

                f'<div style="flex:1;display:flex;flex-direction:column;align-items:center;gap:6px;">'
                f'{home_img}'
                f'<div style="font-size:0.95rem;font-weight:700;color:#f1f5f9;text-align:center;">{home}</div>'
                f'<div style="font-size:0.78rem;color:#64748b;">{home_rec}</div>'
                f'</div>'

                f'</div>'

                f'<div style="display:flex;justify-content:space-around;background:#0f0f1e;'
                f'border-radius:10px;padding:10px 0;margin-top:16px;">'
                f'<div style="text-align:center;flex:1;">'
                f'<div style="font-size:0.65rem;color:#475569;text-transform:uppercase;'
                f'letter-spacing:0.5px;margin-bottom:3px;">Moneyline</div>'
                f'<div style="font-size:0.82rem;font-weight:600;color:#cbd5e1;">{ml_str}</div>'
                f'</div>'
                f'<div style="text-align:center;flex:1;border-left:1px solid #1e1e3a;">'
                f'<div style="font-size:0.65rem;color:#475569;text-transform:uppercase;'
                f'letter-spacing:0.5px;margin-bottom:3px;">Spread</div>'
                f'<div style="font-size:0.82rem;font-weight:600;color:#cbd5e1;">{sp_str}</div>'
                f'</div>'
                f'<div style="text-align:center;flex:1;border-left:1px solid #1e1e3a;">'
                f'<div style="font-size:0.65rem;color:#475569;text-transform:uppercase;'
                f'letter-spacing:0.5px;margin-bottom:3px;">Total</div>'
                f'<div style="font-size:0.82rem;font-weight:600;color:#cbd5e1;">{total_str}</div>'
                f'</div>'
                f'</div>'

                f'{bet_strip}'
                f'</div>'
            )

            st.markdown(card, unsafe_allow_html=True)

            # ── Book comparison expander ──
            book_data = books_map.get(home.lower(), {})
            if book_data:
                with st.expander(f"📊 Line comparison — {len(book_data)} book(s)"):
                    rows = []
                    for book_name, bd in book_data.items():
                        home_ml = bd.get(f"ml_{home}")
                        away_ml = bd.get(f"ml_{away}")
                        rows.append({
                            "Book":        book_name,
                            f"{away} ML":  fo(away_ml) if away_ml else "—",
                            f"{home} ML":  fo(home_ml) if home_ml else "—",
                            f"{away} Spread": bd.get(f"spread_{away}", "—") or "—",
                            f"{home} Spread": bd.get(f"spread_{home}", "—") or "—",
                            "Over":  bd.get("total_Over",  "—") or "—",
                            "Under": bd.get("total_Under", "—") or "—",
                        })
                    if rows:
                        st.dataframe(
                            pd.DataFrame(rows).set_index("Book"),
                            use_container_width=True,
                        )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Chat

# ═══════════════════════════════════════════════════════════════════════════════

with tab_chat:
    st.markdown("### Ask BetIQ anything")

    # Session state
    if "messages"     not in st.session_state: st.session_state.messages     = []
    if "history"      not in st.session_state: st.session_state.history      = []
    if "show_thinking" not in st.session_state: st.session_state.show_thinking = False
    if "quick_prompt" not in st.session_state: st.session_state.quick_prompt  = None

    # ── Quick action chips ──
    st.markdown("**Quick actions**")
    col1, col2, col3, col4 = st.columns(4)
    if col1.button("🔍 Best bets tonight",    use_container_width=True):
        st.session_state.quick_prompt = "Scan all of today's NBA games and find the best bets with strong edges."
    if col2.button("💰 Bankroll report",       use_container_width=True):
        st.session_state.quick_prompt = "Give me a full bankroll report including open bets, recent performance, and win rate."
    if col3.button("✅ Resolve bets",          use_container_width=True):
        st.session_state.quick_prompt = "Resolve all my open bets and tell me the results."
    if col4.button("📊 Performance summary",   use_container_width=True):
        st.session_state.quick_prompt = "Summarize my overall betting performance, win rate, and what I should improve."

    st.session_state.show_thinking = st.toggle("Show agent reasoning", value=st.session_state.show_thinking)
    st.divider()

    # ── Chat history ──
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant" and msg.get("thinking") and st.session_state.show_thinking:
                with st.expander("Reasoning process", expanded=False):
                    for i, thought in enumerate(msg["thinking"], 1):
                        st.markdown(f"**Step {i}**\n\n{thought}")
                        if i < len(msg["thinking"]):
                            st.divider()
            st.markdown(msg["content"])

    # ── Handle quick prompt or typed input ──
    prompt = st.session_state.quick_prompt or st.chat_input("Ask about bets, bankroll, or any matchup…")
    st.session_state.quick_prompt = None

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing…"):
                try:
                    response_text, updated_history, thinking = run_agent(
                        prompt, st.session_state.history
                    )
                    st.session_state.history = updated_history
                except Exception as exc:
                    response_text = f"**Error:** {exc}\n\nCheck that your API keys are set in `.env`."
                    thinking = []

            if thinking and st.session_state.show_thinking:
                with st.expander("Reasoning process", expanded=False):
                    for i, thought in enumerate(thinking, 1):
                        st.markdown(f"**Step {i}**\n\n{thought}")
                        if i < len(thinking):
                            st.divider()
            st.markdown(response_text)

        st.session_state.messages.append({
            "role":     "assistant",
            "content":  response_text,
            "thinking": thinking,
        })
        # Keep last 50 messages to prevent unbounded memory growth
        if len(st.session_state.messages) > 50:
            st.session_state.messages = st.session_state.messages[-50:]
        st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Bet History
# ═══════════════════════════════════════════════════════════════════════════════

with tab_history:
    if not all_bets:
        st.info("No bets placed yet. Use the chat or wait for the next scheduled scan.")
    else:
        # ── Summary strip ──
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total Bets",  len(all_bets))
        c2.metric("Win Rate",    f"{stats['win_rate']:.1f}%")
        c3.metric("Net P&L",     f"${stats['total_pnl']:+.2f}", delta=f"{stats['roi']:.1f}% ROI")
        c4.metric("Open",        br["open_bets_count"])
        c5.metric("Slots Free",  br["slots_remaining"])

        st.divider()

        # ── Filter controls ──
        col_f1, col_f2, _ = st.columns([1, 1, 3])
        status_filter = col_f1.selectbox("Status", ["All", "Open", "Won", "Lost", "Cancelled"])
        type_filter   = col_f2.selectbox("Type",   ["All", "Moneyline", "Spread", "Total"])

        filtered = all_bets
        if status_filter != "All":
            filtered = [b for b in filtered if b["status"] == status_filter.lower()]
        if type_filter != "All":
            filtered = [b for b in filtered if b["bet_type"] == type_filter.lower()]

        st.markdown(f"**{len(filtered)} bets**")

        # ── Bet cards ──
        for b in filtered:
            pnl_str = f"${b['pnl']:+.2f}" if b["status"] not in ("open", "cancelled") else "—"
            st.markdown(f"""
            <div class="bet-card {b['status']}">
                <div style="display:flex; justify-content:space-between; align-items:flex-start;">
                    <div>
                        <div class="bet-pick">{b['pick']}</div>
                        <div class="bet-meta">{b['matchup']} &nbsp;·&nbsp; {b['game_date']}</div>
                    </div>
                    <div style="text-align:right;">
                        <div class="stat-value {pnl_color(b['pnl'])}" style="font-size:1.1rem;">{pnl_str}</div>
                        <div class="bet-meta">{b['bet_type'].title()} @ {fo(b['odds'])}</div>
                    </div>
                </div>
                <div style="margin-top:8px;">
                    {status_badge(b['status'])}
                    {confidence_badge(b['confidence'])}
                    {edge_badge(b['edge'])}
                    <span class="bet-badge badge-open">Stake ${b['stake']:.2f}</span>
                </div>
            </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Performance
# ═══════════════════════════════════════════════════════════════════════════════

with tab_perf:
    if not resolved:
        st.info("Performance analytics will appear here once bets are settled.")
    else:
        # ── Top KPIs ──
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Win Rate",      f"{stats['win_rate']:.1f}%")
        k2.metric("Net P&L",       f"${stats['total_pnl']:+.2f}")
        k3.metric("ROI",           f"{stats['roi']:.1f}%")
        k4.metric("Bets Resolved", len(resolved))

        st.divider()

        # ── Cumulative P&L chart ──
        st.markdown("### Bankroll Growth")
        running     = 1000.0
        chart_rows  = [{"Date": "", "Bankroll": 1000.0}]
        for b in resolved:
            running += b["pnl"]
            chart_rows.append({
                "Date":     (b.get("resolved_at") or "")[:10],
                "Bankroll": round(running, 2),
            })
        df_chart = pd.DataFrame(chart_rows[1:]).set_index("Date")
        st.line_chart(df_chart)

        st.divider()

        # ── Side by side: by confidence + by type ──
        left, right = st.columns(2)

        def _group_stats(bets, key, values, label_key):
            rows = []
            for val in values:
                group = [b for b in bets if b[key] == val]
                if group:
                    s = bet_stats(group)
                    rows.append({
                        label_key:  val.title() if key == "bet_type" else val,
                        "Bets":     len(group),
                        "Win %":    f"{s['win_rate']:.1f}%",
                        "P&L":      f"${s['total_pnl']:+.2f}",
                        "ROI":      f"{s['roi']:.1f}%",
                    })
            return rows

        with left:
            st.markdown("### By Confidence")
            rows = _group_stats(resolved, "confidence", ["High", "Medium"], "Confidence")
            if rows:
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        with right:
            st.markdown("### By Bet Type")
            rows = _group_stats(resolved, "bet_type", ["moneyline", "spread", "total"], "Type")
            if rows:
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        st.divider()

        # ── Last 10 results ──
        st.markdown("### Recent Results")
        for b in list(reversed(resolved))[:10]:
            st.markdown(f"""
            <div class="bet-card {b['status']}">
                <div style="display:flex; justify-content:space-between;">
                    <div>
                        <div class="bet-pick">{b['pick']}</div>
                        <div class="bet-meta">{b['matchup']} &nbsp;·&nbsp; {(b.get('resolved_at') or '')[:10]}</div>
                    </div>
                    <div style="text-align:right;">
                        <div class="stat-value {pnl_color(b['pnl'])}" style="font-size:1.1rem;">${b['pnl']:+.2f}</div>
                        <div class="bet-meta">{fo(b['odds'])} · ${b['stake']:.2f}</div>
                    </div>
                </div>
                <div style="margin-top:6px;">
                    {status_badge(b['status'])}
                    {confidence_badge(b['confidence'])}
                    {edge_badge(b['edge'])}
                </div>
            </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Daily Reports
# ═══════════════════════════════════════════════════════════════════════════════

with tab_reports:
    st.markdown("### Daily Betting Reports")
    st.caption("Reports are generated automatically when the last bet of each game day resolves.")

    # ── Manual generation ──
    from datetime import date as _date
    all_game_dates = sorted(
        {b["game_date"] for b in all_bets if b["status"] in ("won", "lost", "push", "cancelled") and b.get("game_date")},
        reverse=True,
    )
    if all_game_dates:
        gen_col1, gen_col2 = st.columns([2, 1])
        gen_date = gen_col1.selectbox("Generate report for", all_game_dates, key="gen_date_select")
        if gen_col2.button("Generate Report", use_container_width=True):
            with st.spinner(f"Calling BetIQ analyst to write {gen_date} report…"):
                # Force regeneration by removing existing txt if present
                import os
                txt_path = os.path.join(reporter.REPORTS_DIR, f"{gen_date}.txt")
                if os.path.exists(txt_path):
                    os.remove(txt_path)
                pdf_path = os.path.join(reporter.REPORTS_DIR, f"{gen_date}.pdf")
                if os.path.exists(pdf_path):
                    os.remove(pdf_path)
                ok = reporter.maybe_generate_report(gen_date)
            if ok:
                st.success(f"Report for {gen_date} generated!")
                st.rerun()
            else:
                st.warning("No resolved bets found for that date.")
    else:
        st.info("No resolved bets yet — reports will appear once bets are settled.")

    st.divider()
    reports = reporter.list_reports()

    if not reports:
        st.info("No reports yet. Reports appear here after a full betting day resolves.")
    else:
        # ── Report selector ──
        dates = [r["date"] for r in reports]
        selected_date = st.selectbox("Select report date", dates, index=0)
        selected = next(r for r in reports if r["date"] == selected_date)

        st.divider()

        # ── Download button ──
        if selected["pdf_path"]:
            with open(selected["pdf_path"], "rb") as pdf_file:
                st.download_button(
                    label="Download PDF",
                    data=pdf_file,
                    file_name=f"BetIQ_Report_{selected_date}.pdf",
                    mime="application/pdf",
                )

        # ── Report content ──
        text = reporter.get_report_text(selected_date)
        if text:
            for line in text.splitlines():
                stripped = line.strip()
                if not stripped:
                    st.write("")
                    continue
                if stripped.startswith("BETIQ DAILY REPORT"):
                    st.markdown(f"## {stripped}")
                elif (
                    stripped in ("OVERVIEW", "DAILY SUMMARY")
                    or stripped.startswith("BET:")
                    or stripped.startswith("WHY WE BET")
                    or stripped.startswith("RESULT:")
                    or stripped.startswith("SELF-EVALUATION")
                ):
                    st.markdown(f"**{stripped}**")
                else:
                    st.write(stripped)
        else:
            st.error("Could not read report file.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 — Runner-Up Bets
# ═══════════════════════════════════════════════════════════════════════════════

with tab_runners:
    st.markdown("### Runner-Up Bets")
    st.caption("Picks the agent analysed and liked but chose not to place — and why.")

    candidates = db.get_candidate_bets(limit=100)

    if not candidates:
        st.info("No runner-up bets logged yet. They'll appear here after the next scan.")
    else:
        # ── Filter row ──
        col_d, col_r, _ = st.columns([1, 1, 3])
        dates_avail = sorted({c["game_date"] for c in candidates}, reverse=True)
        date_filter = col_d.selectbox("Date", ["All"] + dates_avail, key="runner_date")

        _SKIP_LABELS = {
            "edge_below_threshold": "Edge below 5%",
            "slots_full":           "Slots full",
            "sharp_money_opposing": "Sharp money opposing",
            "injury_uncertainty":   "Injury uncertainty",
            "line_moved_against":   "Line moved against",
            "low_confidence":       "Low confidence",
            "other":                "Other",
        }
        reason_filter = col_r.selectbox("Skip reason", ["All"] + list(_SKIP_LABELS.values()), key="runner_reason")

        filtered_c = candidates
        if date_filter != "All":
            filtered_c = [c for c in filtered_c if c["game_date"] == date_filter]
        if reason_filter != "All":
            inv = {v: k for k, v in _SKIP_LABELS.items()}
            filtered_c = [c for c in filtered_c if c["skip_reason"] == inv.get(reason_filter)]

        st.markdown(f"**{len(filtered_c)} near-miss(es)**")
        st.divider()

        for c in filtered_c:
            skip_label = _SKIP_LABELS.get(c["skip_reason"], c["skip_reason"])
            odds_fmt   = fo(c["odds"]) if c["odds"] else "—"
            edge_color = "orange" if c["edge_pct"] >= 3 else ""
            conf_cls   = "badge-high" if c["confidence"] == "High" else "badge-med"

            st.markdown(f"""
            <div class="bet-card open" style="border-left-color: #f59e0b;">
                <div style="display:flex; justify-content:space-between; align-items:flex-start;">
                    <div>
                        <div class="bet-pick">{c['pick']}</div>
                        <div class="bet-meta">{c['matchup']} &nbsp;·&nbsp; {c['game_date']}</div>
                    </div>
                    <div style="text-align:right;">
                        <div class="stat-value {edge_color}" style="font-size:1rem;">{c['edge_pct']:+.1f}% edge</div>
                        <div class="bet-meta">{c['bet_type'].title()} @ {odds_fmt}</div>
                    </div>
                </div>
                <div style="margin-top:8px;">
                    <span class="bet-badge {conf_cls}">{c['confidence']}</span>
                    <span class="bet-badge" style="background:#422006;color:#fb923c;">⏭ {skip_label}</span>
                </div>
                {f'<div class="bet-meta" style="margin-top:8px;">{c["reasoning"]}</div>' if c.get("reasoning") else ""}
            </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 6 — Replaced Bets
# ═══════════════════════════════════════════════════════════════════════════════

with tab_replaced:
    st.markdown("### Replaced Bets")
    st.caption("Bets the agent cancelled and swapped out for a stronger pick.")

    pairs = db.get_replaced_bets()

    if not pairs:
        st.info("No replaced bets yet. They appear here when the agent cancels a bet and places a stronger one in its place.")
    else:
        st.markdown(f"**{len(pairs)} swap(s) recorded**")
        st.divider()

        for p in pairs:
            old_odds_fmt = fo(p["old_odds"]) if p["old_odds"] else "—"
            new_odds_fmt = fo(p["new_odds"]) if p["new_odds"] else "—"

            edge_delta = round(p["new_edge"] - p["old_edge"], 1)
            delta_color = "#22c55e" if edge_delta > 0 else "#ef4444"

            new_status = p.get("new_status", "open")
            if new_status == "won":
                result_badge = '<span class="bet-badge badge-high">✓ Won</span>'
            elif new_status == "lost":
                result_badge = '<span class="bet-badge badge-low">✗ Lost</span>'
            elif new_status == "open":
                result_badge = '<span class="bet-badge badge-med">Open</span>'
            else:
                result_badge = f'<span class="bet-badge">{new_status.title()}</span>'

            cancelled_ts = p["cancelled_at"][:10] if p.get("cancelled_at") else "—"

            st.markdown(f"""
            <div class="bet-card open" style="border-left-color:#6366f1; padding:14px 16px;">

                <div style="font-size:0.72rem;color:#64748b;margin-bottom:8px;">
                    Swap on {cancelled_ts}
                </div>

                <!-- CANCELLED BET -->
                <div style="display:flex;align-items:center;gap:10px;margin-bottom:6px;">
                    <span style="font-size:1.1rem;">❌</span>
                    <div>
                        <div class="bet-pick" style="font-size:0.95rem;color:#94a3b8;">#{p['old_id']} &nbsp;{p['old_pick']}</div>
                        <div class="bet-meta">{p['old_matchup']} &nbsp;·&nbsp; {p['old_bet_type'].title()} @ {old_odds_fmt} &nbsp;·&nbsp; {p['old_edge']:.1f}% edge</div>
                    </div>
                </div>

                <!-- REASON -->
                <div style="margin:6px 0 6px 30px;font-size:0.8rem;color:#f59e0b;font-style:italic;">
                    ↳ {p['reason'] if p.get('reason') else 'Replaced by stronger pick'}
                </div>

                <!-- NEW BET -->
                <div style="display:flex;align-items:center;justify-content:space-between;margin-top:4px;">
                    <div style="display:flex;align-items:center;gap:10px;">
                        <span style="font-size:1.1rem;">✅</span>
                        <div>
                            <div class="bet-pick" style="font-size:0.95rem;">#{p['new_id']} &nbsp;{p['new_pick']}</div>
                            <div class="bet-meta">{p['new_matchup']} &nbsp;·&nbsp; {p['new_bet_type'].title()} @ {new_odds_fmt} &nbsp;·&nbsp; {p['new_edge']:.1f}% edge</div>
                        </div>
                    </div>
                    <div style="text-align:right;">
                        <div style="font-size:0.85rem;font-weight:600;color:{delta_color};">edge {'+' if edge_delta >= 0 else ''}{edge_delta}%</div>
                        {result_badge}
                    </div>
                </div>

            </div>""", unsafe_allow_html=True)
