# BetIQ — NBA Betting Analyst & Paper Trader

An autonomous NBA sports betting analysis system powered by Claude. Ask questions in plain English, get structured picks backed by deep multi-source analysis, and track paper bets with a full bankroll ledger.

---

## Features

- **Natural language chat** — "What should I bet tonight?", "Analyze Lakers vs Celtics", "How has my bankroll done?"
- **Deep pre-bet analysis** — season stats, recent form, home/away splits, rest days, injuries, head-to-head, and live odds — all gathered automatically before any pick
- **Autonomous paper trading** — the agent places, tracks, and resolves hypothetical bets against a $1,000 starting bankroll
- **Self-evaluation loop** — reviews its own win rate and mistake patterns before every new recommendation
- **Extended thinking** — Claude reasons step-by-step before responding (toggle visible in the UI)
- **Streamlit dashboard** — live bankroll, open bets sidebar, bet history table, cumulative P&L chart

---

## Setup

### 1. Clone / navigate to the project

```bash
cd C:\Users\cbemb\Documents\BetIQ
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure API keys

```bash
cp .env.example .env
```

Edit `.env` and fill in:

| Key | Where to get it | Required |
|-----|----------------|----------|
| `ANTHROPIC_API_KEY` | [console.anthropic.com](https://console.anthropic.com) | Yes |
| `ODDS_API_KEY` | [the-odds-api.com](https://the-odds-api.com) | Yes (for live odds) |
| `BALLDONTLIE_API_KEY` | [balldontlie.io](https://www.balldontlie.io) | Optional |

### 5. Run the app

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`.

---

## Project structure

```
BetIQ/
├── app.py          # Streamlit frontend — chat, history, performance tabs
├── agent.py        # Agentic loop — Claude + tools + extended thinking
├── tools.py        # All API calls (Balldontlie, Odds API) + bet management
├── database.py     # SQLite schema, CRUD for bankroll and bets
├── betiq.db        # Auto-created on first run (SQLite database)
├── .env            # Your API keys (not committed)
├── .env.example    # Template
├── requirements.txt
└── README.md
```

---

## Bankroll rules

| Condition | Stake |
|-----------|-------|
| Edge ≥ 10% (High confidence) | 5% of current bankroll |
| Edge 5–9.9% (Medium confidence) | 3% of current bankroll |
| Edge < 5% | No bet placed |
| Open bets ≥ 3 | No new bets until one resolves |

Starting bankroll: **$1,000 paper money**

---

## Resolving bets

The agent calls `resolve_bets` automatically during analysis. You can also trigger it manually:
- Click **Resolve Open Bets** in the sidebar
- Ask in chat: *"Resolve my open bets"*

Results are pulled from The Odds API (primary) and Balldontlie (fallback).

---

## Notes

- **Paper trading only** — no real money is ever involved
- Live odds require an active Odds API key; the free tier includes 500 requests/month
- Injury data requires a Balldontlie paid plan; the agent flags this uncertainty and reduces confidence accordingly
- The agent model is `claude-sonnet-4-6` with interleaved extended thinking enabled
