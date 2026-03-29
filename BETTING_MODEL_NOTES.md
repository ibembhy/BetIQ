# Betting Model Notes

## What The Repo Currently Does
- `runner.py` and `manual_trigger.py` prefetch team stats, trends, injuries, odds, public betting, line movement, roster data, and an Elo baseline for each game.
- `agent.py` gives the LLM those inputs and asks it to choose a pick, estimate win probability, and submit an `edge_pct`.
- `agent.py::_dispatch_submit_analysis()` applies a deterministic threshold:
  - `edge_pct >= 5` and fewer than 5 open bets -> call `tools.place_paper_bet()`
  - otherwise -> call `tools.log_candidate_bet()`
- `tools.py` handles deterministic bankroll operations:
  - implied probability conversion
  - Kelly-based stake sizing
  - open-bet limits
  - bet persistence
  - CLV snapshots
- `backtest.py` runs a fully deterministic historical simulation using rolling point differential plus no-vig market normalization.

## Where Probability, Edge, EV, And Bet Sizing Currently Come From
- Probability:
  - Elo baseline comes from `elo.py`
  - final pick win probability is estimated by the LLM from prompt context in `agent.py`
- Edge:
  - the LLM computes and submits `edge_pct` in `submit_analysis`
  - live bet placement decisions in `agent.py` depend on that submitted edge
- EV:
  - prior to this session, EV was not implemented as a shared deterministic utility
  - after this session, deterministic EV helpers exist in `betting_math.py`, and placed-bet responses now include EV derived from the submitted edge plus market odds
- Bet sizing:
  - deterministic Half-Kelly sizing occurs in `tools.py`
  - backtest sizing was separate but is now aligned to the shared helper in `betting_math.py`

## Places Where The LLM Is Still Estimating Quantitative Values
- `agent.py` main system prompt:
  - tells the model to "Estimate your own win probability"
  - tells the model to compute `edge = your_prob - implied_prob`
- `agent.py` prefetch system prompt:
  - tells the model to estimate win probability for the best bet type
  - tells the model to submit `edge_pct`
- `runner.py` and `manual_trigger.py`:
  - pass Elo probability as a baseline, but the LLM still adjusts it subjectively before submitting edge
- `submit_analysis` tool payload:
  - accepts `edge_pct` directly from the LLM, and that number controls whether a bet is placed
- Confidence labels:
  - still LLM-authored, except `tools.place_paper_bet()` coerces `Low` to `Medium` for actual placed bets

## Quantitative Pieces Still Missing
- A deterministic win-probability model for actual live pick selection
- Deterministic edge calculation from stored model probability and market probability
- Database fields for model probability, market implied probability, no-vig fair probability, and EV
- Consistent use of no-vig fair pricing in the live path, not just in the backtest path
- Deterministic candidate ranking across moneyline, spread, and total markets

## What Changed In This Session
- Added `betting_math.py` with deterministic helpers for:
  - American odds to implied probability
  - American odds to decimal odds
  - no-vig probability normalization
  - expected value
  - Kelly fraction and Kelly stake with configurable fractional Kelly
- Updated `tools.py` to reuse the shared math helpers and to include deterministic EV fields in the bet placement result payload.
- Updated `backtest.py` to reuse the shared implied probability, no-vig, and Kelly helpers.
- Updated `betfair.py` to reuse the shared decimal odds conversion.
- Added unit tests covering the deterministic math helpers.

## What Was Intentionally Not Changed
- The `agent.py` system prompts were not replaced or rewritten.
- The current LLM-driven workflow for selecting picks and estimating edge was left intact.
- No DB schema changes were introduced for new model fields.
- No large refactor was done to split `tools.py` or redesign the orchestration layer.
- The UI and reporting flows were not reworked beyond consuming the existing deterministic outputs.

## Recommended Next Change Later
The highest-value next step is to change the `submit_analysis` contract from "LLM submits edge percent" to "LLM submits pick plus reasoning, deterministic code computes implied probability, fair probability, edge, EV, and sizing." That keeps the current architecture recognizable while moving the fragile numeric pieces out of prompt space.
