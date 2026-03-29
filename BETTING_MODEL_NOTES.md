# Betting Model Notes

## What The Repo Currently Does
- `runner.py` and `manual_trigger.py` prefetch matchup context and send one game at a time into `agent.run_agent_prefetch()`.
- `agent.py` still uses the existing prompt-centric flow: the LLM selects a market and explains the pick.
- The downstream recommendation path is now more deterministic:
  - the model still submits an `edge_pct` hint through `submit_analysis`
  - `agent.py` forwards that into `tools.submit_recommendation()`
  - `tools.submit_recommendation()` evaluates the recommendation in code and assigns `BET`, `LEAN`, or `PASS`
  - only `BET` decisions can place a paper bet
- `backtest.py` remains fully deterministic and still uses no-vig normalization plus Kelly sizing.

## Where Win Probability Currently Comes From
- Moneyline live path:
  - base win probability comes from `elo.py`
  - `tools.evaluate_recommendation()` applies small deterministic adjustments for rest, injuries, public money divergence, and line movement
- Spread live path:
  - no coded cover-probability model yet
- Totals live path:
  - no coded total-points probability model yet
- Chat/prompt layer:
  - the LLM still narrates and may estimate probabilities conceptually, but those are no longer the main live trigger for placing a bet

## Where Edge, EV, And Stake Sizing Currently Come From
- Edge:
  - moneyline edge is now computed in code from `model_prob - fair_prob_no_vig` when a no-vig market pair is available
  - if fair probability is unavailable, the code falls back to market implied probability
  - for unsupported spread/total markets, stored edge may still reflect the LLM hint, but it no longer authorizes a `BET`
- EV:
  - deterministic via `betting_math.py`
  - computed from coded `model_prob`, offered odds, and recommended stake
- Stake sizing:
  - deterministic Half-Kelly remains downstream of the coded decision
  - stake is only non-zero for `BET` recommendations

## Where The LLM Still Invents Or Estimates Quantitative Values
- The system prompts in `agent.py` still ask the model to estimate win probability and edge.
- The `submit_analysis` tool payload still accepts `edge_pct` from the LLM.
- Confidence labels are still model-authored.
- Spread and totals recommendations still rely on qualitative LLM reasoning because no deterministic market-specific probability model exists yet.

## What This Pass Changed
- Added `decision_support.py` for:
  - coded moneyline probability adjustment
  - data-quality scoring
  - `BET` / `LEAN` / `PASS` inference
  - deterministic stake recommendation
- Added `scan_context.py` so `runner.py` and `manual_trigger.py` share the same prefetch/context logic.
- Updated `tools.py` so live recommendations flow through `submit_recommendation()` instead of trusting raw LLM edge as the main trigger.
- Added persistence for:
  - `model_prob`
  - `market_implied_prob`
  - `fair_prob_no_vig`
  - `edge_pct`
  - `ev`
  - `stake_pct`
  - `stake_amount`
  - `data_quality_score`
  - `decision`
  - `llm_edge_pct`
- Updated the UI to expose the new quantitative fields in supplemental detail tables.
- Updated report payloads to include the richer pricing and decision metadata.

## What Was Intentionally Left Unchanged
- The current `agent.py` system prompts were not replaced.
- The overall prompt-centric architecture was not rewritten.
- The LLM still chooses and explains a pick rather than receiving a fully deterministic ranked slate.
- No deterministic spread model was invented.
- No deterministic totals model was invented.
- `tools.py` is still a large module; this pass did not split it apart.

## Current Market Discipline
- Moneyline:
  - supported for `BET`, `LEAN`, or `PASS`
  - requires a coded win probability
- Spread:
  - no coded cover model yet
  - can only be `LEAN` or `PASS`
- Totals:
  - no coded totals model yet
  - can only be `LEAN` or `PASS`

## Highest-Value Next Change
Change the `submit_analysis` contract from "LLM submits edge percent" to "LLM submits market choice plus qualitative rationale, deterministic code computes all quantitative fields." That is the cleanest next step without replacing the existing prompt architecture.
