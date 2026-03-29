# TODO

## Safe Incremental Improvements
- Add deterministic helpers for market-level no-vig calculations on spreads and totals, not just head-to-head pairs.
- Surface deterministic EV, implied probability, and fair probability in the UI for every placed bet and candidate bet.
- Expose `betting_math.py` helpers through small wrapper tools only where the model genuinely needs them.
- Add tests for CLV calculation and any edge-to-probability conversion assumptions in `tools.py`.
- De-duplicate American/decimal odds formatting between `app.py`, `betfair.py`, and `betting_math.py`.

## Medium-Risk Refactors
- Replace the duplicated prefetch context builders in `runner.py` and `manual_trigger.py` with one shared function/module.
- Move the `submit_analysis` payload toward deterministic fields such as `model_win_probability` and let code compute implied probability, EV, and edge.
- Store model probability separately from edge in the database so downstream analytics do not have to reconstruct it.
- Centralize CLV, EV, and staking output formatting so app, reports, and notifications all show the same numbers.
- Tighten fallback behavior when Elo data is unavailable so the system distinguishes "unknown" from a true 50/50 model.

## Major Architectural Replacements
- Replace LLM-authored edge estimation with a deterministic ensemble model that consumes the pre-fetched stats and outputs win probabilities directly.
- Convert the current prompt-led pick selection into a structured ranking pipeline where the model explains deterministic candidates instead of inventing the candidate values.
- Split `tools.py` into domain modules (`odds`, `bets`, `stats`, `clv`, `notifications`) to reduce shared-state coupling.
- Introduce a formal typed schema layer for analysis inputs/outputs instead of passing large JSON blobs through prompt text.
- Rework reporting so narrative generation is downstream of a fully deterministic betting ledger and metrics engine.
