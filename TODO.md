# TODO

## Safe Incremental Improvements
- Finish surfacing deterministic pricing fields directly in the card-style UI components, not just supplemental tables.
- Add tests for CLV calculation and the live recommendation evaluator in `tools.py`.
- Persist and display data-quality penalty reasons alongside the numeric score.
- Tighten stale-odds handling when the submitted odds differ from the current best line.
- De-duplicate American/decimal odds formatting between `app.py`, `betfair.py`, and `betting_math.py`.

## Medium-Risk Refactors
- Move the `submit_analysis` payload toward deterministic fields such as `model_win_probability` or ranked market choices, and stop asking the LLM for an edge percentage at all.
- Split the live evaluator out of `tools.py` into a dedicated service/module once the current pass stabilizes.
- Centralize CLV, EV, and staking output formatting so app, reports, and notifications all show the same numbers.
- Tighten fallback behavior when Elo data is unavailable so the system distinguishes "unknown" from a true 50/50 model.
- Add a real deterministic spread model based on margin distribution before allowing spread bets to become `BET`.
- Add a real deterministic totals model before allowing totals bets to become `BET`.

## Major Architectural Replacements
- Replace LLM-authored edge estimation with a deterministic ensemble model that consumes the pre-fetched stats and outputs win probabilities directly.
- Convert the current prompt-led pick selection into a structured ranking pipeline where the model explains deterministic candidates instead of inventing the candidate values.
- Split `tools.py` into domain modules (`odds`, `bets`, `stats`, `clv`, `notifications`) to reduce shared-state coupling.
- Introduce a formal typed schema layer for analysis inputs/outputs instead of passing large JSON blobs through prompt text.
- Rework reporting so narrative generation is downstream of a fully deterministic betting ledger and metrics engine.
