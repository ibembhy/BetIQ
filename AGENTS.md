# BetIQ Agent Notes

## Purpose
BetIQ is an NBA betting analysis and paper-trading app. The current architecture mixes:
- deterministic data collection and bankroll enforcement in `tools.py`, `database.py`, `elo.py`, `backtest.py`
- LLM-driven qualitative reasoning and edge estimation in `agent.py`
- orchestration and prefetch pipelines in `runner.py` and `manual_trigger.py`
- presentation in `app.py`

## Architecture guardrails
- Do not replace or rewrite `agent.py` system prompts unless the task explicitly requires it.
- Keep the current prefetch flow intact. `runner.py` and `manual_trigger.py` gather data, then `agent.run_agent_prefetch()` submits one per-game context blob to the model.
- Treat `submit_analysis` in `agent.py` as the handoff point from LLM reasoning into deterministic execution.
- `tools.place_paper_bet()` is the enforcement layer for bankroll limits, open-bet caps, and stake sizing.
- `database.py` owns persistence. Avoid bypassing it with ad hoc SQL unless a tiny compatibility patch is needed.

## Quantitative model boundaries
- Current deterministic pieces:
  - American odds conversion
  - implied probability
  - no-vig normalization in backtesting
  - Kelly sizing and bankroll application
  - CLV snapshot calculation
  - Elo win probability baseline
- Current LLM-estimated pieces:
  - final win probability for a pick
  - edge percentage submitted through `submit_analysis`
  - confidence tier and edge rationale
- If you add more quant logic, prefer placing it in `betting_math.py` and calling it from existing flows rather than teaching the prompt more formulas.

## Editing guidance
- Prefer additive changes over architecture replacement.
- Avoid changing DB schema unless necessary for a concrete feature.
- If you touch shared math, update both live flow (`tools.py`) and historical flow (`backtest.py`) so they do not drift.
- Add or update tests for all deterministic math changes.
- Keep docs in sync:
  - `BETTING_MODEL_NOTES.md` for architecture/audit notes
  - `TODO.md` for staged improvement ideas

## Verification
- Use `python -m unittest discover -s tests` for math/unit coverage added in this repo.
- If external API behavior is relevant, document assumptions rather than hard-coding live responses in tests.
