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
- `tools.submit_recommendation()` is now the main enforcement layer for live recommendations. It computes deterministic decision state, pricing, EV, and stake before deciding whether to place or log.
- `tools.place_paper_bet()` still exists for compatibility, but direct calls now run through the same deterministic evaluator instead of trusting raw LLM edge.
- `database.py` owns persistence. Avoid bypassing it with ad hoc SQL unless a tiny compatibility patch is needed.

## Quantitative model boundaries
- Current deterministic pieces:
  - American odds conversion
  - implied probability and no-vig normalization
  - coded moneyline evaluation using Elo plus bounded signal adjustments
  - Kelly sizing and bankroll application
  - BET / LEAN / PASS decision state
  - data-quality scoring
  - CLV snapshot calculation
  - Elo win probability baseline
- Current LLM-estimated pieces:
  - market selection and qualitative reasoning
  - edge hint submitted through `submit_analysis`
  - confidence tier and edge rationale
- Current market discipline:
  - moneyline can be BET / LEAN / PASS because there is a coded probability model
  - spread and total can be LEAN / PASS only until a real cover / totals model exists
- If you add more quant logic, prefer placing pure math in `betting_math.py`, pure decision logic in `decision_support.py`, and shared orchestration in `scan_context.py`.

## Editing guidance
- Prefer additive changes over architecture replacement.
- Schema changes are acceptable when they preserve backward compatibility for existing `betiq.db` files. Use `ALTER TABLE` migrations in `database.init_db()`.
- If you touch shared math, update both live flow (`tools.py`) and historical flow (`backtest.py`) so they do not drift.
- If you touch live recommendation evaluation, keep `submit_recommendation()` and `place_paper_bet()` behavior aligned.
- Add or update tests for all deterministic math changes.
- Keep docs in sync:
  - `BETTING_MODEL_NOTES.md` for architecture/audit notes
  - `TODO.md` for staged improvement ideas

## Verification
- Use `py -m unittest discover -s tests` in this Windows workspace.
- Some environments in this repo cannot write `__pycache__` cleanly; prefer tests over compile commands that require bytecode writes.
- If external API behavior is relevant, document assumptions rather than hard-coding live responses in tests.
