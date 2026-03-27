"""
BetIQ — Agentic loop with extended thinking and tool use.
"""

import json
import os
from anthropic import Anthropic
from dotenv import load_dotenv

import tools as t

load_dotenv()

client = Anthropic()

# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are BetIQ, an elite NBA sports betting analyst and autonomous paper trader.

## Your analytical process (NEVER skip steps)

**Before every recommendation:**
1. Call `get_notes` — recall lessons and patterns from past scans
2. Call `get_bet_history` — review your recent win rate, CLV trend, and any mistake patterns
3. Call `get_bankroll` — confirm balance and open bet slots
4. Call `resolve_bets` — settle any finished games
5. Call `snapshot_closing_odds` — capture pre-game closing lines for open bets (CLV data)

**For any matchup analysis, gather ALL of the following:**
- `get_team_stats` for both teams (current season)
- `get_season_stats` for both teams (3-season trend)
- `get_recent_form` for both teams (last 10 games)
- `get_home_away_splits` for both teams
- `get_rest_days` for both teams
- `get_injury_report` for both teams
- `get_head_to_head` between the two teams
- `get_current_odds` for the game
- `get_public_betting_percentages` for the game — check where public and sharp money are going
- `get_line_movement` for both teams — detect whether the line has moved toward or away from your pick
- `calculate_implied_probability` on the odds of your intended pick

**Reading public % and line movement:**
- Money % significantly higher than ticket % on a team → sharp/professional money on that side → weight your estimate toward them
- 70%+ public tickets on one team + line moving the other way → classic sharp fade → consider the other side
- Line moved 1+ point toward your pick since opening → sharp confirmation; moved against → red flag, reassess
- Never use public % alone — treat it as a confirming or disconfirming signal on top of your statistical edge

**Edge calculation:**
- Estimate your own win probability using all gathered data
- Compare to implied probability from market odds
- Edge = your_prob - implied_prob (as a percentage)
- Only recommend bets with ≥ 5% edge

## Bankroll & stake rules
- Stake sizing is calculated automatically using **Half-Kelly Criterion** based on your edge % and the odds
- Kelly formula: f = (p·b − q) / b where p = your win prob, b = net decimal odds, q = 1−p; then halved for variance reduction and capped at 12% of bankroll
- Simply pass `edge` and `odds` to `place_paper_bet` — the system computes the optimal stake
- Never bet below 5% edge
- Never exceed 5 open bets simultaneously
- Always call `get_bankroll` immediately before `place_paper_bet`

## Closing Line Value (CLV)
- CLV = closing_implied_probability − your_implied_probability (positive = you beat the market = long-term profitability signal)
- Call `snapshot_closing_odds` at every scan to capture pre-game closing lines for open bets
- Check CLV trend in `get_bet_history` — if avg CLV is negative, you are entering bets too late (line has already moved against you); time picks earlier in the day

## Runner-up bets (near misses)
For EVERY game you analyse where you find a potential edge but decide NOT to bet, call `log_candidate_bet`.
Common reasons to log:
- Edge is 2–4% (interesting but below the 5% threshold)
- All 5 slots are full and this pick isn't strong enough to swap in
- Sharp money or line movement contradicts your model
- Injury status unknown and it changes the pick materially
- You like the pick but confidence is too low to stake real money on it

Log these even for games you never fully analysed — if you glanced at the matchup and decided it wasn't worth deep analysis, log it with `edge_below_threshold` or `low_confidence`.
The user uses these to understand what you almost bet and why, and to spot if you're being too conservative.

## Self-evaluation loop
Before reasoning on any new pick, review `get_bet_history`. If recent win rate < 50%, lower confidence tiers. If high-confidence bets are losing, question your probability model. Log what you got right/wrong.

## Learning loop
After every scan — whether you placed bets or not — call `save_note` to record:
- Any hypothesis you formed that could not be verified yet (e.g. "I think rest advantage is being overpriced in tonight's line — will track")
- Any pattern noticed across multiple games (e.g. "home underdogs +4 to +7 have covered 4 of last 5")
- After bets resolve: what your model got right or wrong and why, and what to adjust
- Never repeat a mistake you already logged — check `get_notes` first

## Bet swapping rules
If all 5 bet slots are full but you find a new bet with an edge at least 3% higher than an existing open bet:
1. Call `cancel_bet` on the weakest open bet (lowest edge), stating why it's being replaced
2. Immediately place the new stronger bet with `place_paper_bet`
Never cancel a bet just to free up a slot — only swap when the new edge is meaningfully better.

## Output format for every structured pick:
```
MATCHUP: [Team A vs Team B]
MY EDGE: [Your estimated probability X% vs implied probability Y% = Z% edge]
PICK: [Exact pick — e.g. "Boston Celtics ML" or "Lakers -4.5" or "Over 224.5"]
CONFIDENCE: [High / Medium / Low]
STAKE: [$X of $Y bankroll]
REASONING: [2-3 sentences covering the decisive data points]
PAST PERFORMANCE NOTE: [Relevant pattern from your bet history, or "No history yet"]
```

Be honest about uncertainty. Missing injury data reduces confidence. Never form a probability estimate from a single data source. The more data, the better the edge.
"""

# ── Tool schema ───────────────────────────────────────────────────────────────

TOOLS = [
    {
        "name": "get_todays_games",
        "description": "Fetch today's NBA schedule with matchups, current scores, and game status.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "get_team_stats",
        "description": "Current or historical season averages and win-loss record for a team.",
        "input_schema": {
            "type": "object",
            "properties": {
                "team_name": {"type": "string", "description": "Team name, city, or abbreviation"},
                "season":    {"type": "integer", "description": "Season start year (e.g. 2024). Defaults to current."},
            },
            "required": ["team_name"],
        },
    },
    {
        "name": "get_season_stats",
        "description": "Multi-season stats to identify trends (offense/defense trajectory, improvement/decline).",
        "input_schema": {
            "type": "object",
            "properties": {
                "team_name": {"type": "string"},
                "seasons":   {"type": "array", "items": {"type": "integer"}, "description": "List of season years. Defaults to last 3."},
            },
            "required": ["team_name"],
        },
    },
    {
        "name": "get_recent_form",
        "description": "Last N games for a team: results, scores, opponents, home/away, point margin. Reveals hot/cold streaks.",
        "input_schema": {
            "type": "object",
            "properties": {
                "team_name": {"type": "string"},
                "last_n":    {"type": "integer", "description": "Number of recent games (default 10)"},
            },
            "required": ["team_name"],
        },
    },
    {
        "name": "get_home_away_splits",
        "description": "Separate home vs away record, scoring average, and point differential. Critical for line shopping.",
        "input_schema": {
            "type": "object",
            "properties": {
                "team_name": {"type": "string"},
                "season":    {"type": "integer"},
            },
            "required": ["team_name"],
        },
    },
    {
        "name": "get_rest_days",
        "description": "Days of rest since last game. Identifies back-to-backs and fatigue situations.",
        "input_schema": {
            "type": "object",
            "properties": {
                "team_name": {"type": "string"},
            },
            "required": ["team_name"],
        },
    },
    {
        "name": "get_injury_report",
        "description": "Current injury status for a team's players. Unavailable data is flagged explicitly.",
        "input_schema": {
            "type": "object",
            "properties": {
                "team_name": {"type": "string"},
            },
            "required": ["team_name"],
        },
    },
    {
        "name": "get_head_to_head",
        "description": "Historical matchup results between two teams across multiple seasons including scores, winner, and average totals.",
        "input_schema": {
            "type": "object",
            "properties": {
                "team1_name":  {"type": "string"},
                "team2_name":  {"type": "string"},
                "num_seasons": {"type": "integer", "description": "Seasons of history to pull (default 3)"},
            },
            "required": ["team1_name", "team2_name"],
        },
    },
    {
        "name": "get_advanced_stats",
        "description": (
            "Advanced team stats from local historical data: paint points, fast-break points, "
            "2nd-chance points, team turnovers, points off turnovers, lead changes. "
            "Available for seasons up to 2022-23."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "team_name": {"type": "string"},
                "season":    {"type": "integer", "description": "Season start year (default: current)"},
            },
            "required": ["team_name"],
        },
    },
    {
        "name": "get_public_betting_percentages",
        "description": (
            "Fetch public ticket % and money % from Action Network (free). "
            "High ticket % on one team + money % on the other = sharp money fading the public. "
            "Call this for every matchup you are seriously considering betting."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "team_name": {"type": "string", "description": "Optional: filter to a specific team's game"},
            },
            "required": [],
        },
    },
    {
        "name": "get_line_movement",
        "description": (
            "Show how the moneyline, spread, and total have moved throughout the day for a team's game. "
            "Reads from local snapshots captured during each odds fetch — no extra API call. "
            "Significant movement toward a team = sharp money on that side."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "team_name": {"type": "string"},
            },
            "required": ["team_name"],
        },
    },
    {
        "name": "get_current_odds",
        "description": "Live moneyline, spread, and totals from The Odds API for current NBA games.",
        "input_schema": {
            "type": "object",
            "properties": {
                "team_name": {"type": "string", "description": "Optional: filter to a specific team's game"},
            },
            "required": [],
        },
    },
    {
        "name": "get_historical_odds",
        "description": "Past game results and closing lines for recent games (up to 3 days back). Use for model calibration.",
        "input_schema": {
            "type": "object",
            "properties": {
                "team_name": {"type": "string", "description": "Optional team filter"},
                "days_back": {"type": "integer", "description": "Days to look back (max 3 on free plan)"},
            },
            "required": [],
        },
    },
    {
        "name": "calculate_implied_probability",
        "description": "Convert American odds to implied win probability. Always call this before estimating edge.",
        "input_schema": {
            "type": "object",
            "properties": {
                "odds": {"type": "integer", "description": "American odds (e.g. -110, +150)"},
            },
            "required": ["odds"],
        },
    },
    {
        "name": "place_paper_bet",
        "description": (
            "Log a paper bet to the SQLite ledger. Enforces: max 5 open bets, min 5% edge, "
            "stake sizing by confidence. MUST call get_bankroll first."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "matchup":   {"type": "string", "description": "e.g. 'Lakers vs Celtics'"},
                "pick":      {"type": "string", "description": "e.g. 'Boston Celtics ML' or 'Lakers -4.5' or 'Over 224.5'"},
                "bet_type":  {"type": "string", "enum": ["moneyline", "spread", "total"]},
                "odds":      {"type": "integer", "description": "American odds for this pick"},
                "confidence":{"type": "string", "enum": ["High", "Medium"]},
                "edge":      {"type": "number", "description": "Your calculated edge % (must be >= 5)"},
                "reasoning": {"type": "string", "description": "Why this bet has edge"},
                "game_date": {"type": "string", "description": "YYYY-MM-DD format"},
            },
            "required": ["matchup", "pick", "bet_type", "odds", "confidence", "edge", "reasoning"],
        },
    },
    {
        "name": "cancel_bet",
        "description": (
            "Cancel an open bet and refund its stake to the bankroll. "
            "Use this to replace a weaker open bet with a stronger one just found. "
            "Only swap if the new edge is at least 3% higher than the bet being cancelled."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "bet_id": {"type": "integer", "description": "ID of the open bet to cancel"},
                "reason": {"type": "string",  "description": "Why this bet is being replaced"},
            },
            "required": ["bet_id", "reason"],
        },
    },
    {
        "name": "get_bankroll",
        "description": "Current balance, open bets, and available slots. MUST call before place_paper_bet.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "get_bet_history",
        "description": (
            "All past picks with outcomes, P&L, win rate by confidence, and recent trend. "
            "Review this before every new recommendation."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "resolve_bets",
        "description": "Check completed games against open bets, mark won/lost, and update bankroll.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "save_note",
        "description": (
            "Save a lesson, pattern, or hypothesis to persistent memory across scans. "
            "Use after every scan to record: what you predicted, what actually happened, "
            "and what you'll do differently. Also use to flag patterns like 'back-to-back teams "
            "consistently underperform my model' or 'injury reports overstate impact'."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "note_type": {
                    "type": "string",
                    "enum": ["lesson", "pattern", "hypothesis", "model_update"],
                    "description": "lesson=specific mistake; pattern=recurring trend; hypothesis=unverified theory; model_update=adjustment to probability estimates",
                },
                "content": {
                    "type": "string",
                    "description": "Detailed note. Include: what you predicted, what happened, why you were wrong, what to adjust.",
                },
            },
            "required": ["note_type", "content"],
        },
    },
    {
        "name": "log_candidate_bet",
        "description": (
            "Log a near-miss bet — a pick you analysed and liked but chose NOT to place. "
            "Call this for EVERY game where you found a potential edge but decided not to bet. "
            "These are displayed in the UI as 'Runner-Up Bets' so the user can track near-misses."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "matchup":     {"type": "string", "description": "e.g. 'Boston Celtics vs Miami Heat'"},
                "pick":        {"type": "string", "description": "e.g. 'Boston Celtics ML' or 'Over 224.5'"},
                "bet_type":    {"type": "string", "enum": ["moneyline", "spread", "total"], "description": "Bet type"},
                "odds":        {"type": "integer", "description": "American odds at time of analysis"},
                "edge_pct":    {"type": "number",  "description": "Your estimated edge % (can be < 5)"},
                "confidence":  {"type": "string",  "enum": ["High", "Medium", "Low"]},
                "skip_reason": {
                    "type": "string",
                    "enum": [
                        "edge_below_threshold",
                        "slots_full",
                        "sharp_money_opposing",
                        "injury_uncertainty",
                        "line_moved_against",
                        "low_confidence",
                        "other",
                    ],
                    "description": "Primary reason you did not place this bet",
                },
                "reasoning": {"type": "string", "description": "Brief explanation of what you liked and why you passed"},
            },
            "required": ["matchup", "pick", "bet_type", "odds", "edge_pct", "confidence", "skip_reason"],
        },
    },
    {
        "name": "snapshot_closing_odds",
        "description": (
            "For each open bet, look up current market odds and record them as the closing line. "
            "Calculates CLV (Closing Line Value): positive CLV = you beat the market. "
            "Call this at every scan — the odds captured closest to tip-off are the best CLV benchmark."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "get_notes",
        "description": (
            "Retrieve lessons and patterns saved from previous scans. "
            "ALWAYS call this at the start of every scan before making any picks."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "note_type": {
                    "type": "string",
                    "enum": ["lesson", "pattern", "hypothesis", "model_update"],
                    "description": "Filter by type. Omit to get all notes.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max notes to return (default 30).",
                },
            },
            "required": [],
        },
    },
]

# ── Tool dispatch ─────────────────────────────────────────────────────────────

DISPATCH = {
    "get_todays_games":          lambda a: t.get_todays_games(),
    "get_team_stats":            lambda a: t.get_team_stats(**a),
    "get_season_stats":          lambda a: t.get_season_stats(**a),
    "get_recent_form":           lambda a: t.get_recent_form(**a),
    "get_home_away_splits":      lambda a: t.get_home_away_splits(**a),
    "get_rest_days":             lambda a: t.get_rest_days(**a),
    "get_injury_report":         lambda a: t.get_injury_report(**a),
    "get_head_to_head":          lambda a: t.get_head_to_head(**a),
    "get_advanced_stats":        lambda a: t.get_advanced_stats(**a),
    "get_public_betting_percentages": lambda a: t.get_public_betting_percentages(**a),
    "get_line_movement":         lambda a: t.get_line_movement(**a),
    "get_current_odds":          lambda a: t.get_current_odds(**a),
    "get_historical_odds":       lambda a: t.get_historical_odds(**a),
    "calculate_implied_probability": lambda a: t.calculate_implied_probability(**a),
    "cancel_bet":                lambda a: t.cancel_bet(**a),
    "place_paper_bet":           lambda a: t.place_paper_bet(**a),
    "get_bankroll":              lambda a: t.get_bankroll(),
    "get_bet_history":           lambda a: t.get_bet_history(),
    "resolve_bets":              lambda a: t.resolve_bets(),
    "snapshot_closing_odds":     lambda a: t.snapshot_closing_odds(),
    "log_candidate_bet":         lambda a: t.log_candidate_bet(**a),
    "save_note":                 lambda a: t.save_note(**a),
    "get_notes":                 lambda a: t.get_notes(**a),
}


def _run_tool(name: str, args: dict) -> str:
    fn = DISPATCH.get(name)
    if fn is None:
        return json.dumps({"error": f"Unknown tool: {name}"})
    try:
        result = fn(args)
        return json.dumps(result, default=str)
    except Exception as exc:
        return json.dumps({"error": f"{name} failed: {exc}"})


# ── Agent loop ────────────────────────────────────────────────────────────────

def run_agent(
    user_message: str,
    conversation_history: list,
) -> tuple[str, list, list[str]]:
    """
    Run one turn of the agentic loop.

    Returns:
        response_text      – final assistant reply
        updated_history    – full conversation for the next turn
        thinking_blocks    – collected reasoning strings (for display)
    """
    messages = conversation_history + [{"role": "user", "content": user_message}]
    all_thinking: list[str] = []

    while True:
        # ── Call the model ────────────────────────────────────────────────────
        try:
            response = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=16000,
                thinking={"type": "enabled", "budget_tokens": 8000},
                system=SYSTEM_PROMPT,
                tools=TOOLS,
                messages=messages,
                betas=["interleaved-thinking-2025-05-14"],
            )
        except Exception:
            # Fallback: no extended thinking (older SDK / network issue)
            response = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=8000,
                system=SYSTEM_PROMPT,
                tools=TOOLS,
                messages=messages,
            )

        # ── Collect thinking blocks ───────────────────────────────────────────
        for block in response.content:
            if block.type == "thinking":
                all_thinking.append(block.thinking)

        # ── Add assistant turn to history (preserve thinking blocks) ─────────
        messages.append({"role": "assistant", "content": response.content})

        # ── End of turn ───────────────────────────────────────────────────────
        if response.stop_reason == "end_turn":
            text_parts = [
                block.text for block in response.content
                if getattr(block, "type", None) == "text"
            ]
            final_text = "\n".join(text_parts).strip() or "Analysis complete."
            return final_text, messages, all_thinking

        # ── Tool use turn ─────────────────────────────────────────────────────
        if response.stop_reason == "tool_use":
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    result_str = _run_tool(block.name, block.input)
                    tool_results.append({
                        "type":        "tool_result",
                        "tool_use_id": block.id,
                        "content":     result_str,
                    })
            messages.append({"role": "user", "content": tool_results})
            continue

        # Unexpected stop — bail out
        break

    return "Agent loop ended unexpectedly.", messages, all_thinking
