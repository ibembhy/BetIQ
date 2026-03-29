"""
BetIQ — Agentic loop with extended thinking and tool use.
"""

import json
import os
import time
from anthropic import Anthropic, RateLimitError
from dotenv import load_dotenv

import tools as t
import database as db

load_dotenv()

client = Anthropic()

def _log_claude(model: str, response=None):
    inp = getattr(getattr(response, "usage", None), "input_tokens", 0) or 0
    out = getattr(getattr(response, "usage", None), "output_tokens", 0) or 0
    db.log_api_call("Anthropic", model, input_tokens=inp, output_tokens=out)

# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are BetIQ, an elite NBA sports betting analyst and autonomous paper trader.

## ⚠️ CRITICAL — Anti-hallucination rules (NEVER violate these)
- Your training data about NBA rosters, trades, and player status is OUTDATED. Do NOT rely on it.
- **Only reference players that appear in the injury report or roster data provided to you.**
- If a player is not listed in the provided data, assume they are NOT on that team. Do not mention them.
- Never invent or assume statistics, scores, or outcomes not present in the provided data.
- If key data is missing, say "data unavailable" — do not fill gaps with your training knowledge.
- Ground every claim in the specific numbers from the fetched data. No guessing.

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
- `get_book_discrepancies` for both teams — detect cross-book line disagreements that signal sharp action or stale pricing
- `get_public_betting_percentages` for the game — check where public and sharp money are going
- `get_line_movement` for both teams — detect whether the line has moved toward or away from your pick
- `calculate_implied_probability` on the odds of your intended pick

**Reading public % and line movement:**
- Money % significantly higher than ticket % on a team → sharp/professional money on that side → weight your estimate toward them
- 70%+ public tickets on one team + line moving the other way → classic sharp fade → consider the other side
- Line moved 1+ point toward your pick since opening → sharp confirmation; moved against → red flag, reassess
- Never use public % alone — treat it as a confirming or disconfirming signal on top of your statistical edge

**Reading book discrepancies (`get_book_discrepancies`):**
- ML spread ≥ 15 pts across books → large discrepancy; one book likely has stale pricing — always bet the best available price and flag this as a confirming signal
- ML spread 8–14 pts → moderate; worth noting but not conclusive on its own
- ML spread < 8 pts → books aligned; no additional signal
- Total line spread ≥ 2.5 pts → unusual; may signal injury news or sharp total action not yet priced everywhere
- A large discrepancy that aligns with your model edge (e.g. your model likes Team A and Team A has the best line at one stale book) is a strong double-confirmation — increase confidence tier
- A large discrepancy that contradicts your pick (your model likes Team A but sharp books have moved heavily against them) → red flag, reduce edge estimate or skip

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
- Always set `edge_type` — classify the primary reason for the edge: "injury" (key player out), "line_movement" (sharp money moved the line), "public_fade" (fading heavy public side), "rest_fatigue" (back-to-back or rest advantage), "statistical" (model edge from stats alone), "multiple" (two or more factors)

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
1. Call `cancel_bet` on the weakest open bet (lowest edge), stating why it's being replaced — note the returned `cancelled_bet_id`
2. Immediately place the new stronger bet with `place_paper_bet`, passing `replaces_bet_id` = the cancelled bet's ID
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
        "name": "get_book_discrepancies",
        "description": (
            "Scan all bookmakers for a team's game and flag significant ML, spread, or total line disagreements. "
            "A spread ≥ 15 pts on the ML signals a stale line or unpriced news — strong confirming signal when it aligns with your model edge. "
            "Call this for every game you are seriously considering betting."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "team_name": {"type": "string", "description": "Either team in the matchup"},
            },
            "required": ["team_name"],
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
                "reasoning":       {"type": "string",  "description": "Why this bet has edge"},
                "game_date":       {"type": "string",  "description": "YYYY-MM-DD format"},
                "replaces_bet_id": {"type": "integer", "description": "ID of the cancelled bet this replaces (only set when swapping)"},
                "edge_type":       {"type": "string",  "enum": ["injury", "line_movement", "public_fade", "rest_fatigue", "statistical", "multiple"], "description": "Primary reason for the edge"},
            },
            "required": ["matchup", "pick", "bet_type", "odds", "confidence", "edge", "reasoning", "edge_type"],
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
        "cache_control": {"type": "ephemeral"},
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
    "get_book_discrepancies":    lambda a: t.get_book_discrepancies(**a),
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
        # ── Call the model (with rate-limit retry) ───────────────────────────
        response = None
        for attempt in range(6):
            try:
                response = client.messages.create(
                    model="claude-sonnet-4-6",
                    max_tokens=5000,
                    thinking={"type": "enabled", "budget_tokens": 2000},
                    system=[{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}],
                    tools=TOOLS,
                    messages=messages,
                )
                _log_claude("claude-sonnet-4-6", response)
                break
            except RateLimitError:
                wait = 60 * (attempt + 1)
                print(f"Rate limit hit — waiting {wait}s before retry {attempt + 1}/6...", flush=True)
                time.sleep(wait)
            except Exception:
                # Fallback: no extended thinking (older SDK / network issue)
                for fb_attempt in range(6):
                    try:
                        response = client.messages.create(
                            model="claude-sonnet-4-6",
                            max_tokens=5000,
                            system=[{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}],
                            tools=TOOLS,
                            messages=messages,
                        )
                        _log_claude("claude-sonnet-4-6", response)
                        break
                    except RateLimitError:
                        wait = 60 * (fb_attempt + 1)
                        print(f"Rate limit hit (fallback) — waiting {wait}s...", flush=True)
                        time.sleep(wait)
                break
        if response is None:
            raise RuntimeError("API rate limit exceeded after all retries.")

        # ── Collect thinking blocks ───────────────────────────────────────────
        for block in response.content:
            if block.type == "thinking":
                all_thinking.append(block.thinking)

        # ── Add assistant turn to history (strip thinking blocks to save tokens) ─
        history_content = [b for b in response.content if b.type != "thinking"]
        messages.append({"role": "assistant", "content": history_content})

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


# ── Lite agent (Haiku, DB-only, for floating chat widget) ─────────────────────

_LITE_SYSTEM = """You are BetIQ Assistant. Answer questions about the user's paper betting account.
You can read bet history, bankroll, agent notes, and near-miss bets.
Be direct and short — under 120 words. Use numbers and facts. No NBA analysis, no predictions."""

_LITE_TOOLS = [
    {
        "name": "get_bankroll",
        "description": "Current balance, open bets count, and slots remaining.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "get_bet_history",
        "description": "All past picks with outcomes, P&L, and win rate.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "get_notes",
        "description": "Lessons and patterns the agent saved from previous scans.",
        "input_schema": {
            "type": "object",
            "properties": {"limit": {"type": "integer", "description": "Max notes (default 10)."}},
            "required": [],
        },
    },
    {
        "name": "get_candidate_bets",
        "description": "Near-miss bets the agent considered but chose not to place.",
        "input_schema": {
            "type": "object",
            "properties": {"limit": {"type": "integer", "description": "Max to return (default 20)."}},
            "required": [],
        },
    },
]

_LITE_DISPATCH = {
    "get_bankroll":       lambda a: t.get_bankroll(),
    "get_bet_history":    lambda a: t.get_bet_history(),
    "get_notes":          lambda a: t.get_notes(**a),
    "get_candidate_bets": lambda a: db.get_candidate_bets(limit=a.get("limit", 20)),
}


def run_lite_agent(user_message: str, conversation_history: list) -> tuple[str, list, list]:
    """Cheap Haiku agent with DB-only tools for the in-app floating chat widget."""
    messages = conversation_history + [{"role": "user", "content": user_message}]

    for _ in range(10):
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=800,
            system=_LITE_SYSTEM,
            tools=_LITE_TOOLS,
            messages=messages,
        )
        _log_claude("claude-haiku-4-5 (chat)", response)
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "end_turn":
            text = " ".join(
                b.text for b in response.content if getattr(b, "type", None) == "text"
            ).strip() or "Done."
            return text, messages, []

        if response.stop_reason == "tool_use":
            results = []
            for block in response.content:
                if block.type == "tool_use":
                    fn = _LITE_DISPATCH.get(block.name)
                    try:
                        out = fn(block.input) if fn else {"error": f"Unknown tool: {block.name}"}
                        content = json.dumps(out, default=str)
                    except Exception as exc:
                        content = json.dumps({"error": str(exc)})
                    results.append({"type": "tool_result", "tool_use_id": block.id, "content": content})
            messages.append({"role": "user", "content": results})

    return "Couldn't complete that request.", messages, []


# ── Bet report generator ───────────────────────────────────────────────────────

def generate_bet_report(bet: dict, score: str) -> str:
    """
    Generate a post-game analysis report for a resolved bet using Haiku.
    Single API call, no tools — cheap and fast.
    """
    outcome = bet.get("status", "unknown").upper()
    pnl     = bet.get("pnl", 0)
    pnl_str = f"+${pnl:.2f}" if pnl >= 0 else f"-${abs(pnl):.2f}"

    prompt = f"""You are a sharp NBA betting analyst. Analyse this resolved bet and write a concise post-game report.

**Bet:** {bet['pick']} ({bet['bet_type'].title()}) @ {bet['odds']:+d}
**Matchup:** {bet['matchup']} — {bet['game_date']}
**Stake:** ${bet['stake']:.2f} | **Edge:** {bet['edge']}% | **Confidence:** {bet['confidence']}
**Final Score:** {score}
**Outcome:** {outcome} ({pnl_str})
**Original reasoning:** {bet.get('reasoning', 'N/A')}

Write a 3-section report in plain text (no markdown headers, use bold labels):

**What happened:** Briefly explain the game result vs what was expected.
**Why it {'won' if outcome == 'WON' else 'lost'}:** Key factors that determined the outcome. If lost, what information was missing or misjudged.
**Lesson:** One concrete thing the model should do differently or keep doing for future bets of this type."""

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=600,
        messages=[{"role": "user", "content": prompt}],
    )
    _log_claude("claude-haiku-4-5 (bet report)", response)
    return response.content[0].text.strip()


# ── Pre-fetch agent (Sonnet, write-tools only, data passed in context) ─────────

PREFETCH_SYSTEM = """You are BetIQ, an elite NBA sports betting analyst and autonomous paper trader.

All game data AND bankroll info has been pre-fetched and is provided in the user message. Do NOT call any data tools — everything you need is already there.

## ⚠️ CRITICAL — Anti-hallucination rules (NEVER violate these)
- Your training data about NBA rosters, trades, and player status is OUTDATED. Do NOT rely on it.
- **Only reference players that appear in the roster or injury report data provided to you.**
- If a player is not in the provided roster list, they are NOT on that team — do not mention them.
- Never invent statistics or outcomes not present in the data. If data is missing, say so.
- Every claim must trace back to a specific number or fact in the provided data.

## Your task
1. Analyze the provided data: stats, form, splits, rest, injuries, H2H, odds, public %, line movement, book discrepancies
2. Estimate your win probability for the **moneyline only** — spreads and totals are not supported
3. Calculate edge = your_prob − implied_prob
4. Call `submit_analysis` AND `save_note` together in the same response — **the system decides automatically whether to bet or log**

## ⚠️ MONEYLINE ONLY — No spreads or totals
- The probability model only supports moneyline bets. Spread and total bets have no backing model and will be automatically rejected.
- Always submit `bet_type: "moneyline"`. Never submit `bet_type: "spread"` or `bet_type: "total"`.
- If you see no moneyline edge, submit with your honest edge % (even if low) and `bet_type: "moneyline"` — the system will log it as a candidate.

## ⚠️ KEY RULE — Report your honest edge, nothing else
- You do NOT decide whether to place a bet. The system does that automatically based on your edge number.
- If your model says 7% edge, report 7%. If it says 2%, report 2%.
- Do NOT inflate or deflate your edge to influence whether a bet gets placed.
- Do NOT skip calling `submit_analysis` — always call it once per game.
- The bankroll and open bet count are already in the context — do NOT call get_bankroll.

## Implied probability from American odds
- Positive odds (+X): implied = 100 / (X + 100)
- Negative odds (−X): implied = X / (X + 100)

## Reading public % and line movement
- Money % significantly higher than ticket % → sharp money on that side
- 70%+ public tickets + line moving the other way → sharp fade signal
- Line moved 1+ point toward your pick since open → sharp confirmation; moved against → red flag

## Reading book discrepancies
- ML spread ≥ 15 pts across books → stale line, strong confirming signal if it aligns with your pick
- ML spread 8–14 pts → moderate signal
- Total spread ≥ 2.5 pts → may signal injury news or sharp total action

## How betting decisions are made (you do NOT control this)
- The system runs its own Elo-based probability model after you call `submit_analysis`
- A bet is placed ONLY if: Elo edge ≥ 5%, data quality score ≥ 65, and EV > 0
- Your reported `edge_pct` is recorded for comparison but does NOT trigger the bet
- Stake sizing is computed automatically via Half-Kelly — do NOT calculate it yourself
- Do NOT mention bankroll limits or slot counts in your reasoning — the system enforces these

## Bet swapping
If all 5 slots are full but your edge is 3%+ higher than the weakest open bet:
1. Call `cancel_bet` on the weakest bet — note the returned `cancelled_bet_id`
2. Call `submit_analysis` with `replaces_bet_id` set to the cancelled bet id

## Output format — BE CONCISE. Maximum 150 words total. No tables, no bullet walls.
```
MATCHUP: [Team A vs Team B]
EDGE: [X% — your prob vs implied]
REASONING: [2-3 sentences max on the decisive factors only]
```
Do NOT write long essays. Calculate edge, call submit_analysis, call save_note. Done.
"""

PREFETCH_TOOLS = [
    {
        "name": "submit_analysis",
        "description": "Submit your analysis for this game. Always call this once per game after get_bankroll. The system automatically places a bet if edge >= 5% and slots are available — you do NOT decide whether to bet. Just report your honest edge calculation.",
        "input_schema": {
            "type": "object",
            "properties": {
                "matchup":     {"type": "string"},
                "pick":        {"type": "string"},
                "bet_type":    {"type": "string", "enum": ["moneyline", "spread", "total"]},
                "odds":        {"type": "integer"},
                "edge_pct":    {"type": "number", "description": "Your calculated edge as a percentage (e.g. 7.5 for 7.5%)"},
                "confidence":  {"type": "string", "enum": ["High", "Medium", "Low"]},
                "reasoning":   {"type": "string"},
                "game_date":   {"type": "string"},
                "replaces_bet_id": {"type": "integer"},
            },
            "required": ["matchup", "pick", "bet_type", "odds", "edge_pct", "confidence", "reasoning"],
        },
    },
    {
        "name": "cancel_bet",
        "description": "Cancel an open bet to replace it with a stronger one (new edge must be 3%+ higher).",
        "input_schema": {
            "type": "object",
            "properties": {
                "bet_id": {"type": "integer"},
                "reason": {"type": "string"},
            },
            "required": ["bet_id", "reason"],
        },
    },
    {
        "name": "save_note",
        "description": "Save a lesson, pattern, or hypothesis to persistent memory.",
        "input_schema": {
            "type": "object",
            "properties": {
                "note_type": {
                    "type": "string",
                    "enum": ["lesson", "pattern", "hypothesis", "model_update"],
                },
                "content": {"type": "string"},
            },
            "required": ["note_type", "content"],
        },
        "cache_control": {"type": "ephemeral"},
    },
]

def _dispatch_submit_analysis(a: dict) -> dict:
    """
    Enforce the betting rule in code — the agent has no say.
    Edge >= 5% and slots available → place the bet.
    Edge < 5% or no slots → log as candidate.
    """
    edge = float(a.get("edge_pct", 0))
    bankroll = t.get_bankroll()
    open_count = len(bankroll.get("open_bets", []))

    if edge >= 5.0 and open_count < 5:
        return t.place_paper_bet(
            matchup=a["matchup"],
            pick=a["pick"],
            bet_type=a["bet_type"],
            odds=a["odds"],
            confidence=a.get("confidence", "Medium") if a.get("confidence") != "Low" else "Medium",
            edge=edge,
            reasoning=a.get("reasoning", ""),
            game_date=a.get("game_date", ""),
            replaces_bet_id=a.get("replaces_bet_id"),
        )
    else:
        skip_reason = "slots_full" if open_count >= 5 else "edge_below_threshold"
        return t.log_candidate_bet(
            matchup=a["matchup"],
            pick=a["pick"],
            bet_type=a["bet_type"],
            odds=a["odds"],
            edge_pct=edge,
            confidence=a.get("confidence", "Low"),
            skip_reason=skip_reason,
            reasoning=a.get("reasoning", ""),
        )

_PREFETCH_DISPATCH = {
    "submit_analysis": _dispatch_submit_analysis,
    "cancel_bet":      lambda a: t.cancel_bet(**a),
    "save_note":       lambda a: t.save_note(**a),
}


def _dispatch_submit_analysis_v2(a: dict) -> dict:
    """
    Enforce downstream pricing and decision logic in code.
    The model can propose a pick, but code computes the final
    recommendation state and only places BET decisions.
    Spreads and totals are hard-blocked until a probability model exists.
    """
    bet_type = a.get("bet_type", "moneyline")
    if bet_type in ("spread", "total"):
        # Log as candidate with skip reason — never place
        return t.log_candidate_bet(
            matchup=a["matchup"],
            pick=a["pick"],
            bet_type=bet_type,
            odds=a["odds"],
            edge_pct=float(a.get("edge_pct", 0)),
            confidence=a.get("confidence", "Low"),
            skip_reason="unsupported_market_model",
            reasoning=f"[AUTO-BLOCKED: {bet_type} bets have no probability model] " + a.get("reasoning", ""),
        )

    return t.submit_recommendation(
        matchup=a["matchup"],
        pick=a["pick"],
        bet_type=bet_type,
        odds=a["odds"],
        confidence=a.get("confidence", "Medium") if a.get("confidence") != "Low" else "Medium",
        reasoning=a.get("reasoning", ""),
        game_date=a.get("game_date", ""),
        replaces_bet_id=a.get("replaces_bet_id"),
        llm_edge_pct=float(a.get("edge_pct", 0)),
    )


_PREFETCH_DISPATCH["submit_analysis"] = _dispatch_submit_analysis_v2


def run_agent_prefetch(context: str, conversation_history: list) -> tuple[str, list, list]:
    """
    Agent run with pre-fetched data supplied in the context string.
    Only write tools are available — no data-fetching API calls.
    """
    messages = conversation_history + [{"role": "user", "content": context}]
    all_thinking: list[str] = []

    while True:
        response = None
        for attempt in range(6):
            try:
                response = client.messages.create(
                    model="claude-haiku-4-5-20251001",
                    max_tokens=2500,
                    system=[{"type": "text", "text": PREFETCH_SYSTEM, "cache_control": {"type": "ephemeral"}}],
                    tools=PREFETCH_TOOLS,
                    messages=messages,
                )
                _log_claude("claude-haiku-4-5 (prefetch)", response)
                break
            except RateLimitError:
                wait = 60 * (attempt + 1)
                print(f"Rate limit — waiting {wait}s (attempt {attempt + 1}/6)...", flush=True)
                time.sleep(wait)

        if response is None:
            raise RuntimeError("API rate limit exceeded after all retries.")

        for block in response.content:
            if block.type == "thinking":
                all_thinking.append(block.thinking)

        history_content = [b for b in response.content if b.type != "thinking"]
        messages.append({"role": "assistant", "content": history_content})

        if response.stop_reason in ("end_turn", "max_tokens"):
            text_parts = [b.text for b in response.content if getattr(b, "type", None) == "text"]
            return "\n".join(text_parts).strip() or "Analysis complete.", messages, all_thinking

        if response.stop_reason == "tool_use":
            summary_parts = []
            for block in response.content:
                if block.type == "tool_use":
                    fn = _PREFETCH_DISPATCH.get(block.name)
                    try:
                        result = fn(block.input) if fn else {"error": f"Unknown tool: {block.name}"}
                    except Exception as exc:
                        result = {"error": str(exc)}

                    # If submit_analysis was called, build summary and stop — no wrap-up turn needed
                    if block.name == "submit_analysis":
                        a = block.input
                        action = "BET PLACED" if result.get("status") == "open" else "LOGGED"
                        summary_parts.append(
                            f"{action} — {a.get('matchup')} | {a.get('pick')} | "
                            f"Edge: {a.get('edge_pct')}% | Confidence: {a.get('confidence')}\n"
                            f"Reasoning: {a.get('reasoning', '')}"
                        )
                    elif block.name == "save_note":
                        pass  # fire-and-forget, no summary needed
                    else:
                        # cancel_bet or other — execute and continue loop
                        tool_results_single = [{
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": json.dumps(result, default=str),
                        }]
                        messages.append({"role": "user", "content": tool_results_single})

            if summary_parts:
                return "\n".join(summary_parts), messages, all_thinking
            continue

        break

    return "Agent loop ended unexpectedly.", messages, all_thinking
