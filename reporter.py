"""
BetIQ — Daily report generator.

Generates a narrative report (PDF + text) for a game_date once all bets for
that day are resolved. Called automatically from resolve_bets().
"""

import os
import json
from anthropic import Anthropic
from dotenv import load_dotenv

import database as db

load_dotenv()

REPORTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")

_REPORT_PROMPT = """\
You are BetIQ, an NBA sports betting analyst writing your daily report. \
Write in plain English — the audience is sports fans, not professionals. \
Explain every decision clearly. Be honest, including about mistakes.

Game date: {game_date}

Bets placed and their outcomes:
{bets_json}

CLV (Closing Line Value) explanation for context: CLV measures whether we bet \
at better odds than where the line eventually closed. A positive CLV means we \
beat the market — the line moved against us after we bet, which is a sign of \
genuine edge. Negative CLV means the line moved in our favour, suggesting the \
market agreed we had the wrong side. CLV is expressed as a percentage — e.g. \
+2.5% means our implied probability was 2.5 points better than closing.

Write the report using EXACTLY this structure (keep the headers):

BETIQ DAILY REPORT — {game_date}

OVERVIEW
[2-3 sentences: how many games you scanned, how many bets you placed, \
the final win/loss split, and the net P&L for the day.]

{bet_sections}

CLV PERFORMANCE
[Only include this section if at least one bet has clv data (clv field is not null). \
For each bet that has a clv value, show: "PICK — CLV: +X.X% / -X.X% (closing odds: ODDS)". \
Then write 2-3 sentences explaining what the CLV numbers mean in plain English: \
are we consistently beating the closing line? Is there evidence of real edge or luck? \
If NO bets have clv data, write: "CLV data not yet available for this date — \
closing odds are captured automatically and will appear in future reports."]

DAILY SUMMARY
Bets placed: [N] | Won: [N] | Lost: [N] | Net P&L: $[X]{clv_summary_line}
Key takeaway: [One concrete lesson or observation from today — something \
you will apply to future scans.]

Rules for the bet sections:
- One section per bet, titled: BET: [PICK] — WON or LOST
- Sub-fields: Matchup / Type / Odds / Stake / P&L
- WHY WE BET THIS: 3-5 sentences in plain English covering the key signals \
(team form, rest, injuries, head-to-head, odds value). No numbers without context.
- RESULT: one line stating outcome and dollar amount.
- SELF-EVALUATION (LOST bets only): honest paragraph explaining what the model \
missed or got wrong and what to watch for differently next time.
"""

_BET_SECTION_TEMPLATE = """\
BET: {pick} — {result_label}
Matchup: {matchup} | Type: {bet_type} | Odds: {odds_fmt} | Stake: ${stake:.2f} | P&L: ${pnl:+.2f}

WHY WE BET THIS:
{reasoning}

RESULT: {result_label} — {'Profit' if pnl > 0 else 'Loss'}: ${pnl:+.2f}
"""

_SELF_EVAL_PLACEHOLDER = "SELF-EVALUATION:\n{reasoning}"


def _fmt_american(odds: int) -> str:
    return f"+{odds}" if odds > 0 else str(odds)


def _clean_for_pdf(text: str) -> str:
    """Replace characters that latin-1 can't encode."""
    replacements = {
        "\u2014": "--", "\u2013": "-",
        "\u2019": "'", "\u2018": "'",
        "\u201c": '"', "\u201d": '"',
        "\u2026": "...",
    }
    for char, repl in replacements.items():
        text = text.replace(char, repl)
    # Drop any remaining non-latin-1 chars
    return text.encode("latin-1", errors="replace").decode("latin-1")


def _report_exists(game_date: str) -> bool:
    return os.path.exists(os.path.join(REPORTS_DIR, f"{game_date}.txt"))


def maybe_generate_report(game_date: str) -> bool:
    """
    Generate the daily report for game_date if not already done.
    Returns True if a new report was created.
    """
    if _report_exists(game_date):
        return False

    bets = [
        b for b in db.get_all_bets()
        if b["game_date"] == game_date
        and b["status"] in ("won", "lost", "push", "cancelled")
    ]
    if not bets:
        return False

    narrative = _call_claude(game_date, bets)
    _save_report(game_date, narrative)
    return True


def _call_claude(game_date: str, bets: list) -> str:
    client = Anthropic()

    bets_summary = [
        {
            "matchup":      b["matchup"],
            "pick":         b["pick"],
            "bet_type":     b["bet_type"],
            "odds":         b["odds"],
            "stake":        b["stake"],
            "confidence":   b["confidence"],
            "edge_pct":     b["edge"],
            "reasoning":    b["reasoning"] or "No reasoning recorded.",
            "result":       b["status"],
            "pnl":          b["pnl"],
            "closing_odds": b.get("closing_odds"),
            "clv":          b.get("clv"),   # positive = beat closing line
        }
        for b in bets
    ]

    # Build placeholder section headers so the model knows how many bets to cover
    bet_sections = "\n".join(
        f"[Section for bet {i+1}: {b['pick']} — {b['result'].upper()}]"
        for i, b in enumerate(bets_summary)
    )

    # Pre-compute CLV summary line for DAILY SUMMARY
    clv_values = [b["clv"] for b in bets_summary if b["clv"] is not None]
    if clv_values:
        avg_clv = sum(clv_values) / len(clv_values)
        positive = sum(1 for v in clv_values if v > 0)
        clv_summary_line = (
            f" | Avg CLV: {avg_clv:+.2f}% ({positive}/{len(clv_values)} bets beat closing line)"
        )
    else:
        clv_summary_line = ""

    prompt = _REPORT_PROMPT.format(
        game_date=game_date,
        bets_json=json.dumps(bets_summary, indent=2),
        bet_sections=bet_sections,
        clv_summary_line=clv_summary_line,
    )

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=4000,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


def _save_report(game_date: str, narrative: str):
    os.makedirs(REPORTS_DIR, exist_ok=True)

    txt_path = os.path.join(REPORTS_DIR, f"{game_date}.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(narrative)

    _save_pdf(game_date, narrative)


def _save_pdf(game_date: str, narrative: str):
    try:
        from fpdf import FPDF

        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()

        # Title block
        pdf.set_font("Helvetica", "B", 18)
        pdf.cell(0, 12, "BetIQ Daily Report", ln=True, align="C")
        pdf.set_font("Helvetica", "", 11)
        pdf.cell(0, 7, game_date, ln=True, align="C")
        pdf.ln(6)
        pdf.set_draw_color(60, 60, 100)
        pdf.set_line_width(0.5)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(6)

        for line in _clean_for_pdf(narrative).splitlines():
            stripped = line.strip()
            if not stripped:
                pdf.ln(3)
                continue

            # Section headers (ALL CAPS lines or lines starting with "BET:")
            if (
                stripped.isupper()
                or stripped.startswith("BET:")
                or stripped.startswith("BETIQ")
                or stripped.startswith("OVERVIEW")
                or stripped.startswith("DAILY SUMMARY")
                or stripped.startswith("WHY WE BET")
                or stripped.startswith("RESULT:")
                or stripped.startswith("SELF-EVALUATION")
                or stripped.startswith("CLV PERFORMANCE")
            ):
                pdf.set_font("Helvetica", "B", 10)
            else:
                pdf.set_font("Helvetica", "", 10)

            pdf.multi_cell(0, 5, stripped)

        pdf_path = os.path.join(REPORTS_DIR, f"{game_date}.pdf")
        pdf.output(pdf_path)

    except Exception:
        pass  # PDF failure never blocks the main flow


# ── Public helpers for app.py ─────────────────────────────────────────────────

def list_reports() -> list[dict]:
    """Return available reports sorted newest first."""
    if not os.path.exists(REPORTS_DIR):
        return []
    reports = []
    for fname in sorted(os.listdir(REPORTS_DIR), reverse=True):
        if fname.endswith(".txt"):
            date = fname[:-4]
            pdf = os.path.join(REPORTS_DIR, f"{date}.pdf")
            reports.append({
                "date":     date,
                "txt_path": os.path.join(REPORTS_DIR, fname),
                "pdf_path": pdf if os.path.exists(pdf) else None,
            })
    return reports


def get_report_text(game_date: str) -> str | None:
    path = os.path.join(REPORTS_DIR, f"{game_date}.txt")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return f.read()
