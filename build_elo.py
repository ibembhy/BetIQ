"""
BetIQ — One-time Elo initialization script.

Builds team Elo ratings from 3+ seasons of historical NBA game data.
After running this once, ratings update automatically after each resolved bet.

Usage:
    venv\\Scripts\\activate
    python build_elo.py

Estimated time: ~4-6 minutes (BDL API rate limit pacing).
"""

import logging
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env'))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("build_elo")

import database as db
import elo

def main():
    log.info("BetIQ Elo Builder")
    log.info("=================")
    log.info("Building ratings from seasons 2022-2025...")
    log.info("This takes ~4-6 minutes due to API rate limiting.\n")

    db.init_db()
    result = elo.build_from_history(seasons=[2022, 2023, 2024, 2025])

    log.info(f"\nDone! {result['teams']} teams rated from {result['games_processed']} games.\n")
    log.info("Current Elo Rankings:")
    log.info("-" * 45)
    for team, rating in result["ratings"].items():
        diff = rating - 1500
        sign = "+" if diff >= 0 else ""
        log.info(f"  {team:<32} {rating:>7.1f}  ({sign}{diff:.0f})")
    log.info("\nRatings are live. They update automatically after each resolved bet.")

if __name__ == "__main__":
    main()
