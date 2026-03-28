"""
Re-build Elo ratings with slower paging (2s delay) to avoid BDL rate limits.
Run once: venv/bin/python rebuild_elo.py
"""
import sys, os, time, logging
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env'))

logging.basicConfig(level=logging.INFO, format='%(asctime)s  %(message)s')
log = logging.getLogger('rebuild_elo')

import database as db
import tools as t
import elo
from elo import update_ratings, apply_season_regression, DEFAULT_RATING

db.init_db()

# Clear existing ratings
conn = db.get_connection()
conn.execute('DELETE FROM elo_ratings')
conn.commit()
conn.close()
log.info('Cleared existing ratings. Rebuilding from scratch...')

seasons = [2022, 2023, 2024, 2025]
ratings = {}
total_games = 0

for season in seasons:
    log.info(f'Fetching season {season}...')
    games = []
    cursor = None
    while True:
        params = {'seasons[]': season, 'per_page': 100}
        if cursor is not None:
            params['cursor'] = cursor
        data = t._bdl_get('/games', params)
        if 'error' in data:
            log.warning(f'  BDL error: {data["error"]}')
            break
        page = data.get('data', [])
        games.extend(page)
        log.info(f'  {len(games)} games fetched...')
        cursor = data.get('meta', {}).get('next_cursor')
        if not cursor or not page:
            break
        time.sleep(2.0)  # Slow down to avoid rate limits

    log.info(f'  Season {season}: {len(games)} total games.')

    # Season regression
    for tid in ratings:
        if ratings[tid]['season'] < season:
            ratings[tid]['rating'] = apply_season_regression(ratings[tid]['rating'])
            ratings[tid]['gp'] = 0
            ratings[tid]['season'] = season

    games.sort(key=lambda g: g.get('date', ''))
    for game in games:
        if game.get('status') != 'Final':
            continue
        hs = game.get('home_team_score') or 0
        vs = game.get('visitor_team_score') or 0
        if hs == 0 and vs == 0:
            continue
        ht = game['home_team']
        at = game['visitor_team']
        hid, aid = ht['id'], at['id']
        hr = ratings.get(hid, {}).get('rating', DEFAULT_RATING)
        ar = ratings.get(aid, {}).get('rating', DEFAULT_RATING)
        new_hr, new_ar = update_ratings(hr, ar, hs, vs)
        ratings[hid] = {'rating': new_hr, 'season': season, 'gp': ratings.get(hid, {}).get('gp', 0) + 1, 'name': ht['full_name']}
        ratings[aid] = {'rating': new_ar, 'season': season, 'gp': ratings.get(aid, {}).get('gp', 0) + 1, 'name': at['full_name']}
        total_games += 1

    time.sleep(5)  # Pause between seasons

# Save to DB
current_season = t._current_season()
for tid, data in ratings.items():
    db.upsert_elo_rating(tid, data['name'], data['rating'], current_season, data['gp'])

log.info(f'\nDone! {len(ratings)} teams, {total_games} games processed.\n')
log.info('Final Rankings:')
for name, r in sorted(((d['name'], round(d['rating'],1)) for d in ratings.values()), key=lambda x: -x[1]):
    diff = r - 1500
    log.info(f'  {name:<32} {r}  ({"+"+str(int(diff)) if diff>=0 else int(diff)})')
