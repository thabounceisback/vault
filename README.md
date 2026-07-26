# 🥨 The Vault — A League About Nothing

An interactive Streamlit dashboard for mining an ESPN fantasy league's own history:
draft tendencies, hindsight draft value, schedule luck, positional performance,
manager behavior, and all-time records.

The app works out of the box with generated sample data. To use a private ESPN
league, provide your `LEAGUE_ID`, `SWID`, and `espn_s2` cookie in the sidebar or
environment variables.

## Quickstart

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Or use the project launcher, which avoids accidentally running through a global
or Anaconda Python environment:

```bash
./run_dashboard.sh
```

## ESPN Configuration

Private league history requires ESPN cookies:

```bash
export ESPN_LEAGUE_ID=123456
export ESPN_SWID='{XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX}'
export ESPN_S2='your-long-espn-s2-cookie'
export ESPN_COOKIE='SWID={...}; espn_s2=...; ...'
export ESPN_SEASONS='2021,2022,2023,2024,2025'
```

Then open the dashboard and click **Sync ESPN history** in the sidebar.

The full `ESPN_COOKIE` header is optional, but it is the most reliable auth
path. In browser DevTools, open the Network tab, reload your ESPN Fantasy league
page, select an ESPN API request, and copy the `Cookie` request header.

## Branding

The Vault is themed like a dark bank-vault backdrop lit up with a Seinfeld-inspired
color palette (puffy-shirt mustard, Kramer's-door red, Jerry's-shirt blue, and so
on) - see `.streamlit/config.toml` and the `BRAND` constants at the top of
`app.py`. Section and chart names riff on the show (`🌌 Bizarro World` for
schedule luck, `💼 Vandelay Inc.` for the draft room, `🔮 No Points For You` for
projections, ...), but every tab and chart keeps a plain-English caption or
subtitle right underneath so the joke never gets in the way of reading the data.

## Testing

```bash
pip install -r requirements-dev.txt
pytest
```

## Data Model

The dashboard stores everything in a single local SQLite file, shared by whoever
runs this instance - there is no per-user or per-session isolation. If you run
it somewhere multiple people can access at once, one person's "Load sample" or
"Sync ESPN" click can overwrite another's in-progress data (including data
derived from their private ESPN auth cookies). It's designed to be run locally
by one person/league at a time.

SQLite tables are kept in `data/league_history.sqlite`:

- `managers`: stable manager names
- `teams`: team names by season
- `draft_picks`: draft recap by season
- `matchups`: weekly head-to-head outcomes
- `roster_scores`: slot/player scoring lines where ESPN provides box scores
- `roster_scores.projected_points`: ESPN projections where present, used for expectation analysis
- `transactions`: adds, drops, trades, and other activity

## Notes

ESPN's private API changes shape across seasons and league settings. This app
normalizes the fields used by the dashboard and keeps raw pulls in memory only;
if a league lacks a view, that section simply shows the available data.
