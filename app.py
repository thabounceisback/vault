from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from league_history.analytics import (
    acquisition_source_league_average,
    aggregate_luck,
    all_time_leaderboards,
    all_time_records,
    auction_player_values,
    draft_hindsight,
    draft_scorecard,
    draft_tendencies,
    head_to_head_game_table,
    head_to_head_history,
    injury_loss_detail,
    injury_luck,
    manager_profiles,
    player_profile_frames,
    positional_hall_of_fame,
    positional_performance,
    projection_matchup_matrix,
    projection_performance,
    schedule_luck,
    transaction_scorecard,
)
from league_history.espn_client import EspnConfig, EspnSyncError, cookie_auth_summary, sync_espn_history
from league_history.nflverse_injuries import InjurySourceError, fetch_nflverse_injuries
from league_history.sample_data import seed_sample_database
from league_history.storage import Database


DB_PATH = Path("data/league_history.sqlite")

# --- The Fantasy Vault brand system -----------------------------------------
# A torchlit treasure-hall backdrop (this is "the vault", after all) - obsidian
# stone, aged gold trim, and a jewel-tone accent palette for the relics/records
# on display. Fun names go on the chrome - tab titles, chart titles, headers,
# empty states - never on the functional widget labels/errors, so the app
# stays easy to actually use.
BRAND = {
    "gold": "#D4AF37",        # the vault door, trophies, primary accent
    "old_gold": "#8C6D1F",    # dimmer gold for borders/hover states
    "ruby": "#B0223F",
    "sapphire": "#2A5DB0",
    "emerald": "#1E8F5F",
    "amethyst": "#7B4FA6",
    "topaz": "#E0932A",
    "aquamarine": "#3FA9A4",
    "bronze": "#8C6A3F",
    "pewter": "#B9B2A4",       # muted secondary text
    "obsidian": "#0B0906",     # page background
    "stone": "#17130D",        # panel/card background
    "parchment": "#F0E6D2",    # primary text
}
COLORWAY = [
    BRAND["gold"],
    BRAND["ruby"],
    BRAND["sapphire"],
    BRAND["emerald"],
    BRAND["amethyst"],
    BRAND["topaz"],
    BRAND["aquamarine"],
    BRAND["bronze"],
    BRAND["pewter"],
]
PLOT_TEMPLATE = "plotly_dark"
DISPLAY_FONT = "'Cinzel', 'Georgia', 'Times New Roman', serif"
BODY_FONT = "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif"


st.set_page_config(
    page_title="The Fantasy Vault — League History & Records",
    page_icon="🏆",
    layout="wide",
)

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@500;600;700;900&display=swap');
    .stApp {{
        background:
            radial-gradient(circle at 18% 0%, rgba(212, 175, 55, 0.14), transparent 30rem),
            radial-gradient(circle at 85% 6%, rgba(176, 34, 63, 0.10), transparent 32rem),
            {BRAND["obsidian"]};
    }}
    [data-testid="stSidebar"] {{
        background: {BRAND["stone"]};
        border-right: 1px solid rgba(212, 175, 55, 0.18);
    }}
    h1, h2, h3 {{
        font-family: {DISPLAY_FONT} !important;
        letter-spacing: 0.02em;
        color: {BRAND["parchment"]};
    }}
    p, span, label, div {{ font-family: {BODY_FONT} !important; }}
    [data-testid="stIconMaterial"] {{ font-family: 'Material Symbols Rounded' !important; }}
    div[data-testid="stMetric"] {{
        background: rgba(212, 175, 55, 0.06);
        border: 1px solid rgba(212, 175, 55, 0.28);
        border-radius: 8px;
        padding: 14px 16px;
    }}
    [data-testid="stMetricLabel"], [data-testid="stMetricLabel"] * {{
        color: {BRAND["pewter"]};
        white-space: normal !important;
        overflow: visible !important;
        text-overflow: clip !important;
        display: block !important;
        height: auto !important;
        max-height: none !important;
        -webkit-line-clamp: unset !important;
    }}
    div[data-testid="stMetricValue"], div[data-testid="stMetricValue"] * {{
        white-space: normal !important;
        overflow: visible !important;
        text-overflow: clip !important;
        font-size: 1.1rem;
        line-height: 1.3;
    }}
    div[data-testid="stTabs"] [role="tablist"] {{
        gap: 4px;
        border-bottom: 1px solid rgba(212, 175, 55, 0.22);
        flex-wrap: wrap !important;
        overflow: visible !important;
        height: auto !important;
    }}
    div[data-testid="stTabs"] [data-testid="stTab"] {{
        border-radius: 8px 8px 0 0;
        padding: 8px 10px;
        font-family: {DISPLAY_FONT} !important;
        font-size: 0.92rem;
    }}
    div[data-testid="stTabs"] [aria-selected="true"] {{
        color: {BRAND["gold"]} !important;
    }}
    .stButton > button {{
        border: 1px solid rgba(212, 175, 55, 0.35);
    }}
    .stButton > button:hover {{
        border-color: {BRAND["gold"]};
        color: {BRAND["gold"]};
    }}
    .vault-tagline {{
        color: {BRAND["pewter"]};
        font-style: italic;
        margin-top: -0.6rem;
    }}
    div[data-testid="stVerticalBlock"][style*="border"] {{
        background: rgba(212, 175, 55, 0.035);
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.25);
    }}
    .vault-chart-head {{
        font-family: {DISPLAY_FONT} !important;
        font-size: 1.15rem;
        font-weight: 600;
        color: {BRAND["gold"]};
        line-height: 1.3;
        margin-bottom: 1px;
    }}
    .vault-chart-sub {{
        font-family: {BODY_FONT} !important;
        font-size: 0.85rem;
        color: {BRAND["pewter"]};
        opacity: 0.85;
        margin-bottom: 10px;
        line-height: 1.35;
    }}
    /* Below this width, side-by-side columns get too cramped to read (control
       widgets truncate, chart cards squeeze) - let them stack instead. */
    @media (max-width: 960px) {{
        div[data-testid="stHorizontalBlock"] {{
            flex-wrap: wrap !important;
        }}
        div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"] {{
            min-width: 100% !important;
            flex: 1 1 100% !important;
        }}
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_db() -> Database:
    db = Database(DB_PATH)
    db.initialize()
    return db


@st.cache_data(show_spinner=False)
def load_tables(db_path: str, cache_key: float) -> dict[str, pd.DataFrame]:
    db = Database(Path(db_path))
    return db.read_all()


def db_cache_key() -> float:
    return DB_PATH.stat().st_mtime if DB_PATH.exists() else 0


def apply_owner_aliases(tables: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    aliases = tables.get("owner_aliases", pd.DataFrame())
    if aliases.empty:
        return tables
    alias_map = dict(zip(aliases["manager_id"].astype(str), aliases["display_name"]))
    for name in ("managers", "teams"):
        df = tables.get(name)
        if df is not None and not df.empty and "manager_id" in df.columns:
            tables[name] = df.assign(
                manager_name=df["manager_id"].astype(str).map(alias_map).fillna(df["manager_name"])
            )
    return tables


def manager_filter(df: pd.DataFrame, managers: list[str]) -> pd.DataFrame:
    if "manager_name" not in df.columns:
        return df
    if df["manager_name"].dropna().empty:
        return df
    # An explicitly empty selection should show nothing, not fall back to "no filter".
    return df[df["manager_name"].isin(managers)]


def _pick_column(frame: pd.DataFrame, candidates: list[str]) -> str | None:
    normalized = {str(col).strip().lower().replace(" ", "_"): col for col in frame.columns}
    for candidate in candidates:
        if candidate in normalized:
            return normalized[candidate]
    return None


def normalize_import(frame: pd.DataFrame, import_type: str) -> tuple[pd.DataFrame, str | None]:
    if frame.empty:
        return pd.DataFrame(), "The uploaded file is empty."
    if import_type == "auction":
        mapping = {
            "season": ["season", "year"],
            "player_id": ["player_id", "playerid", "espn_player_id"],
            "player_name": ["player_name", "player", "name"],
            "team_id": ["team_id", "teamid", "espn_team_id"],
            "manager_name": ["manager_name", "owner", "manager"],
            "auction_value": ["auction_value", "price", "cost", "salary", "draft_price"],
        }
        required = ["season", "auction_value"]
        table = "auction_values"
    else:
        mapping = {
            "season": ["season", "year"],
            "week": ["week", "scoring_period", "scoringperiod"],
            "player_id": ["player_id", "playerid", "espn_player_id"],
            "player_name": ["player_name", "player", "name"],
            "injury_status": ["injury_status", "status", "injury", "designation"],
        }
        required = ["season", "week", "injury_status"]
        table = "injuries"

    out: dict[str, pd.Series] = {}
    for target, candidates in mapping.items():
        source = _pick_column(frame, candidates)
        out[target] = frame[source] if source else pd.Series([None] * len(frame))
    result = pd.DataFrame(out)
    missing = [column for column in required if result[column].isna().all()]
    if missing:
        return pd.DataFrame(), f"Missing required column(s) for {table}: {', '.join(missing)}."
    if result.get("player_id") is not None:
        result["player_id"] = pd.to_numeric(result["player_id"], errors="coerce").astype("Int64")
    result["season"] = pd.to_numeric(result["season"], errors="coerce").astype("Int64")
    if "week" in result.columns:
        result["week"] = pd.to_numeric(result["week"], errors="coerce").astype("Int64")
    if "team_id" in result.columns:
        result["team_id"] = pd.to_numeric(result["team_id"], errors="coerce").astype("Int64")
    if "auction_value" in result.columns:
        result["auction_value"] = pd.to_numeric(result["auction_value"], errors="coerce")
    result["source"] = "upload"
    result = result.dropna(subset=required)
    return result, None


def sidebar(db: Database) -> tuple[list[int], list[str]]:
    with st.sidebar:
        st.markdown("## 🏆 The Fantasy Vault")
        st.markdown("<p class='vault-tagline'>Every season becomes a relic.</p>", unsafe_allow_html=True)
        tables = apply_owner_aliases(load_tables(str(DB_PATH), db_cache_key()))
        profiles = tables.get("league_profiles", pd.DataFrame())

        # Filters live up top since they're what you'll actually touch most often -
        # league setup is fussed with once and then mostly left alone.
        with st.container(border=True):
            st.subheader("🔎 Filters")
            seasons = sorted(tables["teams"]["season"].dropna().unique().tolist()) if not tables["teams"].empty else []
            selected_seasons = st.multiselect("Visible seasons", seasons, default=seasons)

            managers = sorted(tables["teams"]["manager_name"].dropna().unique().tolist()) if not tables["teams"].empty else []
            selected_managers = st.multiselect("Managers", managers, default=managers)

        with st.container(border=True):
            st.subheader("⚙️ League Setup")
            st.caption("Sync a real ESPN league, load the sample league, or import your own CSVs.")

            selected_profile = None
            if not profiles.empty:
                def _profile_label(profile_id: int | None) -> str:
                    if profile_id is None:
                        return "New / unsaved"
                    row = profiles.loc[profiles["profile_id"] == profile_id].iloc[0]
                    return f"{row.league_name} ({int(row.league_id)})"

                profile_options: list[int | None] = [None] + profiles["profile_id"].tolist()
                selected_profile_id = st.selectbox("Saved league", profile_options, format_func=_profile_label)
                if selected_profile_id is not None:
                    selected_profile = profiles.loc[profiles["profile_id"] == selected_profile_id].iloc[0]

            default_league_id = str(int(selected_profile["league_id"])) if selected_profile is not None else os.getenv("ESPN_LEAGUE_ID", "")
            default_seasons = str(selected_profile["seasons"]) if selected_profile is not None else os.getenv("ESPN_SEASONS", "2021,2022,2023,2024,2025")
            default_name = str(selected_profile["league_name"]) if selected_profile is not None else "My League"

            league_name = st.text_input("League name", default_name)
            league_id = st.text_input("ESPN league ID", default_league_id)
            seasons_text = st.text_input("Seasons", default_seasons)

            with st.expander("Advanced: ESPN authentication"):
                st.caption("Only needed for private leagues - public leagues can sync with just a league ID.")
                swid = st.text_input("SWID", os.getenv("ESPN_SWID", ""), type="password")
                espn_s2 = st.text_input("espn_s2", os.getenv("ESPN_S2", ""), type="password")
                cookie_header = st.text_input(
                    "Cookie header or JSON",
                    os.getenv("ESPN_COOKIE", ""),
                    type="password",
                    help="Optional. Accepts a browser Cookie header or JSON with swid/espn_s2 keys.",
                )
                if swid or espn_s2 or cookie_header:
                    st.caption(cookie_auth_summary(cookie_header or None, swid or None, espn_s2 or None))

            if st.button("💾 Save league", width="stretch"):
                try:
                    db.save_league_profile(int(league_id), league_name.strip() or "My League", seasons_text)
                    st.cache_data.clear()
                    st.success("League saved to the vault.")
                except ValueError:
                    st.error("League ID must be a number before saving.")

            if st.button("🎲 Load sample", width="stretch"):
                if db.is_empty():
                    seed_sample_database(db, replace=True)
                    st.cache_data.clear()
                    st.success("Sample league loaded. The vault is stocked.")
                else:
                    st.session_state["confirm_load_sample"] = True

            if st.session_state.get("confirm_load_sample"):
                st.warning(
                    "This will permanently replace your current league data "
                    "(synced ESPN history, uploaded CSVs, etc.) with generated sample data."
                )
                confirm_col, cancel_col = st.columns(2)
                if confirm_col.button("✅ Yes, overwrite", width="stretch"):
                    seed_sample_database(db, replace=True)
                    st.cache_data.clear()
                    st.session_state["confirm_load_sample"] = False
                    st.success("Sample league loaded.")
                if cancel_col.button("✖️ Cancel", width="stretch"):
                    st.session_state["confirm_load_sample"] = False

            if st.button("🔄 Sync ESPN", width="stretch"):
                try:
                    seasons = [int(part.strip()) for part in seasons_text.split(",") if part.strip()]
                except ValueError:
                    seasons = []
                    st.error("Seasons must be comma-separated years, like 2021,2022,2023.")
                if not league_id or not seasons:
                    st.error("League ID and at least one season are required.")
                else:
                    try:
                        config = EspnConfig(
                            league_id=int(league_id),
                            seasons=seasons,
                            swid=swid or None,
                            espn_s2=espn_s2 or None,
                            cookie_header=cookie_header or None,
                        )
                        with st.spinner("Pulling ESPN history... the vault door is opening."):
                            result = sync_espn_history(config, db)
                        st.cache_data.clear()
                        st.success(f"Synced {result['seasons_synced']} season(s).")
                        incomplete_weeks = result.get("incomplete_weeks") or {}
                        if incomplete_weeks:
                            details = "; ".join(
                                f"{season}: week(s) {', '.join(str(week) for week in weeks)}"
                                for season, weeks in sorted(incomplete_weeks.items())
                            )
                            st.warning(
                                "Some weekly boxscores failed to load and were skipped, so scoring/injury "
                                f"data for those weeks may be incomplete ({details}). Try syncing again."
                            )
                        failed_seasons = result.get("failed_seasons") or {}
                        if failed_seasons:
                            details = "; ".join(f"{season}: {reason}" for season, reason in sorted(failed_seasons.items()))
                            st.warning(
                                "Some seasons couldn't be synced and were skipped - the rest loaded fine. "
                                f"({details})"
                            )
                    except ValueError:
                        st.error("League ID must be a number.")
                    except EspnSyncError as exc:
                        st.error(str(exc))
                        st.info("Most reliable fix: DevTools -> Network -> reload your ESPN league -> click an ESPN API request -> copy the full `Cookie` request header into the sidebar.")

            with st.expander("Local data imports"):
                auction_file = st.file_uploader(
                    "Auction draft CSV",
                    type=["csv"],
                    help="Headers can be season, player_id/player_name, team_id/manager_name, and auction_value/price.",
                )
                if st.button("💰 Save auction prices", width="stretch", disabled=auction_file is None):
                    try:
                        frame, error = normalize_import(pd.read_csv(auction_file), "auction")
                        if error:
                            st.error(error)
                        else:
                            db.replace_import_table("auction_values", frame)
                            st.cache_data.clear()
                            st.success(f"Saved {len(frame)} auction value row(s).")
                    except Exception as exc:
                        st.error(f"Could not load auction CSV: {exc}")

                injury_file = st.file_uploader(
                    "Player injury CSV",
                    type=["csv"],
                    help="Headers can be season, week, player_id/player_name, and injury_status/status.",
                )
                if st.button("🩹 Save injury history", width="stretch", disabled=injury_file is None):
                    try:
                        frame, error = normalize_import(pd.read_csv(injury_file), "injury")
                        if error:
                            st.error(error)
                        else:
                            db.replace_import_table("injuries", frame)
                            st.cache_data.clear()
                            st.success(f"Saved {len(frame)} injury row(s).")
                    except Exception as exc:
                        st.error(f"Could not load injury CSV: {exc}")

                if st.button("📡 Pull nflverse injuries", width="stretch"):
                    try:
                        seasons = [int(part.strip()) for part in seasons_text.split(",") if part.strip()]
                        with st.spinner("Loading public NFL injury reports..."):
                            frame, source = fetch_nflverse_injuries(seasons)
                        for stale_source in {"nfl_data_py", "nflverse_csv"} - {source}:
                            db.replace_source_rows("injuries", stale_source, pd.DataFrame())
                        db.replace_source_rows("injuries", source, frame)
                        st.cache_data.clear()
                        st.success(f"Saved {len(frame)} public injury row(s) from {source}.")
                    except ValueError:
                        st.error("Seasons must be comma-separated years, like 2021,2022,2023.")
                    except InjurySourceError as exc:
                        st.error(str(exc))

        if not tables["teams"].empty:
            with st.expander("Owner display names"):
                st.caption("Give a manager a new name without erasing their receipts.")
                aliases: dict[str, str] = {}
                owners = (
                    tables["teams"][["manager_id", "manager_name"]]
                    .drop_duplicates("manager_id")
                    .sort_values("manager_name")
                )
                for owner in owners.itertuples(index=False):
                    aliases[str(owner.manager_id)] = st.text_input(
                        str(owner.manager_name),
                        str(owner.manager_name),
                        key=f"owner_alias_{owner.manager_id}",
                    )
                if st.button("✏️ Save owner names", width="stretch"):
                    db.save_owner_aliases(aliases)
                    st.cache_data.clear()
                    st.success("Owner names saved.")

    return selected_seasons, selected_managers


def metric_row(records: list[tuple[str, str, str]]) -> None:
    cols = st.columns(len(records))
    for col, (label, value, help_text) in zip(cols, records):
        col.metric(label, value, help=help_text)


COLUMN_LABELS = {
    "season": "Season",
    "week": "Week",
    "manager_name": "Manager",
    "opponent_manager_name": "Opponent",
    "team_name": "Team",
    "opponent_team_name": "Opponent Team",
    "points_for": "Points",
    "points_against": "Opponent Points",
    "margin": "Margin",
    "win": "Won",
    "player_id": "Player ID",
    "player_name": "Player",
    "auction_value": "Price",
    "source": "Source",
    "transaction_type": "Type",
    "future_points": "Points After",
    "acquisition_source": "Source",
    "points": "Points",
    "point_share": "Share",
    "manager_points": "Manager Points",
    "league_median": "League Median",
    "delta_to_median": "+/- Median",
    "slot_role": "Slot",
    "position": "Position",
    "slot_points": "Total Points",
    "weeks": "Weeks",
    "slot_points_per_week": "Pts/Week",
    "weeks_out": "Weeks Out",
    "injury_statuses": "Injury Statuses",
    "draft_value": "Draft $ Value",
    "value_source": "Value Source",
    "draft_value_score": "Hindsight Score",
    "draft_score": "Draft Score",
    "risk_avoidance_score": "Risk Avoidance",
    "lineup_construction_score": "Lineup Construction",
    "sleeper_score": "Sleeper Score",
    "injury_value_lost": "Injury $ Lost",
    "sleeper_value": "Sleeper Value",
    "transaction_score": "Transaction Score",
    "net_transaction_value": "Net Value",
    "add_value": "Add Value",
    "trade_value": "Trade Value",
    "unfortunate_drop_value": "Drop Regret",
    "move_count": "Moves",
    "actual_points": "Actual Points",
    "projected_points": "Projected Points",
    "average_delta": "Avg Points Over Projection",
    "beat_projection_pct": "Beat Projection %",
    "waiver_adds": "Waiver Adds",
    "trades": "Trades",
    "bench_points_left": "Bench Points Left",
    "optimality_pct": "Lineup Efficiency",
    "luck_wins": "Luck Wins",
    "actual_wins": "Actual Wins",
    "actual_losses": "Actual Losses",
    "all_play_win_pct": "All-Play Win %",
    "actual_win_pct": "Actual Win %",
    "seasons": "Seasons",
    "injured_player_weeks": "Injured Player-Weeks",
    "injury_luck_index": "Injury Luck Index",
    "schedule_luck_index": "Schedule Luck Index",
    "overall_luck_index": "Overall Luck Index",
    "early_picks": "Early Picks",
    "early_pick_share": "Early Pick Share",
    "team_id": "Team ID",
    "manager_id": "Manager ID",
}


def humanize_columns(df: pd.DataFrame, overrides: dict[str, str] | None = None) -> pd.DataFrame:
    """Consistent, readable table headers instead of raw snake_case column names."""
    labels = {**COLUMN_LABELS, **(overrides or {})}

    def _label(col: str) -> str:
        if col in labels:
            return labels[col]
        words = str(col).replace("_", " ").split()
        return " ".join(word.upper() if word.lower() == "id" else word.capitalize() for word in words)

    return df.rename(columns={col: _label(col) for col in df.columns})


def format_percent(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Render fraction columns (0.62) as "62.0%" strings for display tables."""
    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[col] = (pd.to_numeric(out[col], errors="coerce") * 100).round(1).map(
                lambda value: "n/a" if pd.isna(value) else f"{value:.1f}%"
            )
    return out


def show_table(df: pd.DataFrame, label: str = "Show data table", overrides: dict[str, str] | None = None) -> None:
    """A collapsed expander for supporting detail, so the chart above stays the main event."""
    with st.expander(f"📋 {label} ({len(df)} rows)"):
        display_df = humanize_columns(df, overrides)
        # Cap the visible height so a table with hundreds of rows scrolls internally
        # instead of stretching the whole page - past ~11 rows it just scrolls.
        st.dataframe(display_df, width="stretch", hide_index=True, height=min(38 * (len(display_df) + 1) + 3, 420))
        if not display_df.empty:
            st.download_button(
                "⬇️ Download CSV",
                display_df.to_csv(index=False).encode("utf-8"),
                file_name=f"{label.lower().replace(' ', '_')}.csv",
                mime="text/csv",
                key=f"download_{label}_{len(display_df)}",
            )


def _color_map(values: pd.Series) -> dict[object, str]:
    unique = list(dict.fromkeys(values.dropna().tolist()))
    return {value: COLORWAY[index % len(COLORWAY)] for index, value in enumerate(unique)}


def render_chart_header(headline: str, subtitle: str) -> None:
    """A themed headline over a plain-English subtitle, rendered as normal wrapping
    HTML above the chart - unlike Plotly's own title, this never clips or gets cut
    off in a narrow column."""
    st.markdown(
        f"<div class='vault-chart-head'>{headline}</div><div class='vault-chart-sub'>{subtitle}</div>",
        unsafe_allow_html=True,
    )


def polish_figure(fig: go.Figure) -> go.Figure:
    # Most charts render their headline via render_chart_header() now, not a Plotly
    # title - but a few (small multiples, colorbar-only figures) still set one via
    # update_layout(title=...) before calling this. Preserve that text explicitly;
    # otherwise setting title_font alone with no title.text renders as "undefined".
    existing_title = fig.layout.title.text if fig.layout.title is not None else None
    fig.update_layout(
        template=PLOT_TEMPLATE,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(227, 165, 24, 0.03)",
        font={"family": BODY_FONT, "color": BRAND["parchment"]},
        title={"text": existing_title or "", "font": {"family": DISPLAY_FONT, "size": 20, "color": BRAND["gold"]}},
        margin={"l": 40, "r": 20, "t": 24, "b": 42},
        # Horizontal, below the plot: a right-side legend eats plot width, which
        # gets painful once a chart is squeezed into half a card.
        legend={
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
            "orientation": "h",
            "yanchor": "top",
            "y": -0.18,
            "xanchor": "center",
            "x": 0.5,
        },
        hovermode="closest",
    )
    fig.update_xaxes(gridcolor="rgba(227, 165, 24, 0.12)", zerolinecolor="rgba(227, 165, 24, 0.3)")
    fig.update_yaxes(gridcolor="rgba(227, 165, 24, 0.12)", zerolinecolor="rgba(227, 165, 24, 0.3)")
    return fig


def scatter_figure(
    df: pd.DataFrame,
    x: str,
    y: str,
    color: str,
    title: str,
    x_title: str,
    y_title: str,
    size: str | None = None,
) -> go.Figure:
    fig = go.Figure()
    colors = _color_map(df[color])
    if size:
        sizes = pd.to_numeric(df[size], errors="coerce")
        max_size = float(sizes.max()) if not sizes.dropna().empty else 1.0
    else:
        max_size = 1.0

    # Every dot is one (color-group, season) combination - make that explicit on hover
    # instead of leaving it to be inferred from color alone.
    has_season = "season" in df.columns
    hover_lines = [f"<b>{color.replace('_', ' ').title()}: %{{fullData.name}}</b>"]
    if has_season:
        hover_lines.append("Season: %{customdata[0]}")
    hover_lines.append(f"{x_title}: %{{x:,.1f}}")
    hover_lines.append(f"{y_title}: %{{y:,.1f}}")
    hover_lines.append("<extra></extra>")
    hovertemplate = "<br>".join(hover_lines)

    for value, group in df.groupby(color, dropna=False):
        marker: dict[str, object] = {"color": colors.get(value, "#64748b"), "opacity": 0.82}
        if size:
            size_values = pd.to_numeric(group[size], errors="coerce").fillna(0)
            marker["size"] = (size_values / max(max_size, 1) * 24 + 8).tolist()
        fig.add_trace(
            go.Scatter(
                x=group[x],
                y=group[y],
                mode="markers",
                name=str(value),
                customdata=group[["season"]].to_numpy() if has_season else None,
                hovertemplate=hovertemplate,
                marker=marker,
            )
        )
    fig.update_layout(title=title, xaxis_title=x_title, yaxis_title=y_title, legend_title=color.replace("_", " ").title())
    return polish_figure(fig)


def grouped_bar_figure(
    df: pd.DataFrame,
    x: str,
    y: str,
    color: str,
    title: str,
    x_title: str,
    y_title: str,
    barmode: str = "group",
) -> go.Figure:
    fig = go.Figure()
    colors = _color_map(df[color])
    # Highest-total category first, so a dense chart (many managers x many seasons)
    # reads as a leaderboard instead of whatever order rows happened to arrive in.
    category_order = df.groupby(x)[y].sum().sort_values(ascending=False).index.tolist()
    for value, group in df.groupby(color, dropna=False):
        fig.add_trace(
            go.Bar(
                x=group[x],
                y=group[y],
                name=str(value),
                marker_color=colors.get(value, "#64748b"),
            )
        )
    fig.update_layout(
        title=title,
        xaxis_title=x_title,
        yaxis_title=y_title,
        xaxis={"categoryorder": "array", "categoryarray": category_order},
        barmode=barmode,
        legend_title=color.replace("_", " ").title(),
    )
    return polish_figure(fig)


def ranked_bar_figure(df: pd.DataFrame, x: str, y: str, title: str, x_title: str, y_title: str) -> go.Figure:
    """A single sorted bar per category, colored by its own value (red -> green)."""
    ordered = df.sort_values(y, ascending=False)
    fig = go.Figure(
        go.Bar(
            x=ordered[x],
            y=ordered[y],
            marker={"color": ordered[y], "colorscale": "RdYlGn"},
        )
    )
    fig.update_layout(title=title, xaxis_title=x_title, yaxis_title=y_title, showlegend=False)
    return polish_figure(fig)


def line_figure(df: pd.DataFrame, x: str, y: str, color: str, title: str, x_title: str, y_title: str) -> go.Figure:
    fig = go.Figure()
    colors = _color_map(df[color])
    for value, group in df.sort_values(x).groupby(color, dropna=False):
        fig.add_trace(
            go.Scatter(
                x=group[x],
                y=group[y],
                mode="lines+markers",
                name=str(value),
                line={"color": colors.get(value, "#64748b"), "width": 3},
                marker={"size": 8},
            )
        )
    fig.update_layout(title=title, xaxis_title=x_title, yaxis_title=y_title, legend_title=color.replace("_", " ").title())
    return polish_figure(fig)


def main() -> None:
    db = get_db()
    if not DB_PATH.exists() or db.is_empty():
        seed_sample_database(db, replace=True)

    selected_seasons, selected_managers = sidebar(db)
    tables = apply_owner_aliases(load_tables(str(DB_PATH), db_cache_key()))
    has_any_league_data = not tables["teams"].empty

    for name, df in list(tables.items()):
        if "season" in df.columns:
            tables[name] = df[df["season"].isin(selected_seasons)].copy()
    for name, df in list(tables.items()):
        tables[name] = manager_filter(df, selected_managers)
    teams = tables["teams"]

    st.title("🏆 The Fantasy Vault")
    st.caption("Draft hauls, schedule fortune, roster relics, and the all-time hall of fame — every season, cataloged.")

    if not teams.empty:
        season_count = teams["season"].nunique()
        manager_count = teams["manager_name"].nunique()
        game_count = (
            tables["matchups"].drop_duplicates(subset=["season", "week", "matchup_id"]).shape[0]
            if not tables["matchups"].empty and "matchup_id" in tables["matchups"].columns
            else 0
        )
        st.markdown(
            f"<span style='color:{BRAND['pewter']}'>📊 "
            f"{season_count} season{'s' if season_count != 1 else ''} &nbsp;·&nbsp; "
            f"{manager_count} manager{'s' if manager_count != 1 else ''} &nbsp;·&nbsp; "
            f"{game_count} game{'s' if game_count != 1 else ''} logged</span>",
            unsafe_allow_html=True,
        )

    if teams.empty:
        if has_any_league_data:
            st.info(
                "No data matches your current **Visible seasons** / **Managers** filters — "
                "select more of either in the sidebar."
            )
        else:
            with st.container(border=True):
                st.markdown("## 🗝️ The vault is empty.")
                st.write(
                    "No treasure has been logged yet. Load the generated sample league below to "
                    "explore the halls, or open **League Setup** in the sidebar to sync your real "
                    "ESPN league."
                )
                if st.button("🎲 Load the sample league", type="primary"):
                    seed_sample_database(db, replace=True)
                    st.cache_data.clear()
                    st.rerun()
        return

    records = all_time_records(tables["matchups"], tables["teams"])
    metric_row(
        [
            ("Legendary Haul", records.get("highest_score", "n/a"), "Highest single-week score in league history."),
            ("The Devastating Loss", records.get("worst_loss", "n/a"), "The lowest score that still somehow took the loss."),
            ("The Reign", records.get("longest_win_streak", "n/a"), "Longest winning streak in a single season."),
            ("The Coin Flip", records.get("closest_loss", "n/a"), "The closest margin of defeat on record."),
        ]
    )

    tab_luck, tab_h2h, tab_draft, tab_transactions, tab_player, tab_projection, tab_positions, tab_profiles, tab_records = st.tabs(
        [
            "⚖️ Fortune's Favor",
            "⚔️ Rivalries",
            "⛏️ The Excavation",
            "💰 The Trading Post",
            "🔍 The Appraisal",
            "🔮 The Oracle",
            "💎 The Collection",
            "🧭 The Expedition",
            "🏆 Hall of Fame",
        ]
    )

    with tab_luck:
        st.caption(
            "How much of your record is the schedule's fault, not yours — actual wins vs. what you'd "
            "get if you played everyone, every week."
        )
        luck = schedule_luck(tables["matchups"], tables["teams"])
        enriched_luck = injury_luck(
            luck,
            tables["draft_picks"],
            tables["auction_values"],
            tables["roster_scores"],
            tables["injuries"],
            tables["teams"],
            tables["transactions"],
        )
        injury_detail = injury_loss_detail(
            tables["draft_picks"],
            tables["auction_values"],
            tables["roster_scores"],
            tables["injuries"],
            tables["teams"],
            tables["transactions"],
        )
        if luck.empty:
            st.info("No matchup data available yet.")
        else:
            with st.container(border=True):
                c_luck_a, c_luck_b, c_luck_c = st.columns([1, 1, 1])
                granularity = c_luck_a.multiselect(
                    "Group luck by",
                    ["season", "manager_name", "team_name"],
                    default=["season", "manager_name"],
                    help="How to bucket the chart below - by season, by manager, or both.",
                )
                view_mode = c_luck_b.selectbox(
                    "Luck view",
                    ["Trend", "Ranked bars", "Actual vs all-play"],
                    index=0,
                    help=(
                        "Trend: the metric over time. Ranked bars: a leaderboard for one metric. "
                        "Actual vs all-play: your real record vs. what you'd get facing the whole league every week."
                    ),
                )
                score_metric = c_luck_c.selectbox(
                    "Luck metric",
                    ["luck_wins", "overall_luck_index", "injury_value_lost"],
                    format_func=lambda value: {
                        "luck_wins": "Schedule luck wins",
                        "overall_luck_index": "Overall luck index",
                        "injury_value_lost": "Dollars lost to injury",
                    }[value],
                    help=(
                        "Schedule luck wins: wins above/below what your scoring deserved. "
                        "Overall luck index: schedule luck plus injury luck combined. "
                        "Dollars lost to injury: draft/auction value of players out hurt."
                    ),
                )
                chart_luck = enriched_luck if not enriched_luck.empty else luck
                aggregated_luck = aggregate_luck(chart_luck, granularity)
                if view_mode == "Actual vs all-play":
                    render_chart_header(
                        "Fortune vs. Form",
                        "Actual win % vs. all-play win % — the record you'd have if you played everyone, every week.",
                    )
                    fig = scatter_figure(
                        chart_luck,
                        "actual_win_pct",
                        "all_play_win_pct",
                        "manager_name",
                        "",
                        "Actual win %",
                        "All-play win %",
                        size="injury_value_lost" if "injury_value_lost" in chart_luck.columns else "points_for",
                    )
                    fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line={"dash": "dash", "color": "#AAB3C5"})
                    st.plotly_chart(fig)
                    st.caption(
                        "The dashed diagonal is zero luck: your actual win % equals your all-play win %. "
                        "Above the line means a friendlier schedule than your scoring deserved; below means a tougher one."
                    )
                elif view_mode == "Trend" and "season" in aggregated_luck.columns and len(granularity) > 1:
                    color_dim = "manager_name" if "manager_name" in aggregated_luck.columns else granularity[-1]
                    render_chart_header("The Tides of Fortune", f"{score_metric.replace('_', ' ').title()} across seasons.")
                    fig = line_figure(
                        aggregated_luck,
                        "season",
                        score_metric,
                        color_dim,
                        "",
                        "Season",
                        score_metric.replace("_", " ").title(),
                    )
                    st.plotly_chart(fig)
                else:
                    x_dim = "manager_name" if "manager_name" in aggregated_luck.columns else (granularity[0] if granularity else "season")
                    color_dim = "season" if "season" in aggregated_luck.columns and x_dim != "season" else x_dim
                    render_chart_header("The Fortune Rankings", f"{score_metric.replace('_', ' ').title()}, ranked.")
                    fig = grouped_bar_figure(
                        aggregated_luck.sort_values(score_metric, ascending=score_metric == "injury_value_lost"),
                        x_dim,
                        score_metric,
                        color_dim,
                        "",
                        x_dim.replace("_", " ").title(),
                        score_metric.replace("_", " ").title(),
                    )
                    st.plotly_chart(fig)
                show_table(format_percent(aggregated_luck, ["all_play_win_pct", "actual_win_pct"]), "Show luck numbers")
                if not injury_detail.empty:
                    detail_cols = [
                        "season",
                        "manager_name",
                        "team_name",
                        "player_name",
                        "weeks_out",
                        "weeks",
                        "injury_statuses",
                        "draft_value",
                        "injury_value_lost",
                        "value_source",
                    ]
                    show_table(
                        injury_detail[detail_cols].sort_values(
                            ["season", "injury_value_lost", "weeks_out"],
                            ascending=[True, False, False],
                        ),
                        "Show players lost to injury",
                    )
                else:
                    st.caption("No matched injury-player losses for the selected filters.")

    with tab_h2h:
        st.caption("Head-to-head history between any two managers: record, margins, and every meeting week by week.")
        managers = sorted(tables["teams"]["manager_name"].dropna().unique().tolist())
        if len(managers) < 2:
            st.info("Head-to-head history needs at least two managers.")
        else:
            with st.container(border=True):
                c_h2h_a, c_h2h_b = st.columns(2)
                manager_a = c_h2h_a.selectbox("Manager A", managers, index=0)
                manager_b_options = [manager for manager in managers if manager != manager_a]
                manager_b = c_h2h_b.selectbox("Manager B", manager_b_options, index=0)
                h2h_games, h2h_summary = head_to_head_history(tables["matchups"], tables["teams"], manager_a, manager_b)
                if h2h_games.empty:
                    st.info("No matchups found for that pairing in the selected seasons.")
                else:
                    cols = st.columns(2)
                    for col, row in zip(cols, h2h_summary.itertuples(index=False)):
                        col.metric(
                            str(row.manager_name),
                            f"{int(row.wins)}-{int(row.losses)}",
                            f"{row.avg_margin:+.2f} avg margin",
                        )
                    render_chart_header(f"Old Rivals: {manager_a} vs. {manager_b}", f"{manager_a} vs. {manager_b} — points scored in every meeting.")
                    fig = line_figure(
                        h2h_games,
                        "game_label",
                        "points_for",
                        "manager_name",
                        "",
                        "Matchup",
                        "Points",
                    )
                    st.plotly_chart(fig)
                    h2h_overrides = {
                        f"{manager_a}_team": f"{manager_a}'s Team",
                        f"{manager_a}_points": f"{manager_a} Points",
                        f"{manager_b}_team": f"{manager_b}'s Team",
                        f"{manager_b}_points": f"{manager_b} Points",
                    }
                    show_table(
                        head_to_head_game_table(h2h_games, manager_a, manager_b),
                        "Show every meeting",
                        overrides=h2h_overrides,
                    )

    with tab_draft:
        st.caption("Draft tendencies, hindsight value, and a scorecard for how each draft actually turned out.")
        tendencies = draft_tendencies(tables["draft_picks"], tables["teams"])
        hindsight = draft_hindsight(tables["draft_picks"], tables["roster_scores"], tables["teams"])
        scorecard = draft_scorecard(
            tables["draft_picks"],
            tables["roster_scores"],
            tables["teams"],
            tables["auction_values"],
            tables["injuries"],
        )
        auction_values = auction_player_values(tables["draft_picks"], tables["auction_values"], tables["teams"])

        draft_seasons = sorted(tables["draft_picks"]["season"].dropna().unique().tolist()) if not tables["draft_picks"].empty else []
        season_choice = st.selectbox(
            "Season",
            ["All seasons (avg)"] + [str(season) for season in draft_seasons],
            key="draft_season_filter",
            help="Pick one draft year, or average every manager's numbers across all of them.",
        )

        if season_choice == "All seasons (avg)":
            if tendencies.empty:
                tendencies_view = tendencies
            else:
                tendencies_view = tendencies.groupby(["manager_name", "position"], as_index=False)["early_picks"].sum()
                tendencies_view["early_pick_share"] = tendencies_view["early_picks"] / tendencies_view.groupby(
                    "manager_name"
                )["early_picks"].transform("sum")
            hindsight_view = (
                hindsight.groupby("manager_name", as_index=False)["draft_value_score"].mean().round({"draft_value_score": 1})
                if not hindsight.empty
                else hindsight
            )
            scorecard_view = (
                scorecard.groupby("manager_name", as_index=False)["draft_score"].mean().round({"draft_score": 0})
                if not scorecard.empty
                else scorecard
            )
        else:
            season_num = int(season_choice)
            tendencies_view = tendencies[tendencies["season"] == season_num] if not tendencies.empty else tendencies
            hindsight_view = hindsight[hindsight["season"] == season_num] if not hindsight.empty else hindsight
            scorecard_view = scorecard[scorecard["season"] == season_num] if not scorecard.empty else scorecard

        c1, c2 = st.columns([1, 1])
        with c1:
            with st.container(border=True):
                if tendencies_view.empty:
                    st.info("No draft data available.")
                else:
                    render_chart_header("The Excavation Site", "Position mix of each manager's early-round picks.")
                    fig = grouped_bar_figure(
                        tendencies_view,
                        "manager_name",
                        "early_pick_share",
                        "position",
                        "",
                        "Manager",
                        "Share of early picks",
                        barmode="stack",
                    )
                    fig.update_yaxes(tickformat=".0%")
                    st.plotly_chart(fig)
        with c2:
            with st.container(border=True):
                if hindsight_view.empty:
                    st.info("Hindsight draft grades need draft picks plus player scoring lines.")
                else:
                    render_chart_header("Buried Treasure", "Draft value once the season actually happened - higher is better.")
                    fig = ranked_bar_figure(
                        hindsight_view,
                        "manager_name",
                        "draft_value_score",
                        "",
                        "Manager",
                        "Hindsight value score",
                    )
                    st.plotly_chart(fig)
                    show_table(hindsight_view, "Show hindsight numbers")

        if not scorecard_view.empty:
            with st.container(border=True):
                render_chart_header("The Dig Report", "Composite draft score: risk avoidance, lineup construction, sleeper value.")
                fig = ranked_bar_figure(
                    scorecard_view,
                    "manager_name",
                    "draft_score",
                    "",
                    "Manager",
                    "Draft score",
                )
                st.plotly_chart(fig)
                display_columns = [
                    col
                    for col in [
                        "season",
                        "manager_name",
                        "team_name",
                        "draft_score",
                        "risk_avoidance_score",
                        "lineup_construction_score",
                        "sleeper_score",
                        "injury_value_lost",
                        "sleeper_value",
                    ]
                    if col in scorecard_view.columns
                ]
                show_table(scorecard_view[display_columns], "Show scorecard numbers")

        if not auction_values.empty:
            show_table(
                auction_values[
                    ["season", "manager_name", "team_name", "player_name", "auction_value", "source"]
                ],
                "Show auction prices",
                overrides={"auction_value": "Price ($)"},
            )

    with tab_transactions:
        st.caption("Waiver adds, trades, and drops — scored by the points they actually produced afterward.")
        scores, details = transaction_scorecard(tables["transactions"], tables["roster_scores"], tables["teams"])
        if scores.empty:
            st.info("No transaction data available.")
        else:
            with st.container(border=True):
                render_chart_header("The Trading Post Report", "Composite transaction score from adds, trades, and drops.")
                fig = grouped_bar_figure(
                    scores,
                    "manager_name",
                    "transaction_score",
                    "season",
                    "",
                    "Manager",
                    "Score",
                )
                st.plotly_chart(fig)
                show_table(
                    scores[
                        [
                            "season",
                            "manager_name",
                            "team_name",
                            "transaction_score",
                            "net_transaction_value",
                            "add_value",
                            "trade_value",
                            "unfortunate_drop_value",
                            "move_count",
                        ]
                    ],
                    "Show transaction numbers",
                )
            if details.empty:
                st.info("Transaction rows exist, but no future player scoring lines were available to grade them.")
            else:
                drops = details[details["score_type"] == "Unfortunate drop"].sort_values("future_points", ascending=False)
                if not drops.empty:
                    with st.container(border=True):
                        st.subheader("The One That Got Away")
                        st.caption("Players you dropped who kept scoring — just not for you.")
                        show_table(
                            drops[
                                ["season", "week", "manager_name", "player_name", "transaction_type", "future_points"]
                            ].head(25),
                            "Show unfortunate drops",
                            overrides={"transaction_type": "Move Type", "future_points": "Points They Scored After"},
                        )

    with tab_player:
        st.caption("One manager's season under the spotlight: where their points came from and how each slot performed.")
        managers = sorted(tables["teams"]["manager_name"].dropna().unique().tolist())
        if not managers:
            st.info("Player profile needs team and roster data.")
        else:
            selected_profile_manager = st.selectbox("Manager", managers, key="player_profile_manager")
            source_share, weekly, slot_weekly = player_profile_frames(
                selected_profile_manager,
                tables["roster_scores"],
                tables["teams"],
                tables["draft_picks"],
                tables["transactions"],
            )
            if weekly.empty:
                st.info("No weekly starter scoring available for that manager.")
            else:
                c_source, c_hist = st.columns([1, 2])
                with c_source:
                    with st.container(border=True):
                        if source_share.empty:
                            st.info("No acquisition source split available.")
                        else:
                            render_chart_header("Provenance", "Starter points by how each player was acquired.")
                            fig = go.Figure(
                                go.Pie(
                                    labels=source_share["acquisition_source"],
                                    values=source_share["points"],
                                    hole=0.48,
                                    marker={"colors": COLORWAY},
                                )
                            )
                            st.plotly_chart(polish_figure(fig))
                            show_table(
                                format_percent(source_share, ["point_share"]),
                                "Show source breakdown",
                                overrides={"points": "Points"},
                            )
                with c_hist:
                    with st.container(border=True):
                        render_chart_header(
                            "Your Haul vs. The Field", "Weekly starter points for this manager vs. the league median that week."
                        )
                        fig = go.Figure()
                        fig.add_trace(
                            go.Histogram(
                                x=weekly["manager_points"],
                                name=selected_profile_manager,
                                opacity=0.78,
                                marker_color=COLORWAY[0],
                            )
                        )
                        fig.add_trace(
                            go.Histogram(
                                x=weekly["league_median"],
                                name="League median",
                                opacity=0.62,
                                marker_color=COLORWAY[3],
                            )
                        )
                        fig.update_layout(
                            xaxis_title="Points",
                            yaxis_title="Weeks",
                            barmode="overlay",
                        )
                        st.plotly_chart(polish_figure(fig))

                if not source_share.empty:
                    with st.container(border=True):
                        league_avg_share = acquisition_source_league_average(
                            tables["roster_scores"], tables["teams"], tables["draft_picks"], tables["transactions"]
                        )
                        if not league_avg_share.empty:
                            compare = source_share[["acquisition_source", "point_share"]].merge(
                                league_avg_share.rename(columns={"point_share": "league_avg_share"}),
                                on="acquisition_source",
                                how="outer",
                            ).fillna(0)
                            render_chart_header(
                                "Your Provenance vs. The Vault",
                                "This manager's acquisition-source point share vs. the league average.",
                            )
                            fig = go.Figure()
                            fig.add_trace(
                                go.Bar(
                                    x=compare["acquisition_source"],
                                    y=compare["point_share"],
                                    name=selected_profile_manager,
                                    marker_color=COLORWAY[0],
                                )
                            )
                            fig.add_trace(
                                go.Bar(
                                    x=compare["acquisition_source"],
                                    y=compare["league_avg_share"],
                                    name="League average",
                                    marker_color=COLORWAY[3],
                                )
                            )
                            fig.update_layout(
                                xaxis_title="Acquisition source",
                                yaxis_title="Share of starter points",
                                barmode="group",
                            )
                            fig.update_yaxes(tickformat=".0%")
                            st.plotly_chart(polish_figure(fig))

                if not slot_weekly.empty:
                    with st.container(border=True):
                        st.subheader("🏺 The Reliquary")
                        slot_roles = sorted(slot_weekly["slot_role"].dropna().unique().tolist())
                        slot_filter = st.multiselect(
                            "Roster slots",
                            slot_roles,
                            default=slot_roles,
                            help="Show or hide individual roster slots below.",
                        )
                        active_slots = [slot for slot in slot_roles if slot in slot_filter] if slot_filter else slot_roles
                        st.markdown(
                            f"<span style='color:{COLORWAY[0]}'>&#9632;</span> {selected_profile_manager} "
                            f"&nbsp;&nbsp; <span style='color:{COLORWAY[3]}'>&#9632;</span> League median "
                            "&nbsp;&nbsp; <span style='opacity:0.7'>one panel per roster slot, same idea as \"Your Haul vs. The Field\".</span>",
                            unsafe_allow_html=True,
                        )
                        slot_view = slot_weekly[slot_weekly["slot_role"].isin(active_slots)]
                        cols_per_row = 3
                        for start in range(0, len(active_slots), cols_per_row):
                            row_slots = active_slots[start : start + cols_per_row]
                            row_cols = st.columns(len(row_slots))
                            for col, slot_name in zip(row_cols, row_slots):
                                group = slot_view[slot_view["slot_role"] == slot_name]
                                fig = go.Figure()
                                fig.add_trace(
                                    go.Histogram(
                                        x=group["manager_points"],
                                        opacity=0.78,
                                        marker_color=COLORWAY[0],
                                    )
                                )
                                fig.add_trace(
                                    go.Histogram(
                                        x=group["league_median"],
                                        opacity=0.62,
                                        marker_color=COLORWAY[3],
                                    )
                                )
                                fig.update_layout(
                                    title=str(slot_name),
                                    xaxis_title="Points",
                                    yaxis_title="Weeks",
                                    barmode="overlay",
                                    showlegend=False,
                                    height=280,
                                )
                                col.plotly_chart(polish_figure(fig), use_container_width=True)
                        show_table(slot_view, "Show slot-by-week numbers")

    with tab_projection:
        st.caption("Actual points vs. what was projected — who beats their projections, and by how much.")
        projection = projection_performance(tables["roster_scores"], tables["teams"])
        matrix = projection_matchup_matrix(tables["matchups"], tables["roster_scores"], tables["teams"])
        if projection.empty:
            st.info("No projected scoring data available for these seasons.")
        else:
            best_delta = projection.sort_values("average_delta", ascending=False).iloc[0]
            best_beat = projection.sort_values("beat_projection_pct", ascending=False).iloc[0]
            with st.container(border=True):
                c0a, c0b = st.columns(2)
                c0a.metric(
                    "Biggest projection beat",
                    f"{best_delta['manager_name']}, {int(best_delta['season'])}",
                    f"{best_delta['average_delta']:+.2f} pts/week",
                )
                c0b.metric(
                    "Best beat rate",
                    f"{best_beat['manager_name']}, {int(best_beat['season'])}",
                    f"{best_beat['beat_projection_pct']:.1%}",
                )

                c1, c2 = st.columns([1, 1])
                with c1:
                    render_chart_header("Prophecy vs. Reality", "Actual points vs. projected — above the line means you overperformed.")
                    fig = scatter_figure(
                        projection,
                        "projected_points",
                        "actual_points",
                        "manager_name",
                        "",
                        "Projected points",
                        "Actual points",
                    )
                    fig.add_shape(
                        type="line",
                        x0=projection["projected_points"].min(),
                        y0=projection["projected_points"].min(),
                        x1=projection["projected_points"].max(),
                        y1=projection["projected_points"].max(),
                        line={"dash": "dash", "color": "#777"},
                    )
                    st.plotly_chart(polish_figure(fig))
                    st.caption(
                        "Each dot is one manager's full-season total for one year (color = manager, "
                        "hover for the exact season) — not a single week."
                    )
                with c2:
                    beat = (
                        projection.groupby("manager_name", as_index=False)
                        .agg(
                            average_delta=("average_delta", "mean"),
                            beat_projection_pct=("beat_projection_pct", "mean"),
                        )
                        .round({"average_delta": 2, "beat_projection_pct": 3})
                    )
                    render_chart_header(
                        "Prophecy Breakers", "Average points over projection per manager, colored by how often they beat it."
                    )
                    fig = go.Figure(
                        go.Bar(
                            x=beat.sort_values("average_delta", ascending=False)["manager_name"],
                            y=beat.sort_values("average_delta", ascending=False)["average_delta"],
                            marker_color=beat.sort_values("average_delta", ascending=False)["beat_projection_pct"],
                            marker_colorscale="RdYlGn",
                        )
                    )
                    fig.update_layout(
                        xaxis_title="Manager",
                        yaxis_title="Avg points over projection",
                    )
                    st.plotly_chart(polish_figure(fig))

            if not matrix.empty:
                with st.container(border=True):
                    render_chart_header(
                        "As Foretold", "Actual win rate by how big a favorite or underdog you were projected to be."
                    )
                    wins = matrix[matrix["result"] == "Win"]
                    fig = go.Figure(
                        go.Bar(
                            x=wins["projected_bucket"].astype(str),
                            y=wins["win_rate"],
                            marker_color=COLORWAY[: len(wins)],
                        )
                    )
                    fig.update_layout(
                        xaxis_title="Projected matchup bucket",
                        yaxis_title="Actual win rate",
                        showlegend=False,
                    )
                    st.plotly_chart(polish_figure(fig))

            show_table(
                format_percent(
                    projection[
                        [
                            "season",
                            "manager_name",
                            "team_name",
                            "actual_points",
                            "projected_points",
                            "average_delta",
                            "beat_projection_pct",
                        ]
                    ],
                    ["beat_projection_pct"],
                ),
                "Show projection numbers",
            )

    with tab_positions:
        st.caption("Average points per week at every roster position (including Flex), mixed together by manager.")
        positional = positional_performance(tables["roster_scores"], tables["teams"])
        if positional.empty:
            st.info("No roster slot data available.")
        else:
            with st.container(border=True):
                heatmap = positional.pivot_table(
                    index="manager_name",
                    columns="position",
                    values="slot_points_per_week",
                    aggfunc="mean",
                ).fillna(0)
                # Color each position column on its own scale - a shared scale would make low-
                # scoring positions like K/DST look uniformly "bad" next to RB/WR, when what
                # actually matters is how a manager stacks up against the league at that same spot.
                col_min = heatmap.min(axis=0)
                col_max = heatmap.max(axis=0)
                col_span = (col_max - col_min).replace(0, float("nan"))
                normalized = ((heatmap - col_min) / col_span).fillna(0.5)
                fig = go.Figure(
                    go.Heatmap(
                        z=normalized.values,
                        x=heatmap.columns.tolist(),
                        y=heatmap.index.tolist(),
                        zmin=0,
                        zmax=1,
                        colorscale="RdYlGn",
                        text=heatmap.values,
                        texttemplate="%{text:.1f}",
                        hovertemplate="Manager: %{y}<br>Position: %{x}<br>Pts/week: %{text:.1f}<extra></extra>",
                        colorbar={"title": "Rank within<br>position"},
                    )
                )
                fig.update_layout(
                    xaxis_title="Position",
                    yaxis_title="Manager",
                )
                render_chart_header(
                    "The Inventory Ledger",
                    "Points/week by position - color shows rank within that position, not raw magnitude.",
                )
                st.plotly_chart(polish_figure(fig))
                show_table(positional, "Show position numbers")

    with tab_profiles:
        st.caption("Manager behavior: waiver-wire hustle vs. points wasted on the bench.")
        profiles = manager_profiles(tables["transactions"], tables["roster_scores"], tables["teams"])
        if profiles.empty:
            st.info("No transaction or bench data available.")
        else:
            with st.container(border=True):
                render_chart_header("Unearthed vs. Untouched", "Waiver-wire activity vs. points left stranded on the bench.")
                fig = scatter_figure(
                    profiles,
                    "waiver_adds",
                    "bench_points_left",
                    "manager_name",
                    "",
                    "Waiver adds",
                    "Bench points left",
                    size="trades",
                )
                st.plotly_chart(fig)
                show_table(format_percent(profiles, ["optimality_pct"]), "Show manager numbers")

    with tab_records:
        st.caption("Every all-time record in one place, for when someone needs the receipts.")
        top_n_choice = st.selectbox("How many per category?", [3, 5, 10], index=1, key="finale_top_n")
        leaderboards = all_time_leaderboards(tables["matchups"], tables["teams"], top_n=top_n_choice)
        if not leaderboards:
            st.info("No matchup data available yet.")
        else:
            game_renames = {
                "season": "Season",
                "week": "Week",
                "manager_name": "Manager",
                "team_name": "Team",
                "points_for": "Points",
                "opponent_manager_name": "Opponent",
                "points_against": "Opponent Points",
            }
            margin_renames = {**game_renames, "margin": "Margin"}
            del margin_renames["team_name"]
            sections = [
                (
                    "🏆 Legendary Hauls",
                    f"Top {top_n_choice} highest single-week scores",
                    leaderboards["highest_scores"],
                    game_renames,
                ),
                (
                    "😩 The Devastating Losses",
                    f"Top {top_n_choice} worst losses (lowest score that still lost)",
                    leaderboards["worst_losses"],
                    game_renames,
                ),
                (
                    "🪙 The Coin Flip",
                    f"Top {top_n_choice} closest games",
                    leaderboards["closest_games"],
                    margin_renames,
                ),
                (
                    "💥 Total Plunder",
                    f"Top {top_n_choice} biggest margins",
                    leaderboards["biggest_blowouts"],
                    margin_renames,
                ),
                (
                    "👑 The Reign",
                    f"Top {top_n_choice} winning streaks",
                    leaderboards["longest_win_streaks"],
                    {"manager_name": "Manager", "season": "Season", "win_streak": "Streak (weeks)"},
                ),
                (
                    "⛓️ The Dark Ages",
                    f"Top {top_n_choice} losing streaks",
                    leaderboards["longest_loss_streaks"],
                    {"manager_name": "Manager", "season": "Season", "loss_streak": "Streak (weeks)"},
                ),
                (
                    "💎 The Crown Jewel Seasons",
                    f"Top {top_n_choice} single-season point totals",
                    leaderboards["best_season_totals"],
                    {"season": "Season", "manager_name": "Manager", "team_name": "Team", "points_for": "Total Points"},
                ),
                (
                    "📊 The Steady Hand",
                    f"Top {top_n_choice} highest single-season median scores",
                    leaderboards["best_season_medians"],
                    {"season": "Season", "manager_name": "Manager", "team_name": "Team", "median_points": "Median Points"},
                ),
            ]

            _, transaction_details = transaction_scorecard(
                tables["transactions"], tables["roster_scores"], tables["teams"]
            )
            if not transaction_details.empty:
                pickup_cols = ["season", "week", "manager_name", "player_name", "future_points"]
                pickup_renames = {
                    "season": "Season",
                    "week": "Week",
                    "manager_name": "Manager",
                    "player_name": "Player",
                    "future_points": "Points After",
                }
                best_pickups = (
                    transaction_details[transaction_details["score_type"] == "Pickup"]
                    .sort_values("future_points", ascending=False)
                    .head(top_n_choice)[pickup_cols]
                    .reset_index(drop=True)
                )
                best_trades = (
                    transaction_details[transaction_details["score_type"] == "Trade"]
                    .sort_values("future_points", ascending=False)
                    .head(top_n_choice)[pickup_cols]
                    .reset_index(drop=True)
                )
                sections.append(("🎣 The Big Catch", f"Top {top_n_choice} waiver/free-agent pickups by points scored after", best_pickups, pickup_renames))
                sections.append(("🤝 The Heist", f"Top {top_n_choice} trade acquisitions by points scored after", best_trades, pickup_renames))

            medals = {1: "🥇", 2: "🥈", 3: "🥉"}

            for start in range(0, len(sections), 2):
                row_sections = sections[start : start + 2]
                cols = st.columns(len(row_sections))
                for col, (title, caption, df, renames) in zip(cols, row_sections):
                    with col:
                        with st.container(border=True):
                            st.markdown(f"#### {title}")
                            st.caption(caption)
                            if df.empty:
                                st.info("Not enough data yet.")
                            else:
                                display_df = df.rename(columns=renames)
                                display_df.insert(
                                    0, "Rank", [f"{medals.get(i, '')} {i}".strip() for i in range(1, len(display_df) + 1)]
                                )
                                st.dataframe(display_df, width="stretch", hide_index=True)

            position_records = positional_hall_of_fame(tables["roster_scores"], tables["teams"])
            best_position_seasons = position_records.get("best_position_seasons", pd.DataFrame())
            best_position_weeks = position_records.get("best_position_weeks", pd.DataFrame())
            if not best_position_seasons.empty or not best_position_weeks.empty:
                position_renames = {
                    "slot": "Position",
                    "season": "Season",
                    "manager_name": "Manager",
                    "team_name": "Team",
                    "total_points": "Total Points",
                    "week_points": "Points",
                    "week": "Week",
                }
                season_cols = ["slot", "season", "manager_name", "team_name", "total_points"]
                week_cols = ["slot", "season", "week", "manager_name", "team_name", "week_points"]
                col_a, col_b = st.columns(2)
                with col_a:
                    with st.container(border=True):
                        st.markdown("#### 🏛️ Best Season, By Position")
                        st.caption("The single best season ever started at each roster slot.")
                        if best_position_seasons.empty:
                            st.info("Not enough data yet.")
                        else:
                            st.dataframe(
                                best_position_seasons[season_cols].rename(columns=position_renames),
                                width="stretch",
                                hide_index=True,
                            )
                with col_b:
                    with st.container(border=True):
                        st.markdown("#### ⚡ Best Week, By Position")
                        st.caption("The single best week ever started at each roster slot.")
                        if best_position_weeks.empty:
                            st.info("Not enough data yet.")
                        else:
                            st.dataframe(
                                best_position_weeks[week_cols].rename(columns=position_renames),
                                width="stretch",
                                hide_index=True,
                            )


if __name__ == "__main__":
    main()
