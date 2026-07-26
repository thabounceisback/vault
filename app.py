from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from league_history.analytics import (
    aggregate_luck,
    all_time_records,
    auction_player_values,
    draft_hindsight,
    draft_scorecard,
    draft_tendencies,
    head_to_head_history,
    injury_luck,
    manager_profiles,
    player_profile_frames,
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
COLORWAY = [
    "#5B8DEF",
    "#F25F5C",
    "#20BF55",
    "#F7B32B",
    "#7B61FF",
    "#2EC4B6",
    "#FF8C42",
    "#C44569",
    "#6C757D",
    "#9BC53D",
]
PLOT_TEMPLATE = "plotly_dark"


st.set_page_config(
    page_title="League History Dashboard",
    page_icon=":trophy:",
    layout="wide",
)

st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(circle at 16% 0%, rgba(91, 141, 239, 0.16), transparent 28rem),
            radial-gradient(circle at 85% 8%, rgba(46, 196, 182, 0.12), transparent 30rem),
            #0b0f19;
    }
    [data-testid="stSidebar"] {
        background: #151927;
        border-right: 1px solid rgba(255,255,255,0.08);
    }
    h1, h2, h3 { letter-spacing: 0; }
    div[data-testid="stMetric"] {
        background: rgba(255,255,255,0.045);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 8px;
        padding: 14px 16px;
    }
    div[data-testid="stMetricLabel"] { color: #aab3c5; }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        border-bottom: 1px solid rgba(255,255,255,0.08);
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 10px 14px;
    }
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
        st.header("League Source")
        tables = apply_owner_aliases(load_tables(str(DB_PATH), db_cache_key()))
        profiles = tables.get("league_profiles", pd.DataFrame())

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

        col_save, col_a, col_b = st.columns([1, 1, 1])
        if col_save.button("Save league", width="stretch"):
            try:
                db.save_league_profile(int(league_id), league_name.strip() or "My League", seasons_text)
                st.cache_data.clear()
                st.success("League saved.")
            except ValueError:
                st.error("League ID must be a number before saving.")

        if col_a.button("Load sample", width="stretch"):
            if db.is_empty():
                seed_sample_database(db, replace=True)
                st.cache_data.clear()
                st.success("Sample league loaded.")
            else:
                st.session_state["confirm_load_sample"] = True

        if st.session_state.get("confirm_load_sample"):
            st.warning(
                "This will permanently replace your current league data "
                "(synced ESPN history, uploaded CSVs, etc.) with generated sample data."
            )
            confirm_col, cancel_col = st.columns(2)
            if confirm_col.button("Yes, overwrite with sample data", width="stretch"):
                seed_sample_database(db, replace=True)
                st.cache_data.clear()
                st.session_state["confirm_load_sample"] = False
                st.success("Sample league loaded.")
            if cancel_col.button("Cancel", width="stretch"):
                st.session_state["confirm_load_sample"] = False

        if col_b.button("Sync ESPN", width="stretch"):
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
                    with st.spinner("Pulling ESPN history..."):
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
            if st.button("Save auction prices", width="stretch", disabled=auction_file is None):
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
            if st.button("Save injury history", width="stretch", disabled=injury_file is None):
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

            if st.button("Pull nflverse injuries", width="stretch"):
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

        seasons = sorted(tables["teams"]["season"].dropna().unique().tolist()) if not tables["teams"].empty else []
        selected_seasons = st.multiselect("Visible seasons", seasons, default=seasons)

        managers = sorted(tables["teams"]["manager_name"].dropna().unique().tolist()) if not tables["teams"].empty else []
        selected_managers = st.multiselect("Managers", managers, default=managers)

        if not tables["teams"].empty:
            with st.expander("Owner display names"):
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
                if st.button("Save owner names", width="stretch"):
                    db.save_owner_aliases(aliases)
                    st.cache_data.clear()
                    st.success("Owner names saved.")

    return selected_seasons, selected_managers


def metric_row(records: dict[str, str]) -> None:
    cols = st.columns(len(records))
    for col, (label, value) in zip(cols, records.items()):
        col.metric(label, value)


def _color_map(values: pd.Series) -> dict[object, str]:
    unique = list(dict.fromkeys(values.dropna().tolist()))
    return {value: COLORWAY[index % len(COLORWAY)] for index, value in enumerate(unique)}


def polish_figure(fig: go.Figure) -> go.Figure:
    fig.update_layout(
        template=PLOT_TEMPLATE,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.02)",
        font={"family": "Inter, -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif", "color": "#E9EEF8"},
        title_font={"size": 18},
        margin={"l": 40, "r": 20, "t": 56, "b": 42},
        legend={"bgcolor": "rgba(0,0,0,0)", "borderwidth": 0},
        hovermode="closest",
    )
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.08)", zerolinecolor="rgba(255,255,255,0.2)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.08)", zerolinecolor="rgba(255,255,255,0.2)")
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
                customdata=group.drop(columns=[x, y], errors="ignore"),
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
) -> go.Figure:
    fig = go.Figure()
    colors = _color_map(df[color])
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
        barmode="group",
        legend_title=color.replace("_", " ").title(),
    )
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

    for name, df in list(tables.items()):
        if "season" in df.columns:
            tables[name] = df[df["season"].isin(selected_seasons)].copy()
    for name, df in list(tables.items()):
        tables[name] = manager_filter(df, selected_managers)
    teams = tables["teams"]

    st.title("League History Dashboard")
    st.caption("A dashboard for your league's story: draft habits, luck, slot pain, and all-time receipts.")

    if teams.empty:
        st.info("No league data found. Load sample data or sync an ESPN league from the sidebar.")
        return

    records = all_time_records(tables["matchups"], tables["teams"])
    metric_row(
        {
            "Highest score": records.get("highest_score", "n/a"),
            "Worst loss": records.get("worst_loss", "n/a"),
            "Longest streak": records.get("longest_win_streak", "n/a"),
            "Biggest heartbreak": records.get("closest_loss", "n/a"),
        }
    )

    tab_luck, tab_h2h, tab_draft, tab_transactions, tab_player, tab_projection, tab_positions, tab_profiles, tab_records = st.tabs(
        [
            "Schedule Luck",
            "Head to Head",
            "Draft Room",
            "Transactions",
            "Player Profile",
            "Projections",
            "Positions",
            "Profiles",
            "Records",
        ]
    )

    with tab_luck:
        luck = schedule_luck(tables["matchups"], tables["teams"])
        enriched_luck = injury_luck(
            luck,
            tables["draft_picks"],
            tables["auction_values"],
            tables["roster_scores"],
            tables["injuries"],
            tables["teams"],
        )
        if luck.empty:
            st.info("No matchup data available.")
        else:
            c_luck_a, c_luck_b, c_luck_c = st.columns([1, 1, 1])
            granularity = c_luck_a.multiselect(
                "Group luck by",
                ["season", "manager_name", "team_name"],
                default=["season", "manager_name"],
            )
            view_mode = c_luck_b.selectbox(
                "Luck view",
                ["Trend", "Ranked bars", "Actual vs all-play"],
                index=0,
            )
            score_metric = c_luck_c.selectbox(
                "Luck metric",
                ["luck_wins", "overall_luck_index", "injury_value_lost"],
                format_func=lambda value: {
                    "luck_wins": "Schedule luck wins",
                    "overall_luck_index": "Overall luck index",
                    "injury_value_lost": "$ lost to injury",
                }[value],
            )
            chart_luck = enriched_luck if not enriched_luck.empty else luck
            aggregated_luck = aggregate_luck(chart_luck, granularity)
            if view_mode == "Actual vs all-play":
                fig = scatter_figure(
                    chart_luck,
                    "actual_win_pct",
                    "all_play_win_pct",
                    "manager_name",
                    "Schedule Luck: Actual vs All-Play",
                    "Actual win %",
                    "All-play win %",
                    size="injury_value_lost" if "injury_value_lost" in chart_luck.columns else "points_for",
                )
                fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line={"dash": "dash", "color": "#AAB3C5"})
            elif view_mode == "Trend" and "season" in aggregated_luck.columns and len(granularity) > 1:
                color_dim = "manager_name" if "manager_name" in aggregated_luck.columns else granularity[-1]
                fig = line_figure(
                    aggregated_luck,
                    "season",
                    score_metric,
                    color_dim,
                    "Luck Score Over Time",
                    "Season",
                    score_metric.replace("_", " ").title(),
                )
            else:
                x_dim = "manager_name" if "manager_name" in aggregated_luck.columns else (granularity[0] if granularity else "season")
                color_dim = "season" if "season" in aggregated_luck.columns and x_dim != "season" else x_dim
                fig = grouped_bar_figure(
                    aggregated_luck.sort_values(score_metric, ascending=score_metric == "injury_value_lost"),
                    x_dim,
                    score_metric,
                    color_dim,
                    "Schedule Luck Ranked",
                    x_dim.replace("_", " ").title(),
                    score_metric.replace("_", " ").title(),
                )
            st.plotly_chart(fig)

            st.dataframe(
                aggregated_luck,
                width="stretch",
                hide_index=True,
            )

    with tab_h2h:
        managers = sorted(tables["teams"]["manager_name"].dropna().unique().tolist())
        if len(managers) < 2:
            st.info("Head-to-head history needs at least two managers.")
        else:
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
                fig = line_figure(
                    h2h_games,
                    "game_label",
                    "points_for",
                    "manager_name",
                    f"{manager_a} vs {manager_b}: Scores by Week",
                    "Matchup",
                    "Points",
                )
                st.plotly_chart(fig)
                st.dataframe(
                    h2h_games[
                        [
                            "season",
                            "week",
                            "manager_name",
                            "team_name",
                            "points_for",
                            "opponent_manager_name",
                            "points_against",
                            "win",
                            "margin",
                        ]
                    ].sort_values(["season", "week", "manager_name"]),
                    width="stretch",
                    hide_index=True,
                )

    with tab_draft:
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
        c1, c2 = st.columns([1, 1])
        with c1:
            if tendencies.empty:
                st.info("No draft data available.")
            else:
                fig = grouped_bar_figure(
                    tendencies,
                    "manager_name",
                    "early_pick_share",
                    "position",
                    "Early Draft Tendencies",
                    "Manager",
                    "Early pick share",
                )
                st.plotly_chart(fig)
        with c2:
            if hindsight.empty:
                st.info("Hindsight draft grades need draft picks plus player scoring lines.")
            else:
                fig = grouped_bar_figure(
                    hindsight,
                    "manager_name",
                    "draft_value_score",
                    "season",
                    "Hindsight Draft Value",
                    "Manager",
                    "Hindsight value score",
                )
                st.plotly_chart(fig)
                st.dataframe(hindsight, width="stretch", hide_index=True)

        if not scorecard.empty:
            st.subheader("Draft Scorecard")
            fig = grouped_bar_figure(
                scorecard,
                "manager_name",
                "draft_score",
                "season",
                "Draft Score: Risk, Build, Sleepers",
                "Manager",
                "Draft score",
            )
            st.plotly_chart(fig)
            st.dataframe(
                scorecard[
                    [
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
                ],
                width="stretch",
                hide_index=True,
            )

        if not auction_values.empty:
            with st.expander("Auction price detail"):
                st.dataframe(
                    auction_values[
                        [
                            "season",
                            "manager_name",
                            "team_name",
                            "player_name",
                            "player_id",
                            "auction_value",
                            "source",
                        ]
                    ],
                    width="stretch",
                    hide_index=True,
                )

    with tab_transactions:
        scores, details = transaction_scorecard(tables["transactions"], tables["roster_scores"], tables["teams"])
        if scores.empty:
            st.info("No transaction data available.")
        else:
            fig = grouped_bar_figure(
                scores,
                "manager_name",
                "transaction_score",
                "season",
                "Transaction Score",
                "Manager",
                "Score",
            )
            st.plotly_chart(fig)
            st.dataframe(
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
                width="stretch",
                hide_index=True,
            )
            if details.empty:
                st.info("Transaction rows exist, but no future player scoring lines were available to grade them.")
            else:
                drops = details[details["score_type"] == "Unfortunate drop"].sort_values("future_points", ascending=False)
                if not drops.empty:
                    st.subheader("Unfortunate Drops")
                    st.dataframe(
                        drops[
                            [
                                "season",
                                "week",
                                "manager_name",
                                "player_id",
                                "player_name",
                                "transaction_type",
                                "future_points",
                            ]
                        ].head(25),
                        width="stretch",
                        hide_index=True,
                    )

    with tab_player:
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
                    if source_share.empty:
                        st.info("No acquisition source split available.")
                    else:
                        fig = go.Figure(
                            go.Pie(
                                labels=source_share["acquisition_source"],
                                values=source_share["points"],
                                hole=0.48,
                                marker={"colors": COLORWAY},
                            )
                        )
                        fig.update_layout(title="% of Starter Points by Acquisition Source")
                        st.plotly_chart(polish_figure(fig))
                        st.dataframe(source_share, width="stretch", hide_index=True)
                with c_hist:
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
                        title="Weekly Starter Points vs League Median",
                        xaxis_title="Points",
                        yaxis_title="Weeks",
                        barmode="overlay",
                    )
                    st.plotly_chart(polish_figure(fig))

                if not slot_weekly.empty:
                    slot_filter = st.multiselect(
                        "Roster slots",
                        sorted(slot_weekly["slot_role"].dropna().unique().tolist()),
                        default=sorted(slot_weekly["slot_role"].dropna().unique().tolist()),
                    )
                    slot_view = slot_weekly[slot_weekly["slot_role"].isin(slot_filter)].copy() if slot_filter else slot_weekly
                    fig = go.Figure()
                    for slot_name, group in slot_view.groupby("slot_role", dropna=False):
                        fig.add_trace(
                            go.Histogram(
                                x=group["delta_to_median"],
                                name=str(slot_name),
                                opacity=0.68,
                            )
                        )
                    fig.update_layout(
                        title="Slot Performance vs Median",
                        xaxis_title="Points over/under median",
                        yaxis_title="Weeks",
                        barmode="overlay",
                    )
                    st.plotly_chart(polish_figure(fig))
                    st.dataframe(slot_view, width="stretch", hide_index=True)

    with tab_projection:
        projection = projection_performance(tables["roster_scores"], tables["teams"])
        matrix = projection_matchup_matrix(tables["matchups"], tables["roster_scores"], tables["teams"])
        if projection.empty:
            st.info("No projected scoring data available for these seasons.")
        else:
            best_delta = projection.sort_values("average_delta", ascending=False).iloc[0]
            best_beat = projection.sort_values("beat_projection_pct", ascending=False).iloc[0]
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
                fig = scatter_figure(
                    projection,
                    "projected_points",
                    "actual_points",
                    "manager_name",
                    "Actual vs Projected Scoring",
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
            with c2:
                beat = (
                    projection.groupby("manager_name", as_index=False)
                    .agg(
                        average_delta=("average_delta", "mean"),
                        beat_projection_pct=("beat_projection_pct", "mean"),
                    )
                    .round({"average_delta": 2, "beat_projection_pct": 3})
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
                    title="Manager Projection Overperformance",
                    xaxis_title="Manager",
                    yaxis_title="Avg points over projection",
                )
                st.plotly_chart(polish_figure(fig))

            if not matrix.empty:
                wins = matrix[matrix["result"] == "Win"]
                fig = go.Figure(
                    go.Bar(
                        x=wins["projected_bucket"].astype(str),
                        y=wins["win_rate"],
                        marker_color=COLORWAY[: len(wins)],
                    )
                )
                fig.update_layout(
                    title="Projected Favorite Buckets",
                    xaxis_title="Projected matchup bucket",
                    yaxis_title="Actual win rate",
                    showlegend=False,
                )
                st.plotly_chart(polish_figure(fig))

            st.dataframe(
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
                width="stretch",
                hide_index=True,
            )

    with tab_positions:
        positional = positional_performance(tables["roster_scores"], tables["teams"])
        if positional.empty:
            st.info("No roster slot data available.")
        else:
            heatmap = positional.pivot_table(
                index="manager_name",
                columns="position",
                values="slot_points_per_week",
                aggfunc="mean",
            ).fillna(0)
            fig = go.Figure(
                go.Heatmap(
                    z=heatmap.values,
                    x=heatmap.columns.tolist(),
                    y=heatmap.index.tolist(),
                    colorscale="RdYlGn",
                    colorbar={"title": "Pts/week"},
                )
            )
            fig.update_layout(title="Positional Performance", xaxis_title="Position", yaxis_title="Manager")
            st.plotly_chart(polish_figure(fig))
            st.dataframe(positional, width="stretch", hide_index=True)

    with tab_profiles:
        profiles = manager_profiles(tables["transactions"], tables["roster_scores"], tables["teams"])
        if profiles.empty:
            st.info("No transaction or bench data available.")
        else:
            fig = scatter_figure(
                profiles,
                "waiver_adds",
                "bench_points_left",
                "manager_name",
                "Manager Activity vs Lineup Waste",
                "Waiver adds",
                "Bench points left",
                size="trades",
            )
            st.plotly_chart(fig)
            st.dataframe(profiles, width="stretch", hide_index=True)

    with tab_records:
        st.subheader("All-Time Receipts")
        for key, value in records.items():
            st.write(f"**{key.replace('_', ' ').title()}**: {value}")


if __name__ == "__main__":
    main()
