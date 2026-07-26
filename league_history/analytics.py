from __future__ import annotations

import pandas as pd


def _with_team(df: pd.DataFrame, teams: pd.DataFrame) -> pd.DataFrame:
    if df.empty or teams.empty:
        return df.copy()
    cols = ["season", "team_id", "manager_name", "team_name"]
    return df.merge(teams[cols], on=["season", "team_id"], how="inner")


def schedule_luck(matchups: pd.DataFrame, teams: pd.DataFrame) -> pd.DataFrame:
    if matchups.empty:
        return pd.DataFrame()

    rows = []
    for (season, team_id), team_games in matchups.groupby(["season", "team_id"]):
        actual_wins = int(team_games["win"].sum())
        games = len(team_games)
        all_play_wins = 0
        all_play_losses = 0
        for week, game in team_games.set_index("week").iterrows():
            field = matchups[(matchups["season"] == season) & (matchups["week"] == week)]
            opponents = field[field["team_id"] != team_id]
            all_play_wins += int((game["points_for"] > opponents["points_for"]).sum())
            all_play_losses += int((game["points_for"] < opponents["points_for"]).sum())
        all_play_games = all_play_wins + all_play_losses
        rows.append(
            {
                "season": season,
                "team_id": team_id,
                "actual_wins": actual_wins,
                "actual_losses": games - actual_wins,
                "actual_record": f"{actual_wins}-{games - actual_wins}",
                "actual_win_pct": actual_wins / games if games else 0,
                "all_play_wins": all_play_wins,
                "all_play_losses": all_play_losses,
                "all_play_record": f"{all_play_wins}-{all_play_losses}",
                "all_play_win_pct": all_play_wins / all_play_games if all_play_games else 0,
                "luck_wins": round(actual_wins - (all_play_wins / max(all_play_games, 1) * games), 2),
                "points_for": round(team_games["points_for"].sum(), 2),
            }
        )

    result = _with_team(pd.DataFrame(rows), teams)
    if result.empty:
        return result
    result["points_for_rank"] = result.groupby("season")["points_for"].rank(method="min", ascending=False).astype(int)
    return result.sort_values(["season", "luck_wins"], ascending=[True, False])


def aggregate_luck(luck: pd.DataFrame, dimensions: list[str]) -> pd.DataFrame:
    if luck.empty:
        return pd.DataFrame()
    dims = [dim for dim in dimensions if dim in luck.columns]
    if not dims:
        dims = ["manager_name"]
    aggregations = {
        "luck_wins": ("luck_wins", "sum"),
        "actual_wins": ("actual_wins", "sum"),
        "actual_losses": ("actual_losses", "sum"),
        "all_play_win_pct": ("all_play_win_pct", "mean"),
        "actual_win_pct": ("actual_win_pct", "mean"),
        "points_for": ("points_for", "sum"),
        "seasons": ("season", "nunique"),
    }
    for optional in ["injury_value_lost", "injured_player_weeks", "injury_luck_index", "schedule_luck_index", "overall_luck_index"]:
        if optional in luck.columns:
            aggregations[optional] = (optional, "sum" if optional in {"injury_value_lost", "injured_player_weeks"} else "mean")
    result = luck.groupby(dims, dropna=False).agg(**aggregations).reset_index()
    result["luck_wins"] = result["luck_wins"].round(2)
    result["all_play_win_pct"] = result["all_play_win_pct"].round(3)
    result["actual_win_pct"] = result["actual_win_pct"].round(3)
    result["points_for"] = result["points_for"].round(1)
    for optional in ["injury_value_lost", "injury_luck_index", "schedule_luck_index", "overall_luck_index"]:
        if optional in result.columns:
            result[optional] = result[optional].round(2)
    return result.sort_values("luck_wins", ascending=False)


def auction_player_values(draft_picks: pd.DataFrame, auction_values: pd.DataFrame, teams: pd.DataFrame) -> pd.DataFrame:
    frames = []
    if not draft_picks.empty and "auction_value" in draft_picks.columns:
        draft = draft_picks.copy()
        draft["auction_value"] = pd.to_numeric(draft["auction_value"], errors="coerce")
        draft = draft.dropna(subset=["auction_value"])
        if not draft.empty:
            frames.append(
                draft[
                    ["season", "player_id", "player_name", "team_id", "auction_value"]
                ].assign(source="espn_draft")
            )
    if not auction_values.empty:
        upload = auction_values.copy()
        upload["auction_value"] = pd.to_numeric(upload["auction_value"], errors="coerce")
        upload = upload.dropna(subset=["auction_value"])
        if not upload.empty:
            frames.append(
                upload[
                    ["season", "player_id", "player_name", "team_id", "manager_name", "auction_value", "source"]
                ].copy()
            )
    if not frames:
        return pd.DataFrame()

    values = pd.concat(frames, ignore_index=True)
    if "manager_name" not in values.columns:
        values["manager_name"] = None
    values["player_key"] = values["player_name"].fillna("").str.strip().str.lower()

    if not teams.empty:
        team_cols = teams[["season", "team_id", "manager_name", "team_name"]].drop_duplicates()
        values = values.merge(team_cols, on=["season", "team_id"], how="left", suffixes=("", "_team"))
        values["manager_name"] = values["manager_name"].fillna(values["manager_name_team"])
        values = values.drop(columns=["manager_name_team"], errors="ignore")

    return values.sort_values(["season", "auction_value"], ascending=[True, False])


def injury_luck(
    luck: pd.DataFrame,
    draft_picks: pd.DataFrame,
    auction_values: pd.DataFrame,
    roster_scores: pd.DataFrame,
    injuries: pd.DataFrame,
    teams: pd.DataFrame,
) -> pd.DataFrame:
    if luck.empty:
        return pd.DataFrame()

    base = luck.copy()
    base["luck_wins"] = pd.to_numeric(base["luck_wins"], errors="coerce").fillna(0)
    if draft_picks.empty:
        return base.assign(injury_value_lost=0.0, injured_player_weeks=0, injury_luck_index=0.0, overall_luck_index=0.0)

    values = auction_player_values(draft_picks, auction_values, teams)
    if values.empty:
        values = draft_picks[["season", "player_id", "player_name", "team_id"]].copy()
        values["auction_value"] = 0.0
        values["player_key"] = values["player_name"].fillna("").str.strip().str.lower()
    values = values.drop_duplicates(["season", "player_id"], keep="first")

    injury_frames = []
    if not injuries.empty:
        injury_frames.append(injuries.copy())
    if not roster_scores.empty and "injury_status" in roster_scores.columns:
        injury_frames.append(
            roster_scores[["season", "week", "player_id", "player_name", "injury_status"]].copy().assign(source="roster_scores")
        )
    if injury_frames:
        hurt = pd.concat(injury_frames, ignore_index=True)
    else:
        hurt = pd.DataFrame(columns=["season", "week", "player_id", "player_name", "injury_status"])

    if hurt.empty:
        out = base.assign(injury_value_lost=0.0, injured_player_weeks=0)
    else:
        hurt["injury_status"] = hurt["injury_status"].fillna("").astype(str)
        hurt = hurt[
            hurt["injury_status"].str.contains(
                r"\b(?:OUT|IR|INJURED|DNP|DOUBTFUL)\b", case=False, na=False, regex=True
            )
        ].copy()
        hurt["player_key"] = hurt["player_name"].fillna("").str.strip().str.lower()
        hurt = hurt.drop_duplicates(["season", "week", "player_id", "player_key"])
        if hurt.empty:
            out = base.assign(injury_value_lost=0.0, injured_player_weeks=0)
        else:
            max_weeks = luck.groupby("season")["actual_wins"].count().rename("team_rows").reset_index()
            season_weeks = roster_scores.groupby("season")["week"].nunique().reset_index(name="season_weeks") if not roster_scores.empty else pd.DataFrame()
            hurt_values = hurt.merge(values, on=["season", "player_id"], how="left", suffixes=("", "_value"))
            missing = hurt_values["team_id"].isna() & hurt_values["player_key"].ne("")
            if missing.any():
                by_name = values.dropna(subset=["player_key"]).drop_duplicates(["season", "player_key"])
                named = hurt_values.loc[missing, ["season", "player_key"]].merge(
                    by_name[["season", "player_key", "team_id", "auction_value"]],
                    on=["season", "player_key"],
                    how="left",
                )
                hurt_values.loc[missing, "team_id"] = named["team_id"].to_numpy()
                hurt_values.loc[missing, "auction_value"] = named["auction_value"].to_numpy()
            hurt_values = hurt_values.merge(season_weeks, on="season", how="left")
            hurt_values["season_weeks"] = pd.to_numeric(hurt_values["season_weeks"], errors="coerce").fillna(14).clip(lower=1)
            hurt_values["auction_value"] = pd.to_numeric(hurt_values["auction_value"], errors="coerce").fillna(0)
            hurt_values["injury_value_lost"] = hurt_values["auction_value"] / hurt_values["season_weeks"]
            penalty = (
                hurt_values.dropna(subset=["team_id"])
                .groupby(["season", "team_id"], dropna=False)
                .agg(injury_value_lost=("injury_value_lost", "sum"), injured_player_weeks=("week", "count"))
                .reset_index()
            )
            out = base.merge(penalty, on=["season", "team_id"], how="left")
            out["injury_value_lost"] = out["injury_value_lost"].fillna(0)
            out["injured_player_weeks"] = out["injured_player_weeks"].fillna(0).astype(int)

    out["injury_value_lost"] = out["injury_value_lost"].round(2)
    out["injury_luck_index"] = out.groupby("season")["injury_value_lost"].transform(
        lambda s: 0 if s.std(ddof=0) == 0 else -((s - s.mean()) / s.std(ddof=0))
    )
    out["schedule_luck_index"] = out.groupby("season")["luck_wins"].transform(
        lambda s: 0 if s.std(ddof=0) == 0 else (s - s.mean()) / s.std(ddof=0)
    )
    out["overall_luck_index"] = (out["schedule_luck_index"] + out["injury_luck_index"]).round(2)
    out["injury_luck_index"] = out["injury_luck_index"].round(2)
    out["schedule_luck_index"] = out["schedule_luck_index"].round(2)
    return out.sort_values(["season", "overall_luck_index"], ascending=[True, False])


def head_to_head_history(matchups: pd.DataFrame, teams: pd.DataFrame, manager_a: str, manager_b: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    if matchups.empty or teams.empty or not manager_a or not manager_b or manager_a == manager_b:
        return pd.DataFrame(), pd.DataFrame()

    team_lookup = teams[["season", "team_id", "manager_name", "team_name"]].drop_duplicates()
    games = matchups.merge(team_lookup, on=["season", "team_id"], how="inner")
    opponent_lookup = team_lookup.rename(
        columns={
            "team_id": "opponent_id",
            "manager_name": "opponent_manager_name",
            "team_name": "opponent_team_name",
        }
    )
    games = games.merge(opponent_lookup, on=["season", "opponent_id"], how="inner")
    selected = games[
        ((games["manager_name"] == manager_a) & (games["opponent_manager_name"] == manager_b))
        | ((games["manager_name"] == manager_b) & (games["opponent_manager_name"] == manager_a))
    ].copy()
    if selected.empty:
        return pd.DataFrame(), pd.DataFrame()

    selected["margin"] = selected["points_for"] - selected["points_against"]
    selected["game_label"] = selected["season"].astype(str) + " W" + selected["week"].astype(int).astype(str).str.zfill(2)
    selected = selected.sort_values(["season", "week", "manager_name"])

    summary_rows = []
    for manager in [manager_a, manager_b]:
        rows = selected[selected["manager_name"] == manager]
        if rows.empty:
            summary_rows.append(
                {"manager_name": manager, "wins": 0, "losses": 0, "points_for": 0.0, "avg_margin": 0.0}
            )
            continue
        summary_rows.append(
            {
                "manager_name": manager,
                "wins": int(rows["win"].sum()),
                "losses": int((1 - rows["win"]).sum()),
                "points_for": round(rows["points_for"].sum(), 1),
                "avg_margin": round(rows["margin"].mean(), 2),
            }
        )
    return selected, pd.DataFrame(summary_rows)


def head_to_head_game_table(games: pd.DataFrame, manager_a: str, manager_b: str) -> pd.DataFrame:
    """One row per game (not one row per team-perspective) for display purposes.

    `games` (the first frame returned by `head_to_head_history`) has two rows per
    game - one from each manager's perspective - which is what the summary/chart
    need, but reads as duplicated games in a detail table.
    """
    if games.empty:
        return pd.DataFrame()
    rows = games[games["manager_name"] == manager_a].copy()
    if rows.empty:
        return pd.DataFrame()
    rows = rows.rename(
        columns={
            "team_name": f"{manager_a}_team",
            "points_for": f"{manager_a}_points",
            "opponent_team_name": f"{manager_b}_team",
            "points_against": f"{manager_b}_points",
        }
    )
    rows["winner"] = rows["win"].map({1: manager_a, 0: manager_b})
    return rows[
        [
            "season",
            "week",
            f"{manager_a}_team",
            f"{manager_a}_points",
            f"{manager_b}_team",
            f"{manager_b}_points",
            "margin",
            "winner",
        ]
    ].sort_values(["season", "week"])


def draft_tendencies(draft_picks: pd.DataFrame, teams: pd.DataFrame) -> pd.DataFrame:
    if draft_picks.empty:
        return pd.DataFrame()
    early = draft_picks[draft_picks["round"].fillna(99) <= 5].copy()
    if early.empty:
        return pd.DataFrame()
    grouped = (
        early.groupby(["season", "team_id", "position"], dropna=False)
        .size()
        .reset_index(name="early_picks")
    )
    totals = grouped.groupby(["season", "team_id"])["early_picks"].transform("sum")
    grouped["early_pick_share"] = grouped["early_picks"] / totals
    return _with_team(grouped, teams).sort_values(["season", "manager_name", "position"])


def draft_hindsight(draft_picks: pd.DataFrame, roster_scores: pd.DataFrame, teams: pd.DataFrame) -> pd.DataFrame:
    if draft_picks.empty or roster_scores.empty:
        return pd.DataFrame()
    player_points = (
        roster_scores.groupby(["season", "player_id"], dropna=False)["points"]
        .sum()
        .reset_index(name="season_points")
    )
    picks = draft_picks.merge(player_points, on=["season", "player_id"], how="left")
    picks["season_points"] = picks["season_points"].fillna(0)
    replacement = picks.groupby(["season", "position"])["season_points"].transform("median")
    pick_penalty = picks["pick"].fillna(picks.groupby("season")["pick"].transform("max")).fillna(100) * 0.35
    auction_values = pd.to_numeric(picks["auction_value"], errors="coerce").fillna(0)
    auction_penalty = auction_values * 1.4
    picks["value_over_cost"] = picks["season_points"] - replacement - pick_penalty - auction_penalty
    result = (
        picks.groupby(["season", "team_id"], dropna=False)
        .agg(
            draft_value_score=("value_over_cost", "sum"),
            best_pick_points=("season_points", "max"),
            drafted_players=("player_id", "count"),
        )
        .reset_index()
    )
    result["draft_value_score"] = result["draft_value_score"].round(1)
    return _with_team(result, teams).sort_values(["season", "draft_value_score"], ascending=[True, False])


def _score_within_season(df: pd.DataFrame, column: str, higher_is_better: bool = True) -> pd.Series:
    values = pd.to_numeric(df[column], errors="coerce").fillna(0)
    ranks = values.groupby(df["season"]).rank(pct=True, ascending=higher_is_better)
    return (ranks * 100).round(0)


def draft_scorecard(
    draft_picks: pd.DataFrame,
    roster_scores: pd.DataFrame,
    teams: pd.DataFrame,
    auction_values: pd.DataFrame,
    injuries: pd.DataFrame,
) -> pd.DataFrame:
    if draft_picks.empty:
        return pd.DataFrame()

    picks = draft_picks.copy()
    values = auction_player_values(draft_picks, auction_values, teams)
    if not values.empty:
        picks = picks.merge(
            values[["season", "player_id", "auction_value"]].drop_duplicates(["season", "player_id"]),
            on=["season", "player_id"],
            how="left",
            suffixes=("", "_resolved"),
        )
        picks["auction_value"] = pd.to_numeric(picks.get("auction_value"), errors="coerce").fillna(
            pd.to_numeric(picks.get("auction_value_resolved"), errors="coerce")
        )

    if not roster_scores.empty:
        player_points = (
            roster_scores.groupby(["season", "player_id"], dropna=False)["points"]
            .sum()
            .reset_index(name="season_points")
        )
        picks = picks.merge(player_points, on=["season", "player_id"], how="left")
    else:
        picks["season_points"] = 0
    picks["season_points"] = pd.to_numeric(picks["season_points"], errors="coerce").fillna(0)
    picks["round"] = pd.to_numeric(picks["round"], errors="coerce")
    picks["auction_value"] = pd.to_numeric(picks.get("auction_value"), errors="coerce")

    dummy_luck = teams[["season", "team_id", "manager_name", "team_name"]].drop_duplicates().assign(luck_wins=0.0)
    injury = injury_luck(dummy_luck, draft_picks, auction_values, roster_scores, injuries, teams)
    if injury.empty:
        risk = picks.groupby(["season", "team_id"]).size().reset_index(name="drafted_players")
        risk["injury_value_lost"] = 0.0
    else:
        risk = injury[["season", "team_id", "injury_value_lost"]].copy()

    early = picks[(picks["round"].fillna(99) <= 8) | (picks["auction_value"].fillna(0) >= 8)].copy()
    required = ["QB", "RB", "WR", "TE"]
    construction = (
        early.groupby(["season", "team_id", "position"], dropna=False)
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    for pos in required:
        if pos not in construction.columns:
            construction[pos] = 0
    construction["lineup_coverage_raw"] = (
        construction["QB"].clip(upper=1)
        + (construction["RB"].clip(upper=2) / 2)
        + (construction["WR"].clip(upper=2) / 2)
        + construction["TE"].clip(upper=1)
    ) / 4

    pos_medians = picks.groupby(["season", "position"])["season_points"].transform("median")
    cheap = picks[(picks["round"].fillna(0) >= 9) | (picks["auction_value"].fillna(99) <= 5)].copy()
    cheap["sleeper_value"] = (cheap["season_points"] - pos_medians.loc[cheap.index]).clip(lower=0)
    sleeper = (
        cheap.groupby(["season", "team_id"], dropna=False)["sleeper_value"]
        .sum()
        .reset_index()
    )

    base = teams[["season", "team_id", "manager_name", "team_name"]].drop_duplicates()
    out = base.merge(risk, on=["season", "team_id"], how="left")
    out = out.merge(construction[["season", "team_id", "lineup_coverage_raw"]], on=["season", "team_id"], how="left")
    out = out.merge(sleeper, on=["season", "team_id"], how="left")
    out["injury_value_lost"] = out["injury_value_lost"].fillna(0)
    out["lineup_coverage_raw"] = out["lineup_coverage_raw"].fillna(0)
    out["sleeper_value"] = out["sleeper_value"].fillna(0)
    out["risk_avoidance_score"] = _score_within_season(out, "injury_value_lost", higher_is_better=False)
    out["lineup_construction_score"] = (out["lineup_coverage_raw"] * 100).round(0)
    out["sleeper_score"] = _score_within_season(out, "sleeper_value", higher_is_better=True)
    out["draft_score"] = (
        out[["risk_avoidance_score", "lineup_construction_score", "sleeper_score"]].mean(axis=1)
    ).round(0)
    return out.sort_values(["season", "draft_score"], ascending=[True, False])


def positional_performance(roster_scores: pd.DataFrame, teams: pd.DataFrame) -> pd.DataFrame:
    if roster_scores.empty:
        return pd.DataFrame()
    starters = roster_scores[roster_scores["is_starter"] == 1].copy()
    # Bucket by "Flex" (not the underlying position) whenever a player started in a flex
    # slot, so flex usage shows up as its own column instead of being folded into
    # whatever real position happened to fill it that week.
    starters["position"] = starters["slot"].where(starters["slot"] == "Flex", starters["position"])
    result = (
        starters.groupby(["season", "team_id", "position"], dropna=False)
        .agg(slot_points=("points", "sum"), weeks=("week", "nunique"))
        .reset_index()
    )
    result["slot_points_per_week"] = (result["slot_points"] / result["weeks"].clip(lower=1)).round(2)
    return _with_team(result, teams).sort_values(["season", "manager_name", "position"])


def slot_role(position: object, slot: object) -> str:
    slot_name = str(slot or "UNK")
    position_name = str(position or "UNK")
    if slot_name.lower() == "flex":
        return f"Flex-{position_name}"
    return slot_name


def transaction_scorecard(transactions: pd.DataFrame, roster_scores: pd.DataFrame, teams: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if teams.empty:
        return pd.DataFrame(), pd.DataFrame()
    base = teams[["season", "team_id", "manager_name", "team_name"]].drop_duplicates()
    if transactions.empty:
        return base.assign(add_value=0.0, trade_value=0.0, unfortunate_drop_value=0.0, move_count=0, transaction_score=0.0), pd.DataFrame()

    txn = transactions.copy()
    txn["week"] = pd.to_numeric(txn["week"], errors="coerce")
    txn = txn.dropna(subset=["week"]).copy()
    if txn.empty:
        return base.assign(add_value=0.0, trade_value=0.0, unfortunate_drop_value=0.0, move_count=0, transaction_score=0.0), pd.DataFrame()
    txn["week"] = txn["week"].astype(int)
    txn["transaction_type"] = txn["transaction_type"].fillna("").astype(str)
    txn["is_add"] = txn["transaction_type"].str.contains("ADD|WAIVER|FREEAGENT", case=False, na=False)
    txn["is_trade"] = txn["transaction_type"].str.contains("TRADE", case=False, na=False)
    txn["is_drop"] = txn["transaction_type"].str.contains("DROP", case=False, na=False)

    future = roster_scores[["season", "week", "team_id", "player_id", "points", "is_starter"]].copy() if not roster_scores.empty else pd.DataFrame()
    if future.empty:
        details = txn.assign(future_points=0.0, starter_future_points=0.0, score_type="Unscored")
    else:
        detail_rows = []
        for row in txn.itertuples(index=False):
            player_games = future[(future["season"] == row.season) & (future["player_id"] == row.player_id) & (future["week"] >= row.week)]
            if getattr(row, "is_drop"):
                player_games = player_games[player_games["team_id"] != row.team_id]
                score_type = "Unfortunate drop"
            else:
                player_games = player_games[player_games["team_id"] == row.team_id]
                score_type = "Trade" if getattr(row, "is_trade") else "Pickup"
            detail = row._asdict()
            detail["future_points"] = float(player_games["points"].sum()) if not player_games.empty else 0.0
            detail["starter_future_points"] = float(player_games[player_games["is_starter"] == 1]["points"].sum()) if not player_games.empty else 0.0
            detail["score_type"] = score_type
            detail_rows.append(detail)
        details = pd.DataFrame(detail_rows)

    summary = (
        details.groupby(["season", "team_id"], dropna=False)
        .agg(
            add_value=("future_points", lambda s: s[details.loc[s.index, "is_add"]].sum()),
            trade_value=("future_points", lambda s: s[details.loc[s.index, "is_trade"]].sum()),
            unfortunate_drop_value=("future_points", lambda s: s[details.loc[s.index, "is_drop"]].sum()),
            move_count=("player_id", "count"),
        )
        .reset_index()
    )
    out = base.merge(summary, on=["season", "team_id"], how="left").fillna(
        {"add_value": 0, "trade_value": 0, "unfortunate_drop_value": 0, "move_count": 0}
    )
    out["net_transaction_value"] = out["add_value"] + out["trade_value"] - out["unfortunate_drop_value"]
    out["transaction_score"] = _score_within_season(out, "net_transaction_value", higher_is_better=True)
    numeric = ["add_value", "trade_value", "unfortunate_drop_value", "net_transaction_value"]
    out[numeric] = out[numeric].round(1)
    details = _with_team(details, teams) if not details.empty else details
    return out.sort_values(["season", "transaction_score"], ascending=[True, False]), details


def player_profile_frames(
    manager_name: str,
    roster_scores: pd.DataFrame,
    teams: pd.DataFrame,
    draft_picks: pd.DataFrame,
    transactions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if roster_scores.empty or teams.empty or not manager_name:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    rows = _with_team(roster_scores, teams)
    rows = rows[rows["manager_name"] == manager_name].copy()
    if rows.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    drafted = set()
    if not draft_picks.empty:
        draft_team = draft_picks.merge(teams[["season", "team_id", "manager_name"]], on=["season", "team_id"], how="left")
        drafted = {
            (int(row.season), int(row.player_id))
            for row in draft_team[draft_team["manager_name"] == manager_name].dropna(subset=["player_id"]).itertuples()
        }

    trade = set()
    pickup = set()
    if not transactions.empty:
        tx = transactions.merge(teams[["season", "team_id", "manager_name"]], on=["season", "team_id"], how="left")
        tx = tx[tx["manager_name"] == manager_name].dropna(subset=["player_id"])
        for row in tx.itertuples():
            key = (int(row.season), int(row.player_id))
            txn_type = str(row.transaction_type)
            if "TRADE" in txn_type.upper():
                trade.add(key)
            elif any(token in txn_type.upper() for token in ["ADD", "WAIVER", "FREEAGENT"]):
                pickup.add(key)

    def source_for(row: pd.Series) -> str:
        if pd.isna(row["player_id"]):
            return "Unknown"
        key = (int(row["season"]), int(row["player_id"]))
        if key in trade:
            return "Trade"
        if key in pickup:
            return "Pickup"
        if key in drafted:
            return "Draft"
        return "Pickup"

    starters = rows[rows["is_starter"] == 1].copy()
    starters["acquisition_source"] = starters.apply(source_for, axis=1)
    starters["slot_role"] = [slot_role(pos, slot) for pos, slot in zip(starters["position"], starters["slot"])]
    source_share = (
        starters.groupby("acquisition_source", dropna=False)["points"]
        .sum()
        .reset_index(name="points")
        .sort_values("points", ascending=False)
    )
    total = source_share["points"].sum()
    source_share["point_share"] = (source_share["points"] / total).fillna(0).round(3)

    all_starters = _with_team(roster_scores[roster_scores["is_starter"] == 1].copy(), teams)
    weekly = (
        starters.groupby(["season", "week"], dropna=False)["points"].sum().reset_index(name="manager_points")
    )
    league_weekly = (
        all_starters.groupby(["season", "team_id", "week"], dropna=False)["points"].sum().reset_index(name="team_points")
    )
    median_weekly = league_weekly.groupby(["season", "week"])["team_points"].median().reset_index(name="league_median")
    weekly = weekly.merge(median_weekly, on=["season", "week"], how="left")
    weekly["delta_to_median"] = (weekly["manager_points"] - weekly["league_median"]).round(2)

    all_starters["slot_role"] = [slot_role(pos, slot) for pos, slot in zip(all_starters["position"], all_starters["slot"])]
    by_slot = starters.groupby(["season", "week", "slot_role"], dropna=False)["points"].sum().reset_index(name="manager_points")
    league_slot = (
        all_starters.groupby(["season", "team_id", "week", "slot_role"], dropna=False)["points"]
        .sum()
        .reset_index(name="team_slot_points")
    )
    median_slot = (
        league_slot.groupby(["season", "week", "slot_role"], dropna=False)["team_slot_points"]
        .median()
        .reset_index(name="league_median")
    )
    by_slot = by_slot.merge(median_slot, on=["season", "week", "slot_role"], how="left")
    by_slot["delta_to_median"] = (by_slot["manager_points"] - by_slot["league_median"]).round(2)
    return source_share, weekly, by_slot


def acquisition_source_league_average(
    roster_scores: pd.DataFrame,
    teams: pd.DataFrame,
    draft_picks: pd.DataFrame,
    transactions: pd.DataFrame,
) -> pd.DataFrame:
    """Average acquisition-source point share across every manager, for benchmarking one manager against the field.

    Each manager's shares are zero-filled across every source seen league-wide before
    averaging, so a manager who got none of their points from trades (say) correctly
    pulls the trade average down instead of being silently excluded from it.
    """
    if roster_scores.empty or teams.empty:
        return pd.DataFrame()
    manager_shares: dict[str, dict[str, float]] = {}
    all_sources: set[str] = set()
    for manager in teams["manager_name"].dropna().unique().tolist():
        source_share, _weekly, _by_slot = player_profile_frames(manager, roster_scores, teams, draft_picks, transactions)
        if source_share.empty:
            continue
        shares = dict(zip(source_share["acquisition_source"], source_share["point_share"]))
        manager_shares[manager] = shares
        all_sources.update(shares.keys())
    if not manager_shares:
        return pd.DataFrame()
    rows = [
        {
            "acquisition_source": source,
            "point_share": round(
                sum(shares.get(source, 0.0) for shares in manager_shares.values()) / len(manager_shares), 3
            ),
        }
        for source in sorted(all_sources)
    ]
    return pd.DataFrame(rows)


def projection_performance(roster_scores: pd.DataFrame, teams: pd.DataFrame) -> pd.DataFrame:
    if roster_scores.empty or "projected_points" not in roster_scores.columns:
        return pd.DataFrame()
    starters = roster_scores[roster_scores["is_starter"] == 1].copy()
    starters["projected_points"] = pd.to_numeric(starters["projected_points"], errors="coerce")
    starters = starters.dropna(subset=["projected_points"])
    if starters.empty:
        return pd.DataFrame()

    weekly = (
        starters.groupby(["season", "team_id", "week"], dropna=False)
        .agg(actual_points=("points", "sum"), projected_points=("projected_points", "sum"))
        .reset_index()
    )
    weekly["projection_delta"] = weekly["actual_points"] - weekly["projected_points"]
    weekly["beat_projection"] = weekly["projection_delta"] > 0
    result = (
        weekly.groupby(["season", "team_id"], dropna=False)
        .agg(
            weeks=("week", "nunique"),
            actual_points=("actual_points", "sum"),
            projected_points=("projected_points", "sum"),
            average_delta=("projection_delta", "mean"),
            beat_projection_pct=("beat_projection", "mean"),
        )
        .reset_index()
    )
    result["actual_points"] = result["actual_points"].round(1)
    result["projected_points"] = result["projected_points"].round(1)
    result["average_delta"] = result["average_delta"].round(2)
    result["beat_projection_pct"] = result["beat_projection_pct"].round(3)
    return _with_team(result, teams).sort_values(["season", "average_delta"], ascending=[True, False])


def projection_matchup_matrix(matchups: pd.DataFrame, roster_scores: pd.DataFrame, teams: pd.DataFrame) -> pd.DataFrame:
    if matchups.empty or roster_scores.empty or "projected_points" not in roster_scores.columns:
        return pd.DataFrame()
    starters = roster_scores[roster_scores["is_starter"] == 1].copy()
    starters["projected_points"] = pd.to_numeric(starters["projected_points"], errors="coerce")
    starters = starters.dropna(subset=["projected_points"])
    if starters.empty:
        return pd.DataFrame()

    projected = (
        starters.groupby(["season", "team_id", "week"], dropna=False)["projected_points"]
        .sum()
        .reset_index()
    )
    games = matchups.merge(projected, on=["season", "team_id", "week"], how="inner")
    opp = projected.rename(
        columns={"team_id": "opponent_id", "projected_points": "opponent_projected_points"}
    )
    games = games.merge(opp, on=["season", "opponent_id", "week"], how="inner")
    games = _with_team(games, teams)
    if games.empty:
        return pd.DataFrame()
    games["projected_margin"] = games["projected_points"] - games["opponent_projected_points"]
    games["projected_bucket"] = pd.cut(
        games["projected_margin"],
        bins=[-float("inf"), -10, -5, 0, 5, 10, float("inf")],
        labels=["Big dog", "Medium dog", "Little dog", "Little favorite", "Medium favorite", "Big favorite"],
    )
    games["result"] = games["win"].map({1: "Win", 0: "Loss"})
    result = (
        games.groupby(["projected_bucket", "result"], observed=False)
        .size()
        .reset_index(name="games")
    )
    result["win_rate"] = result.groupby("projected_bucket", observed=False)["games"].transform(
        lambda values: values / values.sum()
    )
    return result


def manager_profiles(transactions: pd.DataFrame, roster_scores: pd.DataFrame, teams: pd.DataFrame) -> pd.DataFrame:
    if teams.empty:
        return pd.DataFrame()
    base = teams[["season", "team_id", "manager_name", "team_name"]].drop_duplicates()

    if transactions.empty:
        txn = base.assign(waiver_adds=0, trades=0)
    else:
        t = transactions.copy()
        t["is_add"] = t["transaction_type"].str.contains("ADD|WAIVER|FREEAGENT", case=False, na=False).astype(int)
        t["is_trade"] = t["transaction_type"].str.contains("TRADE", case=False, na=False).astype(int)
        txn = (
            t.groupby(["season", "team_id"])
            .agg(waiver_adds=("is_add", "sum"), trades=("is_trade", "sum"))
            .reset_index()
        )
        txn = base.merge(txn, on=["season", "team_id"], how="left").fillna({"waiver_adds": 0, "trades": 0})

    if roster_scores.empty:
        return txn.assign(bench_points_left=0.0, optimality_pct=1.0)

    rs = roster_scores.copy()
    rs["points"] = pd.to_numeric(rs["points"], errors="coerce").fillna(0)

    starter_totals = (
        rs[rs["is_starter"] == 1]
        .groupby(["season", "team_id", "week"])["points"]
        .sum()
        .reset_index(name="starter_points")
    )

    # How many slots were actually used at each position that week (a "position" here is the
    # player's real position, e.g. RB, even if it filled a Flex slot).
    required_counts = (
        rs[rs["is_starter"] == 1]
        .groupby(["season", "team_id", "week", "position"], dropna=False)
        .size()
        .reset_index(name="required")
    )

    # Best-possible lineup: for each team/week/position, keep the top-scoring players (starters
    # or bench) up to the number of slots actually used at that position that week, and sum them.
    rs["position_rank"] = rs.groupby(["season", "team_id", "week", "position"], dropna=False)["points"].rank(
        method="first", ascending=False
    )
    eligible = rs.merge(required_counts, on=["season", "team_id", "week", "position"], how="left")
    eligible["required"] = eligible["required"].fillna(0)
    optimal_totals = (
        eligible[eligible["position_rank"] <= eligible["required"]]
        .groupby(["season", "team_id", "week"])["points"]
        .sum()
        .reset_index(name="optimal_points")
    )

    weekly = starter_totals.merge(optimal_totals, on=["season", "team_id", "week"], how="left")
    weekly["optimal_points"] = weekly["optimal_points"].fillna(weekly["starter_points"])
    weekly["bench_points_left"] = (weekly["optimal_points"] - weekly["starter_points"]).clip(lower=0)
    weekly["optimality_pct"] = weekly["starter_points"] / weekly["optimal_points"].clip(lower=1)

    opt = (
        weekly.groupby(["season", "team_id"])
        .agg(bench_points_left=("bench_points_left", "sum"), optimality_pct=("optimality_pct", "mean"))
        .reset_index()
    )
    out = txn.merge(opt, on=["season", "team_id"], how="left")
    out["bench_points_left"] = out["bench_points_left"].fillna(0).round(1)
    out["optimality_pct"] = out["optimality_pct"].fillna(1).round(3)
    return out.sort_values(["season", "manager_name"])


def all_time_records(matchups: pd.DataFrame, teams: pd.DataFrame) -> dict[str, str]:
    if matchups.empty:
        return {}
    games = _with_team(matchups, teams)
    high = games.loc[games["points_for"].idxmax()]
    losses = games[games["win"] == 0].copy()
    worst = losses.sort_values("points_for").iloc[0] if not losses.empty else None
    losses["margin"] = (losses["points_against"] - losses["points_for"]).abs()
    close = losses.sort_values("margin").iloc[0] if not losses.empty else None

    longest = ("n/a", 0)
    for (_, _team_id), group in games.sort_values(["season", "week"]).groupby(["season", "team_id"]):
        streak = 0
        best = 0
        manager = group["manager_name"].iloc[0]
        season = group["season"].iloc[0]
        for win in group["win"]:
            streak = streak + 1 if win else 0
            best = max(best, streak)
        if best > longest[1]:
            longest = (f"{manager}, {season}: {best}", best)

    worst_loss = (
        f"{worst['manager_name']} ({int(worst['season'])} W{int(worst['week'])}): {worst['points_for']:.1f}"
        if worst is not None
        else "n/a"
    )
    closest_loss = (
        f"{close['manager_name']} ({int(close['season'])} W{int(close['week'])}): "
        f"{close['points_for']:.1f}-{close['points_against']:.1f}"
        if close is not None
        else "n/a"
    )
    return {
        "highest_score": f"{high['manager_name']} ({int(high['season'])} W{int(high['week'])}): {high['points_for']:.1f}",
        "worst_loss": worst_loss,
        "closest_loss": closest_loss,
        "longest_win_streak": longest[0],
    }


def _games_with_opponents(matchups: pd.DataFrame, teams: pd.DataFrame) -> pd.DataFrame:
    if matchups.empty or teams.empty:
        return pd.DataFrame()
    team_lookup = teams[["season", "team_id", "manager_name", "team_name"]].drop_duplicates()
    games = matchups.merge(team_lookup, on=["season", "team_id"], how="inner")
    opponent_lookup = team_lookup.rename(
        columns={
            "team_id": "opponent_id",
            "manager_name": "opponent_manager_name",
            "team_name": "opponent_team_name",
        }
    )
    return games.merge(opponent_lookup, on=["season", "opponent_id"], how="inner")


def all_time_leaderboards(matchups: pd.DataFrame, teams: pd.DataFrame, top_n: int = 5) -> dict[str, pd.DataFrame]:
    """Top-N leaderboards for the Finale tab - richer than the single-record summary in all_time_records."""
    games = _games_with_opponents(matchups, teams)
    if games.empty:
        return {}
    games["margin"] = (games["points_for"] - games["points_against"]).abs()

    game_cols = ["season", "week", "manager_name", "team_name", "points_for", "opponent_manager_name", "points_against"]
    highest_scores = games.sort_values("points_for", ascending=False).head(top_n)[game_cols].reset_index(drop=True)

    losses = games[games["win"] == 0]
    worst_losses = losses.sort_values("points_for", ascending=True).head(top_n)[game_cols].reset_index(drop=True)

    # One row per unique game (not per team-perspective) for margin-based leaderboards,
    # since every game otherwise appears twice with an identical (symmetric) margin.
    unique_games = games.drop_duplicates(subset=["season", "week", "matchup_id"], keep="first")
    margin_cols = ["season", "week", "manager_name", "points_for", "opponent_manager_name", "points_against", "margin"]
    closest_games = unique_games.sort_values("margin", ascending=True).head(top_n)[margin_cols].reset_index(drop=True)
    biggest_blowouts = unique_games.sort_values("margin", ascending=False).head(top_n)[margin_cols].reset_index(drop=True)

    streak_rows = []
    for (season, _team_id), group in games.sort_values(["season", "week"]).groupby(["season", "team_id"]):
        manager = group["manager_name"].iloc[0]
        best_win_streak = best_loss_streak = win_streak = loss_streak = 0
        for win in group["win"]:
            if win:
                win_streak += 1
                loss_streak = 0
            else:
                loss_streak += 1
                win_streak = 0
            best_win_streak = max(best_win_streak, win_streak)
            best_loss_streak = max(best_loss_streak, loss_streak)
        streak_rows.append(
            {
                "season": season,
                "manager_name": manager,
                "win_streak": best_win_streak,
                "loss_streak": best_loss_streak,
            }
        )
    streaks = pd.DataFrame(streak_rows)
    longest_win_streaks = (
        streaks[streaks["win_streak"] > 0]
        .sort_values("win_streak", ascending=False)
        .head(top_n)[["manager_name", "season", "win_streak"]]
        .reset_index(drop=True)
        if not streaks.empty
        else pd.DataFrame()
    )
    longest_loss_streaks = (
        streaks[streaks["loss_streak"] > 0]
        .sort_values("loss_streak", ascending=False)
        .head(top_n)[["manager_name", "season", "loss_streak"]]
        .reset_index(drop=True)
        if not streaks.empty
        else pd.DataFrame()
    )

    best_season_totals = (
        games.groupby(["season", "manager_name", "team_name"], as_index=False)["points_for"]
        .sum()
        .sort_values("points_for", ascending=False)
        .head(top_n)
        .round({"points_for": 1})
        .reset_index(drop=True)
    )

    return {
        "highest_scores": highest_scores,
        "worst_losses": worst_losses,
        "closest_games": closest_games,
        "biggest_blowouts": biggest_blowouts,
        "longest_win_streaks": longest_win_streaks,
        "longest_loss_streaks": longest_loss_streaks,
        "best_season_totals": best_season_totals,
    }
