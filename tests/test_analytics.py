from __future__ import annotations

import pandas as pd

from league_history.analytics import (
    _score_within_season,
    acquisition_source_league_average,
    all_time_leaderboards,
    all_time_records,
    head_to_head_game_table,
    head_to_head_history,
    manager_profiles,
    positional_performance,
    transaction_scorecard,
)


def test_score_within_season_higher_is_better_true_ranks_max_highest():
    df = pd.DataFrame({"season": [1, 1, 1], "net_transaction_value": [0, 50, 100]})
    scores = _score_within_season(df, "net_transaction_value", higher_is_better=True)
    assert scores.tolist() == [33.0, 67.0, 100.0]


def test_score_within_season_higher_is_better_false_ranks_min_highest():
    df = pd.DataFrame({"season": [1, 1, 1], "injury_value_lost": [0, 50, 100]})
    scores = _score_within_season(df, "injury_value_lost", higher_is_better=False)
    assert scores.tolist() == [100.0, 67.0, 33.0]


def test_all_time_records_returns_na_when_no_losses():
    teams = pd.DataFrame(
        {
            "season": [2024, 2024],
            "team_id": [1, 2],
            "manager_name": ["Alex", "Bailey"],
            "team_name": ["Alex's Team", "Bailey's Team"],
        }
    )
    matchups = pd.DataFrame(
        {
            "season": [2024, 2024],
            "week": [1, 1],
            "team_id": [1, 2],
            "opponent_id": [2, 1],
            "points_for": [110.0, 90.0],
            "points_against": [90.0, 110.0],
            "win": [1, 1],
        }
    )
    records = all_time_records(matchups, teams)
    assert records["worst_loss"] == "n/a"
    assert records["closest_loss"] == "n/a"


def test_transaction_scorecard_excludes_rows_with_missing_week():
    teams = pd.DataFrame(
        {"season": [2024], "team_id": [1], "manager_name": ["Alex"], "team_name": ["Alex's Team"]}
    )
    transactions = pd.DataFrame(
        {
            "season": [2024, 2024],
            "week": [1, None],
            "team_id": [1, 1],
            "transaction_type": ["WAIVER_ADD", "WAIVER_ADD"],
            "player_id": [100, 101],
            "player_name": [None, None],
            "counterparty_team_id": [None, None],
        }
    )
    roster_scores = pd.DataFrame(
        columns=["season", "week", "team_id", "player_id", "points", "is_starter"]
    )
    scores, _details = transaction_scorecard(transactions, roster_scores, teams)
    assert scores.loc[scores["team_id"] == 1, "move_count"].item() == 1


def test_manager_profiles_bench_points_left_measures_actual_optimal_lineup():
    teams = pd.DataFrame(
        {"season": [2024], "team_id": [1], "manager_name": ["Alex"], "team_name": ["Alex's Team"]}
    )
    roster_scores = pd.DataFrame(
        {
            "season": [2024, 2024],
            "team_id": [1, 1],
            "week": [1, 1],
            "player_id": [1, 2],
            "position": ["RB", "RB"],
            "points": [5.0, 10.0],
            "is_starter": [1, 0],
        }
    )
    profiles = manager_profiles(pd.DataFrame(), roster_scores, teams)
    row = profiles.iloc[0]
    assert row["bench_points_left"] == 5.0
    assert row["optimality_pct"] == 0.5


def test_head_to_head_game_table_has_one_row_per_game():
    teams = pd.DataFrame(
        {
            "season": [2024, 2024],
            "team_id": [1, 2],
            "manager_name": ["Alex", "Bailey"],
            "team_name": ["Alex's Team", "Bailey's Team"],
        }
    )
    matchups = pd.DataFrame(
        {
            "season": [2024, 2024],
            "week": [1, 1],
            "team_id": [1, 2],
            "opponent_id": [2, 1],
            "points_for": [120.0, 110.0],
            "points_against": [110.0, 120.0],
            "win": [1, 0],
        }
    )
    games, _summary = head_to_head_history(matchups, teams, "Alex", "Bailey")
    assert len(games) == 2  # both perspectives, needed for the chart/summary

    table = head_to_head_game_table(games, "Alex", "Bailey")
    assert len(table) == 1
    row = table.iloc[0]
    assert row["Alex_points"] == 120.0
    assert row["Bailey_points"] == 110.0
    assert row["winner"] == "Alex"


def test_acquisition_source_league_average_zero_fills_missing_sources():
    teams = pd.DataFrame(
        {
            "season": [2024, 2024],
            "team_id": [1, 2],
            "manager_name": ["Alex", "Bailey"],
            "team_name": ["Alex's Team", "Bailey's Team"],
        }
    )
    roster_scores = pd.DataFrame(
        {
            "season": [2024, 2024],
            "week": [1, 1],
            "team_id": [1, 2],
            "player_id": [10, 20],
            "player_name": ["Drafted Guy", "Waiver Guy"],
            "position": ["QB", "QB"],
            "slot": ["QB", "QB"],
            "is_starter": [1, 1],
            "points": [100.0, 50.0],
        }
    )
    draft_picks = pd.DataFrame({"season": [2024], "team_id": [1], "player_id": [10]})
    transactions = pd.DataFrame(
        {
            "season": [2024],
            "week": [1],
            "team_id": [2],
            "transaction_type": ["WAIVER_ADD"],
            "player_id": [20],
        }
    )
    result = acquisition_source_league_average(roster_scores, teams, draft_picks, transactions)
    shares = dict(zip(result["acquisition_source"], result["point_share"]))
    # Alex: 100% Draft, 0% Pickup. Bailey: 0% Draft, 100% Pickup. Average of each is 0.5,
    # not 1.0 (which a naive groupby-mean would give by skipping each manager's missing source).
    assert shares["Draft"] == 0.5
    assert shares["Pickup"] == 0.5


def test_positional_performance_buckets_flex_separately_from_base_position():
    teams = pd.DataFrame(
        {"season": [2024], "team_id": [1], "manager_name": ["Alex"], "team_name": ["Alex's Team"]}
    )
    roster_scores = pd.DataFrame(
        {
            "season": [2024, 2024],
            "team_id": [1, 1],
            "week": [1, 1],
            "player_id": [1, 2],
            "position": ["RB", "RB"],
            "slot": ["RB", "Flex"],
            "is_starter": [1, 1],
            "points": [10.0, 20.0],
        }
    )
    result = positional_performance(roster_scores, teams)
    positions = set(result["position"])
    assert positions == {"RB", "Flex"}
    flex_row = result[result["position"] == "Flex"].iloc[0]
    assert flex_row["slot_points_per_week"] == 20.0
    rb_row = result[result["position"] == "RB"].iloc[0]
    assert rb_row["slot_points_per_week"] == 10.0


def test_all_time_leaderboards_dedupes_games_for_margin_categories():
    teams = pd.DataFrame(
        {
            "season": [2024, 2024, 2024, 2024],
            "team_id": [1, 2, 1, 2],
            "manager_name": ["Alex", "Bailey", "Alex", "Bailey"],
            "team_name": ["Alex's Team", "Bailey's Team", "Alex's Team", "Bailey's Team"],
        }
    ).drop_duplicates()
    matchups = pd.DataFrame(
        {
            "season": [2024, 2024, 2024, 2024],
            "week": [1, 1, 2, 2],
            "matchup_id": [100, 100, 101, 101],
            "team_id": [1, 2, 1, 2],
            "opponent_id": [2, 1, 2, 1],
            "points_for": [120.0, 119.0, 150.0, 90.0],
            "points_against": [119.0, 120.0, 90.0, 150.0],
            "win": [1, 0, 1, 0],
        }
    )
    boards = all_time_leaderboards(matchups, teams, top_n=5)

    # Week 1 is one game (margin 1), week 2 is one game (margin 60) - each should
    # appear exactly once in the margin-based leaderboards, not twice per perspective.
    assert len(boards["closest_games"]) == 2
    assert len(boards["biggest_blowouts"]) == 2
    assert boards["closest_games"].iloc[0]["margin"] == 1.0
    assert boards["biggest_blowouts"].iloc[0]["margin"] == 60.0

    assert len(boards["highest_scores"]) == 4  # every team-perspective row is its own score
    assert boards["highest_scores"].iloc[0]["points_for"] == 150.0

    assert set(boards["longest_win_streaks"]["manager_name"]) == {"Alex"}
