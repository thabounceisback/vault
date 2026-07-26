from __future__ import annotations

import pandas as pd

from league_history.analytics import (
    _score_within_season,
    all_time_records,
    manager_profiles,
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
