from __future__ import annotations

from league_history.analytics import slot_role
from league_history.espn_client import EspnClient, normalize_season


def _entry(player_id: int, name: str, position_id: int, lineup_slot: int, points: float = 10.0) -> dict:
    return {
        "playerPoolEntry": {
            "appliedStatTotal": points,
            "player": {
                "id": player_id,
                "fullName": name,
                "defaultPositionId": position_id,
            },
        },
        "lineupSlotId": lineup_slot,
    }


def _game(week: int, home_team: int, away_team: int, entries_home: list[dict], entries_away: list[dict]) -> dict:
    return {
        "id": 1,
        "matchupPeriodId": week,
        "home": {
            "teamId": home_team,
            "totalPoints": 100.0,
            "rosterForMatchupPeriod": {"entries": entries_home},
        },
        "away": {
            "teamId": away_team,
            "totalPoints": 90.0,
            "rosterForMatchupPeriod": {"entries": entries_away},
        },
    }


def _base_payload(schedule: list[dict], transactions: list[dict] | None = None) -> dict:
    return {
        "teams": [{"id": 1, "location": "Home", "nickname": "Team"}, {"id": 2, "location": "Away", "nickname": "Team"}],
        "members": [],
        "schedule": schedule,
        "transactions": transactions or [],
        "draftDetail": {"picks": []},
    }


def test_rb_wr_flex_slot_3_normalizes_to_flex():
    entries = [_entry(101, "Some RB", position_id=2, lineup_slot=3)]
    payload = _base_payload([_game(1, 1, 2, entries, [])])
    frames = normalize_season(2024, payload)
    roster = frames["roster_scores"]
    row = roster[roster["player_id"] == 101].iloc[0]
    assert row["slot"] == "Flex"
    assert row["position"] == "RB"
    assert row["is_starter"] == 1
    assert slot_role(row["position"], row["slot"]) == "Flex-RB"


def test_superflex_op_slot_7_normalizes_to_flex():
    entries = [_entry(202, "Some QB", position_id=1, lineup_slot=7)]
    payload = _base_payload([_game(1, 1, 2, entries, [])])
    frames = normalize_season(2024, payload)
    roster = frames["roster_scores"]
    row = roster[roster["player_id"] == 202].iloc[0]
    assert row["slot"] == "Flex"


def test_transaction_player_name_from_schedule_lookup():
    entries = [_entry(303, "Rostered Player", position_id=2, lineup_slot=2)]
    payload = _base_payload(
        [_game(1, 1, 2, entries, [])],
        transactions=[
            {
                "id": "T-1",
                "teamId": 1,
                "type": "WAIVER",
                "scoringPeriodId": 1,
                "items": [{"playerId": 303}],
            }
        ],
    )
    frames = normalize_season(2024, payload)
    txns = frames["transactions"]
    assert txns.loc[0, "player_name"] == "Rostered Player"


def test_transaction_key_dedupes_by_id():
    txn_a = {"id": "T-1", "teamId": 1, "type": "WAIVER", "scoringPeriodId": 1, "items": [{"playerId": 5}]}
    txn_b = {"id": "T-1", "teamId": 1, "type": "WAIVER", "scoringPeriodId": 1, "items": [{"playerId": 5}]}
    assert EspnClient._transaction_key(txn_a) == EspnClient._transaction_key(txn_b)


def test_transaction_key_falls_back_without_id():
    txn = {"teamId": 1, "type": "TRADE", "scoringPeriodId": 3, "items": [{"playerId": 9}, {"playerId": 8}]}
    key = EspnClient._transaction_key(txn)
    assert key == (1, 3, "TRADE", ("8", "9"))


def test_weeks_from_schedule_ignores_zero_and_dedupes():
    schedule = [{"matchupPeriodId": 1}, {"matchupPeriodId": 1}, {"matchupPeriodId": 2}, {"matchupPeriodId": 0}]
    assert EspnClient._weeks_from_schedule(schedule) == [1, 2]
