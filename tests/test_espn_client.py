from __future__ import annotations

from league_history.analytics import slot_role
from league_history.espn_client import EspnClient, EspnConfig, normalize_season


class _FakeResponse:
    def __init__(self, payload: dict, status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code
        self.is_redirect = False
        self.headers: dict = {}
        self.url = "https://example.invalid"

    def json(self) -> dict:
        return self._payload

    def raise_for_status(self) -> None:
        pass


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
    assert slot_role(row["position"], row["slot"]) == "Flex"


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


def test_ir_slot_counts_as_injured_even_without_injury_status():
    entries = [_entry(404, "Hurt Guy", position_id=2, lineup_slot=21)]
    payload = _base_payload([_game(1, 1, 2, entries, [])])
    frames = normalize_season(2024, payload)
    roster = frames["roster_scores"]
    row = roster[roster["player_id"] == 404].iloc[0]
    assert row["injury_status"] == "INJURY_RESERVE"
    assert row["is_starter"] == 0


def test_trade_attributes_each_player_to_the_team_that_received_them():
    payload = _base_payload(
        [_game(1, 1, 2, [], [])],
        transactions=[
            {
                "id": "TR-1",
                "type": "TRADE",
                "teamId": 1,
                "scoringPeriodId": 3,
                "items": [
                    {"playerId": 111, "toTeamId": 2, "fromTeamId": 1},
                    {"playerId": 222, "toTeamId": 1, "fromTeamId": 2},
                ],
            }
        ],
    )
    frames = normalize_season(2024, payload)
    txns = frames["transactions"].set_index("player_id")
    assert txns.loc[111, "team_id"] == 2
    assert txns.loc[111, "counterparty_team_id"] == 1
    assert txns.loc[222, "team_id"] == 1
    assert txns.loc[222, "counterparty_team_id"] == 2


def test_plain_add_still_uses_top_level_team_id_without_to_from_fields():
    payload = _base_payload(
        [_game(1, 1, 2, [], [])],
        transactions=[
            {
                "id": "W-1",
                "type": "WAIVER",
                "teamId": 1,
                "scoringPeriodId": 2,
                "items": [{"playerId": 999}],
            }
        ],
    )
    frames = normalize_season(2024, payload)
    txns = frames["transactions"]
    assert txns.loc[0, "team_id"] == 1


def test_weekly_boxscore_fetch_requests_matchup_score_view():
    # Without mMatchupScore, ESPN tends to leave rosterForMatchupPeriod empty for past
    # weeks, which sends _side_entries() down the rosterForCurrentScoringPeriod fallback -
    # today's roster/lineup slots applied to every historical week. Lock in that the
    # per-week fetch always asks for it.
    client = EspnClient(EspnConfig(league_id=1, seasons=[2024]))
    captured_params: list[list[tuple[str, str]]] = []

    def fake_get(url, params=None, timeout=None, allow_redirects=None):
        captured_params.append(list(params))
        return _FakeResponse({"schedule": []})

    client.session.get = fake_get
    client._fetch_weekly_boxscores(2024, ["https://example.invalid"], [1])

    assert len(captured_params) == 1
    views = [value for name, value in captured_params[0] if name == "view"]
    assert "mMatchupScore" in views
    assert "mBoxscore" in views
    assert "mRoster" in views
