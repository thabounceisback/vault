from __future__ import annotations

import random

import pandas as pd

from .storage import Database


MANAGERS = [
    ("m1", "Alex"),
    ("m2", "Bailey"),
    ("m3", "Casey"),
    ("m4", "Devon"),
    ("m5", "Elliot"),
    ("m6", "Finley"),
    ("m7", "Gray"),
    ("m8", "Harper"),
]

POSITIONS = ["QB", "RB", "WR", "TE", "RB", "WR", "RB", "WR", "TE", "QB"]


def seed_sample_database(db: Database, replace: bool = True) -> None:
    rng = random.Random(42)
    seasons = list(range(2021, 2026))
    managers = pd.DataFrame(MANAGERS, columns=["manager_id", "manager_name"])
    teams = []
    draft_picks = []
    matchups = []
    roster_scores = []
    transactions = []

    for season in seasons:
        drafted_by_team: dict[int, list[dict[str, object]]] = {team_id: [] for team_id in range(1, len(MANAGERS) + 1)}
        for team_id, (manager_id, manager_name) in enumerate(MANAGERS, start=1):
            teams.append(
                {
                    "season": season,
                    "team_id": team_id,
                    "manager_id": manager_id,
                    "manager_name": manager_name,
                    "team_name": f"{manager_name}'s Regrets",
                }
            )

        pick_no = 1
        for round_no in range(1, 11):
            order = list(range(1, len(MANAGERS) + 1))
            if round_no % 2 == 0:
                order.reverse()
            for team_id in order:
                bias = {1: "RB", 2: "WR", 3: "QB", 4: "TE"}.get(team_id)
                position = bias if round_no <= 3 and rng.random() < 0.5 else rng.choice(POSITIONS)
                player_id = season * 100000 + pick_no
                pick = {
                    "season": season,
                    "round": round_no,
                    "pick": pick_no,
                    "team_id": team_id,
                    "player_id": player_id,
                    "player_name": f"{position} Player {pick_no}",
                    "position": position,
                    "keeper": int(round_no <= 2 and rng.random() < 0.08),
                    "auction_value": None,
                }
                draft_picks.append(pick)
                drafted_by_team[team_id].append(pick)
                pick_no += 1

        for week in range(1, 15):
            team_ids = list(range(1, len(MANAGERS) + 1))
            rng.shuffle(team_ids)
            for matchup_id, (home, away) in enumerate(zip(team_ids[::2], team_ids[1::2]), start=1):
                home_points = round(rng.gauss(121 + home * 1.8, 19), 2)
                away_points = round(rng.gauss(121 + away * 1.8, 19), 2)
                matchups.extend(
                    [
                        {
                            "season": season,
                            "week": week,
                            "matchup_id": matchup_id,
                            "team_id": home,
                            "opponent_id": away,
                            "points_for": home_points,
                            "points_against": away_points,
                            "win": int(home_points > away_points),
                        },
                        {
                            "season": season,
                            "week": week,
                            "matchup_id": matchup_id,
                            "team_id": away,
                            "opponent_id": home,
                            "points_for": away_points,
                            "points_against": home_points,
                            "win": int(away_points > home_points),
                        },
                    ]
                )

            for team_id in range(1, len(MANAGERS) + 1):
                starters = [("QB", 1), ("RB", 2), ("WR", 2), ("TE", 1), ("Flex", 1), ("DST", 1), ("K", 1)]
                bench = [("RB", 2), ("WR", 2), ("QB", 1), ("TE", 1)]
                for slot, count in starters:
                    for idx in range(count):
                        position = "RB" if slot == "Flex" else slot
                        player = _choose_player(drafted_by_team[team_id], position, rng)
                        baseline = {"QB": 18, "RB": 13, "WR": 12, "TE": 8, "DST": 7, "K": 8}.get(position, 10)
                        projected = max(0, rng.gauss(baseline, 3))
                        points = max(0, rng.gauss(projected, 6))
                        roster_scores.append(_score_row(season, week, team_id, position, slot, 1, points, projected, idx, player))
                for slot, count in bench:
                    for idx in range(count):
                        player = _choose_player(drafted_by_team[team_id], slot, rng)
                        baseline = {"QB": 15, "RB": 9, "WR": 9, "TE": 6}.get(slot, 8)
                        projected = max(0, rng.gauss(baseline, 3))
                        points = max(0, rng.gauss(projected, 6))
                        roster_scores.append(_score_row(season, week, team_id, slot, "Bench", 0, points, projected, idx + 20, player))

        for team_id in range(1, len(MANAGERS) + 1):
            for move_no in range(rng.randint(8, 34)):
                transactions.append(
                    {
                        "season": season,
                        "week": rng.randint(1, 14),
                        "team_id": team_id,
                        "transaction_type": rng.choice(["WAIVER_ADD", "FREEAGENT_ADD", "DROP", "TRADE"]),
                        "player_id": season * 10000 + team_id * 100 + move_no,
                        "player_name": None,
                        "counterparty_team_id": rng.choice([None, *range(1, len(MANAGERS) + 1)]),
                    }
                )

    frames = {
        "managers": managers,
        "teams": pd.DataFrame(teams),
        "draft_picks": pd.DataFrame(draft_picks),
        "matchups": pd.DataFrame(matchups),
        "roster_scores": pd.DataFrame(roster_scores),
        "transactions": pd.DataFrame(transactions),
    }
    if replace:
        db.replace_tables(frames)
    else:
        db.append_tables(frames, seasons)


def _choose_player(picks: list[dict[str, object]], position: str, rng: random.Random) -> dict[str, object] | None:
    candidates = [pick for pick in picks if pick["position"] == position]
    if not candidates and position in {"DST", "K"}:
        return None
    if not candidates:
        candidates = picks
    return rng.choice(candidates) if candidates else None


def _score_row(
    season: int,
    week: int,
    team_id: int,
    position: str,
    slot: str,
    is_starter: int,
    points: float,
    projected: float,
    idx: int,
    player: dict[str, object] | None,
) -> dict[str, object]:
    fallback_id = season * 1000000 + week * 10000 + team_id * 100 + idx
    player_id = player["player_id"] if player else fallback_id
    player_name = player["player_name"] if player else f"{position} Option {team_id}-{week}-{idx}"
    return {
        "season": season,
        "week": week,
        "team_id": team_id,
        "player_id": player_id,
        "player_name": player_name,
        "position": position,
        "slot": slot,
        "is_starter": is_starter,
        "points": round(points, 2),
        "projected_points": round(projected, 2),
    }
