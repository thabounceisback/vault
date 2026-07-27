from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import pandas as pd
import requests

from .storage import Database


POSITION_MAP = {
    1: "QB",
    2: "RB",
    3: "WR",
    4: "TE",
    5: "K",
    16: "DST",
}

LINEUP_SLOT_MAP = {
    0: "QB",
    1: "QB",     # TQB: a second QB-only slot in 2-QB leagues
    2: "RB",
    3: "Flex",   # RB/WR flex
    4: "WR",
    5: "Flex",   # WR/TE flex
    6: "TE",
    7: "Flex",   # OP/superflex: any offensive position
    16: "DST",
    17: "K",
    20: "Bench",
    21: "IR",
    23: "Flex",  # RB/WR/TE flex - the common "FLEX" slot
    24: "Flex",  # ER: an additional flex-type slot some leagues configure
}


@dataclass(frozen=True)
class EspnConfig:
    league_id: int
    seasons: list[int]
    swid: str | None = None
    espn_s2: str | None = None
    cookie_header: str | None = None


class EspnSyncError(RuntimeError):
    """Raised when ESPN history cannot be fetched with the provided settings."""


class EspnClient:
    def __init__(self, config: EspnConfig) -> None:
        self.config = config
        self.failed_weeks: dict[int, list[int]] = {}
        self.session = requests.Session()
        cookie_header = _clean_cookie_header(config.cookie_header)
        self.session.headers.update(
            {
                "User-Agent": (
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/126.0.0.0 Safari/537.36"
                ),
                "Accept": "application/json,text/plain,*/*",
                "Origin": "https://fantasy.espn.com",
                "Referer": f"https://fantasy.espn.com/football/league?leagueId={config.league_id}",
                "x-fantasy-platform": "kona-PROD-1dc40132e6439917483d19616fe48378bc99c4ed",
                "x-fantasy-source": "kona",
            }
        )
        swid = _clean_cookie_value(config.swid)
        espn_s2 = _clean_cookie_value(config.espn_s2)
        if cookie_header:
            self.session.headers["Cookie"] = _merge_cookie_header(cookie_header, swid, espn_s2)
        else:
            for domain in (None, ".espn.com", "espn.com", "fantasy.espn.com", "www.espn.com"):
                if swid:
                    _set_cookie(self.session, "SWID", swid, domain)
                if espn_s2:
                    _set_cookie(self.session, "espn_s2", espn_s2, domain)

    @staticmethod
    def _has_league_data(response: requests.Response) -> bool:
        if response.status_code >= 400 or response.is_redirect:
            return False
        try:
            data = response.json()
        except ValueError:
            return False
        if isinstance(data, list):
            data = data[0] if data else {}
        return bool(data.get("teams"))

    def fetch_season(self, season: int) -> dict[str, Any]:
        urls = self._modern_urls(season)
        params = self._season_params()
        response = self._get_first_available(urls, params=params)
        if not self._has_league_data(response):
            # Seasons from before ESPN's permanent per-league IDs (roughly pre-2018)
            # only live under the legacy leagueHistory endpoint - and it often answers
            # with a plain 200 and no teams rather than a clean 404, so a hard failure
            # isn't the only trigger for trying it.
            legacy_urls = self._legacy_urls()
            legacy_params = [("seasonId", str(season)), *self._season_params()]
            legacy_response = self._get_first_available(legacy_urls, params=legacy_params)
            if self._has_league_data(legacy_response) or response.status_code == 404 or response.status_code >= 500:
                urls = legacy_urls
                params = legacy_params
                response = legacy_response
        if response.is_redirect:
            location = response.headers.get("location", "unknown location")
            auth_state = self._auth_state()
            raise EspnSyncError(
                f"ESPN redirected season {season} to {location}. "
                "That usually means ESPN did not accept the auth cookies for this API request. "
                f"Auth sent: {auth_state}."
            )
        if response.status_code in {401, 403}:
            cookie_hint = "a full Cookie header or both SWID and espn_s2" if not self._has_auth() else "fresh cookies"
            raise EspnSyncError(
                f"ESPN refused season {season} with HTTP {response.status_code}. "
                f"Use {cookie_hint} cookies from an ESPN account that can open league {self.config.league_id}."
            )
        try:
            response.raise_for_status()
            payload = response.json()
            if isinstance(payload, list):
                payload = payload[0] if payload else {}
        except requests.HTTPError as exc:
            raise EspnSyncError(
                f"ESPN request failed for season {season}: HTTP {response.status_code} at {response.url}."
            ) from exc
        except ValueError as exc:
            raise EspnSyncError(
                f"ESPN returned a non-JSON response for season {season}. Check the league ID, season, and cookies."
            ) from exc

        if not payload.get("teams"):
            raise EspnSyncError(
                f"ESPN has no league data for season {season} under league {self.config.league_id} on either the "
                "current or legacy history API. Seasons from before ESPN introduced permanent league IDs "
                "(~2018) are sometimes only reachable under that season's own league ID from that year, "
                "not the current one."
            )

        weeks = self._weeks_from_schedule(payload.get("schedule", []) or [])
        weekly_schedule = self._fetch_weekly_boxscores(season, urls, weeks, params)
        if weekly_schedule:
            payload["schedule"] = weekly_schedule

        payload["transactions"] = self._fetch_weekly_transactions(
            season, urls, weeks, payload.get("transactions", []) or [], params
        )
        return payload

    def _modern_urls(self, season: int) -> list[str]:
        return [
            f"https://lm-api-reads.fantasy.espn.com/apis/v3/games/ffl/seasons/{season}/segments/0/leagues/{self.config.league_id}",
            f"https://fantasy.espn.com/apis/v3/games/ffl/seasons/{season}/segments/0/leagues/{self.config.league_id}",
        ]

    def _legacy_urls(self) -> list[str]:
        return [
            f"https://lm-api-reads.fantasy.espn.com/apis/v3/games/ffl/leagueHistory/{self.config.league_id}",
            f"https://fantasy.espn.com/apis/v3/games/ffl/leagueHistory/{self.config.league_id}",
        ]

    @staticmethod
    def _season_params() -> list[tuple[str, str]]:
        return [
            ("view", "mSettings"),
            ("view", "mTeam"),
            ("view", "mDraftDetail"),
            ("view", "mMatchupScore"),
            ("view", "mRoster"),
            ("view", "mBoxscore"),
            ("view", "mTransactions2"),
        ]

    def _get_first_available(self, urls: list[str], params: list[tuple[str, str]]) -> requests.Response:
        last_response: requests.Response | None = None
        for url in urls:
            response = self.session.get(url, params=params, timeout=30, allow_redirects=False)
            last_response = response
            if not response.is_redirect:
                return response
        if last_response is None:
            raise EspnSyncError("No ESPN API hosts were configured.")
        return last_response

    @staticmethod
    def _weeks_from_schedule(schedule: list[dict[str, Any]]) -> list[int]:
        return sorted(
            {
                int(game.get("matchupPeriodId") or game.get("scoringPeriodId") or 0)
                for game in schedule
                if int(game.get("matchupPeriodId") or game.get("scoringPeriodId") or 0) > 0
            }
        )

    def _fetch_weekly_boxscores(
        self,
        season: int,
        urls: list[str],
        weeks: list[int],
        base_params: list[tuple[str, str]] | None = None,
    ) -> list[dict[str, Any]]:
        if not weeks:
            return []

        weekly_games: list[dict[str, Any]] = []
        for week in weeks:
            # All three views are required. mRoster alone still drops per-player metadata
            # like injuryStatus from this week's snapshot. More importantly, mMatchupScore
            # is what actually scopes rosterForMatchupPeriod to *this* scoring period -
            # without it, ESPN tends to leave that field empty for past weeks, and
            # _side_entries() silently falls back to rosterForCurrentScoringPeriod: today's
            # roster and lineup-slot assignment, retroactively applied to every historical
            # week. That single gap explains both under-reported injuries (only whoever is
            # hurt *right now* ever shows up) and wrong/missing roster slots (a slot that's
            # empty today, like Flex in the off-season, never appears in history at all).
            params = [
                ("view", "mMatchupScore"),
                ("view", "mBoxscore"),
                ("view", "mRoster"),
                ("scoringPeriodId", str(week)),
            ]
            if base_params and any(name == "seasonId" for name, _ in base_params):
                params = [("seasonId", str(season)), *params]
            response = self._get_first_available(
                urls,
                params=params,
            )
            if response.status_code >= 400 or response.is_redirect:
                self.failed_weeks.setdefault(season, []).append(week)
                continue
            try:
                week_payload = response.json()
                if isinstance(week_payload, list):
                    week_payload = week_payload[0] if week_payload else {}
            except ValueError:
                self.failed_weeks.setdefault(season, []).append(week)
                continue
            for game in week_payload.get("schedule", []) or []:
                game_week = int(game.get("matchupPeriodId") or game.get("scoringPeriodId") or 0)
                if game_week == week and _game_has_roster_entries(game):
                    weekly_games.append(game)
        return weekly_games

    @staticmethod
    def _transaction_key(txn: dict[str, Any]) -> Any:
        txn_id = txn.get("id")
        if txn_id is not None:
            return txn_id
        player_ids = tuple(sorted(str(item.get("playerId")) for item in txn.get("items", []) or []))
        return (txn.get("teamId"), txn.get("scoringPeriodId"), txn.get("type"), player_ids)

    def _fetch_weekly_transactions(
        self,
        season: int,
        urls: list[str],
        weeks: list[int],
        initial_transactions: list[dict[str, Any]],
        base_params: list[tuple[str, str]] | None = None,
    ) -> list[dict[str, Any]]:
        # A single league-level fetch with mTransactions2 only returns ESPN's small
        # "recent activity" window, not the full season - so pull week by week like
        # boxscores, and de-duplicate since a transaction can surface in more than
        # one week's window.
        all_transactions: list[dict[str, Any]] = []
        seen: set[Any] = set()
        for txn in initial_transactions:
            key = self._transaction_key(txn)
            if key not in seen:
                seen.add(key)
                all_transactions.append(txn)

        for week in weeks:
            params = [("view", "mTransactions2"), ("scoringPeriodId", str(week))]
            if base_params and any(name == "seasonId" for name, _ in base_params):
                params = [("seasonId", str(season)), *params]
            response = self._get_first_available(urls, params=params)
            if response.status_code >= 400 or response.is_redirect:
                continue
            try:
                week_payload = response.json()
                if isinstance(week_payload, list):
                    week_payload = week_payload[0] if week_payload else {}
            except ValueError:
                continue
            for txn in week_payload.get("transactions", []) or []:
                key = self._transaction_key(txn)
                if key not in seen:
                    seen.add(key)
                    all_transactions.append(txn)
        return all_transactions

    def _has_auth(self) -> bool:
        names = _cookie_names(_clean_cookie_header(self.config.cookie_header))
        if {"swid", "espn_s2"}.issubset(names):
            return True
        return bool(_clean_cookie_value(self.config.swid) and _clean_cookie_value(self.config.espn_s2))

    def _auth_state(self) -> str:
        if _clean_cookie_header(self.config.cookie_header):
            names = _cookie_names(_clean_cookie_header(self.config.cookie_header))
            return f"Cookie header with SWID={_yes_no('swid' in names)}, espn_s2={_yes_no('espn_s2' in names)}"
        return f"SWID={_yes_no(bool(_clean_cookie_value(self.config.swid)))}, espn_s2={_yes_no(bool(_clean_cookie_value(self.config.espn_s2)))}"


def _yes_no(value: bool) -> str:
    return "yes" if value else "no"


def _clean_cookie_value(value: str | None) -> str | None:
    if not value:
        return None
    cleaned = value.strip().strip('"').strip("'")
    if "=" in cleaned and ";" not in cleaned:
        cleaned = cleaned.split("=", 1)[1].strip()
    return cleaned or None


def _clean_cookie_header(value: str | None) -> str | None:
    if not value:
        return None
    cleaned = value.strip().strip('"').strip("'")
    json_cookie = _cookie_header_from_json(cleaned)
    if json_cookie:
        return json_cookie
    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    for line in lines:
        lowered = line.lower()
        if lowered.startswith("cookie:"):
            cookie_line = line.split(":", 1)[1].strip()
            return _cookie_header_from_json(cookie_line) or cookie_line or None
        if lowered == "cookie":
            continue
    if len(lines) >= 2:
        for index, line in enumerate(lines[:-1]):
            if line.lower() == "cookie":
                cookie_line = lines[index + 1].strip()
                return _cookie_header_from_json(cookie_line) or cookie_line or None
    if cleaned.lower().startswith("cookie:"):
        cleaned = cleaned.split(":", 1)[1].strip()
    if not _cookie_names(cleaned):
        return None
    return cleaned or None


def _cookie_names(cookie_header: str | None) -> set[str]:
    if not cookie_header:
        return set()
    return {
        part.split("=", 1)[0].strip().lower()
        for part in cookie_header.split(";")
        if "=" in part
    }


def _cookie_header_from_json(value: str) -> str | None:
    if not value.strip().startswith("{"):
        return None
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None

    cookie_values = {str(key).lower(): str(cookie_value) for key, cookie_value in parsed.items() if cookie_value}
    parts = []
    if "swid" in cookie_values:
        parts.append(f"SWID={cookie_values['swid']}")
    if "espn_s2" in cookie_values:
        parts.append(f"espn_s2={cookie_values['espn_s2']}")
    return "; ".join(parts) if parts else None


def cookie_auth_summary(cookie_header: str | None, swid: str | None, espn_s2: str | None) -> str:
    names = _cookie_names(_clean_cookie_header(cookie_header))
    has_swid = "swid" in names or bool(_clean_cookie_value(swid))
    has_s2 = "espn_s2" in names or bool(_clean_cookie_value(espn_s2))
    source = "header" if names else "fields"
    return f"Auth detected from {source}: SWID={_yes_no(has_swid)}, espn_s2={_yes_no(has_s2)}"


def _merge_cookie_header(cookie_header: str, swid: str | None, espn_s2: str | None) -> str:
    parts = [part.strip() for part in cookie_header.split(";") if part.strip()]
    names = {part.split("=", 1)[0].strip().lower() for part in parts if "=" in part}
    if swid and "swid" not in names:
        parts.append(f"SWID={swid}")
    if espn_s2 and "espn_s2" not in names:
        parts.append(f"espn_s2={espn_s2}")
    return "; ".join(parts)


def _set_cookie(session: requests.Session, name: str, value: str, domain: str | None) -> None:
    if domain:
        session.cookies.set(name, value, domain=domain)
    else:
        session.cookies.set(name, value)


def _member_name(member: dict[str, Any]) -> str:
    first = member.get("firstName") or ""
    last = member.get("lastName") or ""
    display = member.get("displayName") or f"{first} {last}".strip()
    return display or str(member.get("id") or "Unknown Manager")


def _manager_for_team(team: dict[str, Any], members_by_id: dict[str, dict[str, Any]]) -> tuple[str, str]:
    primary_owner = team.get("primaryOwner")
    if primary_owner:
        member = members_by_id.get(str(primary_owner))
        if member:
            return str(member.get("id") or primary_owner), _member_name(member)
        return str(primary_owner), team.get("name") or f"Team {team.get('id')}"

    owners = team.get("owners") or []
    if owners:
        owner = owners[0]
        if isinstance(owner, dict):
            return str(owner.get("id") or team.get("id")), _member_name(owner)
        member = members_by_id.get(str(owner))
        if member:
            return str(member.get("id") or owner), _member_name(member)
        return str(owner), team.get("name") or f"Team {team.get('id')}"
    return str(team.get("id")), team.get("name") or f"Team {team.get('id')}"


def _game_has_roster_entries(game: dict[str, Any]) -> bool:
    for side_name in ("home", "away"):
        side = game.get(side_name) or {}
        if _side_entries(side):
            return True
    return False


def _side_entries(side: dict[str, Any]) -> list[dict[str, Any]]:
    # rosterForMatchupPeriod reflects what was actually rostered/started during that specific
    # scoring period; rosterForCurrentScoringPeriod reflects the team's roster as of the API call
    # and would misattribute today's roster/injury status to past weeks if preferred.
    matchup = side.get("rosterForMatchupPeriod", {}).get("entries", []) or []
    current = side.get("rosterForCurrentScoringPeriod", {}).get("entries", []) or []
    return matchup or current


def _lineup_slot_id(entry: dict[str, Any], pool_entry: dict[str, Any]) -> int | None:
    for container in (entry, pool_entry):
        for key in ("lineupSlotId", "lineup_slot_id", "lineupSlot"):
            value = container.get(key)
            if value is not None:
                try:
                    return int(value)
                except (TypeError, ValueError):
                    return None
    return None


def _projected_points(player: dict[str, Any], week: int) -> float | None:
    for stat in player.get("stats", []) or []:
        if (
            stat.get("statSourceId") == 1
            and stat.get("statSplitTypeId") == 1
            and int(stat.get("scoringPeriodId") or week) == week
            and stat.get("appliedTotal") is not None
        ):
            return float(stat["appliedTotal"])
    return None


def _actual_points(player: dict[str, Any], week: int) -> float | None:
    for stat in player.get("stats", []) or []:
        if (
            stat.get("statSourceId") == 0
            and stat.get("statSplitTypeId") == 1
            and int(stat.get("scoringPeriodId") or week) == week
            and stat.get("appliedTotal") is not None
        ):
            return float(stat["appliedTotal"])
    return None


def _player_lookup_from_schedule(schedule: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    players: dict[int, dict[str, Any]] = {}
    for game in schedule:
        for side_name in ("home", "away"):
            for entry in _side_entries(game.get(side_name) or {}):
                player = entry.get("playerPoolEntry", {}).get("player", {}) or {}
                player_id = player.get("id") or entry.get("playerId")
                if player_id is not None and player_id not in players:
                    players[int(player_id)] = player
    return players


def normalize_season(season: int, payload: dict[str, Any]) -> dict[str, pd.DataFrame]:
    teams_rows = []
    managers_rows = []
    members_by_id = {str(member.get("id")): member for member in payload.get("members", []) if member.get("id")}
    for team in payload.get("teams", []):
        manager_id, manager_name = _manager_for_team(team, members_by_id)
        managers_rows.append({"manager_id": manager_id, "manager_name": manager_name})
        location = team.get("location") or ""
        nickname = team.get("nickname") or team.get("name") or f"Team {team.get('id')}"
        team_name = f"{location} {nickname}".strip()
        teams_rows.append(
            {
                "season": season,
                "team_id": int(team.get("id")),
                "manager_id": manager_id,
                "manager_name": manager_name,
                "team_name": team_name,
            }
        )

    schedule = payload.get("schedule", []) or []
    player_lookup = _player_lookup_from_schedule(schedule)
    draft_rows = []
    auction_rows = []
    for pick in payload.get("draftDetail", {}).get("picks", []) or []:
        player_id = pick.get("playerId")
        player = pick.get("playerPoolEntry", {}).get("player", {}) or player_lookup.get(int(player_id), {}) if player_id else {}
        position_id = player.get("defaultPositionId")
        auction_value = pick.get("bidAmount")
        draft_rows.append(
            {
                "season": season,
                "round": pick.get("roundId"),
                "pick": pick.get("overallPickNumber"),
                "team_id": pick.get("teamId"),
                "player_id": player.get("id") or player_id,
                "player_name": player.get("fullName"),
                "position": POSITION_MAP.get(position_id, str(position_id or "UNK")),
                "keeper": int(bool(pick.get("keeper"))),
                "auction_value": auction_value,
            }
        )
        if auction_value is not None:
            auction_rows.append(
                {
                    "season": season,
                    "player_id": player.get("id") or player_id,
                    "player_name": player.get("fullName"),
                    "team_id": pick.get("teamId"),
                    "manager_name": None,
                    "auction_value": auction_value,
                    "source": "espn_draft",
                }
            )

    matchup_rows = []
    roster_rows = []
    injury_rows = []
    for game in schedule:
        week = int(game.get("matchupPeriodId") or game.get("scoringPeriodId") or 0)
        matchup_id = int(game.get("id") or 0)
        sides = []
        for side_name in ("home", "away"):
            side = game.get(side_name)
            if side and side.get("teamId"):
                sides.append((side_name, side))
        for side_name, side in sides:
            opponent = next((other for other_name, other in sides if other_name != side_name), None)
            points_for = float(side.get("totalPoints") or side.get("totalPointsLive") or 0)
            points_against = float(opponent.get("totalPoints") or opponent.get("totalPointsLive") or 0) if opponent else 0
            matchup_rows.append(
                {
                    "season": season,
                    "week": week,
                    "matchup_id": matchup_id,
                    "team_id": int(side.get("teamId")),
                    "opponent_id": int(opponent.get("teamId")) if opponent else None,
                    "points_for": points_for,
                    "points_against": points_against,
                    "win": int(points_for > points_against),
                }
            )
            for entry in _side_entries(side):
                player = entry.get("playerPoolEntry", {}).get("player", {})
                pool_entry = entry.get("playerPoolEntry", {})
                lineup_slot = _lineup_slot_id(entry, pool_entry)
                injury_status = (
                    player.get("injuryStatus")
                    or player.get("injury_status")
                    or pool_entry.get("injuryStatus")
                )
                # The IR roster slot is itself a reliable injury signal - the slot
                # assignment is accurate for that historical week even in seasons/weeks
                # where ESPN's own injuryStatus field on the player wasn't populated.
                if not injury_status and lineup_slot == 21:
                    injury_status = "INJURY_RESERVE"
                applied_total = pool_entry.get("appliedStatTotal")
                if applied_total is None:
                    applied_total = _actual_points(player, week)
                projected_total = pool_entry.get("appliedProjectedStatTotal")
                if projected_total is None:
                    projected_total = _projected_points(player, week)
                roster_rows.append(
                    {
                        "season": season,
                        "week": week,
                        "team_id": int(side.get("teamId")),
                        "player_id": player.get("id"),
                        "player_name": player.get("fullName"),
                        "position": POSITION_MAP.get(player.get("defaultPositionId"), "UNK"),
                        "slot": LINEUP_SLOT_MAP.get(lineup_slot, str(lineup_slot)),
                        "is_starter": int(lineup_slot not in (20, 21)),
                        "points": float(applied_total or 0),
                        "projected_points": float(projected_total) if projected_total is not None else None,
                        "injury_status": injury_status,
                    }
                )
                if injury_status:
                    injury_rows.append(
                        {
                            "season": season,
                            "week": week,
                            "player_id": player.get("id"),
                            "player_name": player.get("fullName"),
                            "injury_status": injury_status,
                            "source": "espn_roster",
                        }
                    )

    transaction_rows = []
    for item in payload.get("transactions", []) or []:
        default_team_id = item.get("teamId")
        txn_type = item.get("type")
        for txn_item in item.get("items", []) or []:
            player_id = txn_item.get("playerId")
            looked_up = player_lookup.get(int(player_id), {}) if player_id is not None else {}
            # A trade's own top-level teamId only reflects whoever proposed it, which
            # would attribute every player in the trade (both sides given away and
            # received) to that one team. Each item carries its own to/from team for
            # exactly this reason - prefer whichever team actually ended up with the
            # player, falling back to the transaction-level teamId for plain
            # adds/drops that don't set per-item team fields.
            to_team = txn_item.get("toTeamId")
            from_team = txn_item.get("fromTeamId")
            if to_team not in (None, -1):
                item_team_id = to_team
            elif from_team not in (None, -1):
                item_team_id = from_team
            else:
                item_team_id = default_team_id
            counterparty = item.get("proposedTo")
            if to_team not in (None, -1) and from_team not in (None, -1):
                counterparty = from_team if item_team_id == to_team else to_team
            transaction_rows.append(
                {
                    "season": season,
                    "week": item.get("scoringPeriodId"),
                    "team_id": item_team_id,
                    "transaction_type": txn_item.get("type") or txn_type,
                    "player_id": player_id,
                    "player_name": looked_up.get("fullName"),
                    "counterparty_team_id": counterparty,
                }
            )

    return {
        "managers": pd.DataFrame(managers_rows).drop_duplicates("manager_id") if managers_rows else pd.DataFrame(),
        "teams": pd.DataFrame(teams_rows),
        "draft_picks": pd.DataFrame(draft_rows),
        "auction_values": pd.DataFrame(auction_rows),
        "matchups": pd.DataFrame(matchup_rows),
        "roster_scores": pd.DataFrame(roster_rows),
        "injuries": pd.DataFrame(injury_rows),
        "transactions": pd.DataFrame(transaction_rows),
    }


def sync_espn_history(config: EspnConfig, db: Database) -> dict[str, Any]:
    client = EspnClient(config)
    merged: dict[str, list[pd.DataFrame]] = {
        "managers": [],
        "teams": [],
        "draft_picks": [],
        "auction_values": [],
        "matchups": [],
        "roster_scores": [],
        "injuries": [],
        "transactions": [],
    }
    failed_seasons: dict[int, str] = {}
    synced_seasons: list[int] = []
    for season in config.seasons:
        # One old/unreachable season (pre-permanent-league-ID history, a since-deleted
        # league, etc.) shouldn't sink an otherwise-good multi-season sync - skip it
        # and keep going, same spirit as the per-week failure handling below.
        try:
            normalized = normalize_season(season, client.fetch_season(season))
        except EspnSyncError as exc:
            failed_seasons[season] = str(exc)
            continue
        synced_seasons.append(season)
        for name, frame in normalized.items():
            if not frame.empty:
                merged[name].append(frame)

    if not synced_seasons:
        raise EspnSyncError(
            "Could not sync any of the requested seasons. "
            + "; ".join(f"{season}: {reason}" for season, reason in failed_seasons.items())
        )

    frames = {
        name: pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        for name, parts in merged.items()
    }
    db.append_tables(frames, synced_seasons)
    return {
        "seasons_synced": len(synced_seasons),
        "incomplete_weeks": client.failed_weeks,
        "failed_seasons": failed_seasons,
    }
