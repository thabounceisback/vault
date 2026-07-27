from __future__ import annotations

from io import BytesIO
from typing import Iterable

import pandas as pd
import requests


class InjurySourceError(RuntimeError):
    """Raised when public nflverse injury data cannot be loaded."""


def fetch_nflverse_injuries(seasons: Iterable[int]) -> tuple[pd.DataFrame, str]:
    season_list = sorted({int(season) for season in seasons if int(season) >= 2009})
    if not season_list:
        return pd.DataFrame(columns=["season", "week", "player_id", "player_name", "injury_status", "source"]), "none"

    try:
        import nfl_data_py as nfl  # type: ignore

        raw = nfl.import_injuries(season_list)
        return normalize_nflverse_injuries(raw, "nfl_data_py"), "nfl_data_py"
    except ModuleNotFoundError:
        pass
    except Exception as exc:
        package_error = exc
    else:
        package_error = None

    try:
        frames = []
        for season in season_list:
            frame = _read_injury_csv(season)
            if not frame.empty:
                frames.append(frame)
        raw = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        return normalize_nflverse_injuries(raw, "nflverse_csv"), "nflverse_csv"
    except Exception as exc:
        if "package_error" in locals() and package_error is not None:
            raise InjurySourceError(f"nfl_data_py failed ({package_error}); direct nflverse CSV failed ({exc}).") from exc
        raise InjurySourceError(f"Could not load nflverse injury data: {exc}") from exc


def normalize_nflverse_injuries(frame: pd.DataFrame, source: str) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["season", "week", "player_id", "player_name", "injury_status", "source"])

    cols = {str(col).lower(): col for col in frame.columns}
    season_col = cols.get("season")
    week_col = cols.get("week")
    name_col = cols.get("full_name") or cols.get("player_name") or cols.get("name")
    status_col = cols.get("report_status") or cols.get("practice_status") or cols.get("status")
    injury_col = cols.get("report_primary_injury") or cols.get("practice_primary_injury")
    season_type_col = cols.get("season_type")

    if not season_col or not week_col or not status_col:
        raise InjurySourceError("nflverse injury data did not include season, week, and status columns.")

    data = frame.copy()
    if season_type_col:
        data = data[data[season_type_col].fillna("").astype(str).str.upper().isin(["REG", ""])]

    status = data[status_col].fillna("").astype(str)
    if injury_col:
        injury = data[injury_col].fillna("").astype(str)
        status = status.where(injury.eq(""), status + " - " + injury)

    out = pd.DataFrame(
        {
            "season": pd.to_numeric(data[season_col], errors="coerce").astype("Int64"),
            "week": pd.to_numeric(data[week_col], errors="coerce").astype("Int64"),
            "player_id": None,
            "player_name": data[name_col] if name_col else None,
            "injury_status": status,
            "source": source,
        }
    )
    out = out.dropna(subset=["season", "week", "injury_status"])
    return out[out["injury_status"].fillna("").astype(str).str.strip().ne("")]


def _read_injury_csv(season: int) -> pd.DataFrame:
    url = f"https://github.com/nflverse/nflverse-data/releases/download/injuries/injuries_{season}.csv"
    response = requests.get(url, timeout=30)
    if response.status_code == 404:
        return pd.DataFrame()
    response.raise_for_status()
    return pd.read_csv(BytesIO(response.content))
