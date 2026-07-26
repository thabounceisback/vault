from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Iterable

import pandas as pd


TABLES: dict[str, str] = {
    "managers": """
        CREATE TABLE IF NOT EXISTS managers (
            manager_id TEXT PRIMARY KEY,
            manager_name TEXT NOT NULL
        )
    """,
    "league_profiles": """
        CREATE TABLE IF NOT EXISTS league_profiles (
            profile_id INTEGER PRIMARY KEY AUTOINCREMENT,
            league_id INTEGER NOT NULL,
            league_name TEXT NOT NULL,
            seasons TEXT NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """,
    "owner_aliases": """
        CREATE TABLE IF NOT EXISTS owner_aliases (
            manager_id TEXT PRIMARY KEY,
            display_name TEXT NOT NULL
        )
    """,
    "teams": """
        CREATE TABLE IF NOT EXISTS teams (
            season INTEGER NOT NULL,
            team_id INTEGER NOT NULL,
            manager_id TEXT NOT NULL,
            manager_name TEXT NOT NULL,
            team_name TEXT NOT NULL,
            PRIMARY KEY (season, team_id)
        )
    """,
    "draft_picks": """
        CREATE TABLE IF NOT EXISTS draft_picks (
            season INTEGER NOT NULL,
            round INTEGER,
            pick INTEGER,
            team_id INTEGER,
            player_id INTEGER,
            player_name TEXT,
            position TEXT,
            keeper INTEGER DEFAULT 0,
            auction_value REAL
        )
    """,
    "auction_values": """
        CREATE TABLE IF NOT EXISTS auction_values (
            season INTEGER NOT NULL,
            player_id INTEGER,
            player_name TEXT,
            team_id INTEGER,
            manager_name TEXT,
            auction_value REAL NOT NULL,
            source TEXT DEFAULT 'upload'
        )
    """,
    "matchups": """
        CREATE TABLE IF NOT EXISTS matchups (
            season INTEGER NOT NULL,
            week INTEGER NOT NULL,
            matchup_id INTEGER NOT NULL,
            team_id INTEGER NOT NULL,
            opponent_id INTEGER,
            points_for REAL NOT NULL,
            points_against REAL,
            win INTEGER NOT NULL
        )
    """,
    "roster_scores": """
        CREATE TABLE IF NOT EXISTS roster_scores (
            season INTEGER NOT NULL,
            week INTEGER NOT NULL,
            team_id INTEGER NOT NULL,
            player_id INTEGER,
            player_name TEXT,
            position TEXT,
            slot TEXT,
            is_starter INTEGER NOT NULL,
            points REAL NOT NULL,
            projected_points REAL,
            injury_status TEXT
        )
    """,
    "injuries": """
        CREATE TABLE IF NOT EXISTS injuries (
            season INTEGER NOT NULL,
            week INTEGER NOT NULL,
            player_id INTEGER,
            player_name TEXT,
            injury_status TEXT,
            source TEXT DEFAULT 'upload'
        )
    """,
    "transactions": """
        CREATE TABLE IF NOT EXISTS transactions (
            season INTEGER NOT NULL,
            week INTEGER,
            team_id INTEGER,
            transaction_type TEXT,
            player_id INTEGER,
            player_name TEXT,
            counterparty_team_id INTEGER
        )
    """,
}


class Database:
    def __init__(self, path: Path) -> None:
        self.path = path

    def connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def initialize(self) -> None:
        with self.connect() as conn:
            for ddl in TABLES.values():
                conn.execute(ddl)
            self._migrate(conn)

    def _migrate(self, conn: sqlite3.Connection) -> None:
        roster_columns = {
            row["name"] for row in conn.execute("PRAGMA table_info(roster_scores)").fetchall()
        }
        if "projected_points" not in roster_columns:
            conn.execute("ALTER TABLE roster_scores ADD COLUMN projected_points REAL")
        if "injury_status" not in roster_columns:
            conn.execute("ALTER TABLE roster_scores ADD COLUMN injury_status TEXT")

    def replace_tables(self, frames: dict[str, pd.DataFrame]) -> None:
        self.initialize()
        with self.connect() as conn:
            for name in TABLES:
                if name in {"league_profiles", "owner_aliases"}:
                    continue
                conn.execute(f"DELETE FROM {name}")
            for name, frame in frames.items():
                if name not in TABLES or frame.empty:
                    continue
                frame.to_sql(name, conn, if_exists="append", index=False)

    def append_tables(self, frames: dict[str, pd.DataFrame], seasons: Iterable[int]) -> None:
        self.initialize()
        season_list = list(seasons)
        with self.connect() as conn:
            for name in TABLES:
                if name in {"managers", "league_profiles", "owner_aliases"}:
                    continue
                for season in season_list:
                    if name == "auction_values":
                        conn.execute("DELETE FROM auction_values WHERE season = ? AND source = 'espn_draft'", (season,))
                    elif name == "injuries":
                        conn.execute("DELETE FROM injuries WHERE season = ? AND source = 'espn_roster'", (season,))
                    else:
                        conn.execute(f"DELETE FROM {name} WHERE season = ?", (season,))
            for name, frame in frames.items():
                if name not in TABLES or frame.empty:
                    continue
                if name == "managers":
                    existing = pd.read_sql_query("SELECT * FROM managers", conn)
                    frame = pd.concat([existing, frame], ignore_index=True).drop_duplicates("manager_id", keep="last")
                    conn.execute("DELETE FROM managers")
                frame.to_sql(name, conn, if_exists="append", index=False)

    def read_all(self) -> dict[str, pd.DataFrame]:
        self.initialize()
        with self.connect() as conn:
            return {name: pd.read_sql_query(f"SELECT * FROM {name}", conn) for name in TABLES}

    def is_empty(self) -> bool:
        self.initialize()
        with self.connect() as conn:
            row = conn.execute("SELECT COUNT(*) AS count FROM teams").fetchone()
        return int(row["count"]) == 0

    def save_league_profile(self, league_id: int, league_name: str, seasons: str) -> None:
        self.initialize()
        with self.connect() as conn:
            existing = conn.execute(
                "SELECT profile_id FROM league_profiles WHERE league_id = ?",
                (league_id,),
            ).fetchone()
            if existing:
                conn.execute(
                    """
                    UPDATE league_profiles
                    SET league_name = ?, seasons = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE profile_id = ?
                    """,
                    (league_name, seasons, existing["profile_id"]),
                )
            else:
                conn.execute(
                    "INSERT INTO league_profiles (league_id, league_name, seasons) VALUES (?, ?, ?)",
                    (league_id, league_name, seasons),
                )

    def save_owner_aliases(self, aliases: dict[str, str]) -> None:
        self.initialize()
        with self.connect() as conn:
            for manager_id, display_name in aliases.items():
                cleaned = display_name.strip()
                if not cleaned:
                    conn.execute("DELETE FROM owner_aliases WHERE manager_id = ?", (manager_id,))
                else:
                    conn.execute(
                        """
                        INSERT INTO owner_aliases (manager_id, display_name)
                        VALUES (?, ?)
                        ON CONFLICT(manager_id) DO UPDATE SET display_name = excluded.display_name
                        """,
                        (manager_id, cleaned),
                    )

    def replace_import_table(self, table_name: str, frame: pd.DataFrame) -> None:
        if table_name not in {"auction_values", "injuries"}:
            raise ValueError(f"Unsupported import table: {table_name}")
        self.initialize()
        with self.connect() as conn:
            conn.execute(f"DELETE FROM {table_name} WHERE source = 'upload'")
            if not frame.empty:
                frame.to_sql(table_name, conn, if_exists="append", index=False)

    def replace_source_rows(self, table_name: str, source: str, frame: pd.DataFrame) -> None:
        if table_name not in {"auction_values", "injuries"}:
            raise ValueError(f"Unsupported source table: {table_name}")
        self.initialize()
        with self.connect() as conn:
            conn.execute(f"DELETE FROM {table_name} WHERE source = ?", (source,))
            if not frame.empty:
                frame.to_sql(table_name, conn, if_exists="append", index=False)
