"""Chequeos sobre el DataFrame crudo. Fallan pronto si la fuente cambió."""
from __future__ import annotations

import pandas as pd

REQUIRED = [
    "player_id", "player_name", "team", "season",
    "age", "pos", "games", "games_started", "mp",
    "pts", "trb", "ast", "stl", "blk", "tov",
    "bpm", "obpm", "dbpm", "vorp",
]


def validate_season_df(df: pd.DataFrame, season: int) -> None:
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"[{season}] Faltan columnas: {missing}")
    if df["player_id"].isna().any():
        raise ValueError(f"[{season}] player_id con nulos")
    if (df["games"] > 82).any():
        raise ValueError(f"[{season}] games > 82")
    # porcentajes en rango (algunos vienen como ratio 0-1)
    for c in ("fg_pct", "fg3_pct", "ft_pct", "ts_pct"):
        if c in df.columns and ((df[c] < 0) | (df[c] > 1.5)).any():
            raise ValueError(f"[{season}] {c} fuera de rango")