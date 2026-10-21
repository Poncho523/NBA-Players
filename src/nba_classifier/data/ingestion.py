"""Fachada pública para obtener datos. El resto del proyecto importa de aquí."""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from .scraper import KINDS, RAW_DIR, scrape_season, _session, fetch_season_html, parse_table

logger = logging.getLogger(__name__)

# Columnas de Advanced que también están en Totals (las dropeamos de advanced
# antes de fusionar para no acabar con _x/_y). El resto son métricas
# exclusivas de Advanced.
OVERLAP = ["player_name", "age", "pos", "games", "games_started", "mp", "awards_raw"]

KEY = ["player_id", "team", "season"]


def merge_season(totals: pd.DataFrame, advanced: pd.DataFrame) -> pd.DataFrame:
    """Une Totals y Advanced en una fila por (player_id, team, season)."""
    adv = advanced.drop(columns=[c for c in OVERLAP if c in advanced.columns],
                        errors="ignore")
    merged = totals.merge(
        adv, on=KEY, how="outer", validate="one_to_one",
        suffixes=("", "_adv"), indicator=True,
    )
    bad = merged["_merge"].value_counts().to_dict()
    if set(bad) - {"both"}:
        logger.warning("Filas no apareadas totals↔advanced: %s", bad)
    return merged.drop(columns="_merge")


def load_season_data(season: int, *, force: bool = False) -> pd.DataFrame:
    """Descarga, parsea y fusiona Totals+Advanced de una temporada."""
    dfs = scrape_season(season, force=force)
    return merge_season(dfs["totals"], dfs["advanced"])


def load_seasons(seasons: list[int], *, force: bool = False) -> pd.DataFrame:
    """Concatena varias temporadas en un único DataFrame."""
    frames = [load_season_data(s, force=force) for s in seasons]
    return pd.concat(frames, ignore_index=True, sort=False)