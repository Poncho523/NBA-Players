"""Módulo de ingestión y fusión de datos históricos de la NBA."""
from __future__ import annotations

import logging
import pandas as pd

from .scraper import KINDS, RAW_DIR, scrape_season, _session, fetch_season_html, parse_table

logger = logging.getLogger(__name__)

# Columnas duplicadas en Advanced que omitimos para evitar sufijos (_x, _y)
OVERLAP = ["player_name", "age", "pos", "games", "games_started", "mp", "awards_raw"]

# Claves de unión
KEY = ["player_id", "season"]


def merge_season(totals: pd.DataFrame, advanced: pd.DataFrame) -> pd.DataFrame:
    """Une Totals y Advanced en una fila única por jugador y temporada."""
    # 1. Eliminar columnas duplicadas de advanced
    adv = advanced.drop(columns=[c for c in OVERLAP if c in advanced.columns], errors="ignore")
    
    # 2. Merge de ambas tablas
    merged = totals.merge(
        adv, 
        on=KEY, 
        how="outer", 
        suffixes=("", "_adv"), 
        indicator=True
    )
    
    # 3. Validar registros no apareados
    bad = merged["_merge"].value_counts().to_dict()
    if bad.get("left_only", 0) > 0 or bad.get("right_only", 0) > 0:
        logger.warning("Filas no apareadas totals<->advanced: %s", bad)
    
    # 4. Conservar coincidencias completas
    merged = merged[merged["_merge"] == "both"].drop(columns="_merge")
    
    # 5. Priorizar fila agregada (2TM/3TM/TOT) para jugadores traspasados en la misma temporada
    merged["is_aggregated"] = merged["team"].isin(["2TM", "3TM", "TOT"]).astype(int)
    merged = merged.sort_values(
        by=["player_id", "season", "is_aggregated"], 
        ascending=[True, True, False]
    )
    merged = merged.drop_duplicates(subset=KEY, keep="first").drop(columns=["is_aggregated"])
    
    return merged


def load_season_data(season: int, *, force: bool = False) -> pd.DataFrame:
    """Descarga, parsea y fusiona Totals + Advanced de una temporada."""
    dfs = scrape_season(season, force=force)
    return merge_season(dfs["totals"], dfs["advanced"])


def load_seasons(seasons: list[int], *, force: bool = False) -> pd.DataFrame:
    """Concatena múltiples temporadas en un único DataFrame."""
    frames = [load_season_data(s, force=force) for s in seasons]
    return pd.concat(frames, ignore_index=True, sort=False)