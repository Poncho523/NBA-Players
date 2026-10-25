"""Módulo de limpieza, construcción de target y filtrado de leakage."""
import logging
import pandas as pd

logger = logging.getLogger(__name__)


def clean_season_df(df: pd.DataFrame, min_games: int = 20) -> pd.DataFrame:
    """Limpia los datos agrupando por temporada y maneja valores nulos en porcentajes de tiro."""
    df = df.copy()
    
    # 1. Manejo de equipos (2TM, 3TM, TOT)
    df['is_aggregated'] = df['team'].isin(['2TM', '3TM', 'TOT'])
    counts = df.groupby(['player_id', 'season'])['team'].transform('count')
    
    df = df[(counts == 1) | df['is_aggregated']]
    df = df.drop_duplicates(subset=['player_id', 'season'], keep='first')
    
    # 2. Limpieza de Posición (PG, SG, SF, PF, C)
    df['pos'] = df['pos'].fillna('Unknown').astype(str).str.split('-').str[0]
    
    # 3. Filtrar por mínimo de partidos jugados
    df = df[df['games'] >= min_games].copy()
    
    # 4. Manejo de Nulos en Porcentajes de Tiro (Jugadores con 0 intentos)
    pct_cols = ['fg_pct', 'fg3_pct', 'fg2_pct', 'ft_pct', 'ts_pct', 'eFG_pct']
    for col in pct_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0.0)
            
    return df


def build_target(df: pd.DataFrame, threshold: float = 1.5) -> pd.DataFrame:
    """Crea la variable objetivo con el umbral estricto de impacto (BPM >= 1.5)."""
    df = df.copy()
    df["bpm"] = pd.to_numeric(df["bpm"], errors="coerce")

    faltantes = df["bpm"].isna().sum()
    if faltantes > 0:
        logger.warning("Jugadores sin BPM: %s. Omisión en etiquetado.", faltantes)

    df = df.dropna(subset=["bpm"]).copy()
    df["y"] = (df["bpm"] >= threshold).astype(int)

    return df


def drop_leaky_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Elimina métricas compuestas holísticas y columnas de identificación."""
    leaky = [
        'bpm', 'obpm', 'dbpm', 'vorp', 'per', 'ows', 'dws', 
        'ws', 'ws_per_48', 'awards_raw', 'player_name'
    ]
    cols_to_drop = [c for c in leaky if c in df.columns]
    return df.drop(columns=cols_to_drop)


def split_xy(df: pd.DataFrame):
    """Separa las características (X) del target (y)."""
    y = df['y']
    ignore_cols = ['player_id', 'season', 'team', 'y', 'kind', 'team_adv']
    X = df.drop(columns=[c for c in ignore_cols if c in df.columns], errors='ignore')
    return X, y