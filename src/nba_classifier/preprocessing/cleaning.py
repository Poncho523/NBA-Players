import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def clean_season_df(df: pd.DataFrame, min_games: int = 20) -> pd.DataFrame:
    """Limpia los datos agrupando por temporada para no borrar años anteriores."""
    df = df.copy()
    
    # 1. Manejo de equipos (2TM, 3TM)
    df['is_aggregated'] = df['team'].isin(['2TM', '3TM', 'TOT'])
    
    # Contamos cuántas veces aparece el jugador EN LA MISMA TEMPORADA
    counts = df.groupby(['player_id', 'season'])['team'].transform('count')
    
    # Nos quedamos con la fila agregada o con la única fila de esa temporada
    df = df[(counts == 1) | df['is_aggregated']]
    
    # Eliminamos duplicados estrictamente por jugador y temporada
    df = df.drop_duplicates(subset=['player_id', 'season'], keep='first')
    
    # 2. Limpiar posición (a veces viene como "SF-PF" o "PG-SG")
    df['pos'] = df['pos'].fillna('Unknown').astype(str).str.split('-').str[0]
    
    # 3. Filtrar por mínimo de partidos jugados
    df = df[df['games'] >= min_games].copy()
    
    return df

def build_target(df: pd.DataFrame, threshold: float = 0.0) -> pd.DataFrame:
    """Crea la columna objetivo 'y' basada en el BPM."""
    df = df.copy()
    # y = 1 si BPM > 0, sino 0
    df['y'] = (df['bpm'] > threshold).astype(int)
    return df


def drop_leaky_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Elimina columnas que causan fuga de información."""
    leaky = ['bpm', 'obpm', 'dbpm', 'vorp', 'awards_raw', 'player_name']
    cols_to_drop = [c for c in leaky if c in df.columns]
    return df.drop(columns=cols_to_drop)


def split_xy(df: pd.DataFrame):
    """Separa las características (X) del target (y)."""
    y = df['y']
    # Dejamos fuera columnas que no son predictoras (IDs, temporada, target)
    ignore_cols = ['player_id', 'season', 'team', 'y']
    X = df.drop(columns=[c for c in ignore_cols if c in df.columns])
    return X, y