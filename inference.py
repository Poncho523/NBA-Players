"""Módulo de inferencia para clasificar a un jugador a partir de estadísticas básicas."""
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

def calculate_advanced_metrics(player_stats, team_stats):
    """
    Calcula las métricas avanzadas requeridas por el modelo a partir de estadísticas básicas.
    Se asume que las estadísticas son PROMEDIOS POR PARTIDO.
    """
    p = player_stats
    t = team_stats
    
    # Evitar división por cero
    fga = max(p.get('fga', 0), 0.1)
    fta = max(p.get('fta', 0), 0.1)
    mp = max(p.get('mp', 0), 0.1)
    
    # 1. True Shooting Percentage (ts_pct)
    ts_pct = p['pts'] / (2 * (fga + 0.44 * fta))
    
    # 2. 3-Point Attempt Rate (fg3a_per_fga_pct)
    fg3a_per_fga_pct = p['fg3a'] / fga
    
    # 3. Free Throw Attempt Rate (fta_per_fga_pct)
    fta_per_fga_pct = p['fta'] / fga
    
    # 4. Rebound Percentages (orb_pct, drb_pct, trb_pct)
    orb_pct = 100 * (p['orb'] / (p['orb'] + t['drb'])) if (p['orb'] + t['drb']) > 0 else 0
    drb_pct = 100 * (p['drb'] / (p['drb'] + t['orb'])) if (p['drb'] + t['orb']) > 0 else 0
    trb_pct = 100 * (p['trb'] / (p['trb'] + t['trb'])) if (p['trb'] + t['trb']) > 0 else 0
    
    # 5. Assist Percentage (ast_pct)
    ast_pct = 100 * (p['ast'] / (((mp / (t['mp'] / 5)) * t['fg']) - p['fg'])) if (((mp / (t['mp'] / 5)) * t['fg']) - p['fg']) > 0 else 0
    
    # 6. Steal y Block Percentage (stl_pct, blk_pct)
    # Aproximación de posesiones del equipo para porcentajes
    team_poss = t['fga'] + 0.44 * t['fta'] - t['orb'] + t['tov']
    stl_pct = 100 * (p['stl'] / (mp / (t['mp'] / 5) * team_poss)) if (mp / (t['mp'] / 5) * team_poss) > 0 else 0
    blk_pct = 100 * (p['blk'] / (mp / (t['mp'] / 5) * (t['fga'] - t['fg3a']))) if (mp / (t['mp'] / 5) * (t['fga'] - t['fg3a'])) > 0 else 0
    
    # 7. Turnover Percentage (tov_pct)
    tov_pct = 100 * (p['tov'] / (fga + 0.44 * fta + p['tov']))
    
    # 8. Usage Percentage (usg_pct)
    usg_pct = 100 * ((fga + 0.44 * fta + p['tov']) * (t['mp'] / 5)) / (mp * (t['fga'] + 0.44 * t['fta'] + t['tov']))
    
    return {
        'ts_pct': round(ts_pct, 3),
        'fg3a_per_fga_pct': round(fg3a_per_fga_pct, 3),
        'fta_per_fga_pct': round(fta_per_fga_pct, 3),
        'orb_pct': round(orb_pct, 1),
        'drb_pct': round(drb_pct, 1),
        'trb_pct': round(trb_pct, 1),
        'ast_pct': round(ast_pct, 1),
        'stl_pct': round(stl_pct, 1),
        'blk_pct': round(blk_pct, 1),
        'tov_pct': round(tov_pct, 1),
        'usg_pct': round(usg_pct, 1)
    }

def predict_player(pos, player_stats):
    # Cargar modelo
    model_path = Path("models/classifier_pipeline.joblib")
    if not model_path.exists():
        raise FileNotFoundError("Primero debes entrenar el modelo.")
    pipeline = joblib.load(model_path)
    
    # Promedios de la NBA por partido (para el equipo) - Valores aproximados 2024
    # Si el profe conoce los del equipo real, se pueden cambiar.
    default_team_stats = {
        'mp': 240, 'fg': 42, 'fga': 88, 'fg3a': 35, 
        'fta': 22, 'orb': 10, 'drb': 33, 'trb': 43, 'tov': 13
    }
    
    # Calcular métricas avanzadas
    advanced_stats = calculate_advanced_metrics(player_stats, default_team_stats)
    
    # Agregar posición y crear DataFrame
    advanced_stats['pos'] = pos
    df_player = pd.DataFrame([advanced_stats])
    
    # Predecir
    prediction = pipeline.predict(df_player)[0]
    probabilidad = pipeline.predict_proba(df_player)[0][1]
    
    return prediction, probabilidad, advanced_stats

if __name__ == "__main__":
    print("=== CLASIFICADOR DE JUGADORES NBA ===")
    print("Ingresa los promedios por partido del jugador.\n")
    
    pos = input("Posición (PG, SG, SF, PF, C): ").strip().upper()
    
    try:
        stats = {
            'mp': float(input("Minutos por partido (ej. 30.5): ")),
            'pts': float(input("Puntos por partido (ej. 18.2): ")),
            'fg': float(input("Tiros de campo anotados (ej. 6.5): ")),
            'fga': float(input("Tiros de campo intentados (ej. 14.2): ")),
            'fg3a': float(input("Triples intentados (ej. 4.5): ")),
            'ft': float(input("Tiros libres anotados (ej. 4.2): ")),
            'fta': float(input("Tiros libres intentados (ej. 5.0): ")),
            'orb': float(input("Rebotes ofensivos (ej. 1.2): ")),
            'drb': float(input("Rebotes defensivos (ej. 4.5): ")),
            'trb': float(input("Rebotes totales (ej. 5.7): ")),
            'ast': float(input("Asistencias (ej. 4.1): ")),
            'stl': float(input("Robos (ej. 1.1): ")),
            'blk': float(input("Tapones (ej. 0.3): ")),
            'tov': float(input("Pérdidas (ej. 2.0): "))
        }
    except ValueError:
        print("Error: Por favor ingresa solo números.")
        exit()
        
    pred, prob, adv = predict_player(pos, stats)
    
    print("\n--- MÉTRICAS AVANZADAS CALCULADAS ---")
    for k, v in adv.items():
        if k != 'pos':
            print(f"{k}: {v}")
            
    print("\n--- RESULTADO DEL MODELO ---")
    if pred == 1:
        print(f"✅ CLASIFICACIÓN: BUENO (Impacto Positivo)")
        print(f"Probabilidad de ser jugador de impacto: {prob:.2%}")
    else:
        print(f"❌ CLASIFICACIÓN: MALO (Jugador de rotación/recambio)")
        print(f"Probabilidad de ser jugador de impacto: {prob:.2%}")