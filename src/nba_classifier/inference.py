"""Módulo de inferencia NBA por consola con manejo robusto de datos faltantes."""
import joblib
import pandas as pd
from pathlib import Path

# Promedios NBA aproximados por jugador (2023-24). Se usan si el usuario presiona Enter.
DEFAULT_PLAYER_STATS = {
    'mp': 25.0, 'pts': 10.0, 'fg': 3.7, 'fga': 8.0, 'fg3a': 2.5,
    'ft': 1.5, 'fta': 2.0, 'orb': 0.8, 'drb': 3.0, 'trb': 3.8,
    'ast': 2.0, 'stl': 0.6, 'blk': 0.3, 'tov': 1.2,
}

DEFAULT_TEAM_STATS = {
    'mp': 240, 'fg': 42, 'fga': 88, 'fg3a': 35,
    'fta': 22, 'orb': 10, 'drb': 33, 'trb': 43, 'tov': 13,
}


def ask_float(label, key):
    """Pide un número. Enter en blanco usa el promedio NBA."""
    default = DEFAULT_PLAYER_STATS[key]
    while True:
        raw = input(f"  {label} [{default}]: ").strip()
        if raw == "":
            return default, True
        try:
            return float(raw), False
        except ValueError:
            print("    ⚠️ Ingresa un número o presiona Enter para usar el promedio NBA.")


def fix_consistency(stats):
    """Corrige incoherencias típicas en las estadísticas ingresadas."""
    fixes = []
    if stats['fg'] > stats['fga']:
        stats['fga'] = stats['fg']
        fixes.append("FGA ajustado a FGM (no puede anotar más de lo que tiró).")
    if stats['ft'] > stats['fta']:
        stats['fta'] = stats['ft']
        fixes.append("FTA ajustado a FTM (idem).")
    if abs((stats['orb'] + stats['drb']) - stats['trb']) > 1.0:
        stats['trb'] = stats['orb'] + stats['drb']
        fixes.append("TRB recalculado como ORB + DRB.")
    return stats, fixes


def calculate_advanced_metrics(p, t):
    fga = max(p['fga'], 0.1)
    fta = max(p['fta'], 0.1)
    mp  = max(p['mp'], 0.1)

    ts_pct = p['pts'] / (2 * (fga + 0.44 * fta))
    fg3a_per_fga_pct = p['fg3a'] / fga
    fta_per_fga_pct = p['fta'] / fga

    orb_pct = 100 * (p['orb'] / (p['orb'] + t['drb'])) if (p['orb'] + t['drb']) > 0 else 0
    drb_pct = 100 * (p['drb'] / (p['drb'] + t['orb'])) if (p['drb'] + t['orb']) > 0 else 0
    trb_pct = 100 * (p['trb'] / (p['trb'] + t['trb'])) if (p['trb'] + t['trb']) > 0 else 0

    denom_ast = ((mp / (t['mp'] / 5)) * t['fg']) - p['fg']
    ast_pct = 100 * (p['ast'] / denom_ast) if denom_ast > 0 else 0

    team_poss = t['fga'] + 0.44 * t['fta'] - t['orb'] + t['tov']
    stl_denom = (mp / (t['mp'] / 5)) * team_poss
    stl_pct = 100 * (p['stl'] / stl_denom) if stl_denom > 0 else 0

    blk_denom = (mp / (t['mp'] / 5)) * (t['fga'] - t['fg3a'])
    blk_pct = 100 * (p['blk'] / blk_denom) if blk_denom > 0 else 0

    tov_pct = 100 * (p['tov'] / (fga + 0.44 * fta + p['tov']))
    usg_pct = 100 * ((fga + 0.44 * fta + p['tov']) * (t['mp'] / 5)) / (mp * (t['fga'] + 0.44 * t['fta'] + t['tov']))

    return {
        'ts_pct': round(ts_pct, 3), 'fg3a_per_fga_pct': round(fg3a_per_fga_pct, 3),
        'fta_per_fga_pct': round(fta_per_fga_pct, 3), 'orb_pct': round(orb_pct, 1),
        'drb_pct': round(drb_pct, 1), 'trb_pct': round(trb_pct, 1),
        'ast_pct': round(ast_pct, 1), 'stl_pct': round(stl_pct, 1),
        'blk_pct': round(blk_pct, 1), 'tov_pct': round(tov_pct, 1),
        'usg_pct': round(usg_pct, 1),
    }


def main():
    print("=" * 60)
    print("🏀 CLASIFICADOR NBA — modo flexible (Enter = promedio NBA)")
    print("=" * 60)
    print("Presiona Enter en cualquier campo para usar el promedio de la liga.\n")

    pos = input("Posición (PG, SG, SF, PF, C) o Enter para 'Unknown': ").strip().upper()
    if pos not in ['PG', 'SG', 'SF', 'PF', 'C']:
        pos = 'Unknown'
        print("  → Se usará 'Unknown'.\n")

    print("📋 Promedios por partido del jugador:")
    stats, defaults_used = {}, []
    fields = [
        ('mp', 'Minutos'), ('pts', 'Puntos'),
        ('fg', 'Tiros anotados'), ('fga', 'Tiros intentados'),
        ('fg3a', 'Triples intentados'),
        ('ft', 'Libres anotados'), ('fta', 'Libres intentados'),
        ('orb', 'Rebotes ofensivos'), ('drb', 'Rebotes defensivos'), ('trb', 'Rebotes totales'),
        ('ast', 'Asistencias'), ('stl', 'Robos'),
        ('blk', 'Tapones'), ('tov', 'Pérdidas'),
    ]
    for key, label in fields:
        val, was_default = ask_float(label, key)
        stats[key] = val
        if was_default:
            defaults_used.append(label)

    stats, fixes = fix_consistency(stats)
    if fixes:
        print("\n🔧 Correcciones automáticas aplicadas:")
        for f in fixes:
            print(f"  • {f}")

    if defaults_used:
        print(f"\n⚠️ Se usó el promedio NBA en: {', '.join(defaults_used)}.")
        print("   La predicción es válida, pero su precisión depende de estos datos.")

    model_path = Path("models/classifier_pipeline.joblib")
    if not model_path.exists():
        print(f"\n❌ No se encontró el modelo en {model_path}.")
        return
    pipeline = joblib.load(model_path)

    adv = calculate_advanced_metrics(stats, DEFAULT_TEAM_STATS)
    adv['pos'] = pos
    df_player = pd.DataFrame([adv])

    pred = pipeline.predict(df_player)[0]
    prob = pipeline.predict_proba(df_player)[0][1]

    print("\n" + "=" * 60)
    print("📊 MÉTRICAS AVANZADAS CALCULADAS")
    print("=" * 60)
    for k, v in adv.items():
        if k != 'pos':
            print(f"  {k}: {v}")

    print("\n" + "=" * 60)
    print("🎯 RESULTADO DEL MODELO")
    print("=" * 60)
    if pred == 1:
        print(f"✅ JUGADOR BUENO (probabilidad: {prob:.2%})")
    else:
        print(f"❌ JUGADOR MALO (probabilidad de ser bueno: {prob:.2%})")
    print("=" * 60)


if __name__ == "__main__":
    main()