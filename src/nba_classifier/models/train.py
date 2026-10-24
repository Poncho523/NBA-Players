import pandas as pd
import numpy as np
import joblib
import logging
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, accuracy_score

# Modelos
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

logger = logging.getLogger(__name__)

def build_pipeline(model):
    numeric_features = ['age', 'games', 'games_started', 'mp', 'fg', 'fga', 'fg3', 'fg3a',
                        'fg2', 'fg2a', 'ft', 'fta', 'orb', 'drb', 'trb', 'ast', 'stl', 
                        'blk', 'tov', 'pf', 'pts', 'per', 'ts_pct', 'fg3a_per_fga_pct', 
                        'fta_per_fga_pct', 'orb_pct', 'drb_pct', 'trb_pct', 'ast_pct', 
                        'stl_pct', 'blk_pct', 'tov_pct', 'usg_pct', 'ows', 'dws', 'ws', 
                        'ws_per_48']
    categorical_features = ['pos']

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    return Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])

def train_and_evaluate():
    from nba_classifier.data.ingestion import load_seasons
    from nba_classifier.preprocessing.cleaning import clean_season_df, build_target, drop_leaky_columns, split_xy

    print("Cargando datos de 2015 a 2026...")
    # AQUÍ ESTÁ EL CAMBIO: Le pedimos todas las temporadas
    df = load_seasons(list(range(2015, 2027)))
    
    print("Limpiando...")
    df = clean_season_df(df, min_games=20)
    df = build_target(df)
    df = drop_leaky_columns(df)
    
    X, y = split_xy(df)
    X = X.drop(columns=['kind'], errors='ignore') # Quitamos la columna 'kind' para que no entre al modelo
    
    print("Dividiendo datos (Temporal: Entrenar hasta 2025, probar en 2026)...")
    # AQUÍ ESTÁ EL OTRO CAMBIO: Separación temporal real
    train_mask = df['season'] < 2026
    test_mask = df['season'] == 2026
    
    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    print(f"Datos de entrenamiento: {X_train.shape[0]} filas")
    print(f"Datos de prueba (2026): {X_test.shape[0]} filas")

    modelos = {
        "Regresión Logística": LogisticRegression(max_iter=1000, random_state=42),
        "Árbol de Decisión": DecisionTreeClassifier(max_depth=5, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
        "Support Vector Machine": SVC(kernel='rbf', probability=True, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42)
    }

    print("\n--- Entrenando Modelos ---")
    resultados = []
    best_model = None
    best_acc = 0
    best_name = ""

    for nombre, modelo in modelos.items():
        print(f"Entrenando {nombre}...")
        pipeline = build_pipeline(modelo)
        pipeline.fit(X_train, y_train)
        
        y_pred = pipeline.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        
        resultados.append({"Modelo": nombre, "Accuracy": acc})
        
        if acc > best_acc:
            best_acc = acc
            best_model = pipeline
            best_name = nombre

    df_resultados = pd.DataFrame(resultados).sort_values(by="Accuracy", ascending=False)
    print("\n--- TABLA COMPARATIVA FINAL (Entrenado en pasado, probado en 2026) ---")
    print(df_resultados.to_string(index=False))

    print(f"\n¡El mejor modelo fue {best_name} con {best_acc:.4f} de precisión!")
    
    Path("models").mkdir(exist_ok=True)
    joblib.dump(best_model, "models/classifier_pipeline.joblib")
    print("Modelo guardado en models/classifier_pipeline.joblib")

if __name__ == "__main__":
    train_and_evaluate()