"""Módulo de construcción de pipeline, entrenamiento y evaluación temporal."""
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier

from nba_classifier.data.ingestion import load_seasons
from nba_classifier.preprocessing.cleaning import (
    clean_season_df,
    build_target,
    drop_leaky_columns,
    split_xy
)

logger = logging.getLogger(__name__)


def build_pipeline(model):
    """Pipeline que utiliza únicamente métricas porcentuales/eficiencia y la posición."""
    
    # Únicamente métricas de eficiencia/tasas relativas (sin totales acumulados de volumen)
    numeric_features = [
        #'age', 
        'ts_pct', 'fg3a_per_fga_pct', 'fta_per_fga_pct', 
        'orb_pct', 'drb_pct', 'trb_pct', 'ast_pct', 
        'stl_pct', 'blk_pct', 'tov_pct', 'usg_pct'
    ]
    
    # La posición preprocesada (PG, SG, SF, PF, C)
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
        ]
    )

    return Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])


def train_and_evaluate():
    print("Cargando datos de 2015 a 2026...")
    df_raw = load_seasons(list(range(2015, 2027)))
    
    print("Limpiando datos y construyendo target (BPM >= 1.5)...")
    df = clean_season_df(df_raw, min_games=20)
    df = build_target(df, threshold=1.5)
    df = drop_leaky_columns(df)
    
    X, y = split_xy(df)
    
    # Eliminación explícita de columnas basura residuales si existen en X
    X = X.drop(columns=['kind', 'team_adv'], errors='ignore')
    
    print("Dividiendo datos (Entrenamiento: < 2026, Test: == 2026)...")
    train_mask = df['season'] < 2026
    test_mask = df['season'] == 2026
    
    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    grupos_train = df.loc[train_mask, 'season']

    print(f"Muestras de Entrenamiento: {X_train.shape[0]}")
    print(f"Muestras de Prueba (2026): {X_test.shape[0]}")
    print(f"Distribución del target en Test: {np.bincount(y_test)}")

    modelos = {
        "Regresión Logística": LogisticRegression(max_iter=1000, random_state=42),
        "Árbol de Decisión": DecisionTreeClassifier(max_depth=5, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
        "Support Vector Machine": CalibratedClassifierCV(SVC(kernel='rbf', random_state=42), ensemble=False),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "XGBoost": XGBClassifier(n_estimators=100,learning_rate=0.05, max_depth=4, random_state=42,scale_pos_weight=3,eval_metric='logloss')
    }

    print("\n--- Validación Cruzada Agrupada (GroupKFold por Temporada) ---")
    gkf = GroupKFold(n_splits=5)
    
    resultados = []
    best_model = None
    best_cv_acc = 0
    best_name = ""

    for nombre, modelo in modelos.items():
        pipeline = build_pipeline(modelo)
        
        scores = cross_val_score(
            pipeline, X_train, y_train, cv=gkf, groups=grupos_train, scoring='accuracy'
        )
        media_cv = scores.mean()
        std_cv = scores.std()
        
        pipeline.fit(X_train, y_train)
        y_pred_test = pipeline.predict(X_test)
        acc_test = accuracy_score(y_test, y_pred_test)
        
        resultados.append({
            "Modelo": nombre, 
            "CV Media (Pasado)": media_cv, 
            "CV Desv": std_cv,
            "Test 2026 (Futuro)": acc_test
        })
        
        if media_cv > best_cv_acc:
            best_cv_acc = media_cv
            best_model = pipeline
            best_name = nombre

    df_resultados = pd.DataFrame(resultados).sort_values(by="CV Media (Pasado)", ascending=False)
    print("\n--- TABLA COMPARATIVA FINAL ---")
    print(df_resultados.to_string(index=False))

    print(f"\n¡El mejor modelo fue {best_name} con {best_cv_acc:.4f} en CV!")
    
    y_pred_best = best_model.predict(X_test)
    print("\n--- Matriz de Confusión (Test 2026) ---")
    print(confusion_matrix(y_test, y_pred_best))
    print("\n--- Reporte de Clasificación (Test 2026) ---")
    print(classification_report(y_test, y_pred_best, target_names=["Malo (0)", "Bueno (1)"]))

    Path("models").mkdir(exist_ok=True)
    joblib.dump(best_model, "models/classifier_pipeline.joblib")
    print("Modelo guardado en models/classifier_pipeline.joblib")


if __name__ == "__main__":
    train_and_evaluate()