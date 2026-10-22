# NBA Player Classifier

Este proyecto tiene como objetivo construir un clasificador de Machine Learning que prediga si un jugador de la NBA tendrá un rendimiento "bueno" o "malo" basándose en sus estadísticas de temporada.

La definición operativa del target es: **Un jugador es "bueno" si su BPM (Box Plus/Minus) es mayor a 0**, y "malo" si es menor o igual a 0. El BPM es una métrica avanzada que mide la contribución del jugador respecto al promedio de la liga.

---

## 🏗️ Arquitectura y Fisiología del Proyecto

El proyecto sigue una **arquitectura modular orientada al flujo de datos**, evitando el código espagueti y facilitando que varios integrantes trabajen en paralelo.

El flujo de ejecución (la "fisiología") funciona así:

1. **Ingesta (`data/ingestion.py` + `data/scraper.py`)**: 
   Extrae las estadísticas (Totals y Advanced) de Basketball-Reference. Guarda una caché en HTML (`data/raw/`) para no saturar la página y evita bloqueos.
2. **Validación (`data/validation.py`)**:
   Revisa que los datos crudos tengan las columnas necesarias y no tengan valores imposibles (ej. porcentajes mayores a 1 o jugadores con más de 82 partidos).
3. **Limpieza (`preprocessing/cleaning.py`)**:
   Se encarga del control de calidad. Elimina las filas duplicadas de jugadores que cambiaron de equipo a mitad de temporada (dejando solo su fila agregada `2TM`/`3TM`), arregla las posiciones múltiples (`PG-SG` -> `PG`) y filtra a los jugadores con pocos minutos.
4. **Feature Engineering (`preprocessing/features.py`)**:
   Crea el target (`y = BPM > 0`) y elimina las columnas con *Data Leakage* (BPM, OBPM, DBPM y VORP), ya que son las variables que queremos predecir.
5. **Entrenamiento (`models/train.py`)**:
   Es el **director de orquesta**. Llama a los módulos anteriores, arma un `Pipeline` de Scikit-Learn (que imputa nulos, escala datos y entrena) y compara 6 clasificadores distintos (Regresión Logística, Random Forest, SVM, etc.).
6. **Predicción (`models/predict.py`)**:
   Carga el mejor modelo guardado (`models/classifier_pipeline.joblib`) y recibe las estadísticas de un nuevo jugador para predecir si es bueno o malo.

**Diagrama de Flujo:**
```text
[data/raw/*.html]
       |
       v
  (ingestion.py) ---> [DataFrame Sucio]
       |
       v
  (cleaning.py)  ---> [DataFrame Limpio, 1 fila por jugador/temporada]
       |
       v
  (features.py)  ---> [X e y, sin variables con leakage]
       |
       v
  (train.py)     ---> [Pipeline Entrenado] ---> [models/classifier_pipeline.joblib]
                                                       |
                                                       v
                                                 (predict.py)

📂 Estructura de Directorios

nba_players/
├── configs/                  # Parámetros globales (YAML)
├── data/
│   ├── raw/                  # Archivos HTML descargados de B-Ref (NO se suben a Git)
│   └── processed/            # Datos listos para el modelo (opcional)
├── models/                   # Archivo .joblib con el pipeline entrenado
├── notebooks/                # Jupyter Notebooks para exploración y pruebas
├── src/
│   └── nba_classifier/       # Código fuente principal
│       ├── data/             # Scraper, ingesta y validación
│       ├── preprocessing/    # Limpieza y creación de variables
│       ├── models/           # Entrenamiento y predicción
│       ├── evaluation/       # Métricas y gráficas
│       └── main.py           # Script de consola (Train / Predict)
├── tests/                    # Pruebas unitarias
├── requirements.txt          # Dependencias
└── README.md


Como  

🚀 Cómo empezar (Setup)

    Clona el repositorio y entra a la carpeta:
    bash

    git clone <url-del-repo>
    cd nba_players

    Crea y activa un entorno virtual:
    bash

    python3 -m venv nba_players_entorno
    source nba_players_entorno/bin/activate  # En Windows: nba_players_entorno\Scripts\activate

    Instala las dependencias:
    bash

    pip install -r requirements.txt

    (Importante) Para que Python reconozca el código fuente dentro de src/, debes correr los comandos usando PYTHONPATH=src.
    Alternativa recomendada: crear un archivo pyproject.toml y correr pip install -e .

🛠️ Guía para el Equipo: ¿Dónde modificar qué?

Si necesitan hacer cambios o experimentar, no toquen todo el código. Vayan directo al archivo que corresponde:

    "Quiero probar un modelo nuevo (ej. XGBoost)":
    Ve a src/nba_classifier/models/train.py, importa el modelo y agrégalo al diccionario modelos. El Pipeline se encargará del resto.

    "Quiero cambiar el umbral de lo que consideramos un 'buen' jugador":
    Ve a src/nba_classifier/preprocessing/cleaning.py y cambia el threshold en la función build_target (ej. threshold=2.0).

    "Quiero quitar o agregar variables predictoras":
    Ve a src/nba_classifier/models/train.py y modifica las listas numeric_features y categorical_features.

    "Quiero filtrar a los jugadores con menos partidos":
    Ve a src/nba_classifier/preprocessing/cleaning.py y cambia el min_games=20 por el número que deseen.

    "Quiero cambiar cómo se descargan los datos":
    Ve a src/nba_classifier/data/scraper.py o ingestion.py.

⚠️ Decisiones Metodológicas Clave (No modificar a la ligera)

    Cero Data Leakage: Las variables bpm, obpm, dbpm y vorp se eliminan explícitamente antes de entrenar. El modelo debe predecir usando el resto de estadísticas, no usando la respuesta.

    Validación Temporal: El modelo siempre se entrena con temporadas pasadas (ej. 2015-2025) y se prueba con la temporada más reciente (2026). Esto simula un escenario del mundo real y evita que el modelo "haga trampa" memorizando a los jugadores.

    Un solo registro por jugador/temporada: Si un jugador es traspasado (ej. aparece como 2TM, LAC y CLE), la limpieza se queda únicamente con la fila de totales (2TM) para no duplicar su impacto en el entrenamiento.

    Pipeline Integrado: La limpieza final (imputación de nulos, escalado y One-Hot Encoding) viaja dentro del archivo .joblib. El script de predicción no requiere hacer limpieza manual, el Pipeline lo hace solo.

💻 Comandos Útiles

Para descargar datos (Scraper):
(Asegúrate de tener los HTML en data/raw/ para evitar bloqueos de Cloudflare)
bash

PYTHONPATH=src python -m nba_classifier.data.scraper --seasons 2026

Para entrenar el modelo y ver la tabla comparativa:
bash

PYTHONPATH=src python -m nba_classifier.models.train

Para hacer una predicción de prueba (lo que hará el profesor):
bash

PYTHONPATH=src python -m nba_classifier.models.predict

Proyecto desarrollado para la asignatura de AyVD.