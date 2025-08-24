import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline

print("Iniciando generación del modelo")

if not os.path.exists('models'):
    os.makedirs('models')
    print("Directorio 'models' creado")

print("Cargando datos...")
url_csv_corazon = "https://docs.google.com/spreadsheets/d/1fIGT8WBvWGVO80Eczjib9tcfMaAa58SN7GNDFJfcKr8/export?format=csv&gid=998197115"

try:
    df = pd.read_csv(url_csv_corazon)
    print(f"Datos cargados: {len(df)} filas, {len(df.columns)} columnas")
    print("Columnas originales:", df.columns.tolist())
except Exception as e:
    print(f"Error al cargar datos: {e}")
    exit(1)

df.columns = ['Age', 'Gender', 'Heart_rate', 'Systolic_blood_pressure',
              'Diastolic_blood_pressure', 'Blood_sugar', 'CK_MB', 'Troponin', 'Result']

print("Columnas renombradas")

columnas_a_convertir = ['Blood_sugar', 'CK_MB', 'Troponin', 'Heart_rate',
                        'Systolic_blood_pressure', 'Diastolic_blood_pressure', 'Age']

for col in columnas_a_convertir:
    if df[col].dtype == 'object':
        df[col] = df[col].str.replace('.', '', regex=False).str.replace(',', '.', regex=False)
    df[col] = pd.to_numeric(df[col], errors='coerce')

print("Columnas convertidas a numérico")

df['Result_num'] = df['Result'].map({'negative': 0, 'positive': 1})

df.dropna(inplace=True)
print(f"Datos limpiados: {len(df)} filas restantes")

features = ['Age', 'Gender', 'Heart_rate', 'Systolic_blood_pressure',
            'Diastolic_blood_pressure', 'Blood_sugar', 'CK_MB', 'Troponin']
target = 'Result_num'

X = df[features]
y = df[target]

print(f"Datos preparados: {X.shape[0]} muestras, {X.shape[1]} características")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

pipeline = make_pipeline(
    StandardScaler(),
    RandomForestClassifier(random_state=42, n_estimators=100)
)

print("Entrenando modelo")
pipeline.fit(X_train, y_train)

score = pipeline.score(X_test, y_test)
print(f"Modelo entrenado con precisión: {score * 100:.2f}%")

ruta_modelo = 'models/pipeline_riesgo_cardiaco.joblib'
joblib.dump(pipeline, ruta_modelo)

print(f"Modelo guardado en: {ruta_modelo}")
print("Proceso completado exitosamente!")
