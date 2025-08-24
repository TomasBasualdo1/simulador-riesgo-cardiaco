import streamlit as st
import pandas as pd
import joblib

# CONFIGURACIÓN DE LA PÁGINA

st.set_page_config(
    page_title="Simulador de Riesgo Cardíaco",
    layout="centered"
)

# CARGAR EL MODELO ENTRENADO

@st.cache_resource
def load_model():
    import os
    model_path = "models/pipeline_riesgo_cardiaco.joblib"
    
    if not os.path.exists(model_path):
        st.info("Generando modelo por primera vez... Esto puede tomar unos segundos.")
        
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.pipeline import make_pipeline
        import os
        
        if not os.path.exists('models'):
            os.makedirs('models')
        
        url_csv_corazon = "https://docs.google.com/spreadsheets/d/1fIGT8WBvWGVO80Eczjib9tcfMaAa58SN7GNDFJfcKr8/export?format=csv&gid=998197115"
        df = pd.read_csv(url_csv_corazon)
        
        df.columns = ['Age', 'Gender', 'Heart_rate', 'Systolic_blood_pressure',
                      'Diastolic_blood_pressure', 'Blood_sugar', 'CK_MB', 'Troponin', 'Result']
        
        columnas_a_convertir = ['Blood_sugar', 'CK_MB', 'Troponin', 'Heart_rate',
                                'Systolic_blood_pressure', 'Diastolic_blood_pressure', 'Age']
        
        for col in columnas_a_convertir:
            if df[col].dtype == 'object':
                df[col] = df[col].str.replace('.', '', regex=False).str.replace(',', '.', regex=False)
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df['Result_num'] = df['Result'].map({'negative': 0, 'positive': 1})
        df.dropna(inplace=True)
        
        features = ['Age', 'Gender', 'Heart_rate', 'Systolic_blood_pressure',
                    'Diastolic_blood_pressure', 'Blood_sugar', 'CK_MB', 'Troponin']
        X = df[features]
        y = df['Result_num']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        pipeline = make_pipeline(
            StandardScaler(),
            RandomForestClassifier(random_state=42, n_estimators=100)
        )
        pipeline.fit(X_train, y_train)
        
        joblib.dump(pipeline, model_path)
        st.success("Modelo generado exitosamente!")
        
        return pipeline
    else:
        return joblib.load(model_path)

pipeline = load_model()


# INTERFAZ DE USUARIO DE LA APLICACIÓN

st.title("🫀 Simulador de Riesgo de Ataque Cardíaco")

st.markdown("""
Esta herramienta utiliza un modelo de Machine Learning (**Random Forest**) para estimar la probabilidad de tener un diagnóstico positivo de ataque cardíaco basado en datos clínicos.
**Importante:** Este es un proyecto educativo y no reemplaza una consulta médica profesional.
""")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Edad", min_value=1, max_value=120, value=50, step=1)
    gender = st.selectbox("Género", options=[0, 1], format_func=lambda x: "Femenino" if x == 0 else "Masculino")
    heart_rate = st.number_input("Frecuencia Cardíaca (latidos/min)", min_value=30, max_value=220, value=75)
    systolic_bp = st.number_input("Presión Sistólica (mmHg)", min_value=70, max_value=250, value=120)

with col2:
    diastolic_bp = st.number_input("Presión Diastólica (mmHg)", min_value=40, max_value=150, value=80)
    blood_sugar = st.number_input("Azúcar en Sangre (mg/dL)", min_value=50.0, max_value=500.0, value=100.0, format="%.1f")
    ck_mb = st.number_input("CK-MB (ng/mL)", min_value=0.0, max_value=50.0, value=3.0, format="%.2f")
    troponin = st.number_input("Troponina (ng/mL)", min_value=0.0, max_value=10.0, value=0.02, format="%.3f")


if st.button("Calcular Riesgo", type="primary"):

    input_data = pd.DataFrame({
        'Age': [age],
        'Gender': [gender],
        'Heart_rate': [heart_rate],
        'Systolic_blood_pressure': [systolic_bp],
        'Diastolic_blood_pressure': [diastolic_bp],
        'Blood_sugar': [blood_sugar],
        'CK_MB': [ck_mb],
        'Troponin': [troponin]
    })

    probabilidad = pipeline.predict_proba(input_data)[0][1]
    
    st.subheader(f"Probabilidad de diagnóstico positivo: {probabilidad * 100:.2f}%")

    if probabilidad > 0.6:
      st.error("**Riesgo Alto:** Se recomienda encarecidamente buscar atención médica.")
    elif probabilidad > 0.3:
      st.warning("**Riesgo Moderado:** Es aconsejable consultar a un profesional de la salud.")
    else:
      st.success("**Riesgo Bajo:** Continúa con un estilo de vida saludable.")