# Simulador de Riesgo de Ataque Cardíaco

Este proyecto es una aplicación web interactiva que utiliza un modelo de Machine Learning para predecir el riesgo de un diagnóstico positivo de ataque cardíaco a partir de 8 indicadores clínicos.

## Demo en Vivo

[**Ver aplicación en funcionamiento**](https://tu-app-name.streamlit.app) _(Disponible después del deploy)_

## Características

- **Análisis Exploratorio de Datos (EDA):** Investigación detallada de la distribución y correlación de las variables.
- **Modelo Predictivo:** Se entrenó un clasificador `RandomForestClassifier` para obtener un balance entre precisión e interpretabilidad.
- **Pipeline de Scikit-Learn:** Se implementó un pipeline que integra el preprocesamiento (escalado de datos) y el entrenamiento para evitar la fuga de datos.
- **Evaluación Robusta:** El modelo fue evaluado utilizando validación cruzada para asegurar su rendimiento.
- **Interfaz Interactiva:** Una aplicación web desarrollada con Streamlit que permite a los usuarios ingresar sus propios datos y obtener una predicción en tiempo real.
- **Auto-generación del Modelo:** La aplicación genera automáticamente el modelo si no existe.

## Disclaimer

Este es un proyecto educativo y **NO reemplaza una consulta médica profesional**. Las predicciones son solo para fines demostrativos.

## Tecnologías Utilizadas

- **Lenguaje:** Python 3.9
- **Librerías de Análisis:** Pandas, NumPy, Matplotlib, Seaborn
- **Machine Learning:** Scikit-Learn
- **Aplicación Web:** Streamlit
- **Gestión de Entorno:** `pip`
