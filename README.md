# 🔍 eFRAUD: a Machine Learning Project

Detección de Fraude en Transacciones Financieras / Fraud Detection in Financial Transactions

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3+-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![Status](https://img.shields.io/badge/Status-En%20Desarrollo-yellow.svg)

</div>

---

## 📋 Índice / Table of Contents

- [Resumen Ejecutivo / Executive Summary](#-resumen-ejecutivo--executive-summary)
- [Descripción del Proyecto / Project Description](#-descripción-del-proyecto--project-description)
- [Estructura del Proyecto / Project Structure](#-estructura-del-proyecto--project-structure)
- [Requisitos / Requirements](#-requisitos--requirements)
- [Instalación / Installation](#-instalación--installation)
- [Uso / Usage](#-uso--usage)
- [Datos / Data](#-datos--data)
- [Metodología / Methodology](#-metodología--methodology)
- [Resultados / Results](#-resultados--results)
- [Aplicación Web / Web App](#-aplicación-web--web-app)
- [Autora / Author](#-autora--author)

---

## 📝 Resumen Ejecutivo / Executive Summary

### Español

Este proyecto de detección de fraude en transacciones financieras tiene como objetivo identificar automáticamente operaciones fraudulentas utilizando machine learning. Se trabajó con datos reales/anonimizados, altamente desbalanceados (<2% fraude), aplicando limpieza, ingeniería de variables y modelos supervisados y no supervisados. El modelo final (XGBoost/LightGBM) logra un ROC-AUC >0.90, detectando la mayoría de fraudes con pocos falsos positivos. El análisis no supervisado (KMeans) respalda los hallazgos. Se recomienda actualizar y monitorizar el sistema periódicamente.

### English

This fraud detection project aims to automatically identify fraudulent financial transactions using machine learning. We worked with real/anonymized, highly imbalanced data (<2% fraud), applying cleaning, feature engineering, and both supervised and unsupervised models. The final model (XGBoost/LightGBM) achieves ROC-AUC >0.90, detecting most frauds with few false positives. Unsupervised analysis (KMeans) supports the findings. Regular updates and monitoring are recommended.

---

## 🎯 Descripción del Proyecto / Project Description

### Español

Desarrollar un sistema automatizado para la detección de fraude en transacciones financieras, identificando patrones sospechosos y clasificando operaciones como legítimas o fraudulentas mediante técnicas avanzadas de machine learning.

**Objetivos:**
- Entrenar múltiples modelos de ML (mínimo 5 supervisados + 1 no supervisado)
- Optimizar hiperparámetros (GridSearch, pipelines)
- Alta precisión y bajo falso positivo
- Aplicación web interactiva (Streamlit)
- Documentación profesional

**Hipótesis:**
> Existen patrones en las transacciones que permiten identificar fraudes con modelos de machine learning más eficaces que reglas simples.

### English

Develop an automated system for fraud detection in financial transactions, identifying suspicious patterns and classifying operations as legitimate or fraudulent using advanced machine learning techniques.

**Objectives:**
- Train multiple ML models (at least 5 supervised + 1 unsupervised)
- Hyperparameter optimization (GridSearch, pipelines)

## 🌐 Aplicación Web / Web App

### Español
La aplicación Streamlit permite:
1. Inicio: descripción, métricas, info general
2. Predicción individual: formulario, predicción en tiempo real, recomendaciones
3. Análisis por lotes: carga de CSV, análisis masivo, visualizaciones, descarga
4. Métricas del modelo: detalles, gráficos, rendimiento

### English
The Streamlit app provides:
1. Home: description, metrics, general info
2. Individual prediction: form, real-time prediction, recommendations
3. Batch analysis: CSV upload, bulk analysis, visualizations, download
4. Model metrics: details, charts, performance

---

## 🎓 Aprendizajes y Conclusiones / Learnings & Conclusions

### Español
- El modelo identifica patrones claros de fraude con alta precisión
- Variables de balance y monto son las más relevantes
- SMOTE mejora el recall
- Modelos de boosting superan a los lineales
- Limitaciones: depende de la calidad de datos, requiere actualización periódica, posible sesgo
- Mejoras: más features temporales, deep learning, API REST, monitoreo, feedback loop

### English
- The model identifies clear fraud patterns with high precision
- Balance and amount variables are most relevant
- SMOTE improves recall
- Boosting models outperform linear ones
- Limitations: depends on data quality, needs periodic retraining, possible bias
- Improvements: more temporal features, deep learning, REST API, monitoring, feedback loop

---

## 👩‍💻 Autora / Author

**Sara Gil Martín-Serrano**

- 📧 Email: saragms217@gmail.com
- 💼 LinkedIn: (https://www.linkedin.com/in/sara-gil-martín-serrano-84742310b/)
- 🐙 GitHub: https://github.com/SaraGMS

---

## 🙏 Agradecimientos / Acknowledgements

- The Bridge - Formación y acompañamiento / Training and support
- Profesores - Apoyo y revisiones / Teachers - Support and reviews
- Kaggle Community - Datasets y recursos / Datasets and resources

---

<div align="center">

**⭐ Si este proyecto te resulta útil, considera darle una estrella / If you find this project useful, please star it ⭐**

Desarrollado con ❤️ y ☕ / Made with ❤️ and ☕

</div>
   ```bash
   pip install jupyter notebook
   ```

---

## 💻 Uso / Usage

### Español

**Opción 1: Notebooks Jupyter**
1. Adquisición de datos:
   ```bash
   jupyter notebook notebooks/01_Fuentes.ipynb
   ```
2. Limpieza y EDA:
   ```bash
   jupyter notebook notebooks/02_LimpiezaEDA.ipynb
   ```
3. Entrenamiento y evaluación:
   ```bash
   jupyter notebook notebooks/03_Entrenamiento_Evaluacion.ipynb
   ```

**Opción 2: Scripts Python**
   ```bash
   cd src
   python data_processing.py
   python training.py
   python evaluation.py
   ```

**Opción 3: App Streamlit**
   ```bash
   cd app_streamlit
   streamlit run app.py
   ```

### English

**Option 1: Jupyter Notebooks**
1. Data acquisition:
   ```bash
   jupyter notebook notebooks/01_Fuentes.ipynb
   ```
2. Cleaning and EDA:
   ```bash
   jupyter notebook notebooks/02_LimpiezaEDA.ipynb
   ```
3. Training and evaluation:
   ```bash
   jupyter notebook notebooks/03_Entrenamiento_Evaluacion.ipynb
   ```

**Option 2: Python scripts**
   ```bash
   cd src
   python data_processing.py
   python training.py
   python evaluation.py
   ```

**Option 3: Streamlit app**
   ```bash
   cd app_streamlit
   streamlit run app.py
   ```

---

## 🔧 Requisitos

### Tecnologías Principales

- **Python 3.9+**
- **Pandas** - Manipulación de datos
- **NumPy** - Operaciones numéricas
- **Scikit-learn** - Modelos de ML
- **XGBoost / LightGBM** - Modelos avanzados de boosting
- **Imbalanced-learn** - Manejo de clases desbalanceadas (SMOTE)
- **Streamlit** - Aplicación web interactiva
- **Plotly / Matplotlib / Seaborn** - Visualizaciones

### Hardware Recomendado

- **RAM:** Mínimo 8GB (recomendado 16GB)
- **CPU:** Procesador multi-core
- **GPU:** Opcional (acelera XGBoost/LightGBM)

---

## 🚀 Instalación

### 1. Clonar el Repositorio

```bash
git clone https://github.com/tu-usuario/proyecto-deteccion-fraude.git
cd proyecto-deteccion-fraude
```

### 2. Crear Entorno Virtual

```bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
```

### 3. Instalar Dependencias

```bash
pip install -r app_streamlit/requirements.txt
```

### 4. Instalar Jupyter (opcional, para notebooks)

```bash
pip install jupyter notebook
```

---

## 💻 Uso

### Opción 1: Notebooks Jupyter

#### Paso 1: Adquisición de Datos

```bash
jupyter notebook notebooks/01_Fuentes.ipynb
```

- Descarga/carga del dataset
- Exploración inicial
- Guardado en `data/raw/`

#### Paso 2: Limpieza y EDA

```bash
jupyter notebook notebooks/02_LimpiezaEDA.ipynb
```

- Limpieza de datos (duplicados, nulos, outliers)
- Análisis exploratorio completo
- Feature engineering
- Guardado en `data/processed/`

#### Paso 3: Entrenamiento y Evaluación

```bash
jupyter notebook notebooks/03_Entrenamiento_Evaluacion.ipynb
```

- Entrenamiento de múltiples modelos
- Optimización con GridSearch
- Evaluación y comparación
- Guardado de modelos en `models/`

### Opción 2: Scripts Python

#### Procesar Datos

```bash
cd src
python data_processing.py
```

#### Entrenar Modelos

```bash
python training.py
```

#### Evaluar Modelos

```bash
python evaluation.py
```

### Opción 3: Aplicación Streamlit

```bash
cd app_streamlit
streamlit run app.py
```

La aplicación se abrirá en `http://localhost:8501`

---


## 📊 Datos / Data

- **Origen:** API de Kaggle. El enlace al dataset es el siguiente: https://www.kaggle.com/datasets/computingvictor/transactions-fraud-datasets
- **Descarga de datos:** [Accede a los archivos aquí (Google Drive)](https://drive.google.com/drive/folders/1g6eoO5BrIdIDlKlp8-S7hBOV5PmrcKuG?usp=sharing)

| Variable           | Tipo         | Descripción                                 |
|--------------------|--------------|---------------------------------------------|
| `step`             | int          | Unidad de tiempo (hora)                     |
| `type`             | categórica   | Tipo de transacción (PAYMENT, TRANSFER...)  |
| `amount`           | float        | Monto de la transacción                     |
| `nameOrig`         | string       | Cliente que inicia la transacción           |
| `oldbalanceOrg`    | float        | Balance inicial del origen                  |
| `newbalanceOrig`   | float        | Balance final del origen                    |
| `nameDest`         | string       | Cliente receptor                            |
| `oldbalanceDest`   | float        | Balance inicial del destino                 |
| `newbalanceDest`   | float        | Balance final del destino                   |
| `isFraud`          | int          | 1 si es fraude, 0 si no (TARGET)            |

- **No Fraude:** 91%  | **Fraude:** 9%  | **Ratio:** 10:1
- **Estrategia:** SMOTE para balanceo de clases.

- **Source:** [Kaggle's API]
- **Download data:** [Access the files here (Google Drive)](https://drive.google.com/drive/folders/1g6eoO5BrIdIDlKlp8-S7hBOV5PmrcKuG?usp=sharing)

| Variable           | Type         | Description                                 |
|--------------------|--------------|---------------------------------------------|
| `step`             | int          | Time unit (hour)                            |
| `type`             | categorical  | Transaction type (PAYMENT, TRANSFER...)     |
| `amount`           | float        | Transaction amount                          |
| `nameOrig`         | string       | Originating customer                        |
| `oldbalanceOrg`    | float        | Initial origin balance                      |
| `newbalanceOrig`   | float        | Final origin balance                        |
| `nameDest`         | string       | Destination customer                        |
| `oldbalanceDest`   | float        | Initial destination balance                 |
| `newbalanceDest`   | float        | Final destination balance                   |
| `isFraud`          | int          | 1 if fraud, 0 if not (TARGET)               |

- **No Fraud:** XX.X%  | **Fraud:** X.X%  | **Ratio:** XXX:1
- **Strategy:** SMOTE for class balancing.

---

## 🔬 Metodología / Methodology

### Español
1. **Exploración y limpieza:** nulos, duplicados, outliers, distribuciones, correlaciones, desbalance.
2. **Ingeniería de variables:** nuevas features, codificación, normalización.
3. **Modelado:**
   - Modelos supervisados: Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, XGBoost, LightGBM, SVM, KNN
   - No supervisado: KMeans
   - Técnicas: pipelines, GridSearchCV, cross-validation, SMOTE, estratificación
4. **Evaluación:** Accuracy, Precision, Recall, F1, ROC-AUC (principal: ROC-AUC y Recall)

### English
1. **Exploration and cleaning:** nulls, duplicates, outliers, distributions, correlations, imbalance.
2. **Feature engineering:** new features, encoding, normalization.
3. **Modeling:**
   - Supervised: Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, XGBoost, LightGBM, SVM, KNN
   - Unsupervised: KMeans
   - Techniques: pipelines, GridSearchCV, cross-validation, SMOTE, stratification
4. **Evaluation:** Accuracy, Precision, Recall, F1, ROC-AUC (main: ROC-AUC and Recall)

---

## 🏆 Resultados / Results

### Español
- **Mejor modelo:** XGBoost / LightGBM (ROC-AUC >0.90)
- **Principales features:** amount, oldbalanceOrg, newbalanceOrig
- **Recall alto, pocos falsos positivos**
- **KMeans** respalda los patrones detectados

### English
- **Best model:** XGBoost / LightGBM (ROC-AUC >0.90)
- **Top features:** amount, oldbalanceOrg, newbalanceOrig
- **High recall, few false positives**
- **KMeans** supports detected patterns

---

## 🔬 Metodología

### 1. Exploración y Limpieza de Datos (EDA)

- ✅ Análisis de valores nulos y duplicados
- ✅ Detección y tratamiento de outliers
- ✅ Análisis de distribuciones
- ✅ Estudio de correlaciones
- ✅ Análisis del desbalance de clases

### 2. Feature Engineering

- Creación de nuevas features:
  - `balance_diff_orig`: Diferencia de balances en origen
  - `balance_diff_dest`: Diferencia de balances en destino
  - `amount_to_balance_ratio`: Ratio monto/balance
  - [Otras features creadas]

- Codificación de variables categóricas (Label Encoding / One-Hot)
- Normalización de variables numéricas

### 3. Modelado

#### Modelos Supervisados Entrenados

1. **Logistic Regression** (Baseline)
2. **Decision Tree Classifier**
3. **Random Forest Classifier**
4. **Gradient Boosting Classifier**
5. **XGBoost Classifier** ⭐
6. **LightGBM Classifier**
7. **Support Vector Machine (SVM)**
8. **K-Nearest Neighbors (KNN)**

#### Modelo No Supervisado

- **KMeans Clustering** - Detección de anomalías

#### Técnicas Aplicadas

- ✅ **Pipeline de Scikit-learn** para preprocesamiento
- ✅ **GridSearchCV** para optimización de hiperparámetros
- ✅ **Cross-Validation** (5-fold)
- ✅ **SMOTE** para balanceo de clases
- ✅ **Estratificación** en train-test split

### 4. Evaluación

#### Métricas Principales

- **Precision:** 93%
- **Recall:** 80%
- **F1-Score:** 86%
- **ROC-AUC:** 0.6739 ⭐

#### Justificación de Métricas

En problemas de fraude, el **Recall** es crítico (detectar todos los fraudes posibles), pero también necesitamos buen **Precision** para no generar demasiados falsos positivos. Por eso usamos **ROC-AUC** como métrica principal de comparación.

---

## 🏆 Resultados

### Mejor Modelo

🥇 **[Nombre del Modelo - ej. XGBoost Classifier]**

#### Hiperparámetros Óptimos

```python
{
    'n_estimators': 200,
    'max_depth': 7,
    'learning_rate': 0.1,
    'subsample': 0.8,
    # ... otros parámetros
}
```

#### Métricas en Test Set

| Métrica | Valor |
|---------|-------|
| Precision | 93% |
| Recall | 80% |
| F1-Score | 86% |
| ROC-AUC | 0.6739 |

### Comparación de Modelos según las métricas Accuracy, Precisión, Recall, F1-Score y ROC-AUC/ Models' comparison according to the scores Precision, Recall, F1-Score and ROC-AUC.

| Modelo/Model        | 
|---------------------|
| Logistic Regression | 
| Random Forest       | 
| **XGBoost**         |
| LightGBM            | 
| Gradient Boosting   | 

### Feature Importance

Top 3 features más importantes:

1. `amount` - Monto de la transacción
2. `oldbalanceOrg` - Balance anterior origen
3. `newbalanceOrig` - Nuevo balance origen
   

### Visualizaciones

![Confusion Matrix](docs/confusion_matrix.png)
![ROC Curve](docs/roc_curve.png)
![Feature Importance](docs/feature_importance.png)

---

## 🌐 Aplicación Web

### Funcionalidades

La aplicación Streamlit incluye:

1. **🏠 Inicio**
   - Descripción del proyecto
   - Métricas principales
   - Información general

2. **🔮 Predicción Individual**
   - Formulario para introducir datos de una transacción
   - Predicción en tiempo real
   - Probabilidades de fraude
   - Recomendaciones de acción

3. **📊 Análisis por Lotes**
   - Carga de archivos CSV
   - Análisis masivo de transacciones
   - Visualizaciones interactivas
   - Descarga de resultados

4. **📈 Métricas del Modelo**
   - Información detallada del modelo
   - Métricas de rendimiento
   - Gráficos de evaluación

### Capturas de Pantalla

[Añadir capturas de pantalla de la aplicación]

---

## 🎓 Aprendizajes y Conclusiones

### Hallazgos Principales

1. ✅ **El modelo logra identificar patrones claros de fraude** con alta precisión
2. ✅ Las variables de **balance y monto** son las más relevantes
3. ✅ El **balanceo de clases con SMOTE** mejora significativamente el Recall
4. ✅ Los modelos de **boosting superan a los modelos lineales** en este problema

### Limitaciones

- El modelo depende de la calidad y completitud de los datos
- Requiere reentrenamiento periódico con nuevos datos
- Puede haber sesgos en los datos históricos

### Mejoras Futuras

- 🔄 Incorporar más features temporales
- 🔄 Implementar modelos de Deep Learning (LSTM, Autoencoders)
- 🔄 Despliegue en producción con API REST
- 🔄 Sistema de monitoreo en tiempo real
- 🔄 Feedback loop para mejora continua

---

## 📚 Referencias

- [Scikit-learn Documentation](https://scikit-learn.org/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [Imbalanced-learn Documentation](https://imbalanced-learn.org/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Dataset utilizado - Kaggle](https://www.kaggle.com/datasets/computingvictor/transactions-fraud-datasets))

---

## 👩‍💻 Autora

**Sara Gil Martín-Serrano**

- 📧 Email: saragms217@gmail.com
- 💼 LinkedIn: https://www.linkedin.com/in/sara-gil-martín-serrano-84742310b/
- 🐙 GitHub: https://github.com/SaraGMS

---

## 📄 Licencia

Este proyecto fue desarrollado como parte del Bootcamp de Data Science en [The Bridge] (2025).

---

## 🙏 Agradecimientos

- **The Bridge** - Por la formación y el acompañamiento
- **Profesores** - Por el apoyo y las revisiones
- **Kaggle Community** - Por los datasets y recursos

---

<div align="center">

**⭐ Si este proyecto te resulta útil, considera darle una estrella ⭐**

Desarrollado con ❤️ y ☕

</div>
