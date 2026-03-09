---
title: "Telecom X – Predicción de Cancelación de Clientes (Churn)"
author: "Johnnatan Aguilera"
---

# 📊 Telecom X – Predicción de Cancelación de Clientes

## 📣 Historia del Desafío

Después del análisis exploratorio inicial sobre cancelación de clientes en **Telecom X**, la compañía busca avanzar hacia una etapa más estratégica: **predecir qué clientes tienen mayor probabilidad de cancelar sus servicios**.

El objetivo es desarrollar un **pipeline de Machine Learning** capaz de anticipar la cancelación (Churn), permitiendo a la empresa tomar acciones preventivas para mejorar la retención de clientes.

---

# 🎯 Objetivos del Proyecto

Este proyecto tiene como propósito desarrollar un modelo predictivo de cancelación de clientes mediante:

- Preparación y limpieza de datos
- Análisis de correlación y selección de variables
- Entrenamiento de modelos de clasificación
- Evaluación de rendimiento
- Interpretación de resultados
- Generación de insights estratégicos

---

# 🧰 Tecnologías Utilizadas

- **Python**
- **Pandas**
- **NumPy**
- **Matplotlib / Seaborn**
- **Scikit-Learn**
- **RMarkdown** para documentación reproducible

---

# 📂 Estructura del Proyecto


telecomx-churn-ml/
│
├── data/
│ ├── raw
│ ├── processed
│
├── notebooks
│
├── src
│ ├── preprocessing.py
│ ├── train.py
│ ├── predict.py
│
├── models
│
├── reports
│
└── README.md


---

# 📥 Carga de Librerías

```python
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
📥 Carga de Datos
df = pd.read_csv("/content/datos_tratados_telecomx.csv")

df.head()
🔎 Exploración Inicial de Datos
df.info()
df.describe()

Verificación de valores nulos:

df.isnull().sum()


📊 Análisis de Correlación

El análisis de correlación permite identificar qué variables tienen mayor relación con la cancelación de clientes.

plt.figure(figsize=(12,8))

sns.heatmap(df.corr(),
            cmap="coolwarm",
            annot=False)

plt.title("Matriz de Correlación")
plt.show()

Este análisis permite identificar variables relevantes que podrían influir en el abandono de clientes.
