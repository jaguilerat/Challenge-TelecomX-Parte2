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

telecomx-parte2/
│
├── notebooks/
├── src/
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


## 📊 Evaluación Comparativa de los Modelos

En este proyecto se desarrollaron distintos modelos de *Machine Learning* con el objetivo de **predecir la cancelación de clientes (Churn)** dentro de Telecom X.

Se evaluaron los siguientes modelos:

- Dummy Classifier (baseline)
- Regresión Logística
- Árbol de Decisión
- Random Forest

El **Dummy Classifier** se utilizó como modelo base de referencia, prediciendo siempre la clase mayoritaria del dataset.

Los resultados muestran que, aunque el modelo obtiene una exactitud cercana al **73%**, no logra identificar clientes que cancelan, ya que presenta **precision, recall y F1-score iguales a cero**.

Esto confirma que el problema de churn requiere modelos predictivos capaces de capturar relaciones entre variables y detectar patrones asociados a la cancelación.

---

## 🧠 Comparación de desempeño de modelos

Los resultados obtenidos para cada modelo fueron los siguientes:

| Modelo | Accuracy | Precision | Recall | F1 Score |
|------|------|------|------|------|
| Regresión Logística | 0.80 | 0.65 | 0.53 | 0.58 |
| Árbol de Decisión | 0.71 | 0.46 | 0.44 | 0.45 |
| Random Forest | 0.79 | 0.63 | 0.49 | 0.55 |

En problemas de churn, **la métrica más relevante es el Recall**, ya que representa la capacidad del modelo para identificar correctamente a los clientes que cancelarán el servicio.

En este análisis, **la Regresión Logística presentó el mejor desempeño general**, alcanzando el mayor recall y el mejor equilibrio entre precisión y capacidad predictiva.

El **Árbol de Decisión** mostró el desempeño más bajo, probablemente debido a problemas de sobreajuste (*overfitting*), mientras que el **Random Forest** mejoró la estabilidad respecto al árbol simple, pero no superó a la regresión logística en la detección de cancelaciones.

Por esta razón, la **Regresión Logística fue seleccionada como modelo principal del proyecto**.

---

## ⚙️ Optimización del Modelo Final

Para mejorar la capacidad del modelo para detectar churn, se aplicó una optimización utilizando el parámetro:

```python
class_weight = "balanced"




