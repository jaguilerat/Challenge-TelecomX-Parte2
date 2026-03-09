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
```
---
# 📥 Carga de Datos

```python
df = pd.read_csv("/content/datos_tratados_telecomx.csv")

df.head()
```
---
# 🔎 Exploración Inicial de Datos
---
```python
df.info()
df.describe()
```
---
Verificación de valores nulos:
---

```python
df.isnull().sum()
```
---

# 📊 Análisis de Correlación
El análisis de correlación permite identificar qué variables tienen mayor relación con la cancelación de clientes.

```python
plt.figure(figsize=(12,8))

sns.heatmap(df.corr(),
            cmap="coolwarm",
            annot=False)

plt.title("Matriz de Correlación")
plt.show()
```
---
Este análisis permite identificar variables relevantes que podrían influir en el abandono de clientes.
---

# 📊 Evaluación Comparativa de los Modelos

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

# 🧠 Comparación de desempeño de modelos

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

# ⚙️ Optimización del Modelo Final

Para mejorar la capacidad del modelo para detectar churn, se aplicó una optimización utilizando el parámetro:

```python
class_weight = "balanced"
```

---
Esta configuración permite compensar el desbalance moderado del dataset (73% vs 27%), otorgando mayor peso a la clase minoritaria.

Los resultados del modelo optimizado fueron los siguientes:

Métrica	Resultado
Accuracy	0.74
Precision	0.51
Recall	0.79
F1 Score	0.62

Aunque la exactitud general disminuyó ligeramente, el recall aumentó significativamente, pasando de aproximadamente 0.53 a 0.79.

Esto significa que el modelo logra identificar casi el 80% de los clientes que cancelarán el servicio, lo cual es altamente valioso desde el punto de vista del negocio.

En términos prácticos, el modelo puede utilizarse como un sistema de alerta temprana para detectar clientes en riesgo de abandono.

# 🔎 Análisis de la Importancia de las Variables

El análisis de coeficientes en la regresión logística y la importancia de variables del modelo Random Forest permitieron identificar los factores que más influyen en la cancelación de clientes.

Factores que aumentan la probabilidad de cancelación

Cargos mensuales elevados

Los clientes con mayores cargos mensuales presentan una mayor probabilidad de cancelar el servicio, lo que puede indicar una percepción de alto costo en relación al valor recibido.

Método de pago electrónico

Los clientes que utilizan pagos electrónicos presentan mayores tasas de cancelación, posiblemente debido a que tienen menos fricción para cancelar el servicio.

Servicio de fibra óptica

Los clientes con internet de fibra tienden a mostrar mayor propensión al churn, lo que podría relacionarse con expectativas más altas de calidad o mayor competencia en el mercado.

Factores que reducen la probabilidad de cancelación

Antigüedad del cliente (tenure)

Los clientes con mayor tiempo en la empresa presentan menor probabilidad de cancelar el servicio, reflejando un mayor nivel de fidelización.

Contratos de largo plazo

Los contratos anuales o bianuales reducen significativamente la probabilidad de cancelación, generando mayor estabilidad en la relación cliente–empresa.

Servicios adicionales

La contratación de servicios como:

soporte técnico

seguridad online

backup online

se asocia con menores tasas de churn, ya que aumenta la integración del cliente con el ecosistema de servicios de la empresa.

# 📊 Factores principales asociados al churn

A partir del análisis exploratorio, la correlación de variables y los modelos predictivos, se identificaron tres grandes grupos de factores asociados a la cancelación.

Experiencia inicial del cliente

Los clientes que cancelan presentan menor antigüedad, concentrándose principalmente en los primeros meses de contrato.

Esto sugiere que el proceso de onboarding y la experiencia inicial del servicio son críticos para la retención.

Percepción del precio del servicio

Los clientes con mayores cargos mensuales presentan mayor probabilidad de cancelar, lo que puede reflejar una percepción de bajo valor del servicio.

Nivel de vinculación con la empresa

Clientes con menos servicios contratados o sin soporte adicional presentan mayor riesgo de cancelación.

Esto indica que el nivel de integración con el ecosistema de servicios influye directamente en la permanencia del cliente.

# 🚀 Estrategias de Retención Basadas en los Resultados

A partir de los resultados obtenidos, Telecom X podría implementar diversas estrategias para reducir la cancelación de clientes.

Programas de retención temprana

Dado que gran parte del churn ocurre durante los primeros meses, se recomienda fortalecer el proceso de onboarding y realizar seguimiento activo durante los primeros 90 días.

Incentivar contratos de mayor duración

Promover contratos anuales mediante descuentos o beneficios adicionales puede ayudar a reducir la cancelación asociada a contratos mensuales.

Estrategias de cross-selling

Promover la contratación de servicios adicionales como soporte técnico, seguridad o backup puede aumentar la fidelización del cliente.

Ofertas personalizadas para clientes en riesgo

El modelo predictivo permite identificar clientes con alta probabilidad de cancelación, sobre los cuales se pueden aplicar acciones específicas como descuentos personalizados, mejoras de plan o beneficios adicionales.

# 📌 Conclusión

El desarrollo de modelos de Machine Learning permite a Telecom X anticiparse a la cancelación de clientes mediante un enfoque predictivo.

El modelo de Regresión Logística balanceada demostró ser el más efectivo para detectar clientes en riesgo, logrando identificar cerca del 80% de los casos de churn.

Además, el análisis de importancia de variables permitió identificar factores clave asociados a la cancelación, como:

baja antigüedad del cliente

altos cargos mensuales

tipo de contrato

servicios adicionales contratados

Estos resultados proporcionan información estratégica que puede ser utilizada para diseñar políticas de retención más efectivas, mejorar la experiencia del cliente y aumentar el valor de vida del cliente (Customer Lifetime Value).

