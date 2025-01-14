# 🧠 ML Moderation Service

![Python](https://img.shields.io/badge/Python-3.13.1-blue?style=flat-square&logo=python)
![Flask](https://img.shields.io/badge/Flask-2.x-green?style=flat-square&logo=flask)
![scikit-learn](https://img.shields.io/badge/Scikit--Learn-1.0-orange?style=flat-square&logo=scikit-learn)
![Imbalanced-learn](https://img.shields.io/badge/Imbalanced--Learn-0.10.0-red?style=flat-square&logo=python)

Un servicio de moderación basado en Machine Learning que detecta mensajes ofensivos o maliciosos. Implementado con Flask y un modelo de **Logistic Regression** entrenado con **scikit-learn**, utilizando preprocesamiento avanzado de texto y técnicas para manejar datos desbalanceados.

---

## 🚀 Características

- **Detección precisa de contenido ofensivo**: Basado en características extraídas mediante **TF-IDF**.
- **API REST**: Fácil de integrar en aplicaciones externas.
- **Técnicas de balanceo de clases**: Uso de **RandomOverSampler** para mejorar la precisión en datasets desbalanceados.
- **Evaluación robusta**: Informes detallados de clasificación y matriz de confusión.

---

## 🛠️ Tecnologías utilizadas

- **Python 3.13.1**: Lenguaje principal del proyecto.
- **Flask**: Framework para la creación de la API REST.
- **scikit-learn**: Para entrenamiento y evaluación del modelo.
- **Imbalanced-learn**: Para manejo de datasets desbalanceados.
- **pandas**: Para manipulación de datos.
- **pickle**: Para serialización del modelo.
