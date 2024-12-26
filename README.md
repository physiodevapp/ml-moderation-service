# 🧠 ML Moderation Service

![Python](https://img.shields.io/badge/Python-3.13.1-blue?style=flat-square&logo=python)
![Flask](https://img.shields.io/badge/Flask-2.x-green?style=flat-square&logo=flask)
![scikit-learn](https://img.shields.io/badge/Scikit--Learn-1.0-orange?style=flat-square&logo=scikit-learn)

Un servicio de moderación basado en Machine Learning que detecta mensajes ofensivos o maliciosos. Implementado con Flask y un modelo Naive Bayes entrenado con scikit-learn.

---

## 🚀 Características

- Predicción de contenido ofensivo en texto.
- API REST sencilla para integración.
- Escalable y modular.

---

## 📂 Estructura del Proyecto

```plaintext
ml-moderation-service/
├── app.py              # Servicio Flask
├── requirements.txt    # Dependencias
├── venv/               # Entorno virtual
├── model/
│   ├── train_model.py  # Entrenamiento del modelo
│   ├── model.pkl       # Modelo entrenado
└── README.md           # Documentación
