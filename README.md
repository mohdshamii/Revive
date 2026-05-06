<div align="center">

<img src="static/logo.png" width="120" alt="Revive Logo" />

<br/>

# REVIVE

### Machine Learning Based Disease Prediction System

<br/>

![Python](https://img.shields.io/badge/Python-3.8+-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Flask](https://img.shields.io/badge/Flask-2.x-000000?style=for-the-badge&logo=flask&logoColor=white)
![Bootstrap](https://img.shields.io/badge/Bootstrap-5.x-7952B3?style=for-the-badge&logo=bootstrap&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Deep_Learning-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML_Models-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![License](https://img.shields.io/badge/License-Apache_2.0-blue?style=for-the-badge)

<br/>

*Bridging clinical intelligence and machine learning — one prediction at a time.*

</div>

---

## Overview

**Revive** is a production-grade, machine learning powered web application capable of predicting seven distinct diseases from patient medical data. Built on Flask, it integrates both classical machine learning models and convolutional neural networks to deliver fast, accurate, and interpretable medical predictions through a clean web interface.

The system was designed to reduce diagnostic friction for doctors, medical researchers, and patients by providing AI-assisted predictions based on validated medical datasets sourced from Kaggle and the UCI Machine Learning Repository.

---

## Screenshots

<div align="center">

**Home**

<img src="images/home.png" width="800" alt="Home Page" />

<br/><br/>

**Available Services**

<img src="images/sevices.png" width="800" alt="Services Page" />

</div>

---

## Disease Prediction Modules

<div align="center">

| Disease | Model Type | Accuracy | Screenshot |
|---|---|---|---|
| Diabetes | Machine Learning | 98.25% | [View](#diabetes-prediction) |
| Breast Cancer | Machine Learning | 98.25% | [View](#breast-cancer-prediction) |
| Heart Disease | Machine Learning | 85.25% | [View](#heart-disease-prediction) |
| Kidney Disease | Machine Learning | 99.00% | [View](#kidney-disease-prediction) |
| Liver Disease | Machine Learning | 78.00% | [View](#liver-disease-prediction) |
| Malaria | CNN Deep Learning | 96.00% | [View](#malaria-detection) |
| Pneumonia | CNN Deep Learning | 95.00% | [View](#pneumonia-detection) |

</div>

---

### Diabetes Prediction

<img src="images/diabetes.png" width="800" alt="Diabetes Prediction" />

A classical machine learning model trained on the PIMA Indian Diabetes dataset. Users input glucose levels, BMI, insulin, age, and other clinical markers to receive an immediate prediction.

---

### Breast Cancer Prediction

<img src="images/breast_cancer.png" width="800" alt="Breast Cancer Prediction" />

Trained on the Wisconsin Breast Cancer dataset. The model analyzes cell nucleus features extracted from digitized fine needle aspirate images to classify tumors as malignant or benign.

---

### Heart Disease Prediction

<img src="images/heart.png" width="800" alt="Heart Disease Prediction" />

A classifier trained on the Cleveland Heart Disease dataset from UCI. Accepts 13 clinical attributes including age, chest pain type, resting blood pressure, and electrocardiographic results.

---

### Kidney Disease Prediction

<img src="images/kidney.png" width="800" alt="Kidney Disease Prediction" />

Trained on the Chronic Kidney Disease dataset. Processes 24 medical features including blood urea, serum creatinine, haemoglobin, and packed cell volume to predict kidney disease presence.

---

### Liver Disease Prediction

<img src="images/liver.png" width="800" alt="Liver Disease Prediction" />

Uses the Indian Liver Patient dataset (ILPD). Input features include age, total bilirubin, direct bilirubin, alkaline phosphatase, and albumin protein levels.

---

### Malaria Detection

<img src="images/malaria.png" width="800" alt="Malaria Detection" />

A Convolutional Neural Network trained on the NIH Malaria dataset containing 27,558 cell images. Users upload a blood smear image; the CNN classifies it as parasitized or uninfected.

---

### Pneumonia Detection

<img src="images/pneumonia.png" width="800" alt="Pneumonia Detection" />

A CNN trained on chest X-ray images from the Guangzhou Women and Children Medical Center dataset. The model distinguishes between normal lungs and pneumonia-affected lungs from a single uploaded X-ray.

---

## System Architecture

```
                    User Browser
                         |
                    HTTP Request
                         |
                   Flask App (app.py)
                    /           \
              GET /           POST /predict
                |                   |
          Render HTML         Receive Input
          Template            (Form / Image)
                                    |
                             Load Trained Model
                             (.pkl / .h5)
                                    |
                             Preprocess Input
                             (Scaler / Normalize)
                                    |
                             Model Inference
                                    |
                             Generate Prediction
                                    |
                             Return Result
                             to HTML Template
```

**Visual Flow Chart:**

<img src="images/flowchart.jpeg" width="700" alt="Application Flow Chart" />

---

## Project Structure

```
Revive/
│
├── app.py                        # Flask application entry point
├── requirements.txt              # Python dependencies
│
├── models/                       # Trained ML and DL model files
│   ├── diabetes_model.pkl
│   ├── breast_cancer_model.pkl
│   ├── heart_model.pkl
│   ├── kidney_model.pkl
│   ├── liver_model.pkl
│   ├── malaria_model.h5
│   └── pneumonia_model.h5
│
├── static/                       # Static assets
│   ├── logo.png
│   ├── css/
│   └── js/
│
├── templates/                    # HTML templates
│   ├── index.html
│   ├── diabetes.html
│   ├── breast_cancer.html
│   ├── heart.html
│   ├── kidney.html
│   ├── liver.html
│   ├── malaria.html
│   └── pneumonia.html
│
└── images/                       # Documentation images
```

---

## Quick Start

**Prerequisites:** Python 3.8 or higher, pip

**Step 1 — Clone the repository**

```bash
git clone https://github.com/codexshami/Revive.git
```

**Step 2 — Navigate to the project directory**

```bash
cd Revive
```

**Step 3 — Install all dependencies**

```bash
pip install -r requirements.txt
```

**Step 4 — Start the application**

```bash
python app.py
```

or alternatively:

```bash
flask run
```

**Step 5 — Open in browser**

```
http://127.0.0.1:5000
```

---

## Technology Stack

| Layer | Technology |
|---|---|
| Backend Framework | Flask (Python) |
| Machine Learning | scikit-learn |
| Deep Learning | TensorFlow / Keras (CNN) |
| Frontend | HTML5, Bootstrap 5 |
| Model Serialization | Pickle (.pkl), HDF5 (.h5) |
| Data Sources | Kaggle, UCI ML Repository |

---

## Key Capabilities

- Seven disease prediction modules in a single unified application
- Dual modality input — structured clinical data for ML models, image upload for CNN models
- Sub-second inference time on trained model files
- Preprocessing pipelines baked into the prediction route for consistent results
- Responsive web interface accessible from any modern browser

---

## Why This Matters

Clinical diagnosis is demanding. Fatigue, time pressure, and cognitive load can introduce errors in manual assessment. Machine learning systems trained on thousands of validated patient records identify statistical patterns that complement clinical judgment — not replace it.

This project demonstrates that disease prediction models with high accuracy can be packaged into accessible, user-friendly tools that reduce barriers to early detection and support better health outcomes.

---

## License

```
Copyright 2026 Mohd Shami

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

---

<div align="center">

**Mohd Shami**

[LinkedIn](https://www.linkedin.com/in/codexshami) &nbsp;|&nbsp; [GitHub](https://github.com/codexshami) &nbsp;|&nbsp; [Email](mailto:codexshami@gmail.com)

<sub>Built with precision. Trained on data. Deployed for impact.</sub>

</div>
