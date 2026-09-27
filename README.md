# FytoPod 🌿
### Multimodal Plant Health Intelligence Using Predictive Analytics and Explainable Deep Learning

> B.Tech Data Science and Engineering — Final Year Capstone  
> Manipal Academy of Higher Education, Dubai — June 2026  
> Presented at AEIT 2026

---

## Overview

FytoPod is a dual-modal plant health intelligence system for urban farming.  
It combines sensor-based LSTM stress prediction with MobileNetV2 + SE Attention  
CNN disease detection, unified through a single health score and delivered via  
a Gradio demo and FastAPI backend.

---

## System Architecture

<img width="2779" height="3580" alt="fytopod_architecture" src="https://github.com/user-attachments/assets/4ab19de3-f42a-47ea-a567-d4d1a6a7dacb" />

---

## Modules

### Sensor Modal — LSTM Stress Predictor
- 6 months of simulated ESP32 sensor data (4,320 hourly readings, 6 channels)
- LSTM: 32 units, seq_len=12, dropout=0.2
- Tuned across 9 hyperparameter configurations
- Benchmarked against Threshold, Logistic Regression, Random Forest
- **95% stress recall** vs 73% threshold baseline
- 506 early warnings, avg 3-hour lead time

### Vision Modal — CNN Disease Classifier
- MobileNetV2 + Squeeze-and-Excitation attention block
- Transfer learning from ImageNet, fine-tuned on 37,662 PlantVillage images
- 25 disease and healthy classes across 7 urban crop species
- **98% validation accuracy, F1-score: 0.98**
- Grad-CAM explainability heatmaps for every prediction
- Severity: Mild (>85%), Moderate (60-85%), Severe (<60%)

### Novel Modules
- Cellular Automata disease spread simulation (Day 0 to Day 14)
- Random Forest plant lifespan estimator (treated vs untreated)
- Treatment impact simulator (health trajectory comparison)

### RAG Treatment Pipeline
- Disease + severity passed to LLaMA 3 via Groq API
- Targeted treatment recommendations + conversational Q&A

### Unified Health Score
- Sensor score: soil moisture (40%), temperature (30%), light (20%), pH (10%)
- Minus LSTM stress penalty
- Minus CNN disease severity penalty
- Final score: 0 to 100

---

## Results

### LSTM Stress Prediction

| Model | Stress Recall | Weighted F1 |
|---|---|---|
| Threshold Baseline | 73% | 0.96 |
| Logistic Regression | 99% | 0.92 |
| Random Forest | 85% | 0.92 |
| **LSTM (FytoPod)** | **95%** | **0.93** |

Model Comparison
<img width="2100" height="900" alt="model_comparison" src="https://github.com/user-attachments/assets/2116f908-a245-4327-bd9e-553424ab4289" />



### CNN Disease Detection

| Metric | Result |
|---|---|
| Model | MobileNetV2 + SE Attention |
| Dataset | PlantVillage (25 classes, 37,662 images) |
| Validation Accuracy | 98% |
| Macro F1-score | 0.98 |
| Weighted F1-score | 0.98 |

Confusion Matrix Comparison
<img width="1841" height="1790" alt="confusion_matrix png" src="https://github.com/user-attachments/assets/f1024c1e-5c04-474d-a21c-04e0e4024846" />

Per Class F1
<img width="1238" height="553" alt="per_class_f1" src="https://github.com/user-attachments/assets/9f8c01f1-67f2-4572-aa85-60e073b9171c" />

### Grad-CAM Explainability

Grad-CAM
<img width="1350" height="2381" alt="gradcam_output png" src="https://github.com/user-attachments/assets/f9a120a6-da42-425c-8395-e70af8e2426d" />


### Novel Modules

Cellular Automata Spread
<img width="2720" height="1520" alt="ca_spread_timeline png" src="https://github.com/user-attachments/assets/eaa2f586-79a6-46c9-9807-a83c63c93874" />

Lifespan Predictor
<img width="1590" height="691" alt="lifespan_predictor png" src="https://github.com/user-attachments/assets/c17b3792-4443-43eb-a4bf-32ab2b4adc18" />

Treatment Impact
<img width="1492" height="675" alt="treatment_impact" src="https://github.com/user-attachments/assets/fb1cc31d-91b0-4e5e-8c0d-1c77349f2080" />

---

## EDA

Class Distribution
<img width="2700" height="1800" alt="eda_class_distribution png" src="https://github.com/user-attachments/assets/07dcd9a7-1648-4c0b-8206-6ab009034e79" />

Sample Grid

<img width="1010" height="7498" alt="eda_sample_grid png" src="https://github.com/user-attachments/assets/d331a363-2b0c-454d-86c6-eae8497d73bb" />

Augmentation
<img width="3407" height="2242" alt="segmentation_lifecycle png" src="https://github.com/user-attachments/assets/00fbbe50-e844-4698-aba3-f48ea97fa402" />


---

## Tech Stack

| Layer | Technology |
|---|---|
| Sensor ML | TensorFlow, Keras LSTM, scikit-learn |
| Vision ML | TensorFlow, MobileNetV2, SE Attention, Grad-CAM |
| Novel Modules | Cellular Automata (NumPy), Random Forest (scikit-learn) |
| Treatment AI | Groq API, LLaMA 3, RAG |
| Backend | FastAPI, PostgreSQL, Supabase |
| Demo | Gradio, Streamlit |

---

## Dataset

PlantVillage dataset — 37,662 images across 25 classes.  
Source: https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset

Trained model weights (.keras, .pkl) are not included due to file size.  
They are available on request or via Supabase Storage.

---

## Project Structure
FytoPod/
├── app/ FastAPI backend
├── sensor_modal/ LSTM stress prediction modules
├── vision_modal/ CNN inference, Grad-CAM, severity
├── novel_modules/ Spread simulation, lifespan, treatment
├── rag_pipeline/ Groq + LLaMA 3 treatment assistant
├── demo/ Gradio demo app
├── database/ PostgreSQL schema
├── notebooks/ Training notebooks (LSTM + CNN)
├── models/ Model placeholders (weights on Supabase)
├── data/sample/ Sample sensor CSV files
└── docs/ Architecture diagrams + all output graphs


---

## Setup

```bash
git clone https://github.com/4rthye/FytoPod.git
cd FytoPod

python -m venv venv
venv\Scripts\activate

pip install -r requirements.txt

cp .env.example .env
# Fill in your keys in .env

# Run backend
cd app
uvicorn main:app --reload --port 8000
```

---

## Team

| Name | Role |
|---|---|
| Arthye Sridharan | Sensor modal, RAG pipeline, novel modules, backend, database |
| Mridul Chelladurai | CNN training, Grad-CAM, Gradio demo, Streamlit app |

**Supervisor:** Dr. M.I. Jawid Nazir, SOEIT, MAHE Dubai
