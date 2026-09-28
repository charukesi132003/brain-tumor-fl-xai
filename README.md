@'
# Brain Tumor FL-XAI

Brain Tumor MRI Classification using Federated Learning, Privacy-Oriented Training and Explainable AI.

## Overview

This project investigates brain tumor MRI classification using ResNet-18 under centralized and federated training configurations.

Four classes are supported:

- Glioma
- Meningioma
- No Tumor
- Pituitary

The project additionally provides explainability using:

- Grad-CAM
- SHAP GradientExplainer
- L-FEA (project-specific cross-model explanation aggregation)

## Dataset

Total images: 7,200

Training:
- Glioma: 1,400
- Meningioma: 1,400
- No Tumor: 1,400
- Pituitary: 1,400
- Total: 5,600

Testing:
- Glioma: 400
- Meningioma: 400
- No Tumor: 400
- Pituitary: 400
- Total: 1,600

## Model

Backbone: ResNet-18

Input:
Brain MRI image

Output classes:
1. Glioma
2. Meningioma
3. No Tumor
4. Pituitary

## Experimental Results

| Model | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Centralized | 77.38% | 77.70% | 77.38% | 76.43% |
| Federated | 71.69% | 71.06% | 71.69% | 69.53% |
| Federated + DP Experiment | 41.19% | 40.02% | 41.19% | 38.86% |

## Federated Learning

Five simulated hospitals are used.

Training follows the FedAvg workflow:

Local Training
→ Local Model Updates
→ Federated Aggregation
→ Global Model

The best federated model obtained 71.69% test accuracy.

## Privacy Experiment

A privacy-oriented federated experiment using gradient clipping and noise was also evaluated.

The experiment obtained 41.19% test accuracy, demonstrating a substantial privacy/perturbation versus model-utility trade-off under the selected configuration.

## Explainable AI

### Grad-CAM

Grad-CAM visualizes convolutional regions influencing the ResNet-18 prediction.

### SHAP

SHAP GradientExplainer generates pixel-level attribution information.

### L-FEA

L-FEA is the project's experimental cross-model explanation aggregation method.

It aggregates normalized explanation maps from:

- Centralized model
- Federated model
- Federated + privacy-trained model

For one example MRI, overall explanation agreement was 75.07%.

This value is an example result for that image, not a dataset-wide performance metric.

## Application

The Gradio application provides:

- MRI upload
- Tumor prediction
- Class probabilities
- Grad-CAM
- SHAP
- L-FEA comparison
- Model-performance information

## Installation

Create and activate a virtual environment.

Windows:

```powershell
py -3.10 -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt