# COIT29224 – Assessment 1: Enhancing Neural Network Performance using PSO

**Student Name:** Harshvardhan Gosaliya  
**Student ID:** 12231591  
**Term:** T1 2025  
**Course:** COIT29224 – Evolutionary Computation  
**Assignment Title:** Enhancing Neural Network Performance with Particle Swarm Optimization (PSO)

---

## 📘 Project Overview

This project explores how Particle Swarm Optimization (PSO) can be utilized to improve neural network performance through hyperparameter tuning. The focus is on predicting football match outcomes (Win, Draw, Loss) based on match and team statistics using a classification model.

Two models were implemented:
- A **Baseline Neural Network** (manually tuned)
- A **PSO-Optimized Neural Network** (auto-tuned)

---

## 🎯 Problem Domain & Objectives

- **Domain**: Sports analytics – Football match outcome prediction.
- **Goal**: Predict whether a football match will result in a Win, Draw, or Loss for the home team.
- **Key Metric**: Accuracy and F1-score per class (Win, Draw, Loss).

---

## 🗃 Dataset Details

- **Source**: [Kaggle European Soccer Dataset](https://www.kaggle.com/datasets/hugomathien/soccer)
- **Tables Used**: `Match`, `Team_Attributes`
- **Features Selected**: 
  - Build-up play speed
  - Defense aggression
  - League ID
  - Match year
- **Labels**:  
  - `0` = Loss  
  - `1` = Draw  
  - `2` = Win

- **Preprocessing Steps**:
  - Dropped missing values
  - Feature normalization using `StandardScaler`
  - Label encoding using `to_categorical`

---

## 🔧 Model Architectures

### 🔹 Baseline Neural Network
- 2 Hidden Layers: 32 neurons each
- Activation: ReLU
- Optimizer: Adam
- Dropout: 0.3
- Epochs: 30
- Batch Size: 16

### 🔹 PSO-Optimized Neural Network (PSO-NN)
PSO was used to optimize 8 hyperparameters:
- Number of neurons in two layers
- Learning rate
- Activation function (`relu`, `tanh`, `selu`)
- Dropout rate
- Optimizer (`Adam`, `SGD`, `RMSprop`, `Nadam`)
- Batch size
- Third hidden layer (enabled/disabled)

Dynamic informant sizes (3, 4, 5) were used across multiple PSO runs to explore diversity vs. convergence performance.

---

## 📊 Results Snapshot

| Model         | Accuracy | Notable Observation                |
|---------------|----------|-----------------------------------|
| Baseline NN   | 53.64%   | Manual tuning                     |
| Best PSO-NN   | 69.21%   | Achieved with 4 informants        |
| Avg PSO-NN    | 66.14%   | Mean accuracy over 3 PSO runs     |

**Best Hyperparameters Found**:
- [128, 77, 0.0035, tanh, 0.0, RMSprop, 20, use_third_layer=True]


**F1-Score Improvement (PSO vs Baseline)**:
- Loss: +0.116
- Draw: +0.263
- Win: +0.007

---

## 🛠 How to Run This Project

### 1. [Click here for GitHub repository](https://github.com/12231591/COIT29224-PSO-NN-Optimization)

### 2. Required Files
- [`football_match_predictor_pso.py`](./football_match_predictor_pso.py) — Full Python script with baseline and PSO models.
- `requirements.txt` — List of required Python libraries.

### 3. Setup Instructions

#### ▶ Option 1: Local VM with Visual Studio Code
Refer to [VM_Code_Guide.docx](https://github.com/12231591/COIT29224-PSO-NN-Optimization/blob/249a70d496293b603fd70bdbb3cf032e30c1355f/VM_Code_Guide.docx)

#### ▶ Option 2: Google Colab (Recommended for Simplicity)
Refer to [Google_Colab_Run_Guide.docx]([./Google_Colab_Run_Guide.docx](https://github.com/12231591/COIT29224-PSO-NN-Optimization/blob/249a70d496293b603fd70bdbb3cf032e30c1355f/Google_Colab_Run_Guide.docx))

#### ▶ Option 3: Kaggle (Recommended for Long Training)
Refer to [Running_PSO_NN_on_Kaggle.docx](https://github.com/12231591/COIT29224-PSO-NN-Optimization/blob/33309e2e7a68ea275f0463f6754bb117e13e6c7f/Running_PSO_NN_on_Kaggle.docx)

---

## 🔗 Key Links

- [Click here for GitHub repository](https://github.com/12231591/COIT29224-PSO-NN-Optimization)  
- [Click here for Kaggle football dataset](https://www.kaggle.com/datasets/hugomathien/soccer)  
- [Click here for Colab notebook](https://colab.research.google.com/)  
- [Click here for Kaggle Notebook 1](https://www.kaggle.com/code/your_notebook_1_link)  
- [Click here for Kaggle Notebook 2](https://www.kaggle.com/code/your_notebook_2_link)  

---

## 🧠 Discussion & Related Work

The PSO-based optimization significantly outperformed the manually tuned MLP baseline, especially on underperforming classes such as Draw. This approach aligns with previous studies such as:

- [MLP-PSO Hybrid Algorithm for Heart Disease Prediction](https://www.mdpi.com/2075-4426/12/8/1208)
- [Cardiovascular disease prediction with PSO-NN](https://www.kaggle.com/code/zzettrkalpakbal/cardiovascular-disease-prediction-with-pso-nn)

---

## 📚 References

Chollet, F. (2015) Keras. Available at: https://keras.io/  
Pedregosa, F. et al. (2011) Scikit-learn. https://scikit-learn.org/  
Abadi, M. et al. (2016) TensorFlow. https://www.tensorflow.org/  
Mathien, H. (2016) European Soccer Dataset. https://www.kaggle.com/datasets/hugomathien/soccer  
Google Research (n.d.) Colab. https://colab.research.google.com/  
PyGIS. Setting up a Python Environment. https://pygis.io/docs/b_getting_started.html  
Kalami, S.M. (n.d.) PSO in Python. https://github.com/smkalami/path-planning  
Zzettrkalpakbal. Cardiovascular Disease Prediction. https://www.kaggle.com/code/zzettrkalpakbal/  
Gupta, S. et al. (2022) MLP-PSO Hybrid Algorithm. https://www.mdpi.com/2075-4426/12/8/1208  
Wang, Z. et al. (2023) Double PSO-CatBoost Model. https://www.sciencedirect.com/science/article/pii/S0952197623008333  
Nurdiansyah, A. et al. (2020) PSO for Time Series. https://kinetik.umm.ac.id/index.php/kinetik/article/view/1330  
Khan, M. et al. (2022) PSO for Vehicle Classification. https://www.techscience.com/csse/v40n1/44231/html

---

**Note**: All documentation, visuals, guides, and performance screenshots are included in this repository for review and validation.


