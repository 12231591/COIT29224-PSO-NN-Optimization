# COIT29224 – Assessment 1: Enhancing Neural Network Performance using PSO

**Student Name:** Harshvardhan Gosaliya  
**Student ID:** 12231591  
**Term:** T1 2025  
**Course:** COIT29224 – Evolutionary Computation  
**Assignment Title:** Enhancing Neural Network Performance with Particle Swarm Optimization (PSO)

---

## 📘 Project Overview

This project demonstrates how Particle Swarm Optimization (PSO) can be effectively used to enhance the performance of a neural network by optimizing its hyperparameters. The PSO-NN model is tested against a traditionally tuned neural network to showcase its superiority in classification performance.

The application domain selected is **football match outcome prediction**, using a dataset extracted from a SQLite database containing match statistics.

---

## 🎯 Problem Domain & Objectives

- **Domain**: Sports analytics – Football match outcome classification.
- **Objective**: Predict match results (Win, Draw, Loss) for home teams using team attributes.
- **Success Metric**: Classification Accuracy and F1-Score across 3 classes.

---

## 🗃 Dataset & Pre-processing

- Source: `database.sqlite` containing `Match` and `Team_Attributes` tables.
- Selected Features: build-up play speed, defense aggression, match year, and league info.
- Label Classes:  
  - `0`: Loss  
  - `1`: Draw  
  - `2`: Win  
- Missing values dropped. Features normalized using `StandardScaler`.

---

## 🔧 Models Implemented

### 🔹 Baseline Neural Network
- 2 hidden layers (32 neurons each), ReLU, Adam optimizer, Dropout = 0.3.
- Manual hyperparameter tuning with fixed values.

### 🔹 PSO-Optimized Neural Network (PSO-NN)
- Hyperparameters tuned using PSO:
  - Neurons (n1, n2)
  - Learning rate
  - Activation function (`relu`, `tanh`, `selu`)
  - Dropout rate
  - Optimizer (`Adam`, `SGD`, `RMSprop`, `Nadam`)
  - Batch size
  - Third hidden layer (on/off)
- Informant variations tested: 3, 4, and 5.

---

## 📊 Evaluation & Results

- **Metrics Used**:
  - Accuracy
  - F1-Score
  - Confusion Matrix
  - PSO Convergence Plot
  - Runtime per PSO run
- **Best Model Accuracy**: 
- **Baseline Accuracy**: 
- **F1-score improvement** observed in all three classes with PSO over Baseline.
- Early stopping implemented to avoid overfitting.

---

## 🖼 Screenshots (in `/images` folder)
- Confusion matrix comparisons
- PSO convergence plots
- F1-score bar charts

---

## 📁 Repository Structure

