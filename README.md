# Iris TensorFlow Classifier

An end-to-end, production-style machine learning project for Iris flower classification using **TensorFlow/Keras**, **scikit-learn**, **TensorBoard**, and **FastAPI**.

The project demonstrates the complete ML workflow rather than only training a model:

**data preparation → preprocessing → neural-network training → experiment tracking → model artifacts → evaluation → REST API inference → testing → Docker**

---

## Project Overview

The Iris dataset contains measurements of Iris flowers and their corresponding species.

For each flower, the model receives four numerical features:

* Sepal length
* Sepal width
* Petal length
* Petal width

The system predicts one of three Iris species:

* Setosa
* Versicolor
* Virginica

The dataset itself is intentionally simple. The main purpose of this project is not to solve a difficult classification problem, but to demonstrate how a machine-learning model can be structured as a reproducible and deployable application.

Instead of keeping everything inside a single notebook, the project separates data processing, model creation, training, evaluation, inference, configuration, API serving, and testing.

---

# Architecture

```text
                    Iris Dataset
                         │
                         ▼
              ┌─────────────────────┐
              │    Data Pipeline    │
              │                     │
              │ • Load dataset      │
              │ • Train/test split  │
              │ • Stratification    │
              │ • Feature scaling   │
              └──────────┬──────────┘
                         │
                         ▼
                TensorFlow Dataset
                         │
                         ▼
              ┌─────────────────────┐
              │   Keras MLP Model   │
              │                     │
              │ • Dense layers      │
              │ • Dropout           │
              │ • Classification    │
              └──────────┬──────────┘
                         │
                         ▼
                    Training
                         │
              ┌──────────┴──────────┐
              │                     │
              ▼                     ▼
        Model Artifacts        TensorBoard
              │                     │
              │              • Loss/accuracy
              │              • Validation metrics
              │              • Graphs
              │              • Histograms
              │              • Confusion matrices
              │
              ▼
       FastAPI Inference API
              │
              ▼
        POST /predict
              │
              ▼
       Iris Classification
```

---

# Why This Project?

A basic Iris project can be completed with only a few lines of Python.

That is not the goal here.

This project was structured to demonstrate what happens **after the model is created**.

A real machine-learning system needs to answer questions such as:

* How is the data processed?
* How are training and test data separated?
* How is preprocessing reproduced during inference?
* Where is the trained model stored?
* How are experiments monitored?
* How is model performance evaluated?
* How can another application use the model?
* How can the project be tested?
* How can the application be reproduced on another machine?

This project addresses those concerns with a modular Python package, YAML configuration, TensorBoard logging, persisted preprocessing artifacts, automated tests, FastAPI serving, and Docker support.

---

# Technology Stack

| Technology         | Purpose                                          |
| ------------------ | ------------------------------------------------ |
| Python             | Core programming language                        |
| TensorFlow / Keras | Neural-network development and training          |
| scikit-learn       | Dataset handling, preprocessing and evaluation   |
| NumPy              | Numerical operations                             |
| YAML               | External project configuration                   |
| TensorBoard        | Training monitoring and experiment visualization |
| Joblib             | Saving and loading the trained scaler            |
| FastAPI            | REST API for model inference                     |
| Uvicorn            | ASGI server for FastAPI                          |
| Pytest             | Automated testing                                |
| Docker             | Reproducible application environment             |
| Git                | Version control                                  |

---

# Why These Technologies?

## TensorFlow / Keras

TensorFlow/Keras is used to build and train the multi-layer perceptron (MLP).

The model is intentionally small because the Iris dataset is small. The purpose is to demonstrate the complete neural-network lifecycle rather than requiring a complex architecture.

---

## Scikit-learn

Scikit-learn provides useful machine-learning utilities around the TensorFlow model.

It is used for tasks such as:

* Loading the Iris dataset
* Train/test splitting
* Stratification
* Feature preprocessing
* StandardScaler
* Evaluation utilities

TensorFlow handles the neural-network portion while scikit-learn handles traditional preprocessing utilities.

---

## YAML

The project keeps configurable values outside the Python source code.

For example:

```yaml
epochs:
hidden_units:
dropout:
learning_rate:
batch_size:
```

This means training configuration can be changed without modifying the training implementation.

This is cleaner and more maintainable than hardcoding every hyperparameter inside Python.

---

## TensorBoard

TensorBoard is used to monitor the trai
