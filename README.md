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

TensorBoard is used to monitor the training process.

The project records information such as:

* Training loss
* Validation loss
* Training accuracy
* Validation accuracy
* Model graphs
* Weight distributions
* Confusion matrices
* Embeddings/projector information where available

This makes it possible to inspect how the model behaves during training rather than looking only at the final accuracy.

---

## FastAPI

The trained model is exposed through a REST API.

Instead of requiring another developer to import Python modules directly, an external application can send HTTP requests to:

```text
POST /predict
```

The API also provides:

```text
GET /health
GET /ready
```

and automatically generates interactive API documentation.

---

## Joblib

The model does not operate directly on the original feature values.

The same `StandardScaler` used during training must also be used during inference.

Therefore the scaler is saved as an artifact:

```text
artifacts/scaler.joblib
```

This prevents the training and inference preprocessing steps from becoming inconsistent.

---

# Project Structure

```text
.
├── configs/
│   └── default.yaml
│
├── src/
│   └── iris_classifier/
│       ├── __init__.py
│       ├── data.py
│       ├── model.py
│       ├── train.py
│       ├── tensorboard_utils.py
│       ├── evaluate.py
│       ├── predict.py
│       ├── api.py
│       └── cli.py
│
├── scripts/
│   ├── ...
│   └── serve.py
│
├── tests/
│   └── ...
│
├── requirements.txt
├── requirements-dev.txt
├── pyproject.toml
├── docker-compose.yml
├── .gitignore
└── README.md
```

---

# ML Pipeline

## 1. Load the data

The project uses the standard Iris dataset.

Each sample contains four numerical features:

```text
sepal_length
sepal_width
petal_length
petal_width
```

---

## 2. Train/Test Split

The dataset is divided into training and testing data.

Stratification is used so that the class distribution remains representative between the splits.

---

## 3. Feature Scaling

The features are standardized using `StandardScaler`.

Conceptually:

```text
x_scaled = (x - mean) / standard_deviation
```

The scaler is fitted using the training data.

It is then saved so that exactly the same transformation can be applied when the FastAPI service receives new data.

---

## 4. TensorFlow Dataset

The processed data is converted into a TensorFlow-compatible data pipeline.

This keeps the training code separated from the raw dataset representation.

---

## 5. MLP Model

The classifier uses a feed-forward neural network implemented with Keras.

The architecture contains configurable hidden layers/units and dropout according to the YAML configuration.

The output layer produces probabilities for the three Iris classes.

---

## 6. Training

Training is handled by:

```text
src/iris_classifier/train.py
```

Training includes:

* Configurable hyperparameters
* TensorBoard logging
* Model checkpointing
* Validation monitoring
* Artifact generation

---

# Generated Artifacts

After training, the project generates model artifacts.

```text
artifacts/
├── iris_mlp.keras
├── saved_model/
├── scaler.joblib
└── metadata.json
```

### `iris_mlp.keras`

The trained Keras model.

### `saved_model/`

TensorFlow SavedModel representation for serving/deployment workflows.

### `scaler.joblib`

The `StandardScaler` fitted during training.

### `metadata.json`

Stores information about the training run, metrics and TensorBoard logging information.

---

# TensorBoard

Start TensorBoard with:

```powershell
tensorboard --logdir logs/tensorboard --port 6006
```

Then open:

```text
http://localhost:6006
```

The dashboard can contain:

* Scalars
* Graphs
* Histograms
* Images
* Projector embeddings

This provides visibility into the training process.

---

# Evaluation

After training:

```powershell
python -m iris_classifier.evaluate
```

The evaluation process generates:

```text
artifacts/eval_report.json
```

This provides a separate evaluation of the trained model on the test dataset.

---

# FastAPI Inference Service

The trained model can be exposed through FastAPI.

Start the service with:

```powershell
python -m iris_classifier.cli serve
```

Alternatively:

```powershell
python scripts/serve.py
```

Once running, the API will be available at:

```text
http://localhost:8000
```

Interactive API documentation:

```text
http://localhost:8000/docs
```

---

# API Endpoints

## Health

```text
GET /health
```

Used to check whether the application is running.

---

## Readiness

```text
GET /ready
```

Used to determine whether the required model artifacts are available and the service is ready to perform inference.

---

## Prediction

```text
POST /predict
```

Example request:

```json
{
  "samples": [
    {
      "sepal_length": 5.1,
      "sepal_width": 3.5,
      "petal_length": 1.4,
      "petal_width": 0.2
    }
  ]
}
```

The API processes the input using the saved scaler and sends the transformed features to the trained model.

---

# Local Setup

## Requirements

Recommended:

* Python 3.x
* Git
* Optional: Docker Desktop

---

## 1. Clone the repository

```powershell
git clone YOUR_GITHUB_REPOSITORY_URL
cd iris-tensorflow-classifier
```

Replace `YOUR_GITHUB_REPOSITORY_URL` with the actual repository URL.

---

## 2. Create a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
```

Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks script execution:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Then:

```powershell
.\.venv\Scripts\Activate.ps1
```

---

## 3. Install dependencies

```powershell
python -m pip install --upgrade pip
```

```powershell
pip install -r requirements-dev.txt
```

Install the project in editable mode:

```powershell
pip install -e .
```

---

# Train the Model

Run:

```powershell
python -m iris_classifier.train
```

Training will generate the required model and preprocessing artifacts.

---

# Evaluate

Run:

```powershell
python -m iris_classifier.evaluate
```

---

# Start TensorBoard

In another terminal:

```powershell
tensorboard --logdir logs/tensorboard --port 6006
```

Open:

```text
http://localhost:6006
```

---

# Start the API

After completing training:

```powershell
python -m iris_classifier.cli serve
```

Open:

```text
http://localhost:8000/docs
```

---

# Run Tests

```powershell
pytest -q
```

The tests verify important parts of the application without requiring manual testing of every component.

---

# Docker

The project also contains Docker configuration for reproducible execution.

Training:

```powershell
docker compose up --build train
```

API and TensorBoard:

```powershell
docker compose up api tensorboard
```

Services:

```text
FastAPI      → http://localhost:8000
Swagger      → http://localhost:8000/docs
TensorBoard  → http://localhost:6006
```

---

# Configuration

Training parameters are stored in:

```text
configs/default.yaml
```

This allows parameters such as:

* Number of epochs
* Hidden-layer configuration
* Dropout
* Learning rate
* Batch size
* Artifact locations

to be changed without modifying the training source code.

---

# Reproducibility

The project separates:

```text
Configuration
     +
Data processing
     +
Model definition
     +
Training
     +
Artifacts
     +
Inference
```

This makes the training and serving process easier to reproduce and maintain.

The preprocessing scaler is persisted alongside the model so that inference uses the same feature transformation as training.

---

# What This Project Demonstrates

This project demonstrates practical knowledge of:

### Machine Learning

* Classification
* Neural networks
* Train/test splitting
* Stratification
* Feature scaling
* Model evaluation

### Deep Learning

* TensorFlow
* Keras
* Multi-layer perceptrons
* Training callbacks
* Checkpointing

### MLOps / ML Engineering

* Configuration-driven training
* Experiment tracking
* Model artifacts
* Reproducible preprocessing
* Model evaluation
* Structured project architecture

### Backend Development

* FastAPI
* REST APIs
* Model serving
* Health/readiness endpoints
* OpenAPI/Swagger documentation

### Software Engineering

* Python package structure
* Automated testing
* Docker
* Environment isolation
* Git version control

---

# Why Iris?

The Iris dataset is deliberately simple.

A complex dataset would make it difficult to distinguish whether problems came from the model, the data, or the infrastructure.

Because Iris is small and well understood, the project can focus on demonstrating the **engineering lifecycle of a machine-learning system**:

```text
Train
  ↓
Track
  ↓
Evaluate
  ↓
Save
  ↓
Serve
  ↓
Test
```

The same engineering principles can later be applied to larger and more complex ML systems.

---

# Future Improvements

Possible future extensions include:

* CI/CD using GitHub Actions
* Model versioning
* ML experiment tracking with MLflow
* Cloud deployment
* Authentication for the inference API
* Monitoring inference requests
* Model performance monitoring
* Automated retraining
* Container registry deployment
* Kubernetes deployment

---

# License

This project is available under the license included in this repository.

---

## Author

**Atul Gaud**

Engineering student focused on Machine Learning, AI and software development.

---