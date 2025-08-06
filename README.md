---
title: End-to-End Automated MLOps Framework
colorFrom: gray
colorTo: gray
sdk: gradio
sdk_version: 5.41.1
app_file: app.py
pinned: false
license: mit
short_description: End-to-End Automated MLOps Framework
---

# End-to-End Automated MLOps Framework

**Author**: Spencer Purdy

This project is a comprehensive, enterprise-grade MLOps platform that demonstrates a complete, automated lifecycle for machine learning models. It handles everything from automated training and hyperparameter optimization to versioning, production deployment, drift detection, A/B testing, and ongoing performance monitoring.

The entire system is orchestrated by a central engine and managed through a powerful, multi-tab Gradio interface, providing a single pane of glass for all MLOps activities.

## Core Features

* **Automated Model Training**: The system features a `ModelTrainer` that automatically trains a custom PyTorch neural network on tabular data. It includes support for handling class imbalance with SMOTE and integrates `Optuna` for sophisticated hyperparameter optimization.
* **Model Registry and Versioning**: A robust `ModelRegistry` tracks all trained model versions, their performance metrics, and metadata. Models are persisted to disk and logged in a SQLite database, with functionality to promote any version to the "production" stage.
* **Data and Concept Drift Detection**: The platform integrates both `Evidently` and `Alibi-Detect` (with a statistical fallback) to continuously monitor for data drift between the reference training data and live inference data. Drift scores are tracked over time.
* **Automated Retraining**: A background process can be enabled to periodically check for significant data drift. If the drift threshold is exceeded, it automatically triggers a new model training cycle and initiates an A/B test against the current production model.
* **Live A/B Testing**: The `ABTestManager` allows for controlled experiments between the current production model and a challenger. It routes inference traffic, records performance metrics for both models, and determines a statistical winner.
* **Comprehensive Monitoring & Cost Tracking**:
    * **Performance**: The `PerformanceMonitor` uses Prometheus-compatible metrics to track prediction latency, accuracy, and throughput. It also logs detailed performance data to a database for historical analysis.
    * **Cost**: The `CostTracker` provides reports on estimated operational costs, breaking them down by training, inference, and model storage based on configurable rates.
* **Model Cards and Explainability**: The system can generate detailed model cards that consolidate metadata, performance metrics, and operational history. It also has `SHAP` integrated as a dependency for future explainability features.
* **Hugging Face Hub Integration**: Models can be exported directly from the registry to the Hugging Face Hub, with an automatically generated model card (`README.md`).

## How It Works

The platform operates as a cohesive system of specialized components orchestrated by the main `MLOpsEngine`:

1.  **Training**: A user initiates a training job from the UI. The `ModelTrainer` uses `Optuna` to find the best hyperparameters and then trains a `CustomNeuralNetwork` model.
2.  **Registration**: The newly trained model, along with its performance metrics and metadata, is registered in the `ModelRegistry`. The model artifact is saved, and its details are recorded in the SQLite database.
3.  **Promotion**: A user can review all registered models and promote a specific version to be the active "production" model via the UI.
4.  **Prediction**: When a prediction request is made, the engine retrieves the current production model (or routes to an A/B test model if active) to perform inference. Latency and other performance metrics are logged by the `PerformanceMonitor`.
5.  **Monitoring & Drift Detection**: In the background, the `DriftDetector` continuously compares incoming data against a reference dataset. If drift is detected and auto-retraining is enabled, it triggers the training of a new "challenger" model.
6.  **A/B Testing**: The new challenger model is automatically placed into an A/B test against the current production model. Live traffic is split between them until a statistically significant winner is found, which can then be automatically promoted.

## Technical Stack

* **Machine Learning & Training**: scikit-learn, PyTorch, imbalanced-learn
* **MLOps & Experiment Tracking**: MLflow, Optuna, Hugging Face Hub, W&B
* **Drift & Anomaly Detection**: Evidently, Alibi-Detect, SHAP
* **Web Interface & Visualization**: Gradio, Matplotlib, Seaborn, Plotly, Yellowbrick
* **Infrastructure & Utilities**: Prometheus Client, Joblib, SQLite

## How to Use the Demo

The Gradio interface is organized into tabs that follow a logical MLOps workflow.

1.  **Train a Model**: Navigate to the **Model Training** tab, select the number of training samples, and click **Train New Model**. This will create the first version in the registry.
2.  **Manage Models**: Go to the **Model Registry** tab. Click **Refresh Model List** to see all trained models. Select a version from the dropdown and click **Promote to Production** to make it active.
3.  **Make Predictions**: In the **Make Predictions** tab, enter values for the features and click **Predict**. The result from the current production model will be displayed.
4.  **Detect Drift**: Go to the **Drift Detection** tab and click **Check for Data Drift** to simulate checking a new batch of data against the original training data.
5.  **Run an A/B Test**: In the **A/B Testing** tab, click **Start New A/B Test**. This will train a new challenger model and run it against the current production model. To generate results, make several predictions in the "Make Predictions" tab with the "Use A/B Test" checkbox ticked.
6.  **Monitor Performance**: Check the **Performance Monitoring** and **Cost Tracking** tabs to see live operational dashboards for the system.

## Disclaimer

This project is an advanced demonstration of MLOps principles and is intended for educational and portfolio purposes. It uses synthetically generated data for its training and drift detection processes. While built to be robust, it is not intended for direct use in a live production environment without extensive testing and validation.