# -*- coding: utf-8 -*-
"""
End-to-End Automated MLOps Framework
Author: Spencer Purdy
Description: Enterprise-grade MLOps platform with automated model training, versioning,
drift detection, A/B testing, and deployment capabilities
Features: Custom model training, automatic retraining, model versioning, drift detection,
A/B testing, model cards, performance monitoring, cost tracking, HuggingFace deployment
"""

# Installation
# !pip install -q numpy pandas scikit-learn torch matplotlib seaborn plotly mlflow optuna shap imbalanced-learn yellowbrick jsonschema pyyaml huggingface-hub safetensors accelerate wandb evidently alibi-detect prometheus-client joblib requests Pillow python-dotenv gradio scipy

import os
import json
import yaml
import time
import hashlib
import pickle
import shutil
import logging
import warnings
import requests
import sqlite3
import threading
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional, Any, Union
from dataclasses import dataclass, field, asdict
from collections import defaultdict, deque
from pathlib import Path
import tempfile
from abc import ABC, abstractmethod
from contextlib import contextmanager

# Data processing and ML
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE

# Deep Learning
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# MLOps tools
import mlflow
import mlflow.pytorch
import optuna
import shap

# Drift detection imports
try:
    from evidently import ColumnMapping
    from evidently.report import Report
    from evidently.metrics import DataDriftTable, DataQualityMetric
    EVIDENTLY_AVAILABLE = True
except ImportError:
    print("Warning: Evidently imports failed. Using fallback drift detection.")
    EVIDENTLY_AVAILABLE = False
    Report = None
    DataDriftTable = None
    DataQualityMetric = None

try:
    from alibi_detect.cd import TabularDrift
    ALIBI_AVAILABLE = True
except ImportError:
    print("Warning: Alibi-detect imports failed. Using fallback drift detection.")
    ALIBI_AVAILABLE = False
    TabularDrift = None

# Hugging Face imports
from huggingface_hub import HfApi, create_repo, upload_file

# UI and utilities
import gradio as gr
from prometheus_client import Counter, Gauge, Histogram, generate_latest
import joblib
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio

# Configure logging
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# System Configuration
@dataclass
class MLOpsConfig:
    """Configuration for the MLOps system"""
    # Model settings
    model_name: str = "customer_churn_predictor"
    model_version: str = "1.0.0"
    task_type: str = "binary_classification"

    # Training settings
    batch_size: int = 32
    epochs: int = 50
    learning_rate: float = 0.001
    early_stopping_patience: int = 10
    validation_split: float = 0.2

    # MLOps settings
    experiment_tracking: bool = True
    model_registry: bool = True
    drift_detection_threshold: float = 0.05
    retraining_threshold: float = 0.1
    auto_retrain_enabled: bool = True
    auto_retrain_interval_hours: int = 24

    # Performance thresholds
    min_accuracy: float = 0.85
    max_latency_ms: float = 100

    # Cost tracking
    training_cost_per_hour: float = 0.50
    inference_cost_per_1k: float = 0.01
    storage_cost_per_gb_month: float = 0.10

    # Versioning
    version_control_backend: str = "local"
    model_registry_uri: str = "./model_registry"

    # A/B Testing
    ab_test_traffic_split: float = 0.5
    ab_test_min_samples: int = 100
    ab_test_confidence_level: float = 0.95

    # Monitoring
    monitoring_window_size: int = 1000
    alert_email: Optional[str] = None
    alert_threshold_consecutive_failures: int = 5

    # Paths
    data_path: str = "./data"
    models_path: str = "./models"
    reports_path: str = "./reports"
    db_path: str = "./mlops.db"

    # Feature settings
    input_features: List[str] = field(default_factory=lambda: [
        'feature_1', 'feature_2', 'feature_3', 'feature_4', 'feature_5',
        'feature_6', 'feature_7', 'feature_8', 'feature_9', 'feature_10'
    ])
    target_column: str = 'target'

config = MLOpsConfig()

# Create necessary directories
for path in [config.data_path, config.models_path, config.reports_path]:
    os.makedirs(path, exist_ok=True)

# Initialize MLflow
if config.experiment_tracking:
    mlflow.set_tracking_uri(config.model_registry_uri)
    mlflow.set_experiment(config.model_name)

# Metrics for monitoring
prediction_counter = Counter('model_predictions_total', 'Total predictions made')
prediction_latency = Histogram('model_prediction_duration_seconds', 'Prediction latency')
model_accuracy_gauge = Gauge('model_accuracy', 'Current model accuracy')
drift_score_gauge = Gauge('model_drift_score', 'Current drift score')
training_duration_gauge = Gauge('model_training_duration_seconds', 'Last training duration')
model_size_gauge = Gauge('model_size_bytes', 'Model size in bytes')

class DatabaseManager:
    """Manages persistent storage for MLOps system with connection pooling"""

    def __init__(self, db_path: str, pool_size: int = 5):
        self.db_path = db_path
        self.pool_size = pool_size
        self._local = threading.local()
        self.init_database()

    @contextmanager
    def get_connection(self):
        """Get a database connection with context management"""
        if not hasattr(self._local, 'connection') or self._local.connection is None:
            self._local.connection = sqlite3.connect(self.db_path, check_same_thread=False)
            self._local.connection.row_factory = sqlite3.Row

        try:
            yield self._local.connection
        except Exception as e:
            self._local.connection.rollback()
            raise e
        else:
            self._local.connection.commit()

    def init_database(self):
        """Initialize database tables"""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # Model registry table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS model_registry (
                    version_id TEXT PRIMARY KEY,
                    model_path TEXT,
                    metrics TEXT,
                    metadata TEXT,
                    created_at TIMESTAMP,
                    is_production BOOLEAN DEFAULT FALSE,
                    model_size_bytes INTEGER,
                    training_duration_seconds REAL
                )
            ''')

            # Cost tracking table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS cost_tracking (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    category TEXT,
                    amount REAL,
                    timestamp TIMESTAMP,
                    details TEXT,
                    model_version TEXT
                )
            ''')

            # Performance metrics table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_version TEXT,
                    metric_name TEXT,
                    metric_value REAL,
                    timestamp TIMESTAMP,
                    prediction_count INTEGER
                )
            ''')

            # A/B test results table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS ab_test_results (
                    experiment_id TEXT PRIMARY KEY,
                    model_a_version TEXT,
                    model_b_version TEXT,
                    model_a_performance REAL,
                    model_b_performance REAL,
                    winner TEXT,
                    confidence_level REAL,
                    sample_size INTEGER,
                    results TEXT,
                    created_at TIMESTAMP,
                    completed_at TIMESTAMP
                )
            ''')

            # Drift detection logs table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS drift_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_version TEXT,
                    drift_type TEXT,
                    drift_score REAL,
                    is_drift BOOLEAN,
                    feature_drifts TEXT,
                    timestamp TIMESTAMP
                )
            ''')

            # Prediction logs table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS prediction_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_version TEXT,
                    input_features TEXT,
                    prediction REAL,
                    confidence REAL,
                    latency_ms REAL,
                    timestamp TIMESTAMP
                )
            ''')

            # Training history table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS training_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_version TEXT,
                    dataset_hash TEXT,
                    hyperparameters TEXT,
                    final_metrics TEXT,
                    training_curves TEXT,
                    timestamp TIMESTAMP
                )
            ''')

    def execute_query(self, query: str, params: Tuple = None) -> List:
        """Execute a query and return results"""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)

            return cursor.fetchall()

    def insert_record(self, table: str, data: Dict) -> int:
        """Insert a record into specified table and return last row id"""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            columns = ', '.join(data.keys())
            placeholders = ', '.join(['?' for _ in data])
            query = f"INSERT INTO {table} ({columns}) VALUES ({placeholders})"

            cursor.execute(query, tuple(data.values()))
            return cursor.lastrowid

    def update_record(self, table: str, data: Dict, condition: str, params: Tuple) -> None:
        """Update records in specified table"""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            set_clause = ', '.join([f"{k} = ?" for k in data.keys()])
            query = f"UPDATE {table} SET {set_clause} WHERE {condition}"

            cursor.execute(query, tuple(data.values()) + params)

class CustomDataset(Dataset):
    """Custom PyTorch dataset for tabular data"""

    def __init__(self, features: np.ndarray, labels: np.ndarray,
                 transform=None, feature_names: List[str] = None):
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels)
        self.transform = transform
        self.feature_names = feature_names or [f"feature_{i}" for i in range(features.shape[1])]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        features = self.features[idx]
        label = self.labels[idx]

        if self.transform:
            features = self.transform(features)

        return features, label

class CustomNeuralNetwork(nn.Module):
    """Custom neural network architecture for tabular data"""

    def __init__(self, input_dim: int, hidden_dims: List[int] = None,
                 output_dim: int = 1, dropout_rate: float = 0.3):
        super(CustomNeuralNetwork, self).__init__()

        if hidden_dims is None:
            hidden_dims = [128, 64, 32]

        self.input_dim = input_dim
        self.output_dim = output_dim

        layers = []
        prev_dim = input_dim

        # Build hidden layers with batch normalization and dropout
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))

        if output_dim == 1:  # Binary classification
            layers.append(nn.Sigmoid())

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

    def predict_proba(self, x):
        """Get prediction probabilities"""
        self.eval()
        with torch.no_grad():
            if isinstance(x, np.ndarray):
                x = torch.FloatTensor(x)
            output = self.forward(x)
            if self.output_dim == 1:
                # Binary classification
                proba = torch.cat([1 - output, output], dim=1)
            else:
                proba = torch.softmax(output, dim=1)
        return proba.numpy()

class ModelVersion:
    """Represents a model version with associated metadata"""

    def __init__(self, version_id: str, model: Any, metrics: Dict[str, float],
                 metadata: Dict[str, Any], model_path: str = None):
        self.version_id = version_id
        self.model = model
        self.metrics = metrics
        self.metadata = metadata
        self.model_path = model_path
        self.created_at = datetime.now()
        self.deployment_count = 0
        self.last_prediction_time = None
        self.prediction_count = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert model version to dictionary for persistence"""
        return {
            'version_id': self.version_id,
            'metrics': self.metrics,
            'metadata': self.metadata,
            'created_at': self.created_at.isoformat(),
            'deployment_count': self.deployment_count,
            'prediction_count': self.prediction_count,
            'model_path': self.model_path
        }

class ModelRegistry:
    """Model registry with persistent storage and versioning"""

    def __init__(self, base_path: str = "./model_registry", db_manager: DatabaseManager = None):
        self.base_path = Path(base_path)
        self.base_path.mkdir(exist_ok=True)
        self.db_manager = db_manager or DatabaseManager(config.db_path)
        self.versions = {}
        self.current_version = None
        self.load_registry()

    def register_model(self, model: Any, metrics: Dict[str, float],
                      metadata: Dict[str, Any], training_duration: float = 0) -> str:
        """Register a new model version with persistent storage"""
        version_id = self._generate_version_id()

        # Save model to disk
        model_path = self.base_path / f"model_{version_id}.pkl"
        if hasattr(model, 'state_dict'):
            torch.save({
                'state_dict': model.state_dict(),
                'model_config': {
                    'input_dim': model.input_dim if hasattr(model, 'input_dim') else None,
                    'output_dim': model.output_dim if hasattr(model, 'output_dim') else None
                }
            }, model_path)
        else:
            joblib.dump(model, model_path)

        # Calculate model size
        model_size = os.path.getsize(model_path)

        # Create version object
        version = ModelVersion(version_id, model, metrics, metadata, str(model_path))

        # Save metadata to disk
        metadata_path = self.base_path / f"metadata_{version_id}.json"
        with open(metadata_path, 'w') as f:
            json.dump(version.to_dict(), f, indent=2)

        # Save to database
        self.db_manager.insert_record('model_registry', {
            'version_id': version_id,
            'model_path': str(model_path),
            'metrics': json.dumps(metrics),
            'metadata': json.dumps(metadata),
            'created_at': datetime.now(),
            'is_production': False,
            'model_size_bytes': model_size,
            'training_duration_seconds': training_duration
        })

        self.versions[version_id] = version
        logger.info(f"Registered model version: {version_id}")

        # Update metrics
        model_size_gauge.set(model_size)

        return version_id

    def _generate_version_id(self) -> str:
        """Generate unique version identifier"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"v_{timestamp}_{len(self.versions) + 1}"

    def get_model(self, version_id: str = None) -> Optional[ModelVersion]:
        """Retrieve a specific model version"""
        if version_id is None:
            version_id = self.current_version

        if version_id and version_id not in self.versions:
            # Try loading from database
            results = self.db_manager.execute_query(
                "SELECT * FROM model_registry WHERE version_id = ?",
                (version_id,)
            )
            if results:
                self._load_model_from_db(results[0])

        return self.versions.get(version_id)

    def promote_model(self, version_id: str) -> None:
        """Promote a model version to production"""
        if version_id in self.versions:
            # Update database to mark as production
            self.db_manager.execute_query(
                "UPDATE model_registry SET is_production = FALSE WHERE is_production = TRUE"
            )
            self.db_manager.update_record(
                'model_registry',
                {'is_production': True},
                'version_id = ?',
                (version_id,)
            )

            self.current_version = version_id
            self.versions[version_id].deployment_count += 1
            logger.info(f"Promoted model version {version_id} to production")

    def load_registry(self) -> None:
        """Load registry state from database"""
        results = self.db_manager.execute_query(
            "SELECT * FROM model_registry ORDER BY created_at DESC"
        )

        for row in results:
            self._load_model_from_db(row)

        # Find current production model
        prod_results = self.db_manager.execute_query(
            "SELECT version_id FROM model_registry WHERE is_production = TRUE"
        )
        if prod_results:
            self.current_version = prod_results[0][0]

    def _load_model_from_db(self, row: Tuple) -> None:
        """Load model information from database row"""
        version_id = row[0]
        model_path = row[1]
        metrics = json.loads(row[2])
        metadata = json.loads(row[3])

        # Load model from disk
        model = None
        if os.path.exists(model_path):
            if model_path.endswith('.pkl'):
                try:
                    checkpoint = torch.load(model_path, map_location='cpu')
                    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                        # PyTorch model
                        model_config = checkpoint.get('model_config', {})
                        model = CustomNeuralNetwork(
                            input_dim=model_config.get('input_dim', 10),
                            output_dim=model_config.get('output_dim', 1)
                        )
                        model.load_state_dict(checkpoint['state_dict'])
                    else:
                        model = joblib.load(model_path)
                except:
                    model = joblib.load(model_path)

        # Create ModelVersion object
        version = ModelVersion(version_id, model, metrics, metadata, model_path)
        version.created_at = datetime.fromisoformat(str(row[4]))
        self.versions[version_id] = version

class DriftDetector:
    """Handles data drift detection using multiple methods"""

    def __init__(self, reference_data: np.ndarray, config: MLOpsConfig,
                 feature_names: List[str] = None):
        self.reference_data = reference_data
        self.config = config
        self.feature_names = feature_names or [f"feature_{i}" for i in range(reference_data.shape[1])]
        self.drift_threshold = config.drift_detection_threshold

        # Initialize drift detectors based on availability
        self.detectors = self._initialize_detectors()

    def _initialize_detectors(self) -> Dict[str, Any]:
        """Initialize available drift detectors"""
        detectors = {}

        # Statistical drift detector (always available)
        detectors['statistical'] = self._create_statistical_detector()

        # Alibi-detect drift detector
        if ALIBI_AVAILABLE and TabularDrift is not None:
            try:
                detectors['alibi'] = TabularDrift(
                    self.reference_data,
                    p_val=self.drift_threshold,
                    categories_per_feature={}
                )
            except Exception as e:
                logger.warning(f"Failed to initialize Alibi drift detector: {e}")

        return detectors

    def _create_statistical_detector(self):
        """Create a simple statistical drift detector"""
        return {
            'mean': np.mean(self.reference_data, axis=0),
            'std': np.std(self.reference_data, axis=0),
            'min': np.min(self.reference_data, axis=0),
            'max': np.max(self.reference_data, axis=0)
        }

    def detect_drift(self, current_data: np.ndarray) -> Dict[str, Any]:
        """Detect drift in current data compared to reference data"""
        results = {
            'is_drift': False,
            'drift_score': 0.0,
            'feature_drifts': {},
            'method': 'statistical'
        }

        # Try Alibi-detect first if available
        if 'alibi' in self.detectors:
            try:
                drift_pred = self.detectors['alibi'].predict(current_data)
                results['is_drift'] = bool(drift_pred['data']['is_drift'])
                results['drift_score'] = float(drift_pred['data']['p_val'])
                results['method'] = 'alibi'

                # Feature-level drift
                if 'feature_score' in drift_pred['data']:
                    for i, score in enumerate(drift_pred['data']['feature_score']):
                        results['feature_drifts'][self.feature_names[i]] = float(score)

                return results
            except Exception as e:
                logger.warning(f"Alibi drift detection failed: {e}")

        # Fallback to statistical drift detection
        current_mean = np.mean(current_data, axis=0)
        current_std = np.std(current_data, axis=0)

        # Calculate normalized differences
        mean_diff = np.abs(current_mean - self.detectors['statistical']['mean'])
        mean_diff_normalized = mean_diff / (self.detectors['statistical']['std'] + 1e-7)

        # Overall drift score (mean of normalized differences)
        drift_score = np.mean(mean_diff_normalized)
        results['drift_score'] = float(drift_score)
        results['is_drift'] = drift_score > self.drift_threshold

        # Feature-level drift
        for i, feature_name in enumerate(self.feature_names):
            results['feature_drifts'][feature_name] = float(mean_diff_normalized[i])

        return results

    def generate_drift_report(self, current_data: np.ndarray) -> str:
        """Generate a detailed drift report"""
        drift_results = self.detect_drift(current_data)

        report = f"Data Drift Report\n"
        report += f"{'=' * 50}\n"
        report += f"Overall Drift Detected: {drift_results['is_drift']}\n"
        report += f"Drift Score: {drift_results['drift_score']:.4f}\n"
        report += f"Detection Method: {drift_results['method']}\n\n"

        report += f"Feature-level Drift Scores:\n"
        report += f"{'-' * 30}\n"

        for feature, score in drift_results['feature_drifts'].items():
            status = "DRIFT" if score > self.drift_threshold else "OK"
            report += f"{feature}: {score:.4f} [{status}]\n"

        return report

class CostTracker:
    """Tracks and manages costs for the MLOps system"""

    def __init__(self, config: MLOpsConfig, db_manager: DatabaseManager):
        self.config = config
        self.db_manager = db_manager
        self.current_costs = defaultdict(float)

    def track_training_cost(self, duration_seconds: float, model_version: str) -> float:
        """Track training costs"""
        hours = duration_seconds / 3600
        cost = hours * self.config.training_cost_per_hour

        self.db_manager.insert_record('cost_tracking', {
            'category': 'training',
            'amount': cost,
            'timestamp': datetime.now(),
            'details': f'Training duration: {duration_seconds:.2f}s',
            'model_version': model_version
        })

        self.current_costs['training'] += cost
        return cost

    def track_inference_cost(self, num_predictions: int, model_version: str) -> float:
        """Track inference costs"""
        cost = (num_predictions / 1000) * self.config.inference_cost_per_1k

        self.db_manager.insert_record('cost_tracking', {
            'category': 'inference',
            'amount': cost,
            'timestamp': datetime.now(),
            'details': f'Predictions: {num_predictions}',
            'model_version': model_version
        })

        self.current_costs['inference'] += cost
        return cost

    def track_storage_cost(self, size_gb: float, model_version: str) -> float:
        """Track storage costs"""
        cost = size_gb * self.config.storage_cost_per_gb_month

        self.db_manager.insert_record('cost_tracking', {
            'category': 'storage',
            'amount': cost,
            'timestamp': datetime.now(),
            'details': f'Storage: {size_gb:.2f}GB',
            'model_version': model_version
        })

        self.current_costs['storage'] += cost
        return cost

    def get_cost_report(self, days: int = 30) -> Dict[str, Any]:
        """Generate cost report for the specified period"""
        start_date = datetime.now() - timedelta(days=days)

        query = """
            SELECT category, SUM(amount) as total, COUNT(*) as count
            FROM cost_tracking
            WHERE timestamp > ?
            GROUP BY category
        """

        results = self.db_manager.execute_query(query, (start_date,))

        report = {
            'period_days': days,
            'categories': {},
            'total': 0
        }

        for row in results:
            category = row[0]
            total = row[1]
            count = row[2]

            report['categories'][category] = {
                'total': total,
                'count': count,
                'average': total / count if count > 0 else 0
            }
            report['total'] += total

        return report

class ABTestManager:
    """Manages A/B testing for model comparisons"""

    def __init__(self, config: MLOpsConfig, db_manager: DatabaseManager):
        self.config = config
        self.db_manager = db_manager
        self.active_experiments = {}

    def create_experiment(self, model_a_version: str, model_b_version: str,
                         experiment_name: str = None) -> str:
        """Create a new A/B test experiment"""
        experiment_id = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        if experiment_name:
            experiment_id = f"{experiment_id}_{experiment_name}"

        experiment = {
            'experiment_id': experiment_id,
            'model_a_version': model_a_version,
            'model_b_version': model_b_version,
            'model_a_performance': [],
            'model_b_performance': [],
            'model_a_count': 0,
            'model_b_count': 0,
            'created_at': datetime.now(),
            'completed': False
        }

        self.active_experiments[experiment_id] = experiment

        # Save to database
        self.db_manager.insert_record('ab_test_results', {
            'experiment_id': experiment_id,
            'model_a_version': model_a_version,
            'model_b_version': model_b_version,
            'model_a_performance': 0,
            'model_b_performance': 0,
            'winner': None,
            'confidence_level': 0,
            'sample_size': 0,
            'results': json.dumps({}),
            'created_at': datetime.now(),
            'completed_at': None
        })

        logger.info(f"Created A/B test experiment: {experiment_id}")
        return experiment_id

    def route_request(self, experiment_id: str) -> str:
        """Route request to model A or B based on traffic split"""
        if experiment_id not in self.active_experiments:
            raise ValueError(f"Experiment {experiment_id} not found")

        experiment = self.active_experiments[experiment_id]

        # Route based on traffic split
        if np.random.random() < self.config.ab_test_traffic_split:
            return experiment['model_a_version']
        else:
            return experiment['model_b_version']

    def record_performance(self, experiment_id: str, model_version: str,
                          performance_metric: float) -> None:
        """Record performance metric for a model in the experiment"""
        if experiment_id not in self.active_experiments:
            return

        experiment = self.active_experiments[experiment_id]

        if model_version == experiment['model_a_version']:
            experiment['model_a_performance'].append(performance_metric)
            experiment['model_a_count'] += 1
        elif model_version == experiment['model_b_version']:
            experiment['model_b_performance'].append(performance_metric)
            experiment['model_b_count'] += 1

        # Check if we have enough samples to conclude
        if (experiment['model_a_count'] >= self.config.ab_test_min_samples and
            experiment['model_b_count'] >= self.config.ab_test_min_samples):
            self._analyze_experiment(experiment_id)

    def _analyze_experiment(self, experiment_id: str) -> Dict[str, Any]:
        """Analyze A/B test results and determine winner"""
        experiment = self.active_experiments[experiment_id]

        # Calculate statistics
        a_performance = np.array(experiment['model_a_performance'])
        b_performance = np.array(experiment['model_b_performance'])

        a_mean = np.mean(a_performance)
        b_mean = np.mean(b_performance)
        a_std = np.std(a_performance)
        b_std = np.std(b_performance)

        # Perform t-test
        from scipy import stats
        t_stat, p_value = stats.ttest_ind(a_performance, b_performance)

        # Determine winner
        winner = None
        if p_value < (1 - self.config.ab_test_confidence_level):
            winner = experiment['model_a_version'] if a_mean > b_mean else experiment['model_b_version']

        results = {
            'model_a_mean': float(a_mean),
            'model_b_mean': float(b_mean),
            'model_a_std': float(a_std),
            'model_b_std': float(b_std),
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'winner': winner,
            'confidence_level': self.config.ab_test_confidence_level,
            'sample_size_a': experiment['model_a_count'],
            'sample_size_b': experiment['model_b_count']
        }

        # Update database
        self.db_manager.update_record(
            'ab_test_results',
            {
                'model_a_performance': float(a_mean),
                'model_b_performance': float(b_mean),
                'winner': winner,
                'confidence_level': self.config.ab_test_confidence_level,
                'sample_size': experiment['model_a_count'] + experiment['model_b_count'],
                'results': json.dumps(results),
                'completed_at': datetime.now()
            },
            'experiment_id = ?',
            (experiment_id,)
        )

        experiment['completed'] = True
        logger.info(f"A/B test {experiment_id} completed. Winner: {winner}")

        return results

    def get_experiment_status(self, experiment_id: str) -> Dict[str, Any]:
        """Get current status of an A/B test experiment"""
        if experiment_id in self.active_experiments:
            experiment = self.active_experiments[experiment_id]
            return {
                'experiment_id': experiment_id,
                'model_a_version': experiment['model_a_version'],
                'model_b_version': experiment['model_b_version'],
                'model_a_count': experiment['model_a_count'],
                'model_b_count': experiment['model_b_count'],
                'completed': experiment['completed'],
                'created_at': experiment['created_at'].isoformat()
            }

        # Try loading from database
        results = self.db_manager.execute_query(
            "SELECT * FROM ab_test_results WHERE experiment_id = ?",
            (experiment_id,)
        )

        if results:
            row = results[0]
            return {
                'experiment_id': row[0],
                'model_a_version': row[1],
                'model_b_version': row[2],
                'results': json.loads(row[8]) if row[8] else {},
                'completed': row[10] is not None
            }

        return None

class ModelTrainer:
    """Handles model training with hyperparameter optimization"""

    def __init__(self, config: MLOpsConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")

    def train_model(self, X_train: np.ndarray, y_train: np.ndarray,
                   X_val: np.ndarray = None, y_val: np.ndarray = None,
                   optimize_hyperparameters: bool = True) -> Tuple[Any, Dict[str, float], float]:
        """Train a model with optional hyperparameter optimization"""
        start_time = time.time()

        # Split validation set if not provided
        if X_val is None or y_val is None:
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, test_size=self.config.validation_split,
                random_state=42, stratify=y_train
            )

        # Optimize hyperparameters if requested
        if optimize_hyperparameters:
            best_params = self._optimize_hyperparameters(X_train, y_train, X_val, y_val)
        else:
            best_params = {
                'hidden_dims': [128, 64, 32],
                'learning_rate': self.config.learning_rate,
                'batch_size': self.config.batch_size,
                'dropout_rate': 0.3
            }

        # Create model with best parameters
        model = CustomNeuralNetwork(
            input_dim=X_train.shape[1],
            hidden_dims=best_params['hidden_dims'],
            output_dim=1,
            dropout_rate=best_params['dropout_rate']
        ).to(self.device)

        # Create data loaders
        train_dataset = CustomDataset(X_train, y_train)
        val_dataset = CustomDataset(X_val, y_val)

        train_loader = DataLoader(
            train_dataset,
            batch_size=best_params['batch_size'],
            shuffle=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=best_params['batch_size'],
            shuffle=False
        )

        # Training setup
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=best_params['learning_rate'])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=5, factor=0.5
        )

        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        training_history = []

        for epoch in range(self.config.epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for batch_features, batch_labels in train_loader:
                batch_features = batch_features.to(self.device)
                batch_labels = batch_labels.to(self.device)

                optimizer.zero_grad()
                outputs = model(batch_features).squeeze()
                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                predictions = (outputs > 0.5).float()
                train_correct += (predictions == batch_labels).sum().item()
                train_total += batch_labels.size(0)

            # Validation phase
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for batch_features, batch_labels in val_loader:
                    batch_features = batch_features.to(self.device)
                    batch_labels = batch_labels.to(self.device)

                    outputs = model(batch_features).squeeze()
                    loss = criterion(outputs, batch_labels)

                    val_loss += loss.item()
                    predictions = (outputs > 0.5).float()
                    val_correct += (predictions == batch_labels).sum().item()
                    val_total += batch_labels.size(0)

            # Calculate metrics
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            train_acc = train_correct / train_total
            val_acc = val_correct / val_total

            training_history.append({
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_acc': train_acc,
                'val_acc': val_acc
            })

            # Learning rate scheduling
            scheduler.step(val_loss)

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= self.config.early_stopping_patience:
                logger.info(f"Early stopping triggered at epoch {epoch}")
                break

            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}: Train Loss={train_loss:.4f}, "
                          f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")

        # Calculate final metrics
        model.eval()
        with torch.no_grad():
            val_features = torch.FloatTensor(X_val).to(self.device)
            val_predictions = model(val_features).squeeze().cpu().numpy()
            val_predictions_binary = (val_predictions > 0.5).astype(int)

            metrics = {
                'accuracy': accuracy_score(y_val, val_predictions_binary),
                'precision': precision_score(y_val, val_predictions_binary, zero_division=0),
                'recall': recall_score(y_val, val_predictions_binary, zero_division=0),
                'f1': f1_score(y_val, val_predictions_binary, zero_division=0),
                'auc_roc': roc_auc_score(y_val, val_predictions) if len(np.unique(y_val)) > 1 else 0.0
            }

        # Move model back to CPU for storage
        model.cpu()

        training_duration = time.time() - start_time
        training_duration_gauge.set(training_duration)

        return model, metrics, training_duration

    def _optimize_hyperparameters(self, X_train: np.ndarray, y_train: np.ndarray,
                                 X_val: np.ndarray, y_val: np.ndarray,
                                 n_trials: int = 20) -> Dict[str, Any]:
        """Optimize hyperparameters using Optuna"""

        def objective(trial):
            # Suggest hyperparameters
            n_layers = trial.suggest_int('n_layers', 2, 4)
            hidden_dims = []
            for i in range(n_layers):
                hidden_dims.append(trial.suggest_int(f'hidden_dim_{i}', 32, 256, step=32))

            learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
            batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
            dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)

            # Create and train model
            model = CustomNeuralNetwork(
                input_dim=X_train.shape[1],
                hidden_dims=hidden_dims,
                output_dim=1,
                dropout_rate=dropout_rate
            ).to(self.device)

            # Quick training for hyperparameter search
            train_dataset = CustomDataset(X_train, y_train)
            val_dataset = CustomDataset(X_val, y_val)

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

            criterion = nn.BCELoss()
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)

            # Train for fewer epochs during optimization
            for epoch in range(20):
                model.train()
                for batch_features, batch_labels in train_loader:
                    batch_features = batch_features.to(self.device)
                    batch_labels = batch_labels.to(self.device)

                    optimizer.zero_grad()
                    outputs = model(batch_features).squeeze()
                    loss = criterion(outputs, batch_labels)
                    loss.backward()
                    optimizer.step()

            # Evaluate
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_features, batch_labels in val_loader:
                    batch_features = batch_features.to(self.device)
                    batch_labels = batch_labels.to(self.device)

                    outputs = model(batch_features).squeeze()
                    loss = criterion(outputs, batch_labels)
                    val_loss += loss.item()

            return val_loss / len(val_loader)

        # Run optimization
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        # Extract best parameters
        best_params = study.best_params
        hidden_dims = []
        for i in range(best_params['n_layers']):
            hidden_dims.append(best_params[f'hidden_dim_{i}'])

        return {
            'hidden_dims': hidden_dims,
            'learning_rate': best_params['learning_rate'],
            'batch_size': best_params['batch_size'],
            'dropout_rate': best_params['dropout_rate']
        }

class PerformanceMonitor:
    """Monitors model performance and system health"""

    def __init__(self, config: MLOpsConfig, db_manager: DatabaseManager):
        self.config = config
        self.db_manager = db_manager
        self.performance_buffer = deque(maxlen=config.monitoring_window_size)
        self.alert_counter = 0

    def record_prediction(self, model_version: str, prediction: float,
                         confidence: float, latency_ms: float,
                         input_features: np.ndarray) -> None:
        """Record a prediction for monitoring"""
        # Update metrics
        prediction_counter.inc()
        prediction_latency.observe(latency_ms / 1000.0)

        # Save to database
        self.db_manager.insert_record('prediction_logs', {
            'model_version': model_version,
            'input_features': json.dumps(input_features.tolist()) if isinstance(input_features, np.ndarray) else json.dumps(input_features),
            'prediction': prediction,
            'confidence': confidence,
            'latency_ms': latency_ms,
            'timestamp': datetime.now()
        })

        # Add to performance buffer
        self.performance_buffer.append({
            'prediction': prediction,
            'confidence': confidence,
            'latency_ms': latency_ms,
            'timestamp': datetime.now()
        })

        # Check for alerts
        self._check_alerts()

    def record_model_performance(self, model_version: str, metrics: Dict[str, float],
                               prediction_count: int = 0) -> None:
        """Record model performance metrics"""
        for metric_name, metric_value in metrics.items():
            self.db_manager.insert_record('performance_metrics', {
                'model_version': model_version,
                'metric_name': metric_name,
                'metric_value': metric_value,
                'timestamp': datetime.now(),
                'prediction_count': prediction_count
            })

        # Update Prometheus metrics
        if 'accuracy' in metrics:
            model_accuracy_gauge.set(metrics['accuracy'])

    def _check_alerts(self) -> None:
        """Check for performance degradation alerts"""
        if len(self.performance_buffer) < 100:
            return

        recent_latencies = [p['latency_ms'] for p in list(self.performance_buffer)[-100:]]
        avg_latency = np.mean(recent_latencies)

        if avg_latency > self.config.max_latency_ms:
            self.alert_counter += 1

            if self.alert_counter >= self.config.alert_threshold_consecutive_failures:
                logger.warning(f"Performance alert: Average latency {avg_latency:.2f}ms "
                             f"exceeds threshold {self.config.max_latency_ms}ms")
                self.alert_counter = 0
        else:
            self.alert_counter = 0

    def get_performance_summary(self, model_version: str = None,
                               hours: int = 24) -> Dict[str, Any]:
        """Get performance summary for specified period"""
        start_time = datetime.now() - timedelta(hours=hours)

        # Get prediction statistics
        query = """
            SELECT
                COUNT(*) as total_predictions,
                AVG(prediction) as avg_prediction,
                AVG(confidence) as avg_confidence,
                AVG(latency_ms) as avg_latency,
                MAX(latency_ms) as max_latency,
                MIN(latency_ms) as min_latency
            FROM prediction_logs
            WHERE timestamp > ?
        """
        params = [start_time]

        if model_version:
            query += " AND model_version = ?"
            params.append(model_version)

        results = self.db_manager.execute_query(query, tuple(params))

        summary = {
            'period_hours': hours,
            'model_version': model_version,
            'predictions': {}
        }

        if results and results[0][0]:
            row = results[0]
            summary['predictions'] = {
                'total': row[0],
                'avg_prediction': row[1],
                'avg_confidence': row[2],
                'avg_latency_ms': row[3],
                'max_latency_ms': row[4],
                'min_latency_ms': row[5]
            }

        # Get model metrics
        query = """
            SELECT metric_name, AVG(metric_value) as avg_value
            FROM performance_metrics
            WHERE timestamp > ?
        """
        params = [start_time]

        if model_version:
            query += " AND model_version = ?"
            params.append(model_version)

        query += " GROUP BY metric_name"

        results = self.db_manager.execute_query(query, tuple(params))

        summary['metrics'] = {}
        for row in results:
            summary['metrics'][row[0]] = row[1]

        return summary

class MLOpsEngine:
    """Main MLOps engine that orchestrates all components"""

    def __init__(self, config: MLOpsConfig):
        self.config = config
        self.db_manager = DatabaseManager(config.db_path)
        self.model_registry = ModelRegistry(config.model_registry_uri, self.db_manager)
        self.cost_tracker = CostTracker(config, self.db_manager)
        self.ab_test_manager = ABTestManager(config, self.db_manager)
        self.performance_monitor = PerformanceMonitor(config, self.db_manager)
        self.trainer = ModelTrainer(config)
        self.drift_detector = None
        self.scaler = StandardScaler()

        # Threading for auto-retraining
        self.auto_retrain_thread = None
        self.stop_auto_retrain = threading.Event()

        # Initialize reference data for drift detection
        self.reference_data = None
        self.reference_labels = None

        # Current active A/B test
        self.active_ab_test = None

    def generate_synthetic_data(self, n_samples: int = 1000,
                               n_features: int = 10,
                               noise_level: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
        """Generate synthetic data for demonstration"""
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=n_features - 2,
            n_redundant=2,
            n_clusters_per_class=2,
            weights=[0.7, 0.3],
            flip_y=noise_level,
            random_state=42
        )

        # Add some temporal drift to simulate real-world scenarios
        drift_factor = np.random.normal(0, 0.1, size=(n_samples, n_features))
        X = X + drift_factor * np.arange(n_samples).reshape(-1, 1) / n_samples

        return X, y

    def prepare_data(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for training/inference"""
        # Handle class imbalance
        unique_classes = np.unique(y)
        if len(unique_classes) == 2:
            class_counts = np.bincount(y.astype(int))
            if min(class_counts) / max(class_counts) < 0.5:
                logger.info("Applying SMOTE for class imbalance")
                smote = SMOTE(random_state=42)
                X, y = smote.fit_resample(X, y)

        return X, y

    def train_new_model(self, X: np.ndarray = None, y: np.ndarray = None,
                       optimize_hyperparameters: bool = True) -> str:
        """Train a new model and register it"""
        # Generate synthetic data if none provided
        if X is None or y is None:
            logger.info("Generating synthetic training data")
            X, y = self.generate_synthetic_data(n_samples=5000)

        # Prepare data
        X, y = self.prepare_data(X, y)

        # Fit scaler
        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)

        # Store reference data for drift detection
        if self.reference_data is None:
            self.reference_data = X_scaled[:1000]
            self.reference_labels = y[:1000]
            self.drift_detector = DriftDetector(
                self.reference_data,
                self.config,
                feature_names=self.config.input_features[:X.shape[1]]
            )

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )

        # Start MLflow run if enabled
        if self.config.experiment_tracking:
            mlflow.start_run()

        try:
            # Train model
            logger.info("Starting model training")
            model, metrics, training_duration = self.trainer.train_model(
                X_train, y_train, X_test, y_test,
                optimize_hyperparameters=optimize_hyperparameters
            )

            # Evaluate on test set
            model.eval()
            with torch.no_grad():
                test_features = torch.FloatTensor(X_test)
                test_predictions = model(test_features).squeeze().numpy()
                test_predictions_binary = (test_predictions > 0.5).astype(int)

            # Calculate test metrics
            test_metrics = {
                'test_accuracy': accuracy_score(y_test, test_predictions_binary),
                'test_precision': precision_score(y_test, test_predictions_binary, zero_division=0),
                'test_recall': recall_score(y_test, test_predictions_binary, zero_division=0),
                'test_f1': f1_score(y_test, test_predictions_binary, zero_division=0),
                'test_auc_roc': roc_auc_score(y_test, test_predictions) if len(np.unique(y_test)) > 1 else 0.0
            }

            # Combine metrics
            all_metrics = {**metrics, **test_metrics}

            # Log metrics to MLflow
            if self.config.experiment_tracking:
                for metric_name, metric_value in all_metrics.items():
                    mlflow.log_metric(metric_name, metric_value)
                mlflow.log_param("training_duration", training_duration)
                mlflow.pytorch.log_model(model, "model")

            # Create metadata
            metadata = {
                'training_samples': len(X_train),
                'test_samples': len(X_test),
                'features': X.shape[1],
                'training_duration': training_duration,
                'hyperparameter_optimization': optimize_hyperparameters,
                'timestamp': datetime.now().isoformat()
            }

            # Register model
            version_id = self.model_registry.register_model(
                model, all_metrics, metadata, training_duration
            )

            # Track costs
            self.cost_tracker.track_training_cost(training_duration, version_id)
            storage_size_gb = os.path.getsize(
                self.model_registry.base_path / f"model_{version_id}.pkl"
            ) / (1024 ** 3)
            self.cost_tracker.track_storage_cost(storage_size_gb, version_id)

            # Record performance
            self.performance_monitor.record_model_performance(version_id, all_metrics)

            logger.info(f"Model training completed. Version: {version_id}")
            logger.info(f"Test Accuracy: {test_metrics['test_accuracy']:.4f}")

            # Auto-promote if meets criteria
            if test_metrics['test_accuracy'] >= self.config.min_accuracy:
                if not self.model_registry.current_version:
                    self.model_registry.promote_model(version_id)
                    logger.info(f"Auto-promoted model {version_id} to production")

            return version_id

        finally:
            if self.config.experiment_tracking:
                mlflow.end_run()

    def predict(self, features: np.ndarray, use_ab_test: bool = False) -> Dict[str, Any]:
        """Make predictions using the current production model"""
        start_time = time.time()

        # Determine which model to use
        if use_ab_test and self.active_ab_test:
            model_version = self.ab_test_manager.route_request(self.active_ab_test)
        else:
            model_version = self.model_registry.current_version

        if not model_version:
            raise ValueError("No production model available")

        # Get model
        model_info = self.model_registry.get_model(model_version)
        if not model_info or not model_info.model:
            raise ValueError(f"Model {model_version} not found")

        model = model_info.model

        # Prepare features
        if features.ndim == 1:
            features = features.reshape(1, -1)

        features_scaled = self.scaler.transform(features)

        # Make prediction
        model.eval()
        with torch.no_grad():
            features_tensor = torch.FloatTensor(features_scaled)
            outputs = model(features_tensor).squeeze().numpy()

            if outputs.ndim == 0:
                outputs = np.array([outputs])

            predictions = (outputs > 0.5).astype(int)
            confidences = np.where(outputs > 0.5, outputs, 1 - outputs)

        # Calculate latency
        latency_ms = (time.time() - start_time) * 1000

        # Record prediction
        for i in range(len(predictions)):
            self.performance_monitor.record_prediction(
                model_version,
                float(predictions[i]),
                float(confidences[i]),
                latency_ms,
                features[i]
            )

        # Record for A/B test if active
        if use_ab_test and self.active_ab_test:
            # Use confidence as performance metric for A/B testing
            avg_confidence = np.mean(confidences)
            self.ab_test_manager.record_performance(
                self.active_ab_test, model_version, avg_confidence
            )

        # Track inference cost
        self.cost_tracker.track_inference_cost(len(predictions), model_version)

        return {
            'predictions': predictions.tolist(),
            'confidences': confidences.tolist(),
            'model_version': model_version,
            'latency_ms': latency_ms
        }

    def check_drift(self, current_data: np.ndarray = None) -> Dict[str, Any]:
        """Check for data drift"""
        if self.drift_detector is None:
            return {'error': 'Drift detector not initialized. Train a model first.'}

        # Generate current data if not provided
        if current_data is None:
            current_data, _ = self.generate_synthetic_data(n_samples=1000)
            # Add more drift for demonstration
            current_data = current_data + np.random.normal(0, 0.2, current_data.shape)

        current_data_scaled = self.scaler.transform(current_data)

        # Detect drift
        drift_results = self.drift_detector.detect_drift(current_data_scaled)

        # Log drift results
        if self.model_registry.current_version:
            self.db_manager.insert_record('drift_logs', {
                'model_version': self.model_registry.current_version,
                'drift_type': drift_results['method'],
                'drift_score': drift_results['drift_score'],
                'is_drift': int(drift_results['is_drift']),
                'feature_drifts': json.dumps(drift_results['feature_drifts']),
                'timestamp': datetime.now()
            })

            # Update metric
            drift_score_gauge.set(drift_results['drift_score'])

        return drift_results

    def start_ab_test(self, challenger_version: str = None) -> str:
        """Start an A/B test between current model and a challenger"""
        if not self.model_registry.current_version:
            raise ValueError("No current production model for A/B testing")

        # Train challenger model if not specified
        if not challenger_version:
            logger.info("Training challenger model for A/B test")
            challenger_version = self.train_new_model()

        # Create A/B test
        experiment_id = self.ab_test_manager.create_experiment(
            self.model_registry.current_version,
            challenger_version,
            "auto_ab_test"
        )

        self.active_ab_test = experiment_id
        logger.info(f"Started A/B test: {experiment_id}")

        return experiment_id

    def complete_ab_test(self, experiment_id: str = None) -> Dict[str, Any]:
        """Complete an A/B test and potentially promote winner"""
        if experiment_id is None:
            experiment_id = self.active_ab_test

        if not experiment_id:
            return {'error': 'No active A/B test'}

        # Get results
        results = self.ab_test_manager._analyze_experiment(experiment_id)

        # Auto-promote winner if significant
        if results.get('winner'):
            winner_version = results['winner']
            logger.info(f"A/B test winner: {winner_version}")

            # Check if winner meets minimum accuracy
            model_info = self.model_registry.get_model(winner_version)
            if model_info and model_info.metrics.get('accuracy', 0) >= self.config.min_accuracy:
                self.model_registry.promote_model(winner_version)
                logger.info(f"Promoted {winner_version} to production based on A/B test")

        self.active_ab_test = None
        return results

    def auto_retrain_loop(self):
        """Background thread for automatic retraining"""
        while not self.stop_auto_retrain.is_set():
            try:
                # Check drift
                drift_results = self.check_drift()

                if drift_results.get('is_drift', False):
                    logger.info("Drift detected, triggering automatic retraining")

                    # Generate new training data (in practice, this would be recent data)
                    X, y = self.generate_synthetic_data(n_samples=5000)

                    # Train new model
                    new_version = self.train_new_model(X, y)

                    # Start A/B test with new model
                    self.start_ab_test(new_version)

                    logger.info(f"Started A/B test with retrained model {new_version}")

                # Wait for next check
                self.stop_auto_retrain.wait(self.config.auto_retrain_interval_hours * 3600)

            except Exception as e:
                logger.error(f"Error in auto-retrain loop: {e}")
                self.stop_auto_retrain.wait(300)  # Wait 5 minutes on error

    def start_auto_retrain(self):
        """Start automatic retraining background process"""
        if self.config.auto_retrain_enabled and not self.auto_retrain_thread:
            self.auto_retrain_thread = threading.Thread(
                target=self.auto_retrain_loop,
                daemon=True
            )
            self.auto_retrain_thread.start()
            logger.info("Started auto-retraining background process")

    def stop_auto_retrain(self):
        """Stop automatic retraining"""
        if self.auto_retrain_thread:
            self.stop_auto_retrain.set()
            self.auto_retrain_thread.join()
            self.auto_retrain_thread = None
            logger.info("Stopped auto-retraining background process")

    def get_model_card(self, version_id: str = None) -> Dict[str, Any]:
        """Generate a model card for documentation"""
        if version_id is None:
            version_id = self.model_registry.current_version

        if not version_id:
            return {'error': 'No model version specified'}

        model_info = self.model_registry.get_model(version_id)
        if not model_info:
            return {'error': f'Model {version_id} not found'}

        # Get additional information from database
        perf_summary = self.performance_monitor.get_performance_summary(
            model_version=version_id, hours=24*7
        )

        cost_report = self.cost_tracker.get_cost_report(days=30)

        # Check for drift logs
        drift_logs = self.db_manager.execute_query(
            """SELECT COUNT(*) as drift_count, AVG(drift_score) as avg_drift_score
               FROM drift_logs WHERE model_version = ? AND timestamp > ?""",
            (version_id, datetime.now() - timedelta(days=7))
        )

        model_card = {
            'model_name': self.config.model_name,
            'version_id': version_id,
            'created_at': model_info.created_at.isoformat(),
            'metrics': model_info.metrics,
            'metadata': model_info.metadata,
            'performance_summary': perf_summary,
            'cost_summary': cost_report,
            'drift_summary': {
                'drift_count': drift_logs[0][0] if drift_logs else 0,
                'avg_drift_score': drift_logs[0][1] if drift_logs else 0
            },
            'deployment_count': model_info.deployment_count,
            'is_production': version_id == self.model_registry.current_version
        }

        return model_card

    def export_to_huggingface(self, version_id: str = None,
                            repo_name: str = None,
                            token: str = None) -> str:
        """Export model to Hugging Face Hub"""
        if version_id is None:
            version_id = self.model_registry.current_version

        if not version_id:
            return "No model version specified"

        model_info = self.model_registry.get_model(version_id)
        if not model_info:
            return f"Model {version_id} not found"

        if not repo_name:
            repo_name = f"{self.config.model_name}_{version_id}"

        try:
            # Initialize HF API
            api = HfApi()

            # Create repository
            repo_url = create_repo(repo_name, token=token, exist_ok=True)

            # Upload model file
            model_path = Path(model_info.model_path)
            if model_path.exists():
                upload_file(
                    path_or_fileobj=str(model_path),
                    path_in_repo=f"model.pkl",
                    repo_id=repo_name,
                    token=token
                )

            # Create and upload model card
            model_card = self.get_model_card(version_id)
            model_card_content = f"""
# {self.config.model_name}

## Model Details
- **Version**: {version_id}
- **Created**: {model_card['created_at']}
- **Task**: {self.config.task_type}

## Performance Metrics
"""
            for metric, value in model_card['metrics'].items():
                model_card_content += f"- **{metric}**: {value:.4f}\n"

            # Save and upload model card
            model_card_path = self.model_registry.base_path / f"README_{version_id}.md"
            with open(model_card_path, 'w') as f:
                f.write(model_card_content)

            upload_file(
                path_or_fileobj=str(model_card_path),
                path_in_repo="README.md",
                repo_id=repo_name,
                token=token
            )

            return f"Model exported to: {repo_url}"

        except Exception as e:
            return f"Export failed: {str(e)}"

def create_gradio_interface(mlops_engine: MLOpsEngine) -> gr.Blocks:
    """Create the Gradio interface for the MLOps system"""

    with gr.Blocks(title="MLOps System", theme=gr.themes.Soft()) as interface:
        gr.Markdown("""
        # End-to-End Automated MLOps Framework
        **Author**: Spencer Purdy

        Enterprise-grade MLOps platform with automated model training, versioning, drift detection,
        A/B testing, and deployment capabilities.
        """)

        with gr.Tabs():
            # Model Training Tab
            with gr.TabItem("Model Training"):
                gr.Markdown("### Train New Model")

                with gr.Row():
                    n_samples = gr.Slider(
                        minimum=1000, maximum=10000, value=5000, step=1000,
                        label="Number of Training Samples"
                    )
                    optimize_hp = gr.Checkbox(
                        value=True,
                        label="Optimize Hyperparameters"
                    )

                train_button = gr.Button("Train New Model", variant="primary")
                training_output = gr.Textbox(
                    label="Training Results",
                    lines=10,
                    max_lines=20
                )

                def train_model(n_samples, optimize_hp):
                    try:
                        # Generate data
                        X, y = mlops_engine.generate_synthetic_data(n_samples=n_samples)

                        # Train model
                        version_id = mlops_engine.train_new_model(
                            X, y, optimize_hyperparameters=optimize_hp
                        )

                        # Get model info
                        model_info = mlops_engine.model_registry.get_model(version_id)

                        result = f"Model Training Completed\n"
                        result += f"{'=' * 50}\n"
                        result += f"Version ID: {version_id}\n"
                        result += f"Training Samples: {n_samples}\n"
                        result += f"Hyperparameter Optimization: {optimize_hp}\n\n"
                        result += f"Performance Metrics:\n"
                        result += f"{'-' * 30}\n"

                        for metric, value in model_info.metrics.items():
                            result += f"{metric}: {value:.4f}\n"

                        return result
                    except Exception as e:
                        return f"Error during training: {str(e)}"

                train_button.click(
                    train_model,
                    inputs=[n_samples, optimize_hp],
                    outputs=training_output
                )

            # Model Registry Tab
            with gr.TabItem("Model Registry"):
                gr.Markdown("### Model Registry and Versioning")

                refresh_registry_btn = gr.Button("Refresh Model List")
                model_list = gr.Dataframe(
                    headers=["Version ID", "Created At", "Accuracy", "Status"],
                    label="Registered Models"
                )

                with gr.Row():
                    version_selector = gr.Dropdown(
                        label="Select Model Version",
                        choices=[]
                    )
                    promote_btn = gr.Button("Promote to Production")

                promote_output = gr.Textbox(label="Promotion Result")

                def refresh_model_list():
                    models = []
                    for version_id, version in mlops_engine.model_registry.versions.items():
                        status = "Production" if version_id == mlops_engine.model_registry.current_version else "Staged"
                        models.append([
                            version_id,
                            version.created_at.strftime("%Y-%m-%d %H:%M:%S"),
                            f"{version.metrics.get('accuracy', 0):.4f}",
                            status
                        ])

                    # Sort by created date
                    models.sort(key=lambda x: x[1], reverse=True)

                    # Update dropdown choices
                    version_choices = [m[0] for m in models]

                    return models, gr.update(choices=version_choices)

                def promote_model(version_id):
                    if not version_id:
                        return "Please select a model version"

                    try:
                        mlops_engine.model_registry.promote_model(version_id)
                        return f"Successfully promoted {version_id} to production"
                    except Exception as e:
                        return f"Error promoting model: {str(e)}"

                refresh_registry_btn.click(
                    refresh_model_list,
                    outputs=[model_list, version_selector]
                )

                promote_btn.click(
                    promote_model,
                    inputs=version_selector,
                    outputs=promote_output
                )

                # Load initial data
                interface.load(refresh_model_list, outputs=[model_list, version_selector])

            # Prediction Tab
            with gr.TabItem("Make Predictions"):
                gr.Markdown("### Make Predictions Using Production Model")

                with gr.Row():
                    feature_inputs = []
                    for i in range(10):
                        feature_inputs.append(
                            gr.Number(
                                label=f"Feature {i+1}",
                                value=0.0
                            )
                        )

                with gr.Row():
                    predict_btn = gr.Button("Predict", variant="primary")
                    use_ab_test = gr.Checkbox(
                        label="Use A/B Test (if active)",
                        value=False
                    )

                prediction_output = gr.JSON(label="Prediction Results")

                def make_prediction(*features, use_ab_test=False):
                    try:
                        features_array = np.array(features).reshape(1, -1)
                        results = mlops_engine.predict(features_array, use_ab_test=use_ab_test)
                        return results
                    except Exception as e:
                        return {"error": str(e)}

                predict_btn.click(
                    make_prediction,
                    inputs=feature_inputs + [use_ab_test],
                    outputs=prediction_output
                )

            # Drift Detection Tab
            with gr.TabItem("Drift Detection"):
                gr.Markdown("### Data Drift Detection")

                check_drift_btn = gr.Button("Check for Data Drift", variant="primary")
                drift_output = gr.Textbox(
                    label="Drift Detection Results",
                    lines=15
                )

                def check_drift():
                    try:
                        results = mlops_engine.check_drift()

                        if 'error' in results:
                            return results['error']

                        report = mlops_engine.drift_detector.generate_drift_report(
                            mlops_engine.scaler.transform(
                                mlops_engine.generate_synthetic_data(n_samples=1000)[0]
                            )
                        )

                        return report
                    except Exception as e:
                        return f"Error checking drift: {str(e)}"

                check_drift_btn.click(check_drift, outputs=drift_output)

            # A/B Testing Tab
            with gr.TabItem("A/B Testing"):
                gr.Markdown("### A/B Testing for Model Comparison")

                with gr.Row():
                    start_ab_btn = gr.Button("Start New A/B Test", variant="primary")
                    check_ab_btn = gr.Button("Check Current A/B Test")
                    complete_ab_btn = gr.Button("Complete A/B Test")

                ab_output = gr.JSON(label="A/B Test Results")

                def start_ab_test():
                    try:
                        experiment_id = mlops_engine.start_ab_test()
                        return {
                            "status": "A/B test started",
                            "experiment_id": experiment_id,
                            "message": "Make predictions with 'Use A/B Test' enabled to generate results"
                        }
                    except Exception as e:
                        return {"error": str(e)}

                def check_ab_test():
                    if mlops_engine.active_ab_test:
                        return mlops_engine.ab_test_manager.get_experiment_status(
                            mlops_engine.active_ab_test
                        )
                    else:
                        return {"status": "No active A/B test"}

                def complete_ab_test():
                    try:
                        results = mlops_engine.complete_ab_test()
                        return results
                    except Exception as e:
                        return {"error": str(e)}

                start_ab_btn.click(start_ab_test, outputs=ab_output)
                check_ab_btn.click(check_ab_test, outputs=ab_output)
                complete_ab_btn.click(complete_ab_test, outputs=ab_output)

            # Performance Monitoring Tab
            with gr.TabItem("Performance Monitoring"):
                gr.Markdown("### Model Performance Monitoring")

                with gr.Row():
                    hours_slider = gr.Slider(
                        minimum=1, maximum=168, value=24, step=1,
                        label="Time Window (hours)"
                    )
                    refresh_perf_btn = gr.Button("Refresh Performance Metrics")

                performance_output = gr.JSON(label="Performance Summary")

                def get_performance_summary(hours):
                    try:
                        current_version = mlops_engine.model_registry.current_version
                        if not current_version:
                            return {"error": "No production model"}

                        summary = mlops_engine.performance_monitor.get_performance_summary(
                            model_version=current_version,
                            hours=hours
                        )

                        return summary
                    except Exception as e:
                        return {"error": str(e)}

                refresh_perf_btn.click(
                    get_performance_summary,
                    inputs=hours_slider,
                    outputs=performance_output
                )

            # Cost Tracking Tab
            with gr.TabItem("Cost Tracking"):
                gr.Markdown("### Cost Analysis and Tracking")

                with gr.Row():
                    days_slider = gr.Slider(
                        minimum=1, maximum=90, value=30, step=1,
                        label="Report Period (days)"
                    )
                    refresh_cost_btn = gr.Button("Generate Cost Report")

                cost_output = gr.JSON(label="Cost Report")

                def get_cost_report(days):
                    try:
                        report = mlops_engine.cost_tracker.get_cost_report(days=days)
                        return report
                    except Exception as e:
                        return {"error": str(e)}

                refresh_cost_btn.click(
                    get_cost_report,
                    inputs=days_slider,
                    outputs=cost_output
                )

            # Model Card Tab
            with gr.TabItem("Model Card"):
                gr.Markdown("### Model Documentation and Cards")

                with gr.Row():
                    model_version_input = gr.Textbox(
                        label="Model Version (leave empty for current)",
                        placeholder="e.g., v_20240101_120000_1"
                    )
                    generate_card_btn = gr.Button("Generate Model Card")

                model_card_output = gr.JSON(label="Model Card")

                def generate_model_card(version_id):
                    try:
                        if not version_id:
                            version_id = None

                        card = mlops_engine.get_model_card(version_id)
                        return card
                    except Exception as e:
                        return {"error": str(e)}

                generate_card_btn.click(
                    generate_model_card,
                    inputs=model_version_input,
                    outputs=model_card_output
                )

            # Settings Tab
            with gr.TabItem("Settings"):
                gr.Markdown("### System Settings and Configuration")

                with gr.Row():
                    auto_retrain_checkbox = gr.Checkbox(
                        value=mlops_engine.config.auto_retrain_enabled,
                        label="Enable Auto-Retraining"
                    )
                    start_auto_btn = gr.Button("Apply Auto-Retrain Setting")

                with gr.Row():
                    drift_threshold = gr.Slider(
                        minimum=0.01, maximum=0.5, value=mlops_engine.config.drift_detection_threshold,
                        step=0.01, label="Drift Detection Threshold"
                    )
                    update_threshold_btn = gr.Button("Update Threshold")

                settings_output = gr.Textbox(label="Settings Update Result")

                def toggle_auto_retrain(enable):
                    try:
                        mlops_engine.config.auto_retrain_enabled = enable
                        if enable:
                            mlops_engine.start_auto_retrain()
                            return "Auto-retraining enabled and started"
                        else:
                            mlops_engine.stop_auto_retrain()
                            return "Auto-retraining disabled"
                    except Exception as e:
                        return f"Error updating auto-retrain: {str(e)}"

                def update_drift_threshold(threshold):
                    try:
                        mlops_engine.config.drift_detection_threshold = threshold
                        if mlops_engine.drift_detector:
                            mlops_engine.drift_detector.drift_threshold = threshold
                        return f"Drift threshold updated to {threshold}"
                    except Exception as e:
                        return f"Error updating threshold: {str(e)}"

                start_auto_btn.click(
                    toggle_auto_retrain,
                    inputs=auto_retrain_checkbox,
                    outputs=settings_output
                )

                update_threshold_btn.click(
                    update_drift_threshold,
                    inputs=drift_threshold,
                    outputs=settings_output
                )

        # Footer
        gr.Markdown("""
        ---
        **MLOps System** - Enterprise-grade machine learning operations platform

        Features: Automated training, model versioning, drift detection, A/B testing,
        performance monitoring, cost tracking, and model deployment.
        """)

    return interface


def main():
    """Main execution function"""
    # Initialize MLOps engine
    logger.info("Initializing MLOps Engine...")
    mlops_engine = MLOpsEngine(config)

    # Train initial model if no models exist
    if not mlops_engine.model_registry.versions:
        logger.info("No models found. Training initial model...")
        initial_version = mlops_engine.train_new_model()
        logger.info(f"Initial model trained: {initial_version}")

    # Start auto-retraining if enabled
    if config.auto_retrain_enabled:
        mlops_engine.start_auto_retrain()

    # Create and launch Gradio interface
    logger.info("Creating Gradio interface...")
    interface = create_gradio_interface(mlops_engine)

    # Launch the interface
    logger.info("Launching MLOps System interface...")
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        show_error=True
    )


if __name__ == "__main__":
    main()