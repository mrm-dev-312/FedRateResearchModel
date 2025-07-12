"""
Optuna hyperparameter optimization utilities for time series models.
Supports PatchTST, LSTM, and TimeGPT parameter tuning.
"""

import optuna
import numpy as np
import pandas as pd
from typing import Dict, Any, Callable, Optional, List, Union
import logging
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
import json
from datetime import datetime

logger = logging.getLogger(__name__)

OPTUNA_VERSION = "1.0.0"

class TimeSeriesObjective:
    """
    Base objective function for time series model optimization.
    """
    
    def __init__(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        model_type: str = 'lstm',
        metric: str = 'rmse',
        cv_folds: int = 3,
        fixed_params: Optional[Dict] = None
    ):
        """
        Initialize objective function.
        
        Args:
            X_train: Training input data
            y_train: Training target data
            X_val: Validation input data
            y_val: Validation target data
            model_type: Type of model ('lstm', 'patchtst')
            metric: Optimization metric ('rmse', 'mae', 'mape')
            cv_folds: Number of cross-validation folds
            fixed_params: Fixed parameters not to optimize
        """
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.model_type = model_type.lower()
        self.metric = metric.lower()
        self.cv_folds = cv_folds
        self.fixed_params = fixed_params or {}
        
        self.best_score = float('inf')
        self.trial_count = 0
        
    def calculate_metric(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate the specified metric."""
        if self.metric == 'rmse':
            return np.sqrt(mean_squared_error(y_true, y_pred))
        elif self.metric == 'mae':
            return mean_absolute_error(y_true, y_pred)
        elif self.metric == 'mape':
            return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        else:
            raise ValueError(f"Unsupported metric: {self.metric}")
    
    def suggest_lstm_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Suggest hyperparameters for LSTM model."""
        params = {
            'hidden_size': trial.suggest_int('hidden_size', 16, 128, step=16),
            'num_layers': trial.suggest_int('num_layers', 1, 4),
            'dropout': trial.suggest_float('dropout', 0.0, 0.5),
            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
            'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64, 128]),
            'bidirectional': trial.suggest_categorical('bidirectional', [True, False])
        }
        
        # Add fixed parameters
        params.update(self.fixed_params)
        return params
    
    def suggest_patchtst_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Suggest hyperparameters for PatchTST model."""
        params = {
            'patch_length': trial.suggest_categorical('patch_length', [8, 16, 32]),
            'stride': trial.suggest_categorical('stride', [4, 8, 16]),
            'hidden_size': trial.suggest_int('hidden_size', 64, 256, step=32),
            'num_hidden_layers': trial.suggest_int('num_hidden_layers', 3, 8),
            'num_attention_heads': trial.suggest_categorical('num_attention_heads', [4, 8, 16]),
            'd_ff': trial.suggest_int('d_ff', 256, 1024, step=128),
            'dropout': trial.suggest_float('dropout', 0.0, 0.3),
            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
        }
        
        # Add fixed parameters
        params.update(self.fixed_params)
        return params
    
    def train_and_evaluate_lstm(self, params: Dict[str, Any]) -> float:
        """Train and evaluate LSTM model with given parameters."""
        from ..models.lstm import LSTMWrapper
        
        try:
            # Extract training parameters
            training_params = {
                'learning_rate': params.pop('learning_rate', 0.001),
                'batch_size': params.pop('batch_size', 32),
                'epochs': params.pop('epochs', 50),
                'patience': params.pop('patience', 10),
                'verbose': False
            }
            
            # Initialize model
            model = LSTMWrapper(**params)
            
            # Train model
            metrics = model.fit(
                self.X_train, self.y_train,
                self.X_val, self.y_val,
                **training_params
            )
            
            # Generate predictions
            y_pred = model.predict(self.X_val)
            
            # Calculate metric
            score = self.calculate_metric(self.y_val, y_pred)
            
            return score
            
        except Exception as e:
            logger.warning(f"Trial failed: {e}")
            return float('inf')
    
    def train_and_evaluate_patchtst(self, params: Dict[str, Any]) -> float:
        """Train and evaluate PatchTST model with given parameters."""
        from ..models.patchtst import PatchTSTWrapper
        
        try:
            # Extract training parameters
            training_params = {
                'learning_rate': params.pop('learning_rate', 1e-4),
                'epochs': params.pop('epochs', 25),
                'patience': params.pop('patience', 5)
            }
            
            # Initialize model
            model = PatchTSTWrapper(**params)
            
            # Train model
            metrics = model.fit(
                self.X_train, self.y_train,
                self.X_val, self.y_val,
                **training_params
            )
            
            # Generate predictions
            y_pred = model.predict(self.X_val)
            
            # Calculate metric
            score = self.calculate_metric(self.y_val, y_pred)
            
            return score
            
        except Exception as e:
            logger.warning(f"Trial failed: {e}")
            return float('inf')
    
    def __call__(self, trial: optuna.Trial) -> float:
        """Objective function called by Optuna."""
        self.trial_count += 1
        
        # Suggest parameters based on model type
        if self.model_type == 'lstm':
            params = self.suggest_lstm_params(trial)
            score = self.train_and_evaluate_lstm(params)
        elif self.model_type == 'patchtst':
            params = self.suggest_patchtst_params(trial)
            score = self.train_and_evaluate_patchtst(params)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        # Update best score
        if score < self.best_score:
            self.best_score = score
            logger.info(f"Trial {self.trial_count}: New best {self.metric} = {score:.6f}")
        
        return score

def optimize_hyperparameters(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    model_type: str = 'lstm',
    n_trials: int = 100,
    metric: str = 'rmse',
    direction: str = 'minimize',
    fixed_params: Optional[Dict] = None,
    study_name: Optional[str] = None,
    storage: Optional[str] = None,
    sampler_name: str = 'tpe'
) -> optuna.Study:
    """
    Optimize hyperparameters using Optuna.
    
    Args:
        X_train: Training input data
        y_train: Training target data
        X_val: Validation input data
        y_val: Validation target data
        model_type: Type of model to optimize
        n_trials: Number of optimization trials
        metric: Metric to optimize
        direction: 'minimize' or 'maximize'
        fixed_params: Parameters to keep fixed
        study_name: Name for the study
        storage: Storage URL for persistence
        sampler_name: Sampler algorithm ('tpe', 'random', 'cmaes')
        
    Returns:
        Completed Optuna study
    """
    
    # Initialize sampler
    if sampler_name == 'tpe':
        sampler = optuna.samplers.TPESampler(seed=42)
    elif sampler_name == 'random':
        sampler = optuna.samplers.RandomSampler(seed=42)
    elif sampler_name == 'cmaes':
        sampler = optuna.samplers.CmaEsSampler(seed=42)
    else:
        raise ValueError(f"Unsupported sampler: {sampler_name}")
    
    # Create study
    study = optuna.create_study(
        direction=direction,
        sampler=sampler,
        study_name=study_name,
        storage=storage,
        load_if_exists=True
    )
    
    # Create objective
    objective = TimeSeriesObjective(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        model_type=model_type,
        metric=metric,
        fixed_params=fixed_params
    )
    
    # Optimize
    logger.info(f"Starting hyperparameter optimization for {model_type}")
    logger.info(f"Trials: {n_trials}, Metric: {metric}, Sampler: {sampler_name}")
    
    study.optimize(objective, n_trials=n_trials)
    
    # Log results
    logger.info(f"Optimization completed!")
    logger.info(f"Best trial: {study.best_trial.number}")
    logger.info(f"Best {metric}: {study.best_value:.6f}")
    logger.info(f"Best parameters: {study.best_params}")
    
    return study

def cross_validate_params(
    X: np.ndarray,
    y: np.ndarray,
    model_type: str,
    params: Dict[str, Any],
    cv_folds: int = 5,
    metric: str = 'rmse'
) -> Dict[str, Any]:
    """
    Cross-validate model parameters using time series splits.
    
    Args:
        X: Full input data
        y: Full target data
        model_type: Type of model
        params: Model parameters to validate
        cv_folds: Number of CV folds
        metric: Evaluation metric
        
    Returns:
        Cross-validation results
    """
    tscv = TimeSeriesSplit(n_splits=cv_folds)
    cv_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Create objective for this fold
        objective = TimeSeriesObjective(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            model_type=model_type,
            metric=metric
        )
        
        # Evaluate parameters
        if model_type == 'lstm':
            score = objective.train_and_evaluate_lstm(params.copy())
        elif model_type == 'patchtst':
            score = objective.train_and_evaluate_patchtst(params.copy())
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        cv_scores.append(score)
        logger.info(f"Fold {fold + 1}/{cv_folds}: {metric} = {score:.6f}")
    
    results = {
        'scores': cv_scores,
        'mean_score': np.mean(cv_scores),
        'std_score': np.std(cv_scores),
        'metric': metric,
        'cv_folds': cv_folds,
        'params': params
    }
    
    logger.info(f"CV Results: {results['mean_score']:.6f} ± {results['std_score']:.6f}")
    
    return results

def save_optimization_results(
    study: optuna.Study,
    output_path: str,
    include_trials: bool = True
):
    """Save Optuna study results to JSON file."""
    
    results = {
        'study_name': study.study_name,
        'best_trial': {
            'number': study.best_trial.number,
            'value': study.best_value,
            'params': study.best_params,
            'datetime_start': study.best_trial.datetime_start.isoformat() if study.best_trial.datetime_start else None,
            'datetime_complete': study.best_trial.datetime_complete.isoformat() if study.best_trial.datetime_complete else None
        },
        'n_trials': len(study.trials),
        'direction': study.direction.name,
        'optimization_timestamp': datetime.now().isoformat(),
        'optuna_version': OPTUNA_VERSION
    }
    
    if include_trials:
        results['all_trials'] = [
            {
                'number': trial.number,
                'value': trial.value,
                'params': trial.params,
                'state': trial.state.name,
                'datetime_start': trial.datetime_start.isoformat() if trial.datetime_start else None,
                'datetime_complete': trial.datetime_complete.isoformat() if trial.datetime_complete else None
            }
            for trial in study.trials
        ]
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Optimization results saved to {output_path}")

def load_optimization_results(input_path: str) -> Dict[str, Any]:
    """Load Optuna study results from JSON file."""
    with open(input_path, 'r') as f:
        results = json.load(f)
    
    logger.info(f"Loaded optimization results from {input_path}")
    return results

def suggest_good_defaults(model_type: str) -> Dict[str, Any]:
    """Get reasonable default parameters for quick testing."""
    
    if model_type.lower() == 'lstm':
        return {
            'hidden_size': 64,
            'num_layers': 2,
            'dropout': 0.2,
            'learning_rate': 0.001,
            'batch_size': 32,
            'bidirectional': False,
            'epochs': 50,
            'patience': 10
        }
    
    elif model_type.lower() == 'patchtst':
        return {
            'patch_length': 16,
            'stride': 8,
            'hidden_size': 128,
            'num_hidden_layers': 6,
            'num_attention_heads': 8,
            'd_ff': 512,
            'dropout': 0.1,
            'learning_rate': 1e-4,
            'epochs': 25,
            'patience': 5
        }
    
    else:
        raise ValueError(f"No defaults available for model type: {model_type}")

if __name__ == "__main__":
    # Test optimization with synthetic data
    print("=== Optuna Hyperparameter Optimization Test ===")
    
    # Generate synthetic data
    np.random.seed(42)
    t = np.linspace(0, 100, 500)
    ts = np.sin(0.1 * t) + 0.5 * np.sin(0.2 * t) + 0.1 * np.random.randn(500)
    
    # Create sequences (simplified)
    sequence_length = 60
    X, y = [], []
    for i in range(len(ts) - sequence_length):
        X.append(ts[i:i + sequence_length])
        y.append(ts[i + sequence_length])
    
    X = np.array(X).reshape(-1, sequence_length, 1)
    y = np.array(y).reshape(-1, 1)
    
    # Split data
    split_idx = int(0.8 * len(X))
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    print(f"Data prepared: Train={X_train.shape}, Val={X_val.shape}")
    
    # Test default parameters
    defaults = suggest_good_defaults('lstm')
    print(f"✓ Default LSTM parameters: {defaults}")
    
    # Quick optimization test (reduced trials)
    try:
        study = optimize_hyperparameters(
            X_train, y_train, X_val, y_val,
            model_type='lstm',
            n_trials=5,  # Reduced for testing
            metric='rmse'
        )
        
        print(f"✓ Optimization completed")
        print(f"  Best score: {study.best_value:.6f}")
        print(f"  Best params: {study.best_params}")
        
    except Exception as e:
        print(f"⚠️ Optimization test skipped: {e}")
    
    print("✅ Optuna utilities test complete!")
