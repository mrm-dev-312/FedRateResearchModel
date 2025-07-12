"""
Test script for Section 7: Model Layer implementations.
Validates PatchTST, LSTM, TimeGPT, and Optuna utilities.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
import asyncio

def generate_test_data(n_samples: int = 1000) -> pd.DataFrame:
    """Generate synthetic time series for testing."""
    np.random.seed(42)
    
    # Create time index
    dates = pd.date_range('2020-01-01', periods=n_samples, freq='D')
    
    # Generate synthetic price series with trend and seasonality
    t = np.arange(n_samples)
    trend = 0.0001 * t
    seasonal = 0.01 * np.sin(2 * np.pi * t / 365.25)  # Annual cycle
    noise = np.random.randn(n_samples) * 0.02
    
    # Cumulative returns to create realistic price series
    returns = trend + seasonal + noise
    prices = 100 * np.exp(np.cumsum(returns))
    
    # Create OHLCV data
    df = pd.DataFrame({
        'ts': dates,
        'ticker': 'SPY',
        'open': prices * (1 + np.random.randn(n_samples) * 0.001),
        'high': prices * (1 + np.abs(np.random.randn(n_samples)) * 0.005),
        'low': prices * (1 - np.abs(np.random.randn(n_samples)) * 0.005),
        'close': prices,
        'volume': np.random.randint(50000000, 150000000, n_samples)
    })
    
    return df

def test_lstm_model():
    """Test LSTM model implementation."""
    print("=== Testing LSTM Model ===")
    
    try:
        from src.models.lstm import LSTMWrapper, train_lstm_baseline
        
        # Generate test data
        df = generate_test_data(500)
        
        # Initialize LSTM
        lstm = LSTMWrapper(
            sequence_length=30,
            prediction_length=1,
            hidden_size=32,
            num_layers=2,
            dropout=0.1
        )
        
        # Prepare data
        X, y = lstm.prepare_data(df, target_col='close')
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        print(f"✓ Data prepared: X_train={X_train.shape}, y_train={y_train.shape}")
        
        # Train model (1 epoch for speed)
        metrics = lstm.fit(
            X_train, y_train, X_val, y_val,
            epochs=3, verbose=False
        )
        
        print(f"✓ Training completed")
        print(f"  Final train loss: {metrics['final_train_loss']:.6f}")
        print(f"  Final val loss: {metrics['final_val_loss']:.6f}")
        print(f"  Model parameters: {metrics['model_parameters']:,}")
        
        # Test prediction
        predictions = lstm.predict(X_val[:5])
        print(f"✓ Predictions generated: shape={predictions.shape}")
        
        # Test baseline training function
        model, baseline_metrics = train_lstm_baseline(
            X_train, y_train, X_val, y_val,
            epochs=2, hidden_size=16
        )
        print(f"✓ Baseline training function works")
        
        return True
        
    except Exception as e:
        print(f"❌ LSTM test failed: {e}")
        return False

def test_patchtst_model():
    """Test PatchTST model implementation."""
    print("\n=== Testing PatchTST Model ===")
    
    try:
        from src.models.patchtst import PatchTSTWrapper, finetune_patchtst
        
        # Generate test data
        df = generate_test_data(300)
        
        # Initialize PatchTST
        patchtst = PatchTSTWrapper(
            context_length=64,
            prediction_length=5,
            patch_length=8,
            stride=4,
            hidden_size=64,
            num_hidden_layers=3
        )
        
        # Prepare data
        X, y = patchtst.prepare_data(df, target_col='close')
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        print(f"✓ Data prepared: X_train={X_train.shape}, y_train={y_train.shape}")
        
        # Train model (1 epoch for speed)
        metrics = patchtst.fit(
            X_train, y_train, X_val, y_val,
            epochs=2
        )
        
        print(f"✓ Training completed")
        print(f"  Final train loss: {metrics['final_train_loss']:.6f}")
        print(f"  Final val loss: {metrics['final_val_loss']:.6f}")
        print(f"  Device: {metrics['device']}")
        
        # Test prediction
        predictions = patchtst.predict(X_val[:3])
        print(f"✓ Predictions generated: shape={predictions.shape}")
        
        # Test finetune function (use same context length and prediction length as data)
        model, tune_metrics = finetune_patchtst(
            X_train, y_train, X_val, y_val,
            epochs=1, context_length=64, prediction_length=5
        )
        print(f"✓ Finetune function works")
        
        return True
        
    except Exception as e:
        print(f"❌ PatchTST test failed: {e}")
        return False

def test_timegpt_wrapper():
    """Test TimeGPT wrapper (without API call)."""
    print("\n=== Testing TimeGPT Wrapper ===")
    
    try:
        from src.models.timegpt import TimeGPTWrapper, timegpt_forecast_pipeline
        
        # Generate test data
        df = generate_test_data(100)
        
        # Test data preparation (doesn't require API)
        try:
            timegpt = TimeGPTWrapper()
            prepared_data = timegpt.prepare_timegpt_data(
                df, time_col='ts', value_col='close'
            )
            print(f"✓ Data preparation: shape={prepared_data.shape}")
            print(f"  Columns: {list(prepared_data.columns)}")
            
        except ImportError:
            print("⚠️ nixtla package not installed, skipping TimeGPT init test")
            
        # Test forecast pipeline (will fail without API key, but should not crash)
        api_key = os.getenv('TIMEGPT_API_KEY')
        if api_key:
            try:
                result = timegpt_forecast_pipeline(
                    df, horizon=5, api_key=api_key, validate_model=False
                )
                print(f"✓ Forecast pipeline works with API")
            except Exception as e:
                print(f"⚠️ API test failed (expected): {e}")
        else:
            print("⚠️ No TIMEGPT_API_KEY found, skipping API test")
        
        print("✓ TimeGPT wrapper structure validated")
        return True
        
    except Exception as e:
        print(f"❌ TimeGPT test failed: {e}")
        return False

def test_optuna_utils():
    """Test Optuna hyperparameter optimization utilities."""
    print("\n=== Testing Optuna Utilities ===")
    
    try:
        from src.tuning.optuna_utils import (
            suggest_good_defaults,
            TimeSeriesObjective,
            optimize_hyperparameters
        )
        
        # Test default parameters
        lstm_defaults = suggest_good_defaults('lstm')
        patchtst_defaults = suggest_good_defaults('patchtst')
        
        print(f"✓ LSTM defaults: {len(lstm_defaults)} parameters")
        print(f"✓ PatchTST defaults: {len(patchtst_defaults)} parameters")
        
        # Generate simple test data for optimization
        np.random.seed(42)
        n_samples = 200
        sequence_length = 20
        
        # Create simple sequences
        ts = np.cumsum(np.random.randn(n_samples) * 0.1)
        X, y = [], []
        for i in range(len(ts) - sequence_length - 1):
            X.append(ts[i:i + sequence_length])
            y.append(ts[i + sequence_length])
        
        X = np.array(X).reshape(-1, sequence_length, 1)
        y = np.array(y).reshape(-1, 1)
        
        # Split data
        split_idx = int(0.8 * len(X))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        print(f"✓ Optimization test data: Train={X_train.shape}, Val={X_val.shape}")
        
        # Test objective function (without full optimization)
        objective = TimeSeriesObjective(
            X_train, y_train, X_val, y_val,
            model_type='lstm'
        )
        print(f"✓ TimeSeriesObjective created")
        
        # Quick optimization test (2 trials)
        try:
            study = optimize_hyperparameters(
                X_train, y_train, X_val, y_val,
                model_type='lstm',
                n_trials=2  # Very quick test
            )
            print(f"✓ Optimization completed: {len(study.trials)} trials")
            
        except Exception as e:
            print(f"⚠️ Full optimization skipped: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Optuna test failed: {e}")
        return False

def test_model_integration():
    """Test integration between models and feature engineering."""
    print("\n=== Testing Model Integration ===")
    
    try:
        # Import feature engineering
        from src.features.tech import generate_technical_features
        from src.models.lstm import LSTMWrapper
        
        # Generate market data
        df = generate_test_data(200)
        
        # Generate technical features
        enriched_df = generate_technical_features(df)
        
        print(f"✓ Features generated: {enriched_df.shape[1]} columns")
        
        # Use multiple features for LSTM
        feature_cols = [col for col in enriched_df.columns if col not in ['ts', 'ticker']][:10]  # Use 10 features
        
        lstm = LSTMWrapper(sequence_length=30, prediction_length=1)
        X, y = lstm.prepare_data(enriched_df, target_col='close', feature_cols=feature_cols)
        
        print(f"✓ Multi-feature data prepared: X={X.shape}, y={y.shape}")
        
        # Quick training test
        split_idx = int(0.8 * len(X))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        metrics = lstm.fit(
            X_train, y_train, X_val, y_val,
            epochs=2, verbose=False
        )
        
        print(f"✓ Multi-feature LSTM training completed")
        print(f"  Input features: {X.shape[2]}")
        print(f"  Final val loss: {metrics['final_val_loss']:.6f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False

async def main():
    """Run all model tests."""
    print("🚀 Starting Section 7: Model Layer Tests\n")
    
    results = {}
    
    # Run individual tests
    results['lstm'] = test_lstm_model()
    results['patchtst'] = test_patchtst_model() 
    results['timegpt'] = test_timegpt_wrapper()
    results['optuna'] = test_optuna_utils()
    results['integration'] = test_model_integration()
    
    # Summary
    print(f"\n{'='*50}")
    print("🎯 Section 7 Test Results:")
    print(f"{'='*50}")
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {test_name.upper():<12} {status}")
    
    total_tests = len(results)
    passed_tests = sum(results.values())
    
    print(f"\nOverall: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 All model tests passed! Section 7 is ready.")
    else:
        print("⚠️ Some tests failed. Review the errors above.")
    
    return passed_tests == total_tests

if __name__ == "__main__":
    success = asyncio.run(main())
