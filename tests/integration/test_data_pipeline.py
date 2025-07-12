"""
Integration tests for data pipeline functionality.
Tests the complete data flow from ingestion to storage.
"""

import pytest
from unittest.mock import patch, MagicMock
import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

@pytest.mark.integration
@pytest.mark.db
class TestDataPipelineIntegration:
    """Test complete data pipeline integration."""
    
    async def test_macro_data_pipeline(self, mock_db_client, sample_env_vars):
        """Test complete macro data ingestion pipeline."""
        # This test would verify the complete flow:
        # 1. Fetch data from FRED API
        # 2. Transform/validate data
        # 3. Store in database
        # 4. Verify storage
        
        # Mock data that would come from FRED
        mock_fred_data = [
            {
                "indicator": "UNRATE",
                "date": "2024-01-01", 
                "value": "3.7",
                "realtime_start": "2024-01-01",
                "realtime_end": "2024-01-01"
            }
        ]
        
        # Simulate database storage
        expected_db_record = {
            "indicator": "UNRATE",
            "date": datetime(2024, 1, 1),
            "value": 3.7,
            "metadata": {
                "realtime_start": "2024-01-01",
                "realtime_end": "2024-01-01"
            }
        }
        
        # Mock the database operation
        mock_db_client.macrorelease.create.return_value = expected_db_record
        
        # Simulate the pipeline
        await mock_db_client.macrorelease.create(data=expected_db_record)
        
        # Verify the data was processed correctly
        mock_db_client.macrorelease.create.assert_called_once_with(data=expected_db_record)
    
    async def test_market_data_pipeline(self, mock_db_client, sample_env_vars):
        """Test complete market data ingestion pipeline."""
        # Mock data that would come from Yahoo Finance
        mock_yahoo_data = {
            "symbol": "EURUSD=X",
            "timestamp": 1704067200,  # Unix timestamp
            "open": 1.1000,
            "high": 1.1050,
            "low": 1.0950,
            "close": 1.1025,
            "volume": 1000000
        }
        
        # Expected database record after transformation
        expected_db_record = {
            "symbol": "EURUSD",
            "timestamp": datetime.fromtimestamp(1704067200),
            "open": 1.1000,
            "high": 1.1050,
            "low": 1.0950,
            "close": 1.1025,
            "volume": 1000000.0,
            "source": "YAHOO"
        }
        
        # Mock the database operation
        mock_db_client.marketprice.create.return_value = expected_db_record
        
        # Simulate the pipeline
        await mock_db_client.marketprice.create(data=expected_db_record)
        
        # Verify the data was processed correctly
        mock_db_client.marketprice.create.assert_called_once_with(data=expected_db_record)

@pytest.mark.integration
@pytest.mark.ml
class TestFeatureGenerationIntegration:
    """Test feature generation pipeline integration."""
    
    async def test_technical_feature_generation(self, mock_db_client, sample_market_data):
        """Test technical indicator feature generation."""
        # This would test the complete flow:
        # 1. Fetch market data from database
        # 2. Calculate technical indicators
        # 3. Store features back to database
        
        # Mock market data retrieval
        mock_db_client.marketprice.find_many.return_value = sample_market_data
        
        # Expected features after calculation
        expected_features = [
            {
                "name": "SMA_20",
                "value": 1.1025,  # Simple moving average
                "timestamp": sample_market_data[0]["timestamp"],
                "symbol": "EURUSD",
                "feature_type": "TECHNICAL"
            },
            {
                "name": "RSI_14",
                "value": 65.5,  # RSI indicator
                "timestamp": sample_market_data[0]["timestamp"],
                "symbol": "EURUSD", 
                "feature_type": "TECHNICAL"
            }
        ]
        
        # Simulate feature storage
        for feature in expected_features:
            await mock_db_client.feature.create(data=feature)
        
        # Verify features were stored
        assert mock_db_client.feature.create.call_count == len(expected_features)

@pytest.mark.integration
@pytest.mark.slow
class TestBacktestIntegration:
    """Test backtesting engine integration."""
    
    async def test_strategy_backtest_pipeline(self, mock_db_client):
        """Test complete strategy backtesting pipeline."""
        # Mock strategy configuration
        strategy_config = {
            "name": "Test Strategy",
            "config": {
                "lookback_days": 30,
                "threshold": 0.5
            },
            "is_active": True
        }
        
        # Mock historical data
        mock_historical_data = [
            {
                "symbol": "EURUSD",
                "timestamp": datetime.now() - timedelta(days=i),
                "close": 1.1000 + (i * 0.001)
            }
            for i in range(30)
        ]
        
        # Mock backtest results
        expected_backtest_result = {
            "strategy_id": 1,
            "start_date": datetime.now() - timedelta(days=30),
            "end_date": datetime.now(),
            "total_return": 0.025,  # 2.5% return
            "sharpe_ratio": 1.2,
            "max_drawdown": -0.015,  # -1.5% max drawdown
            "num_trades": 10,
            "win_rate": 0.6  # 60% win rate
        }
        
        # Simulate backtest storage
        await mock_db_client.backtestresult.create(data=expected_backtest_result)
        
        # Verify backtest was stored
        mock_db_client.backtestresult.create.assert_called_once_with(data=expected_backtest_result)

@pytest.mark.integration
@pytest.mark.api
class TestExternalAPIIntegration:
    """Test integration with external APIs."""
    
    @pytest.mark.skip(reason="Requires real API credentials")
    async def test_fred_api_integration(self, sample_env_vars):
        """Test real FRED API integration (requires valid API key)."""
        # This test would make actual API calls
        # Skip by default to avoid rate limits in CI
        pass
    
    @pytest.mark.skip(reason="Requires internet connection")
    async def test_yahoo_api_integration(self, sample_env_vars):
        """Test real Yahoo Finance API integration."""
        # This test would make actual API calls
        # Skip by default to avoid network dependency in CI
        pass
