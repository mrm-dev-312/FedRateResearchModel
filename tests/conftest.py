"""
Test configuration and fixtures for MSRK v3
Provides common test utilities, fixtures, and database setup for testing.
"""

import asyncio
import os
import pytest
import tempfile
from pathlib import Path
from typing import AsyncGenerator, Generator
from unittest.mock import MagicMock

# Set test environment
os.environ["TESTING"] = "1"

# Add src to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Provide a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)

@pytest.fixture
def sample_env_vars() -> Generator[dict, None, None]:
    """Provide sample environment variables for testing."""
    original_env = os.environ.copy()
    
    test_env = {
        "DATABASE_URL": "postgresql://test:test@localhost:5432/test_msrk",
        "FRED_API_KEY": "test_fred_key_12345",
        "YAHOO_USER_AGENT": "Mozilla/5.0 (test)",
        "TIMESGPT_API_KEY": "test_timesgpt_key",
        "GEMINI_API_KEY": "test_gemini_key"
    }
    
    # Set test environment variables
    os.environ.update(test_env)
    
    yield test_env
    
    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)

@pytest.fixture
async def mock_db_client():
    """Provide a mock database client for testing."""
    mock_client = MagicMock()
    
    # Mock common database operations
    mock_client.macrorelease.create = MagicMock()
    mock_client.macrorelease.find_many = MagicMock(return_value=[])
    mock_client.macrorelease.find_unique = MagicMock(return_value=None)
    
    mock_client.marketprice.create = MagicMock()
    mock_client.marketprice.find_many = MagicMock(return_value=[])
    
    mock_client.feature.create = MagicMock()
    mock_client.feature.find_many = MagicMock(return_value=[])
    
    mock_client.strategy.create = MagicMock()
    mock_client.strategy.find_many = MagicMock(return_value=[])
    
    mock_client.backtest.create = MagicMock()
    mock_client.backtest.find_many = MagicMock(return_value=[])
    
    mock_client.disconnect = MagicMock()
    
    return mock_client

@pytest.fixture
def sample_macro_data() -> list:
    """Provide sample macro economic data for testing."""
    return [
        {
            "indicator": "UNRATE",
            "description": "Unemployment Rate",
            "frequency": "MONTHLY",
            "unit": "Percent",
            "seasonal_adjustment": "SEASONALLY_ADJUSTED",
            "source": "FRED"
        },
        {
            "indicator": "CPIAUCSL", 
            "description": "Consumer Price Index",
            "frequency": "MONTHLY",
            "unit": "Index 1982-84=100",
            "seasonal_adjustment": "NOT_SEASONALLY_ADJUSTED",
            "source": "FRED"
        }
    ]

@pytest.fixture
def sample_market_data() -> list:
    """Provide sample market price data for testing."""
    from datetime import datetime, timedelta
    
    data = []
    base_date = datetime(2024, 1, 1)
    
    for i in range(5):
        date = base_date + timedelta(days=i)
        data.append({
            "symbol": "EURUSD",
            "timestamp": date,
            "open": 1.1000 + (i * 0.001),
            "high": 1.1050 + (i * 0.001),
            "low": 1.0950 + (i * 0.001),
            "close": 1.1025 + (i * 0.001),
            "volume": 1000000.0,
            "source": "YAHOO"
        })
    
    return data

@pytest.fixture
def sample_features() -> list:
    """Provide sample feature data for testing."""
    from datetime import datetime
    
    return [
        {
            "name": "SMA_20",
            "value": 1.1025,
            "timestamp": datetime(2024, 1, 1),
            "symbol": "EURUSD",
            "feature_type": "TECHNICAL"
        },
        {
            "name": "RSI_14",
            "value": 65.5,
            "timestamp": datetime(2024, 1, 1), 
            "symbol": "EURUSD",
            "feature_type": "TECHNICAL"
        }
    ]

@pytest.fixture
def mock_fred_response() -> dict:
    """Provide mock FRED API response data."""
    return {
        "realtime_start": "2024-01-01",
        "realtime_end": "2024-01-01",
        "observation_start": "1948-01-01",
        "observation_end": "9999-12-31",
        "units": "lin",
        "output_type": 1,
        "file_type": "json",
        "order_by": "observation_date",
        "sort_order": "asc",
        "count": 2,
        "offset": 0,
        "limit": 100000,
        "observations": [
            {
                "realtime_start": "2024-01-01",
                "realtime_end": "2024-01-01", 
                "date": "2023-12-01",
                "value": "3.7"
            },
            {
                "realtime_start": "2024-01-01",
                "realtime_end": "2024-01-01",
                "date": "2024-01-01", 
                "value": "3.5"
            }
        ]
    }

@pytest.fixture
def mock_yahoo_response() -> dict:
    """Provide mock Yahoo Finance response data."""
    return {
        "chart": {
            "result": [
                {
                    "meta": {
                        "symbol": "EURUSD=X",
                        "exchangeTimezoneName": "Europe/London",
                        "timezone": "GMT"
                    },
                    "timestamp": [1704067200, 1704153600],  # Unix timestamps
                    "indicators": {
                        "quote": [
                            {
                                "open": [1.1000, 1.1020],
                                "high": [1.1050, 1.1070],
                                "low": [1.0950, 1.0970],
                                "close": [1.1025, 1.1045],
                                "volume": [1000000, 1100000]
                            }
                        ]
                    }
                }
            ]
        }
    }

# Async fixtures for database testing
@pytest.fixture
async def db_session():
    """Provide a database session for integration tests."""
    # This would set up a test database session
    # For now, we'll use a mock since we don't have a test DB
    mock_session = MagicMock()
    yield mock_session
    await mock_session.disconnect()

# Markers for organizing tests
pytestmark = [
    pytest.mark.asyncio
]
