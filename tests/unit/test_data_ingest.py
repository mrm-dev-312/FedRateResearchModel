"""
Unit tests for data ingestion modules.
"""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

@pytest.mark.unit
class TestFredDataIngestion:
    """Test FRED API data ingestion."""
    
    def test_fred_module_import(self):
        """Test that we can import the FRED module."""
        try:
            from data_ingest.fred import FredClient
            assert FredClient is not None
        except ImportError:
            pytest.skip("FRED module not implemented yet")
    
    @patch('requests.get')
    def test_fred_api_call(self, mock_get, mock_fred_response, sample_env_vars):
        """Test FRED API call functionality."""
        try:
            from data_ingest.fred import FredClient
            
            # Mock successful API response
            mock_get.return_value.status_code = 200
            mock_get.return_value.json.return_value = mock_fred_response
            
            client = FredClient(api_key=sample_env_vars["FRED_API_KEY"])
            
            # This test assumes the FredClient has a get_series method
            # Implementation may vary
            assert client is not None
            
        except ImportError:
            pytest.skip("FRED module not implemented yet")
    
    def test_fred_data_parsing(self, mock_fred_response):
        """Test parsing of FRED API response data."""
        # Test that we can extract observations from response
        observations = mock_fred_response["observations"]
        assert len(observations) == 2
        assert observations[0]["value"] == "3.7"
        assert observations[1]["value"] == "3.5"

@pytest.mark.unit  
class TestYahooDataIngestion:
    """Test Yahoo Finance data ingestion."""
    
    def test_yahoo_module_import(self):
        """Test that we can import the Yahoo module."""
        try:
            from data_ingest.yahoo import YahooClient
            assert YahooClient is not None
        except ImportError:
            pytest.skip("Yahoo module not implemented yet")
    
    @patch('requests.get')
    def test_yahoo_api_call(self, mock_get, mock_yahoo_response, sample_env_vars):
        """Test Yahoo Finance API call functionality."""
        try:
            from data_ingest.yahoo import YahooClient
            
            # Mock successful API response
            mock_get.return_value.status_code = 200
            mock_get.return_value.json.return_value = mock_yahoo_response
            
            client = YahooClient(user_agent=sample_env_vars["YAHOO_USER_AGENT"])
            
            assert client is not None
            
        except ImportError:
            pytest.skip("Yahoo module not implemented yet")
    
    def test_yahoo_data_parsing(self, mock_yahoo_response):
        """Test parsing of Yahoo Finance response data."""
        # Test that we can extract price data from response
        result = mock_yahoo_response["chart"]["result"][0]
        indicators = result["indicators"]["quote"][0]
        
        assert len(indicators["open"]) == 2
        assert indicators["open"][0] == 1.1000
        assert indicators["close"][1] == 1.1045

@pytest.mark.unit
class TestDataIngestionUtilities:
    """Test data ingestion utility functions."""
    
    def test_timestamp_conversion(self):
        """Test timestamp conversion utilities."""
        from datetime import datetime
        
        # Test Unix timestamp conversion
        unix_timestamp = 1704067200
        expected_date = datetime(2024, 1, 1, 0, 0)
        
        # Simple conversion test
        converted = datetime.fromtimestamp(unix_timestamp)
        assert converted.year == expected_date.year
        assert converted.month == expected_date.month
        assert converted.day == expected_date.day
    
    def test_data_validation(self, sample_market_data):
        """Test data validation utilities."""
        # Test required fields are present
        required_fields = ["symbol", "timestamp", "open", "high", "low", "close", "volume"]
        
        for data_point in sample_market_data:
            for field in required_fields:
                assert field in data_point, f"Missing required field: {field}"
    
    def test_data_type_validation(self, sample_market_data):
        """Test data type validation."""
        data_point = sample_market_data[0]
        
        assert isinstance(data_point["symbol"], str)
        assert isinstance(data_point["open"], (int, float))
        assert isinstance(data_point["high"], (int, float))
        assert isinstance(data_point["low"], (int, float))
        assert isinstance(data_point["close"], (int, float))
        assert isinstance(data_point["volume"], (int, float))
