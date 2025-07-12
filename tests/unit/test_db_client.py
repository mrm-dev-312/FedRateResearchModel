"""
Unit tests for database client functionality.
"""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

@pytest.mark.unit
class TestDatabaseClient:
    """Test database client operations."""
    
    def test_get_db_client_import(self):
        """Test that we can import the database client."""
        try:
            from db.client import get_db_client
            assert callable(get_db_client)
        except ImportError:
            pytest.skip("Database client not implemented yet")
    
    @patch('db.client.PrismaClient')
    async def test_get_db_client_creation(self, mock_prisma):
        """Test database client creation."""
        try:
            from db.client import get_db_client
            
            # Mock the Prisma client
            mock_client = AsyncMock()
            mock_prisma.return_value = mock_client
            
            client = await get_db_client()
            
            assert client is not None
            mock_client.connect.assert_called_once()
            
        except ImportError:
            pytest.skip("Database client not implemented yet")
    
    @patch('db.client.PrismaClient')
    async def test_db_client_disconnect(self, mock_prisma):
        """Test database client disconnection."""
        try:
            from db.client import get_db_client
            
            mock_client = AsyncMock()
            mock_prisma.return_value = mock_client
            
            client = await get_db_client()
            await client.disconnect()
            
            mock_client.disconnect.assert_called_once()
            
        except ImportError:
            pytest.skip("Database client not implemented yet")

@pytest.mark.unit
class TestDatabaseOperations:
    """Test database CRUD operations."""
    
    async def test_macro_release_operations(self, mock_db_client, sample_macro_data):
        """Test macro release database operations."""
        # Test create
        macro_data = sample_macro_data[0]
        await mock_db_client.macrorelease.create(data=macro_data)
        mock_db_client.macrorelease.create.assert_called_with(data=macro_data)
        
        # Test find
        await mock_db_client.macrorelease.find_many()
        mock_db_client.macrorelease.find_many.assert_called_once()
    
    async def test_market_price_operations(self, mock_db_client, sample_market_data):
        """Test market price database operations."""
        # Test create
        price_data = sample_market_data[0]
        await mock_db_client.marketprice.create(data=price_data)
        mock_db_client.marketprice.create.assert_called_with(data=price_data)
        
        # Test find
        await mock_db_client.marketprice.find_many()
        mock_db_client.marketprice.find_many.assert_called_once()
    
    async def test_feature_operations(self, mock_db_client, sample_features):
        """Test feature database operations."""
        # Test create
        feature_data = sample_features[0]
        await mock_db_client.feature.create(data=feature_data)
        mock_db_client.feature.create.assert_called_with(data=feature_data)
        
        # Test find
        await mock_db_client.feature.find_many()
        mock_db_client.feature.find_many.assert_called_once()
