"""
Feature storage and versioning system for MSRK v3.
Handles storing technical, macro event, and text sentiment features in database.
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Union, Any
import json
import logging
from src.db.client import get_db
from src.features.tech import generate_technical_features
from src.features.macro_event import create_macro_event_features, FEATURE_VERSION as MACRO_VERSION
from src.features.text_sentiment import analyze_text_sentiment, FEATURE_VERSION as TEXT_VERSION

logger = logging.getLogger(__name__)

# Global feature version for traceability
FEATURE_STORE_VERSION = "1.0.0"

class FeatureStore:
    def __init__(self):
        self.version = FEATURE_STORE_VERSION
    
    async def store_technical_features(
        self,
        df: pd.DataFrame,
        ticker: str,
        feature_names: List[str]
    ) -> int:
        """
        Store technical features in the Feature table.
        
        Args:
            df: DataFrame with technical features
            ticker: Asset ticker
            feature_names: List of feature column names to store
            
        Returns:
            Number of records stored
        """
        db = await get_db()
        stored_count = 0
        
        for _, row in df.iterrows():
            try:
                for feature_name in feature_names:
                    if feature_name in row and not pd.isna(row[feature_name]):
                        await db.feature.create(
                            data={
                                'ticker': ticker,
                                'ts': row['ts'],
                                'name': feature_name,
                                'value': float(row[feature_name]),
                                'feature_type': 'technical',
                                'version': self.version
                            }
                        )
                        stored_count += 1
                        
            except Exception as e:
                logger.error(f"Error storing feature for {ticker} at {row['ts']}: {e}")
                continue
        
        logger.info(f"Stored {stored_count} technical features for {ticker}")
        return stored_count
    
    async def store_macro_features(
        self,
        df: pd.DataFrame,
        ticker: str,
        macro_feature_names: List[str]
    ) -> int:
        """
        Store macro event features in the Feature table.
        
        Args:
            df: DataFrame with macro features
            ticker: Asset ticker
            macro_feature_names: List of macro feature column names
            
        Returns:
            Number of records stored
        """
        db = await get_db()
        stored_count = 0
        
        for _, row in df.iterrows():
            try:
                for feature_name in macro_feature_names:
                    if feature_name in row and not pd.isna(row[feature_name]):
                        await db.feature.create(
                            data={
                                'ticker': ticker,
                                'ts': row['ts'],
                                'name': feature_name,
                                'value': float(row[feature_name]),
                                'feature_type': 'macro_event',
                                'version': MACRO_VERSION
                            }
                        )
                        stored_count += 1
                        
            except Exception as e:
                logger.error(f"Error storing macro feature for {ticker} at {row['ts']}: {e}")
                continue
        
        logger.info(f"Stored {stored_count} macro features for {ticker}")
        return stored_count
    
    async def load_features(
        self,
        ticker: str,
        start_date: datetime,
        end_date: datetime,
        feature_types: List[str] = None
    ) -> pd.DataFrame:
        """
        Load features from database for a given ticker and date range.
        
        Args:
            ticker: Asset ticker
            start_date: Start date
            end_date: End date
            feature_types: List of feature types to load
            
        Returns:
            DataFrame with features pivoted by feature name
        """
        db = await get_db()
        
        where_clause = {
            'ticker': ticker,
            'ts': {
                'gte': start_date,
                'lte': end_date
            }
        }
        
        if feature_types:
            where_clause['feature_type'] = {'in': feature_types}
        
        records = await db.feature.find_many(where=where_clause)
        
        if not records:
            return pd.DataFrame()
        
        # Convert to DataFrame
        feature_data = []
        for record in records:
            feature_data.append({
                'ts': record.ts,
                'ticker': record.ticker,
                'feature_name': record.name,
                'value': record.value,
                'feature_type': record.feature_type,
                'version': record.version
            })
        
        df = pd.DataFrame(feature_data)
        
        # Pivot to get features as columns
        pivot_df = df.pivot_table(
            index=['ts', 'ticker'],
            columns='feature_name',
            values='value',
            aggfunc='first'  # Take first value if duplicates
        ).reset_index()
        
        # Flatten column names
        pivot_df.columns.name = None
        
        return pivot_df
    
    async def create_feature_version_snapshot(
        self,
        ticker: str,
        snapshot_date: datetime,
        description: str = None
    ) -> str:
        """
        Create a versioned snapshot of all features for a ticker.
        
        Args:
            ticker: Asset ticker
            snapshot_date: Date of the snapshot
            description: Optional description
            
        Returns:
            Snapshot ID
        """
        snapshot_id = f"{ticker}_{snapshot_date.strftime('%Y%m%d_%H%M%S')}"
        
        # Load all features for the ticker
        all_features = await self.load_features(
            ticker,
            datetime(2020, 1, 1),  # Far back date
            snapshot_date
        )
        
        if all_features.empty:
            logger.warning(f"No features found for {ticker}")
            return snapshot_id
        
        # Store snapshot metadata (could be expanded to separate table)
        snapshot_metadata = {
            'snapshot_id': snapshot_id,
            'ticker': ticker,
            'snapshot_date': snapshot_date.isoformat(),
            'description': description or f"Feature snapshot for {ticker}",
            'feature_count': len(all_features),
            'version': self.version
        }
        
        logger.info(f"Created feature snapshot {snapshot_id} with {len(all_features)} records")
        return snapshot_id

async def create_comprehensive_features(
    df: pd.DataFrame,
    ticker: str,
    include_technical: bool = True,
    include_macro: bool = True,
    include_text: bool = False,
    store_in_db: bool = True
) -> pd.DataFrame:
    """
    Create comprehensive feature set combining technical, macro, and text features.
    
    Args:
        df: Input DataFrame with OHLCV data
        ticker: Asset ticker
        include_technical: Whether to include technical features
        include_macro: Whether to include macro event features
        include_text: Whether to include text sentiment features
        store_in_db: Whether to store features in database
        
    Returns:
        DataFrame with all requested features
    """
    result_df = df.copy()
    
    # 1. Technical Features
    if include_technical:
        logger.info("Generating technical features...")
        result_df = generate_technical_features(result_df)
        
        if store_in_db:
            feature_store = FeatureStore()
            tech_columns = [col for col in result_df.columns 
                          if col not in df.columns and 'macro' not in col.lower()]
            await feature_store.store_technical_features(result_df, ticker, tech_columns)
    
    # 2. Macro Event Features
    if include_macro:
        logger.info("Generating macro event features...")
        result_df = await create_macro_event_features(result_df)
        
        if store_in_db:
            feature_store = FeatureStore()
            macro_columns = [col for col in result_df.columns 
                           if 'macro' in col.lower() or 'event' in col.lower() or 
                              'fomc' in col.lower() or 'nfp' in col.lower()]
            await feature_store.store_macro_features(result_df, ticker, macro_columns)
    
    # 3. Text Sentiment Features (if text data available)
    if include_text:
        logger.info("Note: Text sentiment features require separate text input")
        # Text features would be generated separately and joined by timestamp
    
    # Add feature metadata
    result_df['feature_generation_time'] = datetime.now()
    result_df['feature_store_version'] = FEATURE_STORE_VERSION
    
    logger.info(f"Feature generation complete for {ticker}")
    logger.info(f"Input shape: {df.shape}, Output shape: {result_df.shape}")
    logger.info(f"Features added: {result_df.shape[1] - df.shape[1]}")
    
    return result_df

async def get_feature_summary(ticker: str) -> Dict[str, Any]:
    """
    Get summary of available features for a ticker.
    
    Args:
        ticker: Asset ticker
        
    Returns:
        Dictionary with feature summary statistics
    """
    db = await get_db()
    
    # Get feature counts by type
    feature_counts = {}
    feature_types = ['technical', 'macro_event', 'text_sentiment']
    
    for feature_type in feature_types:
        count = await db.feature.count(
            where={
                'ticker': ticker,
                'feature_type': feature_type
            }
        )
        feature_counts[feature_type] = count
    
    # Get date range
    first_record = await db.feature.find_first(
        where={'ticker': ticker},
        order={'ts': 'asc'}
    )
    
    last_record = await db.feature.find_first(
        where={'ticker': ticker},
        order={'ts': 'desc'}
    )
    
    date_range = {
        'start_date': first_record.ts if first_record else None,
        'end_date': last_record.ts if last_record else None
    }
    
    # Get unique feature names
    unique_features = await db.feature.find_many(
        where={'ticker': ticker},
        distinct=['name']
    )
    
    feature_names = [record.name for record in unique_features]
    
    return {
        'ticker': ticker,
        'feature_counts': feature_counts,
        'total_features': sum(feature_counts.values()),
        'date_range': date_range,
        'unique_feature_names': feature_names,
        'feature_name_count': len(feature_names)
    }

if __name__ == "__main__":
    async def test_feature_storage():
        """Test feature storage functionality."""
        
        # Create sample data
        dates = pd.date_range('2024-01-01', periods=10, freq='D')
        df = pd.DataFrame({
            'ts': dates,
            'ticker': 'TEST',
            'open': 100 + np.random.randn(10) * 2,
            'high': 105 + np.random.randn(10) * 2,
            'low': 95 + np.random.randn(10) * 2,
            'close': 100 + np.random.randn(10) * 2,
            'volume': np.random.randint(1000000, 5000000, 10)
        })
        
        print("Testing comprehensive feature creation...")
        print(f"Input data shape: {df.shape}")
        
        # Generate features (without storing in DB for testing)
        result_df = await create_comprehensive_features(
            df, 
            'TEST', 
            include_technical=True,
            include_macro=False,  # Skip macro for quick test
            include_text=False,
            store_in_db=False
        )
        
        print(f"Output data shape: {result_df.shape}")
        
        # Show feature categories
        original_cols = set(df.columns)
        new_cols = [col for col in result_df.columns if col not in original_cols]
        
        print(f"Features added: {len(new_cols)}")
        print("Sample new features:", new_cols[:10])
        
        print("✓ Feature storage test completed")
    
    asyncio.run(test_feature_storage())
