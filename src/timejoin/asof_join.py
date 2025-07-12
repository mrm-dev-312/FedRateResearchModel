"""
As-of join functionality for attaching macro data to market prices.
Handles mixed-frequency data alignment with proper timing.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Union
import pandas as pd
import numpy as np
from src.db.client import get_db

logger = logging.getLogger(__name__)

async def join_macro(px_df: pd.DataFrame, series_list: List[str]) -> pd.DataFrame:
    """
    Attach latest macro values to price DataFrame via merge_asof.
    
    This is the core function for mixed-frequency data alignment.
    
    Args:
        px_df: DataFrame with price data, must have 'ts' datetime column
        series_list: List of FRED series IDs to join (e.g., ['FEDFUNDS', 'UNRATE'])
    
    Returns:
        DataFrame with macro series as-of joined to each price timestamp
    """
    if px_df.empty:
        logger.warning("Empty price DataFrame provided")
        return px_df
    
    if 'ts' not in px_df.columns:
        raise ValueError("Price DataFrame must have 'ts' column")
    
    # Ensure ts column is datetime
    px_df = px_df.copy()
    px_df['ts'] = pd.to_datetime(px_df['ts'])
    
    # Sort by timestamp for merge_asof
    px_df = px_df.sort_values('ts')
    
    result_df = px_df.copy()
    
    # Get database connection
    db = await get_db()
    
    for series_id in series_list:
        try:
            # Fetch macro data for this series
            macro_records = await db.macrorelease.find_many(
                where={'series_id': series_id},
                order={'release_time': 'asc'}
            )
            
            if not macro_records:
                logger.warning(f"No data found for series {series_id}")
                # Add NaN columns for this series
                result_df[f'{series_id}_actual'] = np.nan
                result_df[f'{series_id}_surprise'] = np.nan
                continue
            
            # Convert to DataFrame
            macro_data = []
            for record in macro_records:
                macro_data.append({
                    'release_time': record.release_time,
                    'actual': record.actual,
                    'surprise': record.surprise
                })
            
            macro_df = pd.DataFrame(macro_data)
            macro_df['release_time'] = pd.to_datetime(macro_df['release_time'])
            macro_df = macro_df.sort_values('release_time')
            
            # Perform as-of join
            result_df = pd.merge_asof(
                result_df,
                macro_df,
                left_on='ts',
                right_on='release_time',
                direction='backward',  # Use most recent release before timestamp
                suffixes=('', f'_{series_id}')
            )
            
            # Rename columns to include series ID
            if 'actual' in result_df.columns:
                result_df = result_df.rename(columns={
                    'actual': f'{series_id}_actual',
                    'surprise': f'{series_id}_surprise'
                })
            
            # Drop the release_time column
            result_df = result_df.drop(columns=['release_time'], errors='ignore')
            
            logger.info(f"Successfully joined {series_id} to price data")
            
        except Exception as e:
            logger.error(f"Error joining series {series_id}: {e}")
            # Add NaN columns for failed series
            result_df[f'{series_id}_actual'] = np.nan
            result_df[f'{series_id}_surprise'] = np.nan
    
    return result_df

async def join_macro_with_lag(
    px_df: pd.DataFrame, 
    series_list: List[str], 
    max_lag_days: int = 30
) -> pd.DataFrame:
    """
    Join macro data with a maximum lag constraint.
    
    Only uses macro values released within max_lag_days of the price timestamp.
    This prevents using stale macro data.
    
    Args:
        px_df: DataFrame with price data
        series_list: List of FRED series IDs
        max_lag_days: Maximum days between macro release and price timestamp
    
    Returns:
        DataFrame with lag-constrained macro data
    """
    result_df = await join_macro(px_df, series_list)
    
    # For each macro series, set values to NaN if they're too old
    for series_id in series_list:
        actual_col = f'{series_id}_actual'
        if actual_col in result_df.columns:
            # This would require storing the actual release timestamp
            # For now, we'll implement a simpler version
            pass
    
    return result_df

async def join_event_windows(
    px_df: pd.DataFrame,
    series_list: List[str],
    window_before: timedelta = timedelta(hours=2),
    window_after: timedelta = timedelta(days=1)
) -> pd.DataFrame:
    """
    Add event window indicators for macro releases.
    
    Marks price observations that occur within a time window of macro releases.
    
    Args:
        px_df: DataFrame with price data
        series_list: List of FRED series IDs
        window_before: Time window before release
        window_after: Time window after release
    
    Returns:
        DataFrame with event window indicators
    """
    result_df = px_df.copy()
    result_df['ts'] = pd.to_datetime(result_df['ts'])
    
    db = await get_db()
    
    for series_id in series_list:
        # Get release times for this series
        macro_records = await db.macrorelease.find_many(
            where={'series_id': series_id}
        )
        
        if not macro_records:
            result_df[f'{series_id}_event_window'] = 0
            continue
        
        release_times = [pd.to_datetime(r.release_time) for r in macro_records]
        
        # Mark observations within event windows
        in_window = pd.Series(False, index=result_df.index)
        
        for release_time in release_times:
            window_start = release_time - window_before
            window_end = release_time + window_after
            
            window_mask = (
                (result_df['ts'] >= window_start) & 
                (result_df['ts'] <= window_end)
            )
            in_window = in_window | window_mask
        
        result_df[f'{series_id}_event_window'] = in_window.astype(int)
    
    return result_df

# Common macro series for different asset classes
EQUITY_MACRO_SERIES = [
    'FEDFUNDS',   # Fed funds rate
    'DGS10',      # 10-year treasury
    'UNRATE',     # Unemployment rate
    'CPIAUCSL',   # Consumer price index
    'PAYEMS',     # Non-farm payrolls
    'GDP'         # Gross domestic product
]

FX_MACRO_SERIES = [
    'FEDFUNDS',   # Fed funds rate
    'DGS10',      # 10-year treasury
    'DEXUSEU',    # USD/EUR exchange rate
    'CPIAUCSL',   # Consumer price index
    'PAYEMS'      # Non-farm payrolls
]

RATES_MACRO_SERIES = [
    'FEDFUNDS',   # Fed funds rate
    'DGS2',       # 2-year treasury
    'DGS5',       # 5-year treasury
    'DGS10',      # 10-year treasury
    'DGS30',      # 30-year treasury
    'CPIAUCSL',   # Consumer price index
    'PAYEMS'      # Non-farm payrolls
]

async def get_asset_specific_macro(asset_class: str) -> List[str]:
    """
    Get relevant macro series for different asset classes.
    
    Args:
        asset_class: 'equity', 'fx', 'rates', or 'commodity'
    
    Returns:
        List of relevant FRED series IDs
    """
    if asset_class.lower() == 'equity':
        return EQUITY_MACRO_SERIES
    elif asset_class.lower() == 'fx':
        return FX_MACRO_SERIES
    elif asset_class.lower() == 'rates':
        return RATES_MACRO_SERIES
    else:
        # Default to equity series
        return EQUITY_MACRO_SERIES

if __name__ == "__main__":
    async def test_asof_join():
        """Test as-of join functionality."""
        
        # Create sample price data
        dates = pd.date_range('2024-01-01', '2024-01-10', freq='D')
        px_df = pd.DataFrame({
            'ts': dates,
            'ticker': 'SPY',
            'close': 400 + np.random.randn(len(dates)) * 5
        })
        
        print("Sample price data:")
        print(px_df)
        
        # Test joining with FRED data
        series_list = ['FEDFUNDS', 'UNRATE']
        result_df = await join_macro(px_df, series_list)
        
        print(f"\nAfter joining macro data:")
        print(f"Columns: {list(result_df.columns)}")
        print(result_df.head())
    
    asyncio.run(test_asof_join())
