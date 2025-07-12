"""
Macro event feature engineering for economic surprise and event impact analysis.
Handles surprise calculation and event window detection within ±N minutes.
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Tuple
from src.db.client import get_db
import logging

logger = logging.getLogger(__name__)

# Feature version constant for traceability
FEATURE_VERSION = "1.0.0"

class MacroEventFeatures:
    def __init__(self):
        self.feature_version = FEATURE_VERSION
    
    async def calculate_surprises(
        self, 
        df: pd.DataFrame, 
        series_ids: List[str]
    ) -> pd.DataFrame:
        """
        Calculate economic surprises (actual vs consensus) for macro releases.
        
        Args:
            df: DataFrame with market data and timestamps
            series_ids: List of FRED series IDs to calculate surprises for
            
        Returns:
            DataFrame with surprise columns added
        """
        result_df = df.copy()
        
        db = await get_db()
        
        for series_id in series_ids:
            try:
                # Fetch macro releases with consensus data
                macro_records = await db.macrorelease.find_many(
                    where={
                        'series_id': series_id,
                        'consensus': {'not': None}  # Only records with consensus
                    }
                )
                
                if not macro_records:
                    logger.warning(f"No consensus data found for {series_id}")
                    continue
                
                # Convert to DataFrame
                macro_df = pd.DataFrame([
                    {
                        'release_time': record.release_time,
                        'actual': record.actual,
                        'consensus': record.consensus,
                        'surprise': record.surprise
                    } for record in macro_records
                ])
                
                macro_df['release_time'] = pd.to_datetime(macro_df['release_time'])
                
                # Calculate surprises if not already calculated
                if 'surprise' not in macro_df.columns or macro_df['surprise'].isna().all():
                    # Absolute surprise
                    macro_df['surprise'] = macro_df['actual'] - macro_df['consensus']
                    
                    # Relative surprise (percentage)
                    macro_df['surprise_pct'] = (
                        (macro_df['actual'] - macro_df['consensus']) / 
                        macro_df['consensus'].abs()
                    ) * 100
                    
                    # Standardized surprise (z-score)
                    surprise_mean = macro_df['surprise'].mean()
                    surprise_std = macro_df['surprise'].std()
                    macro_df['surprise_zscore'] = (
                        (macro_df['surprise'] - surprise_mean) / surprise_std
                    )
                
                # Store calculated surprises for this series
                result_df[f'{series_id}_surprise'] = np.nan
                result_df[f'{series_id}_surprise_pct'] = np.nan
                result_df[f'{series_id}_surprise_zscore'] = np.nan
                
                # Map surprises to closest timestamps in result_df
                for _, macro_row in macro_df.iterrows():
                    release_time = macro_row['release_time']
                    
                    # Find closest timestamp in result_df
                    time_diffs = abs(pd.to_datetime(result_df['ts']) - release_time)
                    closest_idx = time_diffs.idxmin()
                    
                    # Only assign if within reasonable time window (e.g., 1 day)
                    if time_diffs.iloc[closest_idx] <= timedelta(days=1):
                        result_df.loc[closest_idx, f'{series_id}_surprise'] = macro_row['surprise']
                        result_df.loc[closest_idx, f'{series_id}_surprise_pct'] = macro_row.get('surprise_pct', np.nan)
                        result_df.loc[closest_idx, f'{series_id}_surprise_zscore'] = macro_row.get('surprise_zscore', np.nan)
                
                logger.info(f"Added surprise features for {series_id}")
                
            except Exception as e:
                logger.error(f"Error calculating surprises for {series_id}: {e}")
                continue
        
        return result_df
    
    def create_event_windows(
        self,
        df: pd.DataFrame,
        event_times: List[pd.Timestamp],
        window_before: timedelta = timedelta(minutes=30),
        window_after: timedelta = timedelta(minutes=60),
        event_name: str = "event"
    ) -> pd.DataFrame:
        """
        Create event dummy variables within ±N minutes of releases.
        
        Args:
            df: DataFrame with timestamp column
            event_times: List of event timestamps
            window_before: Time window before event
            window_after: Time window after event
            event_name: Name for the event columns
            
        Returns:
            DataFrame with event window features
        """
        result_df = df.copy()
        result_df['ts'] = pd.to_datetime(result_df['ts'])
        
        # Initialize event columns
        result_df[f'{event_name}_window'] = 0
        result_df[f'{event_name}_minutes_to'] = np.nan
        result_df[f'{event_name}_minutes_since'] = np.nan
        result_df[f'{event_name}_impact_decay'] = 0.0
        
        for event_time in event_times:
            event_time = pd.to_datetime(event_time)
            
            # Define event window
            window_start = event_time - window_before
            window_end = event_time + window_after
            
            # Create masks for different event phases
            in_window = (
                (result_df['ts'] >= window_start) & 
                (result_df['ts'] <= window_end)
            )
            
            pre_event = (
                (result_df['ts'] >= window_start) & 
                (result_df['ts'] < event_time)
            )
            
            post_event = (
                (result_df['ts'] >= event_time) & 
                (result_df['ts'] <= window_end)
            )
            
            # Mark observations in event window
            result_df.loc[in_window, f'{event_name}_window'] = 1
            
            # Calculate minutes to/since event
            time_diff_minutes = (result_df['ts'] - event_time).dt.total_seconds() / 60
            
            # Minutes until event (negative values)
            result_df.loc[pre_event, f'{event_name}_minutes_to'] = time_diff_minutes[pre_event]
            
            # Minutes since event (positive values)
            result_df.loc[post_event, f'{event_name}_minutes_since'] = time_diff_minutes[post_event]
            
            # Calculate impact decay (exponential decay after event)
            half_life_minutes = 30  # Impact halves every 30 minutes
            decay_rate = np.log(2) / half_life_minutes
            
            post_event_decay = np.exp(-decay_rate * time_diff_minutes[post_event])
            result_df.loc[post_event, f'{event_name}_impact_decay'] = np.maximum(
                result_df.loc[post_event, f'{event_name}_impact_decay'],
                post_event_decay
            )
        
        return result_df
    
    async def create_fomc_features(
        self,
        df: pd.DataFrame,
        window_before: timedelta = timedelta(hours=2),
        window_after: timedelta = timedelta(hours=24)
    ) -> pd.DataFrame:
        """
        Create FOMC-specific event features.
        
        Args:
            df: DataFrame with market data
            window_before: Time window before FOMC announcement
            window_after: Time window after FOMC announcement
            
        Returns:
            DataFrame with FOMC event features
        """
        db = await get_db()
        
        # Get FOMC meeting dates (Fed Funds Rate releases)
        fomc_records = await db.macrorelease.find_many(
            where={'series_id': 'FEDFUNDS'}
        )
        
        if not fomc_records:
            logger.warning("No FOMC data found")
            return df
        
        fomc_times = [pd.to_datetime(record.release_time) for record in fomc_records]
        
        # Create FOMC event windows
        result_df = self.create_event_windows(
            df, 
            fomc_times, 
            window_before, 
            window_after, 
            "fomc"
        )
        
        return result_df
    
    async def create_nfp_features(
        self,
        df: pd.DataFrame,
        window_before: timedelta = timedelta(minutes=30),
        window_after: timedelta = timedelta(hours=2)
    ) -> pd.DataFrame:
        """
        Create Non-Farm Payrolls event features.
        
        Args:
            df: DataFrame with market data
            window_before: Time window before NFP release
            window_after: Time window after NFP release
            
        Returns:
            DataFrame with NFP event features
        """
        db = await get_db()
        
        # Get NFP release dates
        nfp_records = await db.macrorelease.find_many(
            where={'series_id': 'PAYEMS'}
        )
        
        if not nfp_records:
            logger.warning("No NFP data found")
            return df
        
        nfp_times = [pd.to_datetime(record.release_time) for record in nfp_records]
        
        # Create NFP event windows
        result_df = self.create_event_windows(
            df,
            nfp_times,
            window_before,
            window_after,
            "nfp"
        )
        
        return result_df
    
    def calculate_event_clustering(
        self,
        df: pd.DataFrame,
        event_columns: List[str],
        window: timedelta = timedelta(hours=4)
    ) -> pd.DataFrame:
        """
        Calculate event clustering features (multiple events close together).
        
        Args:
            df: DataFrame with event features
            event_columns: List of event window column names
            window: Time window to look for clustered events
            
        Returns:
            DataFrame with clustering features
        """
        result_df = df.copy()
        result_df['ts'] = pd.to_datetime(result_df['ts'])
        
        # Count number of concurrent events
        event_sum = result_df[event_columns].sum(axis=1)
        result_df['event_cluster_count'] = event_sum
        
        # Binary indicator for multiple events
        result_df['multiple_events'] = (event_sum > 1).astype(int)
        
        # Rolling event density (events in rolling window)
        window_minutes = int(window.total_seconds() / 60)
        rolling_window = f'{window_minutes}T'  # Convert to pandas frequency
        
        result_df['event_density'] = (
            result_df[event_columns]
            .sum(axis=1)
            .rolling(rolling_window, min_periods=1)
            .sum()
        )
        
        return result_df

async def create_macro_event_features(
    df: pd.DataFrame,
    series_ids: List[str] = None,
    include_fomc: bool = True,
    include_nfp: bool = True,
    include_clustering: bool = True
) -> pd.DataFrame:
    """
    Main function to create comprehensive macro event features.
    
    Args:
        df: DataFrame with market data and timestamps
        series_ids: List of macro series IDs for surprise calculation
        include_fomc: Whether to include FOMC event features
        include_nfp: Whether to include NFP event features
        include_clustering: Whether to include event clustering features
        
    Returns:
        DataFrame with macro event features added
    """
    if series_ids is None:
        series_ids = ['FEDFUNDS', 'PAYEMS', 'UNRATE', 'CPIAUCSL']
    
    event_features = MacroEventFeatures()
    result_df = df.copy()
    
    # Add surprise features
    result_df = await event_features.calculate_surprises(result_df, series_ids)
    
    event_columns = []
    
    # Add FOMC features
    if include_fomc:
        result_df = await event_features.create_fomc_features(result_df)
        event_columns.extend(['fomc_window'])
    
    # Add NFP features
    if include_nfp:
        result_df = await event_features.create_nfp_features(result_df)
        event_columns.extend(['nfp_window'])
    
    # Add event clustering features
    if include_clustering and event_columns:
        result_df = event_features.calculate_event_clustering(result_df, event_columns)
    
    # Add feature version for traceability
    result_df['macro_event_feature_version'] = FEATURE_VERSION
    
    return result_df

if __name__ == "__main__":
    async def test_macro_events():
        """Test macro event feature creation."""
        
        # Create sample data
        dates = pd.date_range('2024-01-01', periods=100, freq='h')
        df = pd.DataFrame({
            'ts': dates,
            'ticker': 'SPY',
            'close': 400 + np.random.randn(100) * 5
        })
        
        print("Testing macro event features...")
        print(f"Input data shape: {df.shape}")
        
        # Test event features
        result_df = await create_macro_event_features(df)
        
        print(f"Output data shape: {result_df.shape}")
        event_cols = [col for col in result_df.columns if 'event' in col.lower() or 'fomc' in col or 'nfp' in col]
        print(f"Event columns added: {event_cols}")
        
        print("✓ Macro event features test completed")
    
    asyncio.run(test_macro_events())
