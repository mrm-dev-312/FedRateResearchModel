"""
Time spine and as-of join functionality for mixed-frequency data.
Handles alignment of macro releases with market data timestamps.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Union
import pandas as pd
from src.db.client import get_db

logger = logging.getLogger(__name__)

class TimeJoiner:
    def __init__(self):
        pass
    
    async def build_time_spine(
        self, 
        start_date: str, 
        end_date: str, 
        frequency: str = "1D"
    ) -> pd.DataFrame:
        """
        Build a continuous time spine for joining data.
        
        Args:
            start_date: Start date 'YYYY-MM-DD'
            end_date: End date 'YYYY-MM-DD' 
            frequency: Pandas frequency string ('1D', '1H', '15T', etc.)
        """
        date_range = pd.date_range(
            start=start_date,
            end=end_date,
            freq=frequency
        )
        
        spine = pd.DataFrame({
            'ts': date_range
        })
        
        return spine
    
    async def asof_join_macro(
        self, 
        market_df: pd.DataFrame,
        macro_series_ids: List[str]
    ) -> pd.DataFrame:
        """
        Perform as-of join to attach latest macro values to each market timestamp.
        
        Args:
            market_df: DataFrame with 'ts' and 'ticker' columns
            macro_series_ids: List of FRED series IDs to join
        """
        db = await get_db()
        
        # Get macro data for specified series
        macro_data = []
        for series_id in macro_series_ids:
            records = await db.macrorelease.find_many(
                where={'series_id': series_id}
            )
            
            for record in records:
                macro_data.append({
                    'series_id': record.series_id,
                    'release_time': record.release_time,
                    'actual': record.actual,
                    'surprise': record.surprise
                })
        
        if not macro_data:
            logger.warning("No macro data found for as-of join")
            return market_df
        
        macro_df = pd.DataFrame(macro_data)
        
        # Perform as-of join for each series
        result_df = market_df.copy()
        
        for series_id in macro_series_ids:
            series_data = macro_df[macro_df['series_id'] == series_id].copy()
            if series_data.empty:
                continue
                
            series_data = series_data.sort_values('release_time')
            
            # Merge as-of join
            result_df = pd.merge_asof(
                result_df.sort_values('ts'),
                series_data[['release_time', 'actual', 'surprise']],
                left_on='ts',
                right_on='release_time',
                direction='backward',
                suffixes=('', f'_{series_id}')
            )
            
            # Rename columns to include series ID
            if 'actual' in result_df.columns:
                result_df = result_df.rename(columns={
                    'actual': f'macro_{series_id}_actual',
                    'surprise': f'macro_{series_id}_surprise'
                })
            
            result_df = result_df.drop(columns=['release_time'], errors='ignore')
        
        return result_df
    
    async def build_feature_spine(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        frequency: str = "1D",
        macro_series: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Build complete feature spine with market data + macro as-of join.
        
        Returns DataFrame ready for feature engineering.
        """
        db = await get_db()
        
        # Get market data for ticker
        market_records = await db.marketprice.find_many(
            where={
                'ticker': ticker,
                'ts': {
                    'gte': datetime.fromisoformat(start_date),
                    'lte': datetime.fromisoformat(end_date)
                }
            }
        )
        
        if not market_records:
            logger.error(f"No market data found for {ticker}")
            return pd.DataFrame()
        
        # Convert to DataFrame
        market_data = []
        for record in market_records:
            market_data.append({
                'ticker': record.ticker,
                'ts': record.ts,
                'open': record.open,
                'high': record.high,
                'low': record.low,
                'close': record.close,
                'volume': record.volume
            })
        
        market_df = pd.DataFrame(market_data)
        
        # Perform as-of join with macro data if requested
        if macro_series:
            result_df = await self.asof_join_macro(market_df, macro_series)
        else:
            result_df = market_df
        
        return result_df

def make_spine(ticker: str, start: str, end: str, freq: str = "D") -> pd.DataFrame:
    """
    Create empty DataFrame index for time spine.
    
    Args:
        ticker: Asset ticker symbol
        start: Start date in 'YYYY-MM-DD' format
        end: End date in 'YYYY-MM-DD' format  
        freq: Frequency string ('D', 'H', '15T', etc.)
    
    Returns:
        DataFrame with datetime index and ticker column
    """
    date_range = pd.date_range(start=start, end=end, freq=freq)
    
    spine_df = pd.DataFrame(index=date_range)
    spine_df.index.name = 'ts'
    spine_df['ticker'] = ticker
    
    return spine_df.reset_index()

# Default macro indicators for joining
DEFAULT_MACRO_SERIES = [
    'FEDFUNDS',  # Fed funds rate
    'DGS10',     # 10-year treasury yield
    'UNRATE',    # Unemployment rate
    'CPIAUCSL',  # CPI
    'VIX'        # Volatility index
]

async def create_feature_spine(
    ticker: str,
    start_date: str = "2020-01-01",
    end_date: str = "2024-12-31",
    frequency: str = "1D",
    include_macro: bool = True
) -> pd.DataFrame:
    """
    Main function to create feature-ready time spine.
    
    Returns DataFrame with market data + macro as-of joined.
    """
    joiner = TimeJoiner()
    
    macro_series = DEFAULT_MACRO_SERIES if include_macro else None
    
    df = await joiner.build_feature_spine(
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
        frequency=frequency,
        macro_series=macro_series
    )
    
    return df

if __name__ == "__main__":
    async def test_spine():
        # Test creating feature spine for SPY
        df = await create_feature_spine(
            ticker="SPY",
            start_date="2024-01-01",
            end_date="2024-12-31"
        )
        print(f"Created feature spine with {len(df)} rows")
        if not df.empty:
            print(f"Columns: {list(df.columns)}")
            print(df.head())
    
    asyncio.run(test_spine())
