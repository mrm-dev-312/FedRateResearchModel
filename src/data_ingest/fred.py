"""
FRED API data ingestion for macro economic indicators.
Fetches macro releases with consensus and surprise data.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import pandas as pd
import fredapi
from src.db.client import get_db

logger = logging.getLogger(__name__)

class FredIngestor:
    def __init__(self, api_key: str):
        self.fred = fredapi.Fred(api_key=api_key)
        
    async def fetch_series(self, series_id: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch FRED series data for given date range."""
        try:
            data = self.fred.get_series(
                series_id, 
                start=start_date, 
                end=end_date
            )
            df = data.reset_index()
            df.columns = ['release_time', 'actual']
            df['series_id'] = series_id
            df['consensus'] = None  # FRED doesn't provide consensus
            df['surprise'] = None   # Will calculate if consensus available
            return df
        except Exception as e:
            logger.error(f"Error fetching {series_id}: {e}")
            return pd.DataFrame()
    
    async def fetch_multiple_series(self, series_configs: Dict[str, dict]) -> pd.DataFrame:
        """
        Fetch multiple FRED series in parallel.
        
        Args:
            series_configs: {series_id: {'start': '2020-01-01', 'end': '2024-01-01'}}
        """
        tasks = []
        for series_id, config in series_configs.items():
            task = self.fetch_series(
                series_id, 
                config['start'], 
                config['end']
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        return pd.concat(results, ignore_index=True) if results else pd.DataFrame()
    
    async def store_macro_releases(self, df: pd.DataFrame) -> int:
        """Store macro release data in database."""
        if df.empty:
            return 0
            
        db = await get_db()
        records = df.to_dict('records')
        
        # Convert datetime objects to ensure proper format
        for record in records:
            if isinstance(record['release_time'], pd.Timestamp):
                record['release_time'] = record['release_time'].to_pydatetime()
        
        try:
            await db.macrorelease.create_many(data=records)
            logger.info(f"Stored {len(records)} macro release records")
            return len(records)
        except Exception as e:
            logger.error(f"Error storing macro releases: {e}")
            return 0

# Key macro indicators to track
DEFAULT_FRED_SERIES = {
    'GDP': {'start': '2020-01-01', 'end': '2024-12-31'},
    'UNRATE': {'start': '2020-01-01', 'end': '2024-12-31'},  # Unemployment rate
    'CPIAUCSL': {'start': '2020-01-01', 'end': '2024-12-31'},  # CPI
    'FEDFUNDS': {'start': '2020-01-01', 'end': '2024-12-31'},  # Fed funds rate
    'DGS10': {'start': '2020-01-01', 'end': '2024-12-31'},     # 10-year treasury
    'DEXUSEU': {'start': '2020-01-01', 'end': '2024-12-31'},   # USD/EUR
    'PAYEMS': {'start': '2020-01-01', 'end': '2024-12-31'},    # Non-farm payrolls
    'INDPRO': {'start': '2020-01-01', 'end': '2024-12-31'},    # Industrial production
}

async def ingest_fred_data(api_key: str, series_configs: Optional[Dict] = None) -> int:
    """
    Main function to ingest FRED data.
    
    Returns number of records stored.
    """
    if series_configs is None:
        series_configs = DEFAULT_FRED_SERIES
    
    ingestor = FredIngestor(api_key)
    df = await ingestor.fetch_multiple_series(series_configs)
    
    if not df.empty:
        return await ingestor.store_macro_releases(df)
    return 0

if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    api_key = os.getenv("FRED_API_KEY")
    
    if not api_key:
        print("Please set FRED_API_KEY environment variable")
        exit(1)
    
    async def test_ingest():
        count = await ingest_fred_data(api_key)
        print(f"Ingested {count} macro release records")
    
    asyncio.run(test_ingest())
