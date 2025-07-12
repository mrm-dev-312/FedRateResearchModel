"""
Yahoo Finance data ingestion for market price data.
Fetches OHLCV data for multiple tickers and timeframes.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import pandas as pd
import yfinance as yf
from src.db.client import get_db

logger = logging.getLogger(__name__)

class YahooIngestor:
    def __init__(self):
        pass
    
    async def fetch_ticker_data(
        self, 
        ticker: str, 
        start_date: str, 
        end_date: str, 
        interval: str = "1d"
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data for a single ticker.
        
        Args:
            ticker: Stock symbol (e.g., 'SPY', 'EURUSD=X')
            start_date: Start date 'YYYY-MM-DD'
            end_date: End date 'YYYY-MM-DD'
            interval: Data interval ('1m', '5m', '1h', '1d', '1wk', '1mo')
        """
        try:
            stock = yf.Ticker(ticker)
            data = stock.history(
                start=start_date,
                end=end_date,
                interval=interval,
                auto_adjust=True,
                prepost=True
            )
            
            if data.empty:
                logger.warning(f"No data found for {ticker}")
                return pd.DataFrame()
            
            # Reset index to get timestamp as column
            df = data.reset_index()
            df['ticker'] = ticker
            
            # Rename columns to match database schema
            column_mapping = {
                'Date': 'ts',
                'Datetime': 'ts',
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            }
            df = df.rename(columns=column_mapping)
            
            # Select only required columns
            required_cols = ['ticker', 'ts', 'open', 'high', 'low', 'close', 'volume']
            df = df[required_cols]
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching {ticker}: {e}")
            return pd.DataFrame()
    
    async def fetch_multiple_tickers(
        self, 
        tickers: List[str], 
        start_date: str, 
        end_date: str, 
        interval: str = "1d"
    ) -> pd.DataFrame:
        """Fetch data for multiple tickers in parallel."""
        tasks = []
        for ticker in tickers:
            task = self.fetch_ticker_data(ticker, start_date, end_date, interval)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        return pd.concat(results, ignore_index=True) if results else pd.DataFrame()
    
    async def store_market_prices(self, df: pd.DataFrame) -> int:
        """Store market price data in database."""
        if df.empty:
            return 0
        
        db = await get_db()
        records = df.to_dict('records')
        
        # Convert datetime and ensure proper types
        for record in records:
            if isinstance(record['ts'], pd.Timestamp):
                record['ts'] = record['ts'].to_pydatetime()
            # Ensure volume is int or None
            if pd.isna(record.get('volume')):
                record['volume'] = None
            else:
                record['volume'] = int(record['volume'])
        
        try:
            await db.marketprice.create_many(data=records)
            logger.info(f"Stored {len(records)} market price records")
            return len(records)
        except Exception as e:
            logger.error(f"Error storing market prices: {e}")
            return 0

# Default tickers to track
DEFAULT_TICKERS = [
    'SPY',      # S&P 500 ETF
    'QQQ',      # Nasdaq 100 ETF
    'IWM',      # Russell 2000 ETF
    'EURUSD=X', # EUR/USD
    'GBPUSD=X', # GBP/USD
    'USDJPY=X', # USD/JPY
    'GLD',      # Gold ETF
    'TLT',      # 20+ Year Treasury Bond ETF
    'VIX',      # Volatility Index
    'DXY',      # Dollar Index
]

async def ingest_yahoo_data(
    tickers: Optional[List[str]] = None,
    start_date: str = "2020-01-01",
    end_date: str = "2024-12-31",
    interval: str = "1d"
) -> int:
    """
    Main function to ingest Yahoo Finance data.
    
    Returns number of records stored.
    """
    if tickers is None:
        tickers = DEFAULT_TICKERS
    
    ingestor = YahooIngestor()
    df = await ingestor.fetch_multiple_tickers(tickers, start_date, end_date, interval)
    
    if not df.empty:
        return await ingestor.store_market_prices(df)
    return 0

if __name__ == "__main__":
    async def test_ingest():
        # Test with a small subset
        test_tickers = ['SPY', 'EURUSD=X']
        count = await ingest_yahoo_data(
            tickers=test_tickers,
            start_date="2024-01-01",
            end_date="2024-12-31"
        )
        print(f"Ingested {count} market price records")
    
    asyncio.run(test_ingest())
