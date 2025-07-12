#!/usr/bin/env python3
"""
Database seeding script for MSRK v3
Populates the database with initial sample data for development and testing.
"""

import asyncio
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from db.client import get_db_client

async def seed_macro_releases() -> List[Dict[str, Any]]:
    """Create sample macro economic releases."""
    releases = [
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
            "description": "Consumer Price Index for All Urban Consumers: All Items",
            "frequency": "MONTHLY", 
            "unit": "Index 1982-84=100",
            "seasonal_adjustment": "NOT_SEASONALLY_ADJUSTED",
            "source": "FRED"
        },
        {
            "indicator": "GDP",
            "description": "Gross Domestic Product",
            "frequency": "QUARTERLY",
            "unit": "Billions of Dollars",
            "seasonal_adjustment": "SEASONALLY_ADJUSTED",
            "source": "FRED"
        },
        {
            "indicator": "FEDFUNDS",
            "description": "Federal Funds Effective Rate",
            "frequency": "MONTHLY",
            "unit": "Percent",
            "seasonal_adjustment": "NOT_SEASONALLY_ADJUSTED", 
            "source": "FRED"
        },
        {
            "indicator": "PAYEMS",
            "description": "All Employees, Total Nonfarm",
            "frequency": "MONTHLY",
            "unit": "Thousands of Persons",
            "seasonal_adjustment": "SEASONALLY_ADJUSTED",
            "source": "FRED"
        }
    ]
    
    print(f"📊 Seeding {len(releases)} macro releases...")
    return releases

async def seed_market_prices() -> List[Dict[str, Any]]:
    """Create sample market price data."""
    symbols = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD"]
    prices = []
    
    # Generate sample daily prices for the last 30 days
    base_date = datetime.now() - timedelta(days=30)
    
    for symbol in symbols:
        for i in range(30):
            date = base_date + timedelta(days=i)
            # Simple random walk for demo data
            base_price = 1.1000 if symbol == "EURUSD" else 1.2500
            price = base_price + (i * 0.001)  # Simple trend for demo
            
            prices.append({
                "symbol": symbol,
                "timestamp": date,
                "open": price,
                "high": price + 0.005,
                "low": price - 0.005,
                "close": price + 0.002,
                "volume": 1000000.0,
                "source": "YAHOO"
            })
    
    print(f"📊 Seeding {len(prices)} market price records...")
    return prices

async def seed_strategies() -> List[Dict[str, Any]]:
    """Create sample trading strategies."""
    strategies = [
        {
            "name": "Macro Momentum",
            "description": "Long/short based on macro economic surprise direction",
            "config": {
                "lookback_days": 30,
                "threshold": 0.5,
                "max_position": 1.0
            },
            "is_active": True
        },
        {
            "name": "Technical Breakout", 
            "description": "Breakout strategy based on technical indicators",
            "config": {
                "ema_period": 20,
                "atr_multiplier": 2.0,
                "stop_loss": 0.02
            },
            "is_active": True
        },
        {
            "name": "ML Ensemble",
            "description": "Machine learning ensemble using PatchTST and LSTM",
            "config": {
                "models": ["PatchTST", "LSTM"],
                "confidence_threshold": 0.7,
                "rebalance_frequency": "DAILY"
            },
            "is_active": False
        }
    ]
    
    print(f"📊 Seeding {len(strategies)} trading strategies...")
    return strategies

async def clear_existing_data(db):
    """Clear existing data from tables (for development)."""
    print("🧹 Clearing existing data...")
    
    # Clear in dependency order
    await db.backtestresult.delete_many({})
    await db.backtest.delete_many({})
    await db.feature.delete_many({})
    await db.marketprice.delete_many({})
    await db.macrorelease.delete_many({})
    await db.strategy.delete_many({})
    
    print("✅ Existing data cleared")

async def seed_database(clear_first: bool = False):
    """Main seeding function."""
    print("🌱 Starting database seeding...")
    
    try:
        db = await get_db_client()
        
        if clear_first:
            await clear_existing_data(db)
        
        # Seed macro releases
        macro_data = await seed_macro_releases()
        for release in macro_data:
            await db.macrorelease.create(data=release)
        
        # Seed market prices
        price_data = await seed_market_prices()
        for price in price_data:
            await db.marketprice.create(data=price)
        
        # Seed strategies
        strategy_data = await seed_strategies()
        for strategy in strategy_data:
            await db.strategy.create(data=strategy)
        
        print("✅ Database seeding completed successfully!")
        
    except Exception as e:
        print(f"❌ Database seeding failed: {e}")
        raise
    finally:
        await db.disconnect()

async def main():
    """CLI interface for seeding."""
    clear_first = "--clear" in sys.argv or "-c" in sys.argv
    
    if clear_first:
        confirm = input("⚠️  This will clear existing data. Type 'CLEAR' to confirm: ").strip()
        if confirm != "CLEAR":
            print("❌ Seeding cancelled")
            return 1
    
    try:
        await seed_database(clear_first)
        return 0
    except Exception as e:
        print(f"❌ Seeding failed: {e}")
        return 1

if __name__ == "__main__":
    print("""
🌱 MSRK v3 Database Seeding

Usage: python scripts/seed_db.py [--clear|-c]

Options:
  --clear, -c    Clear existing data before seeding
    """)
    
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
