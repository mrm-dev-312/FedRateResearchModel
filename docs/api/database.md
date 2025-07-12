# Database Client API Reference

The database client provides a unified interface for interacting with the PostgreSQL database through Prisma ORM.

## Overview

The database client handles all database operations including:
- Connection management
- CRUD operations for all entities
- Transaction support
- Connection pooling

## Usage

```python
from src.db.client import get_db_client

# Get database client instance
db = await get_db_client()

# Use the client for operations
result = await db.macrorelease.find_many()

# Always disconnect when done
await db.disconnect()
```

## API Reference

### get_db_client()

Returns a connected Prisma client instance.

**Returns:** `PrismaClient` - Connected database client

**Raises:** 
- `ConnectionError` - If database connection fails
- `ValueError` - If DATABASE_URL is not configured

### MacroRelease Operations

#### create()
```python
await db.macrorelease.create(data={
    "indicator": "UNRATE",
    "description": "Unemployment Rate",
    "frequency": "MONTHLY",
    "unit": "Percent",
    "seasonal_adjustment": "SEASONALLY_ADJUSTED",
    "source": "FRED"
})
```

#### find_many()
```python
# Find all records
releases = await db.macrorelease.find_many()

# Find with filters
releases = await db.macrorelease.find_many(
    where={"source": "FRED"}
)

# Find with ordering
releases = await db.macrorelease.find_many(
    order_by={"created_at": "desc"}
)
```

#### find_unique()
```python
release = await db.macrorelease.find_unique(
    where={"id": 1}
)
```

#### update()
```python
updated = await db.macrorelease.update(
    where={"id": 1},
    data={"description": "Updated description"}
)
```

#### delete()
```python
deleted = await db.macrorelease.delete(
    where={"id": 1}
)
```

### MarketPrice Operations

#### create()
```python
await db.marketprice.create(data={
    "symbol": "EURUSD",
    "timestamp": datetime.now(),
    "open": 1.1000,
    "high": 1.1050,
    "low": 1.0950,
    "close": 1.1025,
    "volume": 1000000.0,
    "source": "YAHOO"
})
```

#### find_many()
```python
# Find by symbol
prices = await db.marketprice.find_many(
    where={"symbol": "EURUSD"}
)

# Find by date range
prices = await db.marketprice.find_many(
    where={
        "timestamp": {
            "gte": start_date,
            "lte": end_date
        }
    }
)
```

### Feature Operations

#### create()
```python
await db.feature.create(data={
    "name": "SMA_20",
    "value": 1.1025,
    "timestamp": datetime.now(),
    "symbol": "EURUSD",
    "feature_type": "TECHNICAL"
})
```

#### find_many()
```python
# Find by feature type
features = await db.feature.find_many(
    where={"feature_type": "TECHNICAL"}
)

# Find by symbol and date range
features = await db.feature.find_many(
    where={
        "symbol": "EURUSD",
        "timestamp": {
            "gte": start_date,
            "lte": end_date
        }
    }
)
```

### Strategy Operations

#### create()
```python
await db.strategy.create(data={
    "name": "Macro Momentum",
    "description": "Long/short based on macro surprise direction",
    "config": {
        "lookback_days": 30,
        "threshold": 0.5
    },
    "is_active": True
})
```

#### find_many()
```python
# Find active strategies
strategies = await db.strategy.find_many(
    where={"is_active": True}
)
```

### Backtest Operations

#### create()
```python
await db.backtest.create(data={
    "strategy_id": 1,
    "start_date": datetime(2020, 1, 1),
    "end_date": datetime(2023, 12, 31),
    "config": {
        "initial_capital": 100000,
        "commission": 0.001
    }
})
```

#### find_many()
```python
# Find backtests for a strategy
backtests = await db.backtest.find_many(
    where={"strategy_id": 1}
)
```

## Best Practices

### Connection Management
```python
# Always use the client within a try/finally block
db = None
try:
    db = await get_db_client()
    # Perform operations
    result = await db.macrorelease.find_many()
finally:
    if db:
        await db.disconnect()
```

### Batch Operations
```python
# Use create_many for bulk inserts
await db.marketprice.create_many(data=[
    {"symbol": "EURUSD", "timestamp": dt1, ...},
    {"symbol": "EURUSD", "timestamp": dt2, ...},
    {"symbol": "EURUSD", "timestamp": dt3, ...}
])
```

### Transactions
```python
# Use transactions for related operations
async with db.tx() as transaction:
    await transaction.strategy.create(data=strategy_data)
    await transaction.backtest.create(data=backtest_data)
```

## Error Handling

The database client raises specific exceptions for different error conditions:

- `ConnectionError`: Database connection issues
- `ValidationError`: Data validation failures
- `IntegrityError`: Constraint violations
- `NotFoundError`: Record not found errors

Always handle these exceptions appropriately in your application code.
