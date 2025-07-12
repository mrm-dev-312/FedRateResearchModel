"""
Database client for Macro Signal Research Kit v3
Prisma async client wrapper with connection management.
"""

import asyncio
import os
from typing import Optional
from prisma import Prisma
from dotenv import load_dotenv

load_dotenv()

class DatabaseClient:
    def __init__(self):
        self.client: Optional[Prisma] = None
        self._connected = False
    
    async def connect(self) -> Prisma:
        """Connect to database and return client instance."""
        if not self._connected:
            self.client = Prisma()
            await self.client.connect()
            self._connected = True
        return self.client
    
    async def disconnect(self):
        """Disconnect from database."""
        if self._connected and self.client:
            await self.client.disconnect()
            self._connected = False
    
    async def __aenter__(self):
        return await self.connect()
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.disconnect()

# Global instance for easy access
db_client = DatabaseClient()

async def get_db() -> Prisma:
    """Get connected database client."""
    return await db_client.connect()

# Example usage functions
async def health_check() -> dict:
    """Check database connection health."""
    try:
        db = await get_db()
        # Simple query to test connection
        result = await db.query_raw("SELECT 1 as health")
        return {"status": "healthy", "result": result}
    except Exception as e:
        return {"status": "error", "error": str(e)}

if __name__ == "__main__":
    # Test connection
    async def test():
        print("Testing database connection...")
        result = await health_check()
        print(f"Health check result: {result}")
        await db_client.disconnect()
    
    asyncio.run(test())
