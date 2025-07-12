"""
Health check endpoints for MSRK v3
Provides health monitoring for various system components.
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
import psutil
import os

from ..db.client import get_db_client

logger = logging.getLogger("msrk.health")

class HealthChecker:
    """System health checker with various component checks."""
    
    def __init__(self):
        self.last_check_time = None
        self.cached_results = {}
        self.cache_duration = 30  # seconds
    
    async def check_all(self, use_cache: bool = True) -> Dict[str, Any]:
        """Check health of all system components."""
        
        # Use cached results if available and recent
        if (use_cache and self.last_check_time and 
            datetime.now() - self.last_check_time < timedelta(seconds=self.cache_duration)):
            return self.cached_results
        
        logger.info("Running comprehensive health check")
        start_time = time.time()
        
        health_status = {
            "timestamp": datetime.now().isoformat(),
            "overall_status": "healthy",
            "checks": {}
        }
        
        # Run all health checks
        checks = [
            ("system", self.check_system_resources),
            ("database", self.check_database),
            ("environment", self.check_environment),
            ("disk_space", self.check_disk_space),
            ("memory", self.check_memory_usage)
        ]
        
        failed_checks = []
        
        for check_name, check_function in checks:
            try:
                result = await check_function()
                health_status["checks"][check_name] = result
                
                if not result.get("healthy", False):
                    failed_checks.append(check_name)
                    
            except Exception as e:
                logger.error(f"Health check {check_name} failed: {e}")
                health_status["checks"][check_name] = {
                    "healthy": False,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }
                failed_checks.append(check_name)
        
        # Determine overall status
        if failed_checks:
            if len(failed_checks) >= len(checks) // 2:
                health_status["overall_status"] = "unhealthy"
            else:
                health_status["overall_status"] = "degraded"
            
            health_status["failed_checks"] = failed_checks
        
        health_status["check_duration"] = time.time() - start_time
        
        # Cache results
        self.last_check_time = datetime.now()
        self.cached_results = health_status
        
        logger.info(f"Health check completed: {health_status['overall_status']} "
                   f"({health_status['check_duration']:.3f}s)")
        
        return health_status
    
    async def check_database(self) -> Dict[str, Any]:
        """Check database connectivity and basic operations."""
        start_time = time.time()
        
        try:
            db = await get_db_client()
            
            # Test basic query
            result = await db.raw_query("SELECT 1 as test")
            
            # Test table access (should work if schema is set up)
            try:
                count = await db.macrorelease.count()
                table_access = True
            except Exception:
                count = -1
                table_access = False
            
            await db.disconnect()
            
            duration = time.time() - start_time
            
            return {
                "healthy": True,
                "response_time": duration,
                "table_access": table_access,
                "macro_release_count": count,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "healthy": False,
                "error": str(e),
                "response_time": time.time() - start_time,
                "timestamp": datetime.now().isoformat()
            }
    
    async def check_system_resources(self) -> Dict[str, Any]:
        """Check system resource usage."""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Determine if resources are healthy
            cpu_healthy = cpu_percent < 90
            memory_healthy = memory.percent < 90
            disk_healthy = disk.percent < 90
            
            return {
                "healthy": cpu_healthy and memory_healthy and disk_healthy,
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "disk_percent": disk.percent,
                "cpu_healthy": cpu_healthy,
                "memory_healthy": memory_healthy,
                "disk_healthy": disk_healthy,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "healthy": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def check_environment(self) -> Dict[str, Any]:
        """Check environment configuration."""
        required_vars = [
            "DATABASE_URL",
            "FRED_API_KEY",
            "YAHOO_USER_AGENT"
        ]
        
        missing_vars = []
        present_vars = []
        
        for var in required_vars:
            if os.getenv(var):
                present_vars.append(var)
            else:
                missing_vars.append(var)
        
        return {
            "healthy": len(missing_vars) == 0,
            "required_variables": required_vars,
            "present_variables": present_vars,
            "missing_variables": missing_vars,
            "python_version": os.sys.version,
            "timestamp": datetime.now().isoformat()
        }
    
    async def check_disk_space(self) -> Dict[str, Any]:
        """Check available disk space."""
        try:
            disk = psutil.disk_usage('.')
            
            # Check if we have at least 1GB free
            free_gb = disk.free / (1024**3)
            healthy = free_gb > 1.0
            
            return {
                "healthy": healthy,
                "free_space_gb": free_gb,
                "total_space_gb": disk.total / (1024**3),
                "used_percent": disk.percent,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "healthy": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def check_memory_usage(self) -> Dict[str, Any]:
        """Check memory usage details."""
        try:
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()
            
            # Process-specific memory
            process = psutil.Process()
            process_memory = process.memory_info()
            
            return {
                "healthy": memory.percent < 85,
                "system_memory_percent": memory.percent,
                "system_memory_available_gb": memory.available / (1024**3),
                "swap_percent": swap.percent,
                "process_memory_mb": process_memory.rss / (1024**2),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "healthy": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

# Global health checker instance
health_checker = HealthChecker()

async def get_health_status(detailed: bool = False) -> Dict[str, Any]:
    """Get current health status."""
    if detailed:
        return await health_checker.check_all(use_cache=False)
    else:
        # Quick check - just overall status
        full_status = await health_checker.check_all(use_cache=True)
        return {
            "status": full_status["overall_status"],
            "timestamp": full_status["timestamp"],
            "check_duration": full_status.get("check_duration", 0)
        }

async def get_readiness_status() -> Dict[str, Any]:
    """Check if system is ready to serve requests."""
    # For readiness, we primarily care about database connectivity
    db_status = await health_checker.check_database()
    env_status = await health_checker.check_environment()
    
    ready = db_status["healthy"] and env_status["healthy"]
    
    return {
        "ready": ready,
        "database_healthy": db_status["healthy"],
        "environment_healthy": env_status["healthy"],
        "timestamp": datetime.now().isoformat()
    }

async def get_liveness_status() -> Dict[str, Any]:
    """Check if system is alive (basic liveness probe)."""
    # Simple check - if we can respond, we're alive
    return {
        "alive": True,
        "timestamp": datetime.now().isoformat(),
        "uptime_seconds": time.time()  # Simple uptime approximation
    }
