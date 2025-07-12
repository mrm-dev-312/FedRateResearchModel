"""
Logging configuration for MSRK v3
Provides structured logging with different levels and output formats.
"""

import logging
import logging.config
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

def get_log_config() -> Dict[str, Any]:
    """Get logging configuration dictionary."""
    
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Log level from environment or default to INFO
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    
    # Current date for log file rotation
    date_str = datetime.now().strftime("%Y%m%d")
    
    config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "detailed": {
                "format": "[{asctime}] {levelname:8} {name:20} {funcName:15} {lineno:4d} | {message}",
                "style": "{",
                "datefmt": "%Y-%m-%d %H:%M:%S"
            },
            "simple": {
                "format": "{levelname:8} | {message}",
                "style": "{"
            },
            "json": {
                "()": "pythonjsonlogger.jsonlogger.JsonFormatter",
                "format": "%(asctime)s %(name)s %(levelname)s %(funcName)s %(lineno)d %(message)s"
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": log_level,
                "formatter": "simple",
                "stream": sys.stdout
            },
            "file_detailed": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": "DEBUG",
                "formatter": "detailed",
                "filename": f"logs/msrk_{date_str}.log",
                "maxBytes": 10485760,  # 10MB
                "backupCount": 5
            },
            "file_errors": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": "ERROR",
                "formatter": "detailed",
                "filename": f"logs/msrk_errors_{date_str}.log",
                "maxBytes": 10485760,  # 10MB
                "backupCount": 10
            },
            "file_json": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": "INFO",
                "formatter": "json",
                "filename": f"logs/msrk_structured_{date_str}.json",
                "maxBytes": 10485760,  # 10MB
                "backupCount": 5
            }
        },
        "loggers": {
            "msrk": {
                "level": "DEBUG",
                "handlers": ["console", "file_detailed", "file_errors", "file_json"],
                "propagate": False
            },
            "msrk.data_ingest": {
                "level": "INFO",
                "handlers": ["console", "file_detailed", "file_json"],
                "propagate": False
            },
            "msrk.models": {
                "level": "INFO", 
                "handlers": ["console", "file_detailed", "file_json"],
                "propagate": False
            },
            "msrk.backtest": {
                "level": "INFO",
                "handlers": ["console", "file_detailed", "file_json"],
                "propagate": False
            },
            "prisma": {
                "level": "WARNING",
                "handlers": ["file_detailed"],
                "propagate": False
            },
            "urllib3": {
                "level": "WARNING",
                "handlers": ["file_detailed"],
                "propagate": False
            }
        },
        "root": {
            "level": log_level,
            "handlers": ["console", "file_detailed"]
        }
    }
    
    return config

def setup_logging():
    """Initialize logging configuration."""
    config = get_log_config()
    logging.config.dictConfig(config)
    
    # Get the main logger
    logger = logging.getLogger("msrk")
    logger.info("Logging initialized successfully")
    
    # Log environment info
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Working directory: {os.getcwd()}")
    logger.info(f"Log level: {os.getenv('LOG_LEVEL', 'INFO')}")
    
    return logger

def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with the specified name."""
    return logging.getLogger(f"msrk.{name}")

# Convenience functions for common log operations
def log_function_entry(logger: logging.Logger, func_name: str, **kwargs):
    """Log function entry with parameters."""
    params = ", ".join(f"{k}={v}" for k, v in kwargs.items())
    logger.debug(f"Entering {func_name}({params})")

def log_function_exit(logger: logging.Logger, func_name: str, result=None):
    """Log function exit with result."""
    if result is not None:
        logger.debug(f"Exiting {func_name} with result: {result}")
    else:
        logger.debug(f"Exiting {func_name}")

def log_api_call(logger: logging.Logger, method: str, url: str, status_code: int, duration: float):
    """Log API call details."""
    logger.info(f"API {method} {url} -> {status_code} ({duration:.3f}s)")

def log_database_operation(logger: logging.Logger, operation: str, table: str, count: int, duration: float):
    """Log database operation details."""
    logger.info(f"DB {operation} {table}: {count} records ({duration:.3f}s)")

def log_model_performance(logger: logging.Logger, model_name: str, metrics: Dict[str, float]):
    """Log model performance metrics."""
    metrics_str = ", ".join(f"{k}={v:.4f}" for k, v in metrics.items())
    logger.info(f"Model {model_name} performance: {metrics_str}")

# Context manager for logging operations
class LogOperation:
    """Context manager for logging the duration of operations."""
    
    def __init__(self, logger: logging.Logger, operation: str):
        self.logger = logger
        self.operation = operation
        self.start_time = None
    
    def __enter__(self):
        self.start_time = datetime.now()
        self.logger.info(f"Starting {self.operation}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = (datetime.now() - self.start_time).total_seconds()
        
        if exc_type is not None:
            self.logger.error(f"Failed {self.operation} after {duration:.3f}s: {exc_val}")
        else:
            self.logger.info(f"Completed {self.operation} in {duration:.3f}s")

# Initialize logging when module is imported
if __name__ != "__main__":
    setup_logging()
