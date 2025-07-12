#!/usr/bin/env python3
"""
CLI script for registering trading strategies.
Usage: python scripts/register_strategy.py <strategy_file.yaml>
"""

import asyncio
import argparse
import sys
from pathlib import Path
import logging

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from strategies.loader import register_strategy_from_file, validate_strategy_file

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def main():
    parser = argparse.ArgumentParser(
        description="Register trading strategies from YAML configuration files"
    )
    parser.add_argument(
        "strategy_file",
        help="Path to the strategy YAML file"
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate the strategy file without saving to database"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    strategy_file = Path(args.strategy_file)
    
    if not strategy_file.exists():
        logger.error(f"Strategy file not found: {strategy_file}")
        return 1
    
    # Validate strategy file
    logger.info(f"Validating strategy file: {strategy_file}")
    
    if not validate_strategy_file(str(strategy_file)):
        logger.error("Strategy validation failed!")
        return 1
    
    logger.info("✅ Strategy file validation passed")
    
    if args.validate_only:
        logger.info("Validation-only mode. Strategy not saved to database.")
        return 0
    
    # Register strategy to database
    logger.info("Registering strategy to database...")
    
    success = await register_strategy_from_file(str(strategy_file))
    
    if success:
        logger.info("✅ Strategy successfully registered to database")
        return 0
    else:
        logger.error("❌ Failed to register strategy to database")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
