"""
Strategy Registry: Load and manage trading strategies from YAML configuration files.
Provides validation, database integration, and runtime strategy management.
"""

import yaml
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime
import asyncio
from dataclasses import dataclass

# Add parent directory to path
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from db.client import get_db

logger = logging.getLogger(__name__)

@dataclass
class StrategyConfig:
    """Strategy configuration data class."""
    id: str
    name: str
    description: str
    version: str
    active: bool
    config_yaml: str
    
    # Parsed configuration sections
    risk: Dict[str, Any]
    trading: Dict[str, Any]
    data: Dict[str, Any]
    models: Dict[str, Any]
    signals: Dict[str, Any]
    backtest: Dict[str, Any]
    
class StrategyLoader:
    """Load and manage trading strategies from YAML configuration files."""
    
    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self.strategies: Dict[str, StrategyConfig] = {}
        
    def load_strategy_from_file(self, file_path: str) -> StrategyConfig:
        """
        Load a strategy from a YAML file.
        
        Args:
            file_path: Path to the YAML configuration file
            
        Returns:
            StrategyConfig object with parsed configuration
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Strategy file not found: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)
                
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML format in {file_path}: {e}")
        
        # Validate required sections
        required_sections = ['strategy', 'risk', 'trading', 'data', 'models']
        for section in required_sections:
            if section not in config_data:
                raise ValueError(f"Missing required section '{section}' in {file_path}")
        
        # Extract strategy metadata
        strategy_meta = config_data['strategy']
        if 'id' not in strategy_meta:
            raise ValueError("Strategy ID is required")
        
        # Create StrategyConfig object
        strategy_config = StrategyConfig(
            id=strategy_meta['id'],
            name=strategy_meta.get('name', strategy_meta['id']),
            description=strategy_meta.get('description', ''),
            version=strategy_meta.get('version', '1.0.0'),
            active=strategy_meta.get('active', True),
            config_yaml=yaml.dump(config_data, default_flow_style=False),
            risk=config_data.get('risk', {}),
            trading=config_data.get('trading', {}),
            data=config_data.get('data', {}),
            models=config_data.get('models', {}),
            signals=config_data.get('signals', {}),
            backtest=config_data.get('backtest', {})
        )
        
        # Validate configuration
        self._validate_strategy_config(strategy_config)
        
        # Store in memory
        self.strategies[strategy_config.id] = strategy_config
        
        logger.info(f"Loaded strategy: {strategy_config.name} (ID: {strategy_config.id})")
        return strategy_config
    
    def _validate_strategy_config(self, config: StrategyConfig) -> None:
        """Validate strategy configuration for required fields and logical consistency."""
        
        # Validate risk management
        risk = config.risk
        if 'max_positions' not in risk:
            logger.warning(f"Strategy {config.id}: max_positions not specified, defaulting to 1")
        
        if 'position_sizing' in risk:
            valid_sizing = ['fixed', 'kelly', 'volatility_target']
            if risk['position_sizing'] not in valid_sizing:
                raise ValueError(f"Invalid position_sizing: {risk['position_sizing']}. Must be one of {valid_sizing}")
        
        # Validate trading parameters
        trading = config.trading
        if 'initial_capital' not in trading:
            raise ValueError("initial_capital is required in trading section")
        
        if trading['initial_capital'] <= 0:
            raise ValueError("initial_capital must be positive")
        
        # Validate model configuration
        models = config.models
        if 'primary' not in models:
            raise ValueError("Primary model configuration is required")
        
        primary_model = models['primary']
        if 'type' not in primary_model:
            raise ValueError("Primary model type is required")
        
        valid_model_types = ['patchtst', 'lstm', 'timegpt']
        if primary_model['type'] not in valid_model_types:
            raise ValueError(f"Invalid model type: {primary_model['type']}. Must be one of {valid_model_types}")
        
        logger.info(f"Strategy configuration validated: {config.id}")
    
    def load_all_strategies(self) -> List[StrategyConfig]:
        """Load all strategy YAML files from the config directory."""
        strategies = []
        
        # Find all .yaml and .yml files in config directory
        yaml_files = list(self.config_dir.glob("*.yaml")) + list(self.config_dir.glob("*.yml"))
        strategy_files = [f for f in yaml_files if 'strategy' in f.name.lower()]
        
        for file_path in strategy_files:
            try:
                strategy = self.load_strategy_from_file(file_path)
                strategies.append(strategy)
            except Exception as e:
                logger.error(f"Failed to load strategy from {file_path}: {e}")
        
        logger.info(f"Loaded {len(strategies)} strategies from {self.config_dir}")
        return strategies
    
    async def save_strategy_to_db(self, strategy: StrategyConfig) -> bool:
        """
        Save strategy configuration to database.
        
        Args:
            strategy: StrategyConfig object to save
            
        Returns:
            True if successful, False otherwise
        """
        try:
            db = await get_db()
            
            # Check if strategy already exists
            existing = await db.strategy.find_unique(where={'id': strategy.id})
            
            if existing:
                # Update existing strategy
                await db.strategy.update(
                    where={'id': strategy.id},
                    data={
                        'name': strategy.name,
                        'description': strategy.description,
                        'config_yaml': strategy.config_yaml,
                        'active': strategy.active,
                        'updated_at': datetime.utcnow()
                    }
                )
                logger.info(f"Updated strategy in database: {strategy.id}")
            else:
                # Create new strategy
                await db.strategy.create(
                    data={
                        'id': strategy.id,
                        'name': strategy.name,
                        'description': strategy.description,
                        'config_yaml': strategy.config_yaml,
                        'active': strategy.active
                    }
                )
                logger.info(f"Saved new strategy to database: {strategy.id}")
            
            await db.disconnect()
            return True
            
        except Exception as e:
            logger.error(f"Failed to save strategy to database: {e}")
            return False
    
    async def load_strategies_from_db(self) -> List[StrategyConfig]:
        """Load all active strategies from database."""
        strategies = []
        
        try:
            db = await get_db()
            
            # Get all active strategies
            db_strategies = await db.strategy.find_many(
                where={'active': True},
                order_by={'created_at': 'desc'}
            )
            
            for db_strategy in db_strategies:
                try:
                    # Parse YAML configuration
                    config_data = yaml.safe_load(db_strategy.config_yaml)
                    
                    strategy = StrategyConfig(
                        id=db_strategy.id,
                        name=db_strategy.name,
                        description=db_strategy.description or '',
                        version='1.0.0',  # Default version
                        active=db_strategy.active,
                        config_yaml=db_strategy.config_yaml,
                        risk=config_data.get('risk', {}),
                        trading=config_data.get('trading', {}),
                        data=config_data.get('data', {}),
                        models=config_data.get('models', {}),
                        signals=config_data.get('signals', {}),
                        backtest=config_data.get('backtest', {})
                    )
                    
                    strategies.append(strategy)
                    
                except Exception as e:
                    logger.error(f"Failed to parse strategy {db_strategy.id}: {e}")
            
            await db.disconnect()
            logger.info(f"Loaded {len(strategies)} strategies from database")
            
        except Exception as e:
            logger.error(f"Failed to load strategies from database: {e}")
        
        return strategies
    
    def get_strategy(self, strategy_id: str) -> Optional[StrategyConfig]:
        """Get a specific strategy by ID."""
        return self.strategies.get(strategy_id)
    
    def list_strategies(self) -> List[str]:
        """List all loaded strategy IDs."""
        return list(self.strategies.keys())
    
    def create_strategy_template(self, output_path: str) -> None:
        """Create a template strategy YAML file."""
        template = {
            'strategy': {
                'id': 'my_strategy_v1',
                'name': 'My Trading Strategy',
                'description': 'Custom trading strategy description',
                'version': '1.0.0',
                'active': True
            },
            'risk': {
                'max_positions': 1,
                'position_sizing': 'fixed',
                'max_leverage': 1.0,
                'stop_loss_pct': 0.02,
                'take_profit_pct': 0.04
            },
            'trading': {
                'initial_capital': 100000,
                'commission_rate': 0.001,
                'signal_threshold': 0.015
            },
            'data': {
                'primary_ticker': 'EURUSD',
                'timeframe': '1H',
                'lookback_days': 90
            },
            'models': {
                'primary': {
                    'type': 'patchtst',
                    'config': {
                        'context_length': 64,
                        'prediction_length': 5,
                        'hidden_size': 64
                    }
                }
            },
            'backtest': {
                'start_date': '2023-01-01',
                'end_date': '2024-01-01'
            }
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(template, f, default_flow_style=False, indent=2)
        
        logger.info(f"Created strategy template: {output_path}")

# Convenience functions for common operations

async def register_strategy_from_file(file_path: str) -> bool:
    """Register a strategy from file to database."""
    loader = StrategyLoader()
    try:
        strategy = loader.load_strategy_from_file(file_path)
        success = await loader.save_strategy_to_db(strategy)
        return success
    except Exception as e:
        logger.error(f"Failed to register strategy: {e}")
        return False

async def get_active_strategies() -> List[StrategyConfig]:
    """Get all active strategies from database."""
    loader = StrategyLoader()
    return await loader.load_strategies_from_db()

def validate_strategy_file(file_path: str) -> bool:
    """Validate a strategy YAML file without saving it."""
    loader = StrategyLoader()
    try:
        loader.load_strategy_from_file(file_path)
        return True
    except Exception as e:
        logger.error(f"Strategy validation failed: {e}")
        return False

if __name__ == "__main__":
    # Test the strategy loader
    async def test_loader():
        loader = StrategyLoader()
        
        # Load example strategy
        try:
            strategy = loader.load_strategy_from_file("config/example_strategy.yaml")
            print(f"✅ Loaded strategy: {strategy.name}")
            print(f"   ID: {strategy.id}")
            print(f"   Description: {strategy.description}")
            print(f"   Model: {strategy.models['primary']['type']}")
            
            # Save to database
            success = await loader.save_strategy_to_db(strategy)
            print(f"   Database save: {'✅ Success' if success else '❌ Failed'}")
            
        except Exception as e:
            print(f"❌ Error: {e}")
    
    # Run test
    asyncio.run(test_loader())
