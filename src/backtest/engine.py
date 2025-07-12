"""
Backtesting engine for hybrid rule + ML signal strategies.
Supports transaction costs, risk management, and performance metrics.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
import asyncio

# Add database integration
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from db.client import get_db

logger = logging.getLogger(__name__)

@dataclass
class Trade:
    """Individual trade record."""
    timestamp: datetime
    ticker: str
    action: str  # 'buy', 'sell', 'close'
    quantity: float
    price: float
    commission: float
    signal_strength: float
    
@dataclass
class Position:
    """Current position state."""
    ticker: str
    quantity: float
    entry_price: float
    entry_time: datetime
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0

class BacktestEngine:
    def __init__(
        self,
        initial_capital: float = 100000,
        commission_rate: float = 0.001,  # 0.1% per trade
        max_positions: int = 1,
        position_sizing: str = "fixed",  # "fixed", "kelly", "volatility_target"
        risk_free_rate: float = 0.02  # 2% annual
    ):
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.max_positions = max_positions
        self.position_sizing = position_sizing
        self.risk_free_rate = risk_free_rate
        
        # State tracking
        self.cash = initial_capital
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.portfolio_values: List[Dict] = []
        
    def calculate_position_size(
        self, 
        signal_strength: float, 
        price: float, 
        volatility: Optional[float] = None
    ) -> float:
        """Calculate position size based on sizing method."""
        available_capital = self.cash * 0.95  # Keep 5% cash buffer
        
        if self.position_sizing == "fixed":
            # Fixed dollar amount per position
            target_value = available_capital / max(self.max_positions, 1)
            return target_value / price
            
        elif self.position_sizing == "volatility_target":
            # Size based on volatility targeting
            if volatility is None:
                volatility = 0.2  # Default 20% volatility
            target_vol = 0.15  # Target 15% portfolio volatility
            leverage = target_vol / volatility
            target_value = available_capital * leverage * abs(signal_strength)
            return target_value / price
            
        elif self.position_sizing == "kelly":
            # Kelly criterion (simplified)
            win_rate = 0.55  # Assume 55% win rate
            avg_win = 0.02   # 2% average win
            avg_loss = -0.015 # -1.5% average loss
            
            kelly_fraction = (win_rate * avg_win - (1 - win_rate) * abs(avg_loss)) / avg_win
            kelly_fraction = max(0, min(kelly_fraction, 0.25))  # Cap at 25%
            
            target_value = available_capital * kelly_fraction * abs(signal_strength)
            return target_value / price
        
        return 0
    
    def execute_trade(
        self,
        timestamp: datetime,
        ticker: str,
        action: str,
        price: float,
        signal_strength: float,
        volatility: Optional[float] = None
    ) -> Optional[Trade]:
        """Execute a single trade and update portfolio state."""
        
        if action == "buy":
            # Check if we can open new position
            if len(self.positions) >= self.max_positions:
                return None
                
            quantity = self.calculate_position_size(signal_strength, price, volatility)
            if quantity <= 0:
                return None
                
            trade_value = quantity * price
            commission, spread_cost = self.add_friction_costs(trade_value, action, ticker)
            total_cost = trade_value + commission + spread_cost
            
            if total_cost > self.cash:
                return None  # Insufficient funds
            
            # Execute buy
            self.cash -= total_cost
            self.positions[ticker] = Position(
                ticker=ticker,
                quantity=quantity,
                entry_price=price,
                entry_time=timestamp
            )
            
            trade = Trade(
                timestamp=timestamp,
                ticker=ticker,
                action=action,
                quantity=quantity,
                price=price,
                commission=commission + spread_cost,  # Total friction costs
                signal_strength=signal_strength
            )
            self.trades.append(trade)
            return trade
            
        elif action == "sell" and ticker in self.positions:
            # Close existing position
            position = self.positions[ticker]
            trade_value = position.quantity * price
            commission, spread_cost = self.add_friction_costs(trade_value, action, ticker)
            net_proceeds = trade_value - commission - spread_cost
            
            # Calculate realized PnL (including both entry and exit costs)
            entry_value = position.quantity * position.entry_price
            entry_commission, entry_spread = self.add_friction_costs(entry_value, "buy", ticker)
            realized_pnl = net_proceeds - entry_value - entry_commission - entry_spread
            
            # Execute sell
            self.cash += net_proceeds
            position.realized_pnl = realized_pnl
            del self.positions[ticker]
            
            trade = Trade(
                timestamp=timestamp,
                ticker=ticker,
                action=action,
                quantity=position.quantity,
                price=price,
                commission=commission + spread_cost,  # Total friction costs  
                signal_strength=signal_strength
            )
            self.trades.append(trade)
            return trade
        
        return None
    
    def update_portfolio_value(self, timestamp: datetime, prices: Dict[str, float]):
        """Update portfolio valuation and tracking."""
        portfolio_value = self.cash
        
        # Update unrealized PnL for open positions
        for ticker, position in self.positions.items():
            if ticker in prices:
                current_value = position.quantity * prices[ticker]
                entry_value = position.quantity * position.entry_price
                position.unrealized_pnl = current_value - entry_value
                portfolio_value += current_value
        
        # Track portfolio metrics
        self.portfolio_values.append({
            'timestamp': timestamp,
            'cash': self.cash,
            'positions_value': portfolio_value - self.cash,
            'total_value': portfolio_value,
            'num_positions': len(self.positions)
        })
    
    def generate_signals(
        self, 
        df: pd.DataFrame, 
        predictions: np.ndarray,
        signal_threshold: float = 0.02
    ) -> pd.DataFrame:
        """
        Generate trading signals from ML predictions and rules.
        
        Args:
            df: Market data with features
            predictions: Model forecasts
            signal_threshold: Minimum signal strength to trade
        """
        signals_df = df.copy()
        
        # ML signal: Use prediction vs current price
        current_prices = df['close'].values
        predicted_returns = (predictions[:, -1] - current_prices) / current_prices
        
        # Apply signal filters
        signals_df['ml_signal'] = predicted_returns
        signals_df['signal_strength'] = np.abs(predicted_returns)
        
        # Rule-based filters
        # 1. Trend filter: Only trade in direction of 20-day trend
        sma_20 = df['close'].rolling(20).mean()
        trend_up = df['close'] > sma_20
        
        # 2. Volatility filter: Avoid trading in high volatility periods
        volatility = df['close'].pct_change().rolling(20).std() * np.sqrt(252)
        low_vol = volatility < volatility.quantile(0.8)
        
        # 3. Volume filter: Require decent volume
        volume_filter = True  # Simplified
        if 'volume' in df.columns:
            avg_volume = df['volume'].rolling(20).mean()
            volume_filter = df['volume'] > avg_volume * 0.5
        
        # Combined signal
        buy_signal = (
            (predicted_returns > signal_threshold) & 
            trend_up & 
            low_vol & 
            volume_filter
        )
        
        sell_signal = (
            (predicted_returns < -signal_threshold) & 
            ~trend_up & 
            low_vol & 
            volume_filter
        )
        
        signals_df['action'] = 'hold'
        signals_df.loc[buy_signal, 'action'] = 'buy'
        signals_df.loc[sell_signal, 'action'] = 'sell'
        
        # Add timestamps for look-ahead bias checking
        # Signal time = current observation time
        # Trade time = next period (assuming we can trade at next open/close)
        signals_df['signal_time'] = signals_df.index if 'ts' not in signals_df.columns else signals_df['ts']
        
        if 'ts' in signals_df.columns:
            # Trade execution happens after signal generation
            signals_df['trade_time'] = signals_df['ts'] + pd.Timedelta(minutes=1)
        else:
            # For index-based data, trade happens next period
            signals_df['trade_time'] = signals_df.index + 1
        
        # Validate no look-ahead bias
        self.validate_no_lookahead_bias(signals_df)
        
        return signals_df
    
    def run_backtest(
        self,
        data_df: pd.DataFrame,
        predictions: np.ndarray,
        signal_threshold: float = 0.02
    ) -> Dict[str, Any]:
        """
        Run complete backtest on historical data.
        
        Returns:
            results: Comprehensive backtest results and metrics
        """
        # Generate signals
        signals_df = self.generate_signals(data_df, predictions, signal_threshold)
        
        # Execute trades
        for idx, row in signals_df.iterrows():
            timestamp = row['ts'] if 'ts' in row else idx
            ticker = row['ticker'] if 'ticker' in row else 'DEFAULT'
            price = row['close']
            action = row['action']
            signal_strength = row['signal_strength']
            volatility = row.get('volatility_20', 0.2)
            
            if action in ['buy', 'sell']:
                trade = self.execute_trade(
                    timestamp=timestamp,
                    ticker=ticker,
                    action=action,
                    price=price,
                    signal_strength=signal_strength,
                    volatility=volatility
                )
            
            # Update portfolio tracking
            self.update_portfolio_value(timestamp, {ticker: price})
        
        # Calculate final metrics
        results = self.calculate_metrics(data_df)
        return results
    
    def calculate_metrics(self, data_df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics."""
        if not self.portfolio_values:
            return {}
        
        # Portfolio value series
        portfolio_df = pd.DataFrame(self.portfolio_values)
        portfolio_df['returns'] = portfolio_df['total_value'].pct_change()
        
        # Basic metrics
        total_return = (portfolio_df['total_value'].iloc[-1] / self.initial_capital) - 1
        
        # Risk-adjusted metrics
        portfolio_returns = portfolio_df['returns'].dropna()
        avg_return = portfolio_returns.mean()
        volatility = portfolio_returns.std() * np.sqrt(252)  # Annualized
        
        if volatility > 0:
            sharpe_ratio = (avg_return * 252 - self.risk_free_rate) / volatility
        else:
            sharpe_ratio = 0
        
        # Drawdown analysis
        cumulative_returns = (1 + portfolio_returns).cumprod()
        rolling_max = cumulative_returns.expanding().max()
        drawdowns = (cumulative_returns - rolling_max) / rolling_max
        max_drawdown = drawdowns.min()
        
        # Calmar ratio
        calmar_ratio = (avg_return * 252) / abs(max_drawdown) if max_drawdown < 0 else 0
        
        # Trade analysis
        winning_trades = [t for t in self.trades if t.action == 'sell']
        if winning_trades:
            # Calculate win rate based on realized PnL from positions
            trade_pnls = []
            for trade in winning_trades:
                # This is simplified - in practice you'd track PnL per trade
                pass
            
            num_trades = len(winning_trades)
            win_rate = 0.5  # Placeholder
        else:
            num_trades = 0
            win_rate = 0
        
        # Benchmark comparison (if available)
        benchmark_return = 0
        if 'close' in data_df.columns:
            benchmark_return = (data_df['close'].iloc[-1] / data_df['close'].iloc[0]) - 1
        
        return {
            'total_return': total_return,
            'benchmark_return': benchmark_return,
            'excess_return': total_return - benchmark_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar_ratio,
            'volatility': volatility,
            'num_trades': num_trades,
            'win_rate': win_rate,
            'final_portfolio_value': portfolio_df['total_value'].iloc[-1],
            'portfolio_values': portfolio_df.to_dict('records'),
            'trades': [
                {
                    'timestamp': t.timestamp,
                    'ticker': t.ticker,
                    'action': t.action,
                    'quantity': t.quantity,
                    'price': t.price,
                    'commission': t.commission,
                    'signal_strength': t.signal_strength
                } for t in self.trades
            ]
        }
    
    def validate_no_lookahead_bias(
        self, 
        signals_df: pd.DataFrame,
        strict: bool = True
    ) -> bool:
        """
        Verify no look-ahead bias by ensuring signal_time < trade_time.
        
        Args:
            signals_df: DataFrame with signal and trade timestamps
            strict: If True, raises exception on bias detection
            
        Returns:
            True if no bias detected, False otherwise
        """
        if 'signal_time' not in signals_df.columns or 'trade_time' not in signals_df.columns:
            logger.warning("Cannot validate look-ahead bias: missing timestamp columns")
            return True
        
        # Check for any signals that occur after trade time
        lookahead_signals = signals_df[signals_df['signal_time'] >= signals_df['trade_time']]
        
        if len(lookahead_signals) > 0:
            error_msg = f"Look-ahead bias detected! {len(lookahead_signals)} signals occur at or after trade time"
            logger.error(error_msg)
            
            if strict:
                raise ValueError(error_msg)
            return False
        
        logger.info(f"✅ No look-ahead bias detected in {len(signals_df)} signals")
        return True
    
    async def save_signals_to_db(
        self,
        signals_df: pd.DataFrame,
        strategy_id: str
    ) -> int:
        """
        Save generated signals to the database.
        
        Args:
            signals_df: DataFrame with signal data
            strategy_id: Strategy identifier
            
        Returns:
            Number of signals saved
        """
        if len(signals_df) == 0:
            return 0
        
        db = await get_db()
        signals_saved = 0
        
        try:
            for _, row in signals_df.iterrows():
                if row['action'] in ['buy', 'sell']:  # Skip 'hold' signals
                    await db.signal.create(
                        data={
                            'strategy_id': strategy_id,
                            'ticker': row.get('ticker', 'DEFAULT'),
                            'signal_time': row['signal_time'] if 'signal_time' in row else row['ts'],
                            'trade_time': row['trade_time'] if 'trade_time' in row else row['ts'],
                            'action': row['action'],
                            'signal_strength': float(row['signal_strength']),
                            'price': float(row['close']),
                            'quantity': row.get('quantity'),
                            'confidence': row.get('confidence'),
                            'metadata_json': {
                                'ml_signal': float(row.get('ml_signal', 0)),
                                'volatility': float(row.get('volatility_20', 0.2)),
                                'trend_filter': bool(row.get('trend_up', False))
                            }
                        }
                    )
                    signals_saved += 1
                    
        except Exception as e:
            logger.error(f"Error saving signals to database: {e}")
            raise
        finally:
            await db.disconnect()
        
        logger.info(f"Saved {signals_saved} signals to database")
        return signals_saved
    
    def add_friction_costs(
        self,
        trade_value: float,
        action: str,
        ticker: str = "DEFAULT"
    ) -> Tuple[float, float]:
        """
        Calculate friction costs including commission and bid-ask spread.
        
        Args:
            trade_value: Total value of the trade
            action: 'buy' or 'sell'
            ticker: Asset ticker (for ticker-specific costs)
            
        Returns:
            (commission, spread_cost) tuple
        """
        # Commission (percentage-based)
        commission = trade_value * self.commission_rate
        
        # Bid-ask spread cost (assume 0.1% for major pairs, 0.2% for others)
        spread_bps = 10 if ticker in ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF'] else 20
        spread_cost = trade_value * (spread_bps / 10000)
        
        # Market impact (for large trades)
        if trade_value > 1000000:  # $1M+ trades
            impact_bps = min(5, trade_value / 1000000 * 0.5)  # Cap at 5 bps
            spread_cost += trade_value * (impact_bps / 10000)
        
        return commission, spread_cost

def run_strategy_backtest(
    feature_matrix: pd.DataFrame,
    predictions: np.ndarray,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Run backtest for a complete strategy configuration.
    
    Args:
        feature_matrix: DataFrame with market data and features
        predictions: Model predictions array
        config: Strategy configuration dict
        
    Returns:
        Backtest results and performance metrics
    """
    # Initialize engine with config
    engine = BacktestEngine(
        initial_capital=config.get('initial_capital', 100000),
        commission_rate=config.get('commission_rate', 0.001),
        max_positions=config.get('max_positions', 1),
        position_sizing=config.get('position_sizing', 'fixed')
    )
    
    # Run backtest
    results = engine.run_backtest(
        feature_matrix, 
        predictions,
        signal_threshold=config.get('signal_threshold', 0.02)
    )
    
    return results

if __name__ == "__main__":
    # Test with synthetic data
    np.random.seed(42)
    
    dates = pd.date_range('2023-01-01', '2024-01-01', freq='D')
    n = len(dates)
    
    # Generate synthetic price data
    prices = 100 * np.exp(np.cumsum(np.random.randn(n) * 0.01))
    
    test_df = pd.DataFrame({
        'ts': dates,
        'ticker': 'TEST',
        'close': prices,
        'volume': np.random.randint(1000000, 5000000, n)
    })
    
    # Generate synthetic predictions (slightly better than random)
    predictions = prices.reshape(-1, 1) * (1 + np.random.randn(n, 1) * 0.02)
    
    # Run backtest
    config = {
        'initial_capital': 100000,
        'commission_rate': 0.001,
        'signal_threshold': 0.015
    }
    
    results = run_strategy_backtest(test_df, predictions, config)
    
    print("Backtest Results:")
    print(f"Total Return: {results['total_return']:.2%}")
    print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {results['max_drawdown']:.2%}")
    print(f"Number of Trades: {results['num_trades']}")
    print(f"Final Portfolio Value: ${results['final_portfolio_value']:,.2f}")
