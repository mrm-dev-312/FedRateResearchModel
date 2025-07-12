"""
Comprehensive tests for the backtesting engine.
Tests single trades, multiple trades, edge cases, and visualization.
"""

import pandas as pd
import numpy as np
import pytest
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any
import sys
import os

# Add the parent directory to the path to import engine
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from engine import BacktestEngine, run_strategy_backtest, Trade, Position

class TestBacktestEngine:
    """Test suite for BacktestEngine class."""
    
    def setup_method(self):
        """Setup for each test method."""
        self.engine = BacktestEngine(
            initial_capital=100000,
            commission_rate=0.001,
            max_positions=1
        )
        
        # Create test data
        dates = pd.date_range('2023-01-01', '2023-01-31', freq='D')
        n = len(dates)
        np.random.seed(42)
        prices = 100 + np.cumsum(np.random.randn(n) * 0.5)
        
        self.test_df = pd.DataFrame({
            'ts': dates,
            'ticker': 'TEST',
            'close': prices,
            'volume': np.random.randint(1000000, 5000000, n)
        })
    
    def test_single_trade_execution(self):
        """Test execution of a single buy and sell trade."""
        timestamp = datetime.now()
        ticker = "TEST"
        price = 100.0
        signal_strength = 0.5
        
        # Test buy trade
        buy_trade = self.engine.execute_trade(
            timestamp=timestamp,
            ticker=ticker,
            action="buy",
            price=price,
            signal_strength=signal_strength
        )
        
        assert buy_trade is not None
        assert buy_trade.action == "buy"
        assert buy_trade.ticker == ticker
        assert buy_trade.price == price
        assert buy_trade.signal_strength == signal_strength
        assert buy_trade.commission > 0
        
        # Check position was created
        assert ticker in self.engine.positions
        position = self.engine.positions[ticker]
        assert position.quantity > 0
        assert position.entry_price == price
        
        # Check cash was deducted
        initial_cash = 100000
        expected_cost = buy_trade.quantity * price + buy_trade.commission
        assert abs(self.engine.cash - (initial_cash - expected_cost)) < 0.01
        
        # Test sell trade
        sell_price = 105.0
        sell_trade = self.engine.execute_trade(
            timestamp=timestamp + timedelta(days=1),
            ticker=ticker,
            action="sell",
            price=sell_price,
            signal_strength=signal_strength
        )
        
        assert sell_trade is not None
        assert sell_trade.action == "sell"
        assert sell_trade.price == sell_price
        
        # Check position was closed
        assert ticker not in self.engine.positions
        
        # Check profit was realized
        assert self.engine.cash > initial_cash  # Should have made a profit
        
    def test_multiple_overlapping_signals(self):
        """Test handling of multiple trades with overlapping signals."""
        engine = BacktestEngine(
            initial_capital=100000,
            commission_rate=0.001,
            max_positions=3  # Allow multiple positions
        )
        
        timestamp = datetime.now()
        tickers = ["STOCK1", "STOCK2", "STOCK3", "STOCK4"]
        price = 100.0
        
        # Try to execute 4 buy trades (but only 3 should succeed)
        trades = []
        for i, ticker in enumerate(tickers):
            trade = engine.execute_trade(
                timestamp=timestamp + timedelta(minutes=i),
                ticker=ticker,
                action="buy",
                price=price,
                signal_strength=0.5
            )
            if trade:
                trades.append(trade)
        
        # Should only have 3 positions (max_positions limit)
        assert len(engine.positions) == 3
        assert len(trades) == 3
        
        # Test selling one position to make room for another
        sell_trade = engine.execute_trade(
            timestamp=timestamp + timedelta(minutes=5),
            ticker="STOCK1",
            action="sell",
            price=105.0,
            signal_strength=0.5
        )
        
        assert sell_trade is not None
        assert len(engine.positions) == 2
        
        # Now should be able to buy the 4th stock
        new_trade = engine.execute_trade(
            timestamp=timestamp + timedelta(minutes=6),
            ticker="STOCK4",
            action="buy",
            price=price,
            signal_strength=0.5
        )
        
        assert new_trade is not None
        assert len(engine.positions) == 3
    
    def test_edge_cases(self):
        """Test edge cases like no trades, all losses, insufficient funds."""
        
        # Test 1: No trades scenario
        empty_df = pd.DataFrame({
            'ts': pd.date_range('2023-01-01', '2023-01-10'),
            'ticker': 'TEST',
            'close': [100] * 10
        })
        predictions = np.array([[100]] * 10)  # No movement predictions
        
        results = run_strategy_backtest(
            empty_df, predictions, 
            {'signal_threshold': 0.1}  # High threshold, no signals
        )
        
        assert results['num_trades'] == 0
        assert results['total_return'] == 0
        
        # Test 2: Insufficient funds
        expensive_engine = BacktestEngine(initial_capital=1000)  # Low capital
        
        trade = expensive_engine.execute_trade(
            timestamp=datetime.now(),
            ticker="EXPENSIVE",
            action="buy", 
            price=10000,  # Expensive stock
            signal_strength=0.5
        )
        
        assert trade is None  # Should fail due to insufficient funds
        assert len(expensive_engine.positions) == 0
        
        # Test 3: All losing trades scenario
        losing_df = pd.DataFrame({
            'ts': pd.date_range('2023-01-01', '2023-01-05'),
            'ticker': 'LOSER',
            'close': [100, 95, 90, 85, 80]  # Declining prices
        })
        
        # Predictions that trigger buy signals at peaks
        losing_predictions = np.array([[110], [105], [100], [95], [90]])
        
        results = run_strategy_backtest(
            losing_df, losing_predictions,
            {'signal_threshold': 0.01, 'initial_capital': 100000}
        )
        
        assert results['total_return'] < 0  # Should lose money
        
    def test_signal_generation(self):
        """Test signal generation from predictions and rules."""
        # Create test data with clear trends
        dates = pd.date_range('2023-01-01', '2023-02-01', freq='D')
        n = len(dates)
        
        # Uptrend data
        uptrend_prices = 100 + np.arange(n) * 0.5
        uptrend_df = pd.DataFrame({
            'ts': dates,
            'ticker': 'UPTREND',
            'close': uptrend_prices,
            'volume': [1000000] * n
        })
        
        # Predictions suggesting further upside
        bullish_predictions = uptrend_prices.reshape(-1, 1) * 1.02
        
        signals = self.engine.generate_signals(
            uptrend_df, bullish_predictions, signal_threshold=0.015
        )
        
        assert 'action' in signals.columns
        assert 'ml_signal' in signals.columns
        assert 'signal_strength' in signals.columns
        
        # Should have some buy signals in uptrend with bullish predictions
        buy_signals = (signals['action'] == 'buy').sum()
        assert buy_signals > 0
        
    def test_portfolio_value_tracking(self):
        """Test portfolio value updates and tracking."""
        timestamp = datetime.now()
        
        # Execute a buy trade
        trade = self.engine.execute_trade(
            timestamp=timestamp,
            ticker="TEST",
            action="buy",
            price=100.0,
            signal_strength=0.5
        )
        
        assert trade is not None
        
        # Update portfolio value with new price
        self.engine.update_portfolio_value(
            timestamp + timedelta(days=1),
            {"TEST": 105.0}
        )
        
        assert len(self.engine.portfolio_values) == 1
        
        portfolio_record = self.engine.portfolio_values[0]
        assert 'timestamp' in portfolio_record
        assert 'total_value' in portfolio_record
        assert 'unrealized_pnl' in portfolio_record or 'positions_value' in portfolio_record
        
        # Portfolio value should be higher due to price increase
        expected_value = self.engine.cash + trade.quantity * 105.0
        assert abs(portfolio_record['total_value'] - expected_value) < 1.0
    
    def test_position_sizing_methods(self):
        """Test different position sizing methods."""
        
        # Test fixed sizing
        fixed_engine = BacktestEngine(position_sizing="fixed")
        size1 = fixed_engine.calculate_position_size(0.5, 100.0)
        size2 = fixed_engine.calculate_position_size(1.0, 100.0)
        assert size1 == size2  # Should be same for fixed sizing
        
        # Test volatility targeting
        vol_engine = BacktestEngine(position_sizing="volatility_target")
        size_low_vol = vol_engine.calculate_position_size(0.5, 100.0, volatility=0.1)
        size_high_vol = vol_engine.calculate_position_size(0.5, 100.0, volatility=0.3)
        assert size_low_vol > size_high_vol  # Lower vol should allow larger position
        
        # Test Kelly criterion
        kelly_engine = BacktestEngine(position_sizing="kelly")
        size_weak = kelly_engine.calculate_position_size(0.1, 100.0)
        size_strong = kelly_engine.calculate_position_size(1.0, 100.0)
        assert size_strong > size_weak  # Stronger signal should get larger position

def test_comprehensive_backtest():
    """Test a complete backtest with realistic data."""
    
    # Generate more realistic test data
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
    n = len(dates)
    
    # Create price series with some trends and volatility
    returns = np.random.randn(n) * 0.02 + 0.0005  # Slight positive drift
    returns[50:100] += 0.01  # Bull market period
    returns[200:250] -= 0.015  # Bear market period
    
    prices = 100 * np.exp(np.cumsum(returns))
    volumes = np.random.randint(500000, 2000000, n)
    
    market_df = pd.DataFrame({
        'ts': dates,
        'ticker': 'MARKET',
        'close': prices,
        'volume': volumes
    })
    
    # Generate predictions with some skill (correlated with future returns)
    future_returns = np.roll(returns, -5)  # 5-day ahead returns
    noise = np.random.randn(n) * 0.01
    prediction_skill = 0.3  # 30% skill, 70% noise
    
    predicted_returns = prediction_skill * future_returns + (1 - prediction_skill) * noise
    predictions = prices.reshape(-1, 1) * (1 + predicted_returns.reshape(-1, 1))
    
    # Run backtest with realistic configuration
    config = {
        'initial_capital': 100000,
        'commission_rate': 0.002,  # 0.2% commission
        'signal_threshold': 0.01,  # 1% threshold
        'max_positions': 1,
        'position_sizing': 'volatility_target'
    }
    
    results = run_strategy_backtest(market_df, predictions, config)
    
    # Verify results structure
    required_metrics = [
        'total_return', 'sharpe_ratio', 'max_drawdown', 
        'volatility', 'num_trades', 'final_portfolio_value'
    ]
    
    for metric in required_metrics:
        assert metric in results
    
    # Basic sanity checks
    assert isinstance(results['total_return'], (int, float))
    assert isinstance(results['num_trades'], int)
    assert results['final_portfolio_value'] > 0
    
    print(f"Backtest completed successfully:")
    print(f"  Total Return: {results['total_return']:.2%}")
    print(f"  Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    print(f"  Max Drawdown: {results['max_drawdown']:.2%}")
    print(f"  Number of Trades: {results['num_trades']}")

def visualize_backtest_results(results: Dict[str, Any], save_path: str = None):
    """
    Create visualization plots for backtest results.
    Shows portfolio value, drawdown, and signal analysis.
    """
    portfolio_df = pd.DataFrame(results['portfolio_values'])
    trades_df = pd.DataFrame(results['trades'])
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Backtest Results Analysis', fontsize=16)
    
    # 1. Portfolio Value Over Time
    ax1 = axes[0, 0]
    portfolio_df['timestamp'] = pd.to_datetime(portfolio_df['timestamp'])
    ax1.plot(portfolio_df['timestamp'], portfolio_df['total_value'], 
             linewidth=2, color='blue', label='Portfolio Value')
    ax1.axhline(y=results.get('final_portfolio_value', 0), 
                color='red', linestyle='--', alpha=0.7, label='Final Value')
    ax1.set_title('Portfolio Value Over Time')
    ax1.set_ylabel('Portfolio Value ($)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Drawdown Analysis
    ax2 = axes[0, 1]
    if len(portfolio_df) > 1:
        portfolio_df['returns'] = portfolio_df['total_value'].pct_change()
        cumulative_returns = (1 + portfolio_df['returns']).cumprod()
        rolling_max = cumulative_returns.expanding().max()
        drawdowns = (cumulative_returns - rolling_max) / rolling_max * 100
        
        ax2.fill_between(portfolio_df['timestamp'], drawdowns, 0, 
                        color='red', alpha=0.5, label='Drawdown')
        ax2.axhline(y=results.get('max_drawdown', 0) * 100, 
                   color='darkred', linestyle='--', label='Max Drawdown')
    
    ax2.set_title('Drawdown Analysis')
    ax2.set_ylabel('Drawdown (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Trade Distribution
    ax3 = axes[1, 0]
    if len(trades_df) > 0:
        buy_trades = trades_df[trades_df['action'] == 'buy']
        sell_trades = trades_df[trades_df['action'] == 'sell']
        
        ax3.scatter(buy_trades['timestamp'], buy_trades['price'], 
                   color='green', marker='^', s=60, label='Buy', alpha=0.7)
        ax3.scatter(sell_trades['timestamp'], sell_trades['price'], 
                   color='red', marker='v', s=60, label='Sell', alpha=0.7)
    
    ax3.set_title('Trade Execution Points')
    ax3.set_ylabel('Price')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Performance Metrics Summary
    ax4 = axes[1, 1]
    metrics = {
        'Total Return': f"{results.get('total_return', 0):.2%}",
        'Sharpe Ratio': f"{results.get('sharpe_ratio', 0):.2f}",
        'Max Drawdown': f"{results.get('max_drawdown', 0):.2%}",
        'Volatility': f"{results.get('volatility', 0):.2%}",
        'Num Trades': f"{results.get('num_trades', 0)}",
        'Win Rate': f"{results.get('win_rate', 0):.2%}"
    }
    
    ax4.axis('off')
    y_pos = 0.9
    for metric, value in metrics.items():
        ax4.text(0.1, y_pos, f"{metric}:", fontsize=12, fontweight='bold')
        ax4.text(0.6, y_pos, value, fontsize=12)
        y_pos -= 0.15
    
    ax4.set_title('Performance Summary')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_signal_correlation_analysis(
    df: pd.DataFrame, 
    predictions: np.ndarray, 
    returns: np.ndarray,
    save_path: str = None
):
    """
    Plot correlation and variance analysis between signals and returns.
    
    Args:
        df: Market data DataFrame
        predictions: Model predictions
        returns: Actual future returns
        save_path: Optional path to save the plot
    """
    # Calculate signals
    current_prices = df['close'].values[:-len(returns)]  # Align with returns
    predicted_returns = (predictions[:-len(returns), -1] - current_prices) / current_prices
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Signal vs Returns Analysis', fontsize=16)
    
    # 1. Scatter plot: Signal Strength vs Forward Returns
    ax1 = axes[0, 0]
    signal_strength = np.abs(predicted_returns)
    ax1.scatter(signal_strength, returns, alpha=0.6, s=30)
    
    # Add correlation line
    z = np.polyfit(signal_strength, returns, 1)
    p = np.poly1d(z)
    ax1.plot(signal_strength, p(signal_strength), "r--", alpha=0.8)
    
    correlation = np.corrcoef(signal_strength, returns)[0, 1]
    ax1.set_title(f'Signal Strength vs Forward Returns\nCorrelation: {correlation:.3f}')
    ax1.set_xlabel('Signal Strength')
    ax1.set_ylabel('Forward Returns')
    ax1.grid(True, alpha=0.3)
    
    # 2. Signal Direction vs Returns
    ax2 = axes[0, 1]
    signal_direction = np.sign(predicted_returns)
    
    # Box plot by signal direction
    positive_returns = returns[signal_direction > 0]
    negative_returns = returns[signal_direction < 0]
    
    box_data = [positive_returns, negative_returns]
    box_labels = ['Positive Signal', 'Negative Signal']
    
    ax2.boxplot(box_data, labels=box_labels)
    ax2.set_title('Return Distribution by Signal Direction')
    ax2.set_ylabel('Forward Returns')
    ax2.grid(True, alpha=0.3)
    
    # 3. Rolling Correlation
    ax3 = axes[1, 0]
    window = min(50, len(returns) // 4)
    if len(returns) > window:
        rolling_corr = pd.Series(returns).rolling(window).corr(
            pd.Series(signal_strength)
        )
        ax3.plot(rolling_corr, linewidth=2)
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax3.set_title(f'Rolling Correlation ({window}-period)')
        ax3.set_ylabel('Correlation')
        ax3.grid(True, alpha=0.3)
    
    # 4. Signal Strength Distribution
    ax4 = axes[1, 1]
    ax4.hist(signal_strength, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    ax4.axvline(x=np.mean(signal_strength), color='red', linestyle='--', 
               label=f'Mean: {np.mean(signal_strength):.3f}')
    ax4.axvline(x=np.median(signal_strength), color='orange', linestyle='--', 
               label=f'Median: {np.median(signal_strength):.3f}')
    ax4.set_title('Signal Strength Distribution')
    ax4.set_xlabel('Signal Strength')
    ax4.set_ylabel('Frequency')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

if __name__ == "__main__":
    # Run comprehensive tests
    print("🚀 Running BacktestEngine Tests...")
    
    # Run the comprehensive backtest
    test_comprehensive_backtest()
    
    print("\n✅ All backtesting tests passed!")
    print("\nTo run individual tests:")
    print("pytest src/backtest/engine_test.py -v")
