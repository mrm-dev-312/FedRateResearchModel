"""
Technical indicators and feature engineering for time series data.
Includes momentum, volatility, and trend indicators.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
import logging

logger = logging.getLogger(__name__)

class TechnicalFeatures:
    def __init__(self):
        pass
    
    def sma(self, series: pd.Series, window: int) -> pd.Series:
        """Simple Moving Average."""
        return series.rolling(window=window).mean()
    
    def ema(self, series: pd.Series, window: int) -> pd.Series:
        """Exponential Moving Average."""
        return series.ewm(span=window).mean()
    
    def rsi(self, series: pd.Series, window: int = 14) -> pd.Series:
        """Relative Strength Index."""
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def macd(self, series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
        """MACD indicator."""
        ema_fast = self.ema(series, fast)
        ema_slow = self.ema(series, slow)
        macd_line = ema_fast - ema_slow
        signal_line = self.ema(macd_line, signal)
        histogram = macd_line - signal_line
        
        return {
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram
        }
    
    def bollinger_bands(self, series: pd.Series, window: int = 20, num_std: float = 2) -> Dict[str, pd.Series]:
        """Bollinger Bands."""
        sma = self.sma(series, window)
        std = series.rolling(window=window).std()
        upper = sma + (std * num_std)
        lower = sma - (std * num_std)
        
        return {
            'bb_upper': upper,
            'bb_middle': sma,
            'bb_lower': lower,
            'bb_width': upper - lower,
            'bb_position': (series - lower) / (upper - lower)
        }
    
    def atr(self, high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
        """Average True Range."""
        high_low = high - low
        high_close = np.abs(high - close.shift())
        low_close = np.abs(low - close.shift())
        tr = np.maximum(high_low, np.maximum(high_close, low_close))
        return tr.rolling(window=window).mean()
    
    def stochastic(self, high: pd.Series, low: pd.Series, close: pd.Series, k_window: int = 14, d_window: int = 3) -> Dict[str, pd.Series]:
        """Stochastic Oscillator."""
        lowest_low = low.rolling(window=k_window).min()
        highest_high = high.rolling(window=k_window).max()
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_window).mean()
        
        return {
            'stoch_k': k_percent,
            'stoch_d': d_percent
        }
    
    def williams_r(self, high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
        """Williams %R."""
        highest_high = high.rolling(window=window).max()
        lowest_low = low.rolling(window=window).min()
        return -100 * ((highest_high - close) / (highest_high - lowest_low))
    
    def momentum(self, series: pd.Series, window: int = 10) -> pd.Series:
        """Price momentum."""
        return series.pct_change(periods=window) * 100
    
    def volatility(self, series: pd.Series, window: int = 20) -> pd.Series:
        """Rolling volatility (standard deviation of returns)."""
        returns = series.pct_change()
        return returns.rolling(window=window).std() * np.sqrt(252)  # Annualized
    
    def volume_price_trend(self, close: pd.Series, volume: pd.Series) -> pd.Series:
        """Volume Price Trend."""
        price_change = close.pct_change()
        vpt = (price_change * volume).cumsum()
        return vpt
    
    def on_balance_volume(self, close: pd.Series, volume: pd.Series) -> pd.Series:
        """On Balance Volume."""
        price_change = close.diff()
        direction = np.where(price_change > 0, 1, np.where(price_change < 0, -1, 0))
        obv = (direction * volume).cumsum()
        return obv
    
    def rolling_return(self, series: pd.Series, window: int = 1) -> pd.Series:
        """Calculate rolling returns over specified window."""
        return series.pct_change(periods=window)
    
    def z_score(self, series: pd.Series, window: int = 20) -> pd.Series:
        """Calculate rolling z-score (standardized values)."""
        rolling_mean = series.rolling(window=window).mean()
        rolling_std = series.rolling(window=window).std()
        return (series - rolling_mean) / rolling_std
    
    def volatility_percentile(self, series: pd.Series, window: int = 252, lookback: int = 252) -> pd.Series:
        """Calculate volatility percentile rank over lookback period."""
        # Calculate rolling volatility
        returns = series.pct_change()
        volatility = returns.rolling(window=window).std() * np.sqrt(252)  # Annualized
        
        # Calculate percentile rank
        def calc_percentile(x):
            if len(x) < 2:
                return np.nan
            return (x.iloc[-1] <= x).sum() / len(x) * 100
        
        percentile_rank = volatility.rolling(window=lookback).apply(calc_percentile, raw=False)
        return percentile_rank

def generate_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate comprehensive technical indicators for a price DataFrame.
    
    Args:
        df: DataFrame with OHLCV columns (open, high, low, close, volume)
        
    Returns:
        DataFrame with original data plus technical features
    """
    result_df = df.copy()
    tech = TechnicalFeatures()
    
    # Basic price features
    result_df['returns'] = result_df['close'].pct_change()
    result_df['log_returns'] = np.log(result_df['close'] / result_df['close'].shift(1))
    result_df['high_low_ratio'] = result_df['high'] / result_df['low']
    result_df['close_open_ratio'] = result_df['close'] / result_df['open']
    
    # Moving averages
    for window in [5, 10, 20, 50]:
        result_df[f'sma_{window}'] = tech.sma(result_df['close'], window)
        result_df[f'ema_{window}'] = tech.ema(result_df['close'], window)
        result_df[f'close_sma_{window}_ratio'] = result_df['close'] / result_df[f'sma_{window}']
    
    # Momentum indicators
    result_df['rsi_14'] = tech.rsi(result_df['close'], 14)
    
    macd_data = tech.macd(result_df['close'])
    for key, series in macd_data.items():
        result_df[f'macd_{key}'] = series
    
    # Volatility indicators
    bb_data = tech.bollinger_bands(result_df['close'])
    for key, series in bb_data.items():
        result_df[key] = series

    result_df['atr_14'] = tech.atr(result_df['high'], result_df['low'], result_df['close'])
    result_df['volatility_20'] = tech.volatility(result_df['close'], 20)
    result_df['volatility_percentile'] = tech.volatility_percentile(result_df['close'])
    
    # Return features
    result_df['return_1d'] = tech.rolling_return(result_df['close'], 1)
    result_df['return_5d'] = tech.rolling_return(result_df['close'], 5)
    result_df['return_20d'] = tech.rolling_return(result_df['close'], 20)
    
    # Z-score features
    result_df['price_zscore_20'] = tech.z_score(result_df['close'], 20)
    result_df['volume_zscore_20'] = tech.z_score(result_df['volume'], 20) if 'volume' in result_df.columns else np.nan
    
    # Oscillators
    stoch_data = tech.stochastic(result_df['high'], result_df['low'], result_df['close'])
    for key, series in stoch_data.items():
        result_df[key] = series
    
    result_df['williams_r_14'] = tech.williams_r(result_df['high'], result_df['low'], result_df['close'])
    
    # Momentum
    for window in [5, 10, 20]:
        result_df[f'momentum_{window}'] = tech.momentum(result_df['close'], window)
    
    # Volume indicators (if volume available)
    if 'volume' in result_df.columns and not result_df['volume'].isna().all():
        result_df['vpt'] = tech.volume_price_trend(result_df['close'], result_df['volume'])
        result_df['obv'] = tech.on_balance_volume(result_df['close'], result_df['volume'])
        result_df['volume_sma_20'] = tech.sma(result_df['volume'], 20)
        result_df['volume_ratio'] = result_df['volume'] / result_df['volume_sma_20']
    
    # Lagged features
    for lag in [1, 2, 3, 5]:
        result_df[f'close_lag_{lag}'] = result_df['close'].shift(lag)
        result_df[f'returns_lag_{lag}'] = result_df['returns'].shift(lag)
        result_df[f'volume_lag_{lag}'] = result_df['volume'].shift(lag) if 'volume' in result_df.columns else None
    
    return result_df

def macro_surprise_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate macro economic surprise features.
    
    Args:
        df: DataFrame with macro release columns (macro_*_actual, macro_*_surprise)
        
    Returns:
        DataFrame with macro surprise features
    """
    result_df = df.copy()
    
    # Find macro columns
    macro_actual_cols = [col for col in df.columns if col.startswith('macro_') and col.endswith('_actual')]
    macro_surprise_cols = [col for col in df.columns if col.startswith('macro_') and col.endswith('_surprise')]
    
    # Aggregate surprise index
    if macro_surprise_cols:
        # Fill NaN surprises with 0 (no surprise)
        surprise_data = result_df[macro_surprise_cols].fillna(0)
        
        # Simple surprise index (average of standardized surprises)
        surprise_std = surprise_data.std()
        surprise_standardized = surprise_data / surprise_std
        result_df['macro_surprise_index'] = surprise_standardized.mean(axis=1)
        
        # Rolling surprise momentum
        result_df['macro_surprise_momentum_5'] = result_df['macro_surprise_index'].rolling(5).mean()
        result_df['macro_surprise_momentum_20'] = result_df['macro_surprise_index'].rolling(20).mean()
    
    # Macro change features
    for col in macro_actual_cols:
        series_name = col.replace('macro_', '').replace('_actual', '')
        result_df[f'macro_{series_name}_change'] = result_df[col].pct_change()
        result_df[f'macro_{series_name}_change_5'] = result_df[col].pct_change(5)
    
    return result_df

def create_feature_matrix(
    market_df: pd.DataFrame,
    include_technical: bool = True,
    include_macro: bool = True,
    dropna: bool = True
) -> pd.DataFrame:
    """
    Create comprehensive feature matrix from market and macro data.
    
    Args:
        market_df: DataFrame with market data and optional macro columns
        include_technical: Whether to include technical indicators
        include_macro: Whether to include macro features
        dropna: Whether to drop rows with NaN values
        
    Returns:
        Feature matrix ready for ML models
    """
    result_df = market_df.copy()
    
    # Generate technical features
    if include_technical:
        result_df = generate_technical_features(result_df)
    
    # Generate macro features
    if include_macro:
        result_df = macro_surprise_features(result_df)
    
    # Add time-based features
    if 'ts' in result_df.columns:
        result_df['hour'] = pd.to_datetime(result_df['ts']).dt.hour
        result_df['day_of_week'] = pd.to_datetime(result_df['ts']).dt.dayofweek
        result_df['month'] = pd.to_datetime(result_df['ts']).dt.month
        result_df['quarter'] = pd.to_datetime(result_df['ts']).dt.quarter
    
    # Drop NaN values if requested
    if dropna:
        result_df = result_df.dropna()
    
    return result_df

if __name__ == "__main__":
    # Test with synthetic data
    np.random.seed(42)
    
    dates = pd.date_range('2020-01-01', '2024-01-01', freq='D')
    n = len(dates)
    
    # Generate synthetic OHLCV data
    close_prices = 100 + np.cumsum(np.random.randn(n) * 0.01)
    high_prices = close_prices + np.abs(np.random.randn(n) * 0.5)
    low_prices = close_prices - np.abs(np.random.randn(n) * 0.5)
    open_prices = close_prices + np.random.randn(n) * 0.2
    volume = np.random.randint(1000000, 10000000, n)
    
    df = pd.DataFrame({
        'ts': dates,
        'ticker': 'TEST',
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': volume
    })
    
    # Generate features
    feature_df = create_feature_matrix(df, include_macro=False)
    
    print(f"Original shape: {df.shape}")
    print(f"Feature matrix shape: {feature_df.shape}")
    print(f"Number of features added: {feature_df.shape[1] - df.shape[1]}")
    print(f"\nNew features: {[col for col in feature_df.columns if col not in df.columns][:10]}...")
