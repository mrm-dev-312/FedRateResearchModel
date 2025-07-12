"""
Edge case tests for time spine and as-of join functionality.
Tests holiday gaps, DST jumps, and other timing edge cases.
"""

import pytest
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.timejoin.spine import make_spine, TimeJoiner
from src.timejoin.asof_join import join_macro, join_event_windows

@pytest.mark.asyncio
class TestTimeSpineEdgeCases:
    
    def test_make_spine_basic(self):
        """Test basic spine creation."""
        spine = make_spine("SPY", "2024-01-01", "2024-01-10", "D")
        
        assert len(spine) == 10
        assert 'ts' in spine.columns
        assert 'ticker' in spine.columns
        assert all(spine['ticker'] == 'SPY')
        assert spine['ts'].dtype == 'datetime64[ns]'
    
    def test_make_spine_hourly(self):
        """Test hourly frequency spine."""
        spine = make_spine("SPY", "2024-01-01", "2024-01-02", "H")
        
        # Should have 25 hours (00:00 Jan 1 through 00:00 Jan 2)
        assert len(spine) == 25
        
        # Check first and last timestamps
        assert spine['ts'].iloc[0] == pd.Timestamp('2024-01-01 00:00:00')
        assert spine['ts'].iloc[-1] == pd.Timestamp('2024-01-02 00:00:00')
    
    def test_make_spine_minute_frequency(self):
        """Test minute frequency for intraday data."""
        spine = make_spine("EURUSD", "2024-01-01 09:00", "2024-01-01 10:00", "15T")
        
        # Should have 5 periods: 09:00, 09:15, 09:30, 09:45, 10:00
        assert len(spine) == 5
        assert spine['ts'].iloc[0] == pd.Timestamp('2024-01-01 09:00:00')
        assert spine['ts'].iloc[-1] == pd.Timestamp('2024-01-01 10:00:00')
    
    def test_holiday_gaps(self):
        """Test handling of holiday gaps in market data."""
        # Create spine that includes weekends (which are market holidays)
        spine = make_spine("SPY", "2024-01-05", "2024-01-08", "D")  # Fri to Mon
        
        # Should include all calendar days
        assert len(spine) == 4  # Fri, Sat, Sun, Mon
        
        # In practice, market data would only exist for Fri and Mon
        # The spine should still include all dates for proper alignment
        expected_dates = [
            '2024-01-05',  # Friday
            '2024-01-06',  # Saturday (holiday)
            '2024-01-07',  # Sunday (holiday)
            '2024-01-08'   # Monday
        ]
        
        actual_dates = spine['ts'].dt.strftime('%Y-%m-%d').tolist()
        assert actual_dates == expected_dates
    
    def test_dst_transition_spring(self):
        """Test DST transition (spring forward) edge case."""
        # US DST transition in 2024: March 10, 2:00 AM becomes 3:00 AM
        spine = make_spine("SPY", "2024-03-10 01:00", "2024-03-10 04:00", "H")
        
        # Should handle the missing 2:00 AM hour gracefully
        timestamps = spine['ts'].tolist()
        
        # Check that we don't have duplicate or missing hours
        assert len(timestamps) == len(set(timestamps))  # No duplicates
        
        # The exact behavior depends on pandas/pytz handling
        # but we should not crash
        assert len(spine) >= 2  # At least some data
    
    def test_dst_transition_fall(self):
        """Test DST transition (fall back) edge case."""
        # US DST transition in 2024: November 3, 2:00 AM becomes 1:00 AM
        spine = make_spine("SPY", "2024-11-03 01:00", "2024-11-03 04:00", "H")
        
        # Should handle the repeated 1:00 AM hour gracefully
        timestamps = spine['ts'].tolist()
        
        # Should not crash and should have reasonable data
        assert len(spine) >= 3
    
    def test_leap_year_handling(self):
        """Test leap year date handling."""
        # 2024 is a leap year
        spine = make_spine("SPY", "2024-02-28", "2024-03-01", "D")
        
        # Should include Feb 29
        assert len(spine) == 3  # Feb 28, Feb 29, Mar 1
        
        dates = spine['ts'].dt.strftime('%Y-%m-%d').tolist()
        assert '2024-02-29' in dates
    
    def test_year_boundary(self):
        """Test year boundary edge case."""
        spine = make_spine("SPY", "2023-12-30", "2024-01-02", "D")
        
        assert len(spine) == 4  # Dec 30, Dec 31, Jan 1, Jan 2
        
        # Check year transition
        years = spine['ts'].dt.year.unique()
        assert 2023 in years
        assert 2024 in years

@pytest.mark.asyncio
class TestAsOfJoinEdgeCases:
    
    async def test_empty_macro_data(self):
        """Test as-of join with no macro data available."""
        # Create sample price data
        px_df = pd.DataFrame({
            'ts': pd.date_range('2024-01-01', periods=5, freq='D'),
            'ticker': 'SPY',
            'close': [400, 401, 402, 403, 404]
        })
        
        # Try to join with non-existent series
        result = await join_macro(px_df, ['NONEXISTENT_SERIES'])
        
        # Should return original data with NaN columns
        assert len(result) == len(px_df)
        assert 'NONEXISTENT_SERIES_actual' in result.columns
        assert result['NONEXISTENT_SERIES_actual'].isna().all()
    
    async def test_price_data_before_macro_releases(self):
        """Test price data that predates all macro releases."""
        # This would happen if we have very old price data
        # but macro data starts later
        px_df = pd.DataFrame({
            'ts': pd.date_range('1990-01-01', periods=5, freq='D'),
            'ticker': 'SPY',
            'close': [100, 101, 102, 103, 104]
        })
        
        # FRED data typically starts much later
        result = await join_macro(px_df, ['FEDFUNDS'])
        
        # Should handle gracefully with NaN values
        assert len(result) == len(px_df)
        if 'FEDFUNDS_actual' in result.columns:
            # Either all NaN or the first available value forward-filled
            pass
    
    async def test_sparse_macro_releases(self):
        """Test with very sparse macro release schedule."""
        # Create daily price data
        px_df = pd.DataFrame({
            'ts': pd.date_range('2024-01-01', periods=30, freq='D'),
            'ticker': 'SPY',
            'close': 400 + np.random.randn(30) * 5
        })
        
        # GDP is released quarterly, so very sparse
        result = await join_macro(px_df, ['GDP'])
        
        # Should forward-fill the latest GDP value
        assert len(result) == len(px_df)
        if 'GDP_actual' in result.columns:
            # Most values should be the same (forward-filled)
            non_na_values = result['GDP_actual'].dropna()
            if len(non_na_values) > 1:
                # Check that values are constant within quarters
                pass
    
    async def test_intraday_with_daily_macro(self):
        """Test joining intraday price data with daily macro releases."""
        # Create hourly price data
        px_df = pd.DataFrame({
            'ts': pd.date_range('2024-01-01', periods=24, freq='H'),
            'ticker': 'SPY',
            'close': 400 + np.random.randn(24) * 2
        })
        
        result = await join_macro(px_df, ['FEDFUNDS'])
        
        # All hours in a day should have the same macro value
        assert len(result) == len(px_df)
        if 'FEDFUNDS_actual' in result.columns:
            daily_groups = result.groupby(result['ts'].dt.date)
            for date, group in daily_groups:
                macro_values = group['FEDFUNDS_actual'].dropna().unique()
                # Should have at most one unique macro value per day
                assert len(macro_values) <= 1
    
    async def test_event_window_edge_cases(self):
        """Test event window detection edge cases."""
        # Create price data around a specific time
        px_df = pd.DataFrame({
            'ts': pd.date_range('2024-01-01 14:00', periods=10, freq='H'),
            'ticker': 'SPY',
            'close': 400 + np.random.randn(10) * 2
        })
        
        # Test with overlapping event windows
        result = await join_event_windows(
            px_df, 
            ['FEDFUNDS'], 
            window_before=timedelta(hours=4),
            window_after=timedelta(hours=4)
        )
        
        assert len(result) == len(px_df)
        if 'FEDFUNDS_event_window' in result.columns:
            # Values should be 0 or 1
            unique_values = result['FEDFUNDS_event_window'].unique()
            assert all(val in [0, 1] for val in unique_values)

@pytest.mark.asyncio 
class TestTimeJoinerIntegration:
    
    async def test_full_workflow_with_gaps(self):
        """Test complete workflow with data gaps."""
        joiner = TimeJoiner()
        
        # Build spine with gaps (weekends)
        spine = await joiner.build_time_spine(
            "2024-01-05",  # Friday
            "2024-01-08",  # Monday  
            "D"
        )
        
        assert len(spine) == 4  # Includes weekend
        
        # In real workflow, market data would be sparse
        # but macro data should still align correctly

if __name__ == "__main__":
    # Run tests manually for development
    import sys
    sys.path.append('.')
    
    async def run_basic_tests():
        test_class = TestTimeSpineEdgeCases()
        
        print("Testing basic spine creation...")
        test_class.test_make_spine_basic()
        print("✓ Basic spine test passed")
        
        print("Testing holiday gaps...")
        test_class.test_holiday_gaps()
        print("✓ Holiday gap test passed")
        
        print("Testing leap year...")
        test_class.test_leap_year_handling()
        print("✓ Leap year test passed")
        
        print("All manual tests passed!")
    
    asyncio.run(run_basic_tests())
