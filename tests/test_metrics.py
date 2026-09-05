"""
Unit tests for the metrics module.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from lib.metrics import MetricsCalculator, PerformanceMetrics


class TestMetricsCalculator:
    """Test suite for MetricsCalculator."""
    
    @pytest.fixture
    def flat_prices(self):
        """Create flat price data (no change)."""
        dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
        return pd.Series([100.0] * 100, index=dates)
    
    @pytest.fixture
    def uptrend_prices(self):
        """Create upward trending price data."""
        dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
        prices = np.linspace(100, 150, 100)  # +50% over period
        return pd.Series(prices, index=dates)
    
    @pytest.fixture
    def downtrend_prices(self):
        """Create downward trending price data."""
        dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
        prices = np.linspace(100, 80, 100)  # -20% over period
        return pd.Series(prices, index=dates)
    
    @pytest.fixture
    def volatile_prices(self):
        """Create volatile price data."""
        dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.03, 100)  # High volatility
        prices = 100 * (1 + returns).cumprod()
        return pd.Series(prices, index=dates)
    
    @pytest.fixture
    def drawdown_prices(self):
        """Create prices with significant drawdown."""
        dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
        prices = [100] * 20 + list(np.linspace(100, 70, 30)) + [70] * 50
        return pd.Series(prices, index=dates)
    
    def test_calculate_returns(self, uptrend_prices):
        """Test daily returns calculation."""
        returns = MetricsCalculator.calculate_returns(uptrend_prices)
        
        assert isinstance(returns, pd.Series)
        assert len(returns) == len(uptrend_prices) - 1
        assert all(returns >= 0)  # All positive for uptrend
    
    def test_calculate_returns_empty(self):
        """Test returns calculation with empty series."""
        empty = pd.Series([], dtype=float)
        returns = MetricsCalculator.calculate_returns(empty)
        assert len(returns) == 0
    
    def test_calculate_total_return_uptrend(self, uptrend_prices):
        """Test total return calculation for uptrend."""
        total_return = MetricsCalculator.calculate_total_return(uptrend_prices)
        
        assert total_return == pytest.approx(50.0, abs=0.1)  # 50% gain
    
    def test_calculate_total_return_downtrend(self, downtrend_prices):
        """Test total return calculation for downtrend."""
        total_return = MetricsCalculator.calculate_total_return(downtrend_prices)
        
        assert total_return == pytest.approx(-20.0, abs=0.1)  # 20% loss
    
    def test_calculate_total_return_flat(self, flat_prices):
        """Test total return for flat prices."""
        total_return = MetricsCalculator.calculate_total_return(flat_prices)
        
        assert total_return == pytest.approx(0.0, abs=0.01)
    
    def test_calculate_total_return_insufficient_data(self):
        """Test total return with insufficient data."""
        single_price = pd.Series([100.0])
        total_return = MetricsCalculator.calculate_total_return(single_price)
        
        assert total_return == 0.0
    
    def test_calculate_annualized_return(self):
        """Test annualized return calculation."""
        # Create 1-year data with 20% total return
        dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
        prices = pd.Series(np.linspace(100, 120, len(dates)), index=dates)
        
        ann_return = MetricsCalculator.calculate_annualized_return(prices)
        
        # Should be close to 20% for 1-year period
        assert ann_return == pytest.approx(20.0, abs=1.0)
    
    def test_calculate_annualized_return_multi_year(self):
        """Test annualized return over multiple years."""
        # 2 years, 44% total return (20% annualized)
        dates = pd.date_range(start='2023-01-01', end='2024-12-31', freq='D')
        prices = pd.Series(np.linspace(100, 144, len(dates)), index=dates)
        
        ann_return = MetricsCalculator.calculate_annualized_return(prices)
        
        assert ann_return == pytest.approx(20.0, abs=1.0)
    
    def test_calculate_volatility(self, volatile_prices):
        """Test volatility calculation."""
        returns = MetricsCalculator.calculate_returns(volatile_prices)
        vol = MetricsCalculator.calculate_volatility(returns, annualize=True)
        
        assert vol > 0
        assert isinstance(vol, float)
    
    def test_calculate_volatility_flat(self, flat_prices):
        """Test volatility for flat prices."""
        returns = MetricsCalculator.calculate_returns(flat_prices)
        vol = MetricsCalculator.calculate_volatility(returns)
        
        assert vol == 0.0
    
    def test_calculate_volatility_not_annualized(self, uptrend_prices):
        """Test daily volatility (not annualized)."""
        returns = MetricsCalculator.calculate_returns(uptrend_prices)
        daily_vol = MetricsCalculator.calculate_volatility(returns, annualize=False)
        annual_vol = MetricsCalculator.calculate_volatility(returns, annualize=True)
        
        # Annual should be higher (multiplied by sqrt(252))
        assert annual_vol > daily_vol
    
    def test_calculate_sharpe_ratio_positive(self, uptrend_prices):
        """Test Sharpe ratio for positive returns."""
        returns = MetricsCalculator.calculate_returns(uptrend_prices)
        sharpe = MetricsCalculator.calculate_sharpe_ratio(returns)
        
        assert sharpe > 0  # Positive returns should give positive Sharpe
    
    def test_calculate_sharpe_ratio_zero_volatility(self, flat_prices):
        """Test Sharpe ratio with zero volatility."""
        returns = MetricsCalculator.calculate_returns(flat_prices)
        sharpe = MetricsCalculator.calculate_sharpe_ratio(returns)
        
        assert sharpe == 0.0
    
    def test_calculate_max_drawdown(self, drawdown_prices):
        """Test maximum drawdown calculation."""
        max_dd = MetricsCalculator.calculate_max_drawdown(drawdown_prices)
        
        # Drawdown from 100 to 70 = -30%
        assert max_dd == pytest.approx(-30.0, abs=1.0)
        assert max_dd <= 0  # Drawdown is always negative or zero
    
    def test_calculate_max_drawdown_uptrend(self, uptrend_prices):
        """Test max drawdown for pure uptrend."""
        max_dd = MetricsCalculator.calculate_max_drawdown(uptrend_prices)
        
        # Should be close to zero for pure uptrend
        assert max_dd >= -1.0  # Allow small numerical errors
    
    def test_calculate_current_drawdown(self, drawdown_prices):
        """Test current drawdown calculation."""
        current_dd = MetricsCalculator.calculate_current_drawdown(drawdown_prices)
        
        # Currently at 70, peak was 100 = -30%
        assert current_dd == pytest.approx(-30.0, abs=1.0)
    
    def test_calculate_current_drawdown_at_peak(self, uptrend_prices):
        """Test current drawdown when at all-time high."""
        current_dd = MetricsCalculator.calculate_current_drawdown(uptrend_prices)
        
        assert current_dd == pytest.approx(0.0, abs=0.1)
    
    def test_calculate_all_metrics(self, uptrend_prices):
        """Test calculating all metrics at once."""
        data = pd.DataFrame({'Close': uptrend_prices})
        metrics = MetricsCalculator.calculate_all_metrics(data)
        
        assert isinstance(metrics, PerformanceMetrics)
        assert metrics.total_return > 0
        assert metrics.annualized_return > 0
        assert metrics.volatility >= 0
        assert metrics.max_drawdown <= 0
        assert 0 <= metrics.positive_days_pct <= 100
    
    def test_performance_metrics_dataclass(self):
        """Test PerformanceMetrics dataclass creation."""
        metrics = PerformanceMetrics(
            total_return=25.5,
            annualized_return=12.3,
            volatility=15.2,
            sharpe_ratio=0.85,
            max_drawdown=-18.5,
            current_drawdown=-5.2,
            positive_days_pct=58.3
        )
        
        assert metrics.total_return == 25.5
        assert metrics.sharpe_ratio == 0.85
        assert metrics.positive_days_pct == 58.3


class TestMetricsEdgeCases:
    """Test edge cases and error handling."""
    
    def test_single_price_point(self):
        """Test metrics with single data point."""
        single = pd.Series([100.0], index=[datetime.now()])
        
        total_return = MetricsCalculator.calculate_total_return(single)
        assert total_return == 0.0
        
        ann_return = MetricsCalculator.calculate_annualized_return(single)
        assert ann_return == 0.0
    
    def test_two_price_points(self):
        """Test metrics with minimal data."""
        dates = pd.date_range(start='2024-01-01', periods=2, freq='D')
        prices = pd.Series([100.0, 105.0], index=dates)
        
        total_return = MetricsCalculator.calculate_total_return(prices)
        assert total_return == pytest.approx(5.0, abs=0.01)
    
    def test_negative_prices(self):
        """Test handling of negative prices (invalid)."""
        dates = pd.date_range(start='2024-01-01', periods=10, freq='D')
        prices = pd.Series([-100.0] * 10, index=dates)
        
        # Should not crash, but results may be meaningless
        returns = MetricsCalculator.calculate_returns(prices)
        assert len(returns) == 9

