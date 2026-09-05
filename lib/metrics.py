"""
Metrics calculation module for ticker health analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
from dataclasses import dataclass


@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    current_drawdown: float
    positive_days_pct: float


class MetricsCalculator:
    """Calculate various financial metrics for health checking."""
    
    TRADING_DAYS_PER_YEAR = 252
    RISK_FREE_RATE = 0.02  # 2% annual risk-free rate
    
    @staticmethod
    def calculate_returns(prices: pd.Series) -> pd.Series:
        """Calculate daily returns from prices."""
        return prices.pct_change().dropna()
    
    @staticmethod
    def calculate_total_return(prices: pd.Series) -> float:
        """Calculate total return over the period."""
        if len(prices) < 2:
            return 0.0
        return (prices.iloc[-1] / prices.iloc[0] - 1) * 100
    
    @staticmethod
    def calculate_annualized_return(prices: pd.Series) -> float:
        """Calculate annualized return."""
        if len(prices) < 2:
            return 0.0
        
        total_return = prices.iloc[-1] / prices.iloc[0]
        days = (prices.index[-1] - prices.index[0]).days
        years = days / 365.25
        
        if years <= 0:
            return 0.0
        
        return (total_return ** (1 / years) - 1) * 100
    
    @staticmethod
    def calculate_volatility(returns: pd.Series, annualize: bool = True) -> float:
        """Calculate volatility (standard deviation of returns)."""
        vol = returns.std()
        if annualize:
            vol *= np.sqrt(MetricsCalculator.TRADING_DAYS_PER_YEAR)
        return vol * 100
    
    @staticmethod
    def calculate_sharpe_ratio(returns: pd.Series) -> float:
        """Calculate Sharpe ratio."""
        excess_returns = returns - (MetricsCalculator.RISK_FREE_RATE / MetricsCalculator.TRADING_DAYS_PER_YEAR)
        if excess_returns.std() == 0:
            return 0.0
        return np.sqrt(MetricsCalculator.TRADING_DAYS_PER_YEAR) * excess_returns.mean() / excess_returns.std()
    
    @staticmethod
    def calculate_max_drawdown(prices: pd.Series) -> float:
        """Calculate maximum drawdown."""
        cumulative = (1 + prices.pct_change()).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min() * 100
    
    @staticmethod
    def calculate_current_drawdown(prices: pd.Series) -> float:
        """Calculate current drawdown from peak."""
        cumulative = (1 + prices.pct_change()).cumprod()
        running_max = cumulative.max()
        current = cumulative.iloc[-1]
        return ((current - running_max) / running_max) * 100
    
    @classmethod
    def calculate_all_metrics(cls, data: pd.DataFrame) -> PerformanceMetrics:
        """
        Calculate all performance metrics.
        
        Args:
            data: DataFrame with 'Close' column
        
        Returns:
            PerformanceMetrics object with all calculated metrics
        """
        prices = data['Close']
        returns = cls.calculate_returns(prices)
        
        return PerformanceMetrics(
            total_return=cls.calculate_total_return(prices),
            annualized_return=cls.calculate_annualized_return(prices),
            volatility=cls.calculate_volatility(returns),
            sharpe_ratio=cls.calculate_sharpe_ratio(returns),
            max_drawdown=cls.calculate_max_drawdown(prices),
            current_drawdown=cls.calculate_current_drawdown(prices),
            positive_days_pct=(returns > 0).sum() / len(returns) * 100
        )

