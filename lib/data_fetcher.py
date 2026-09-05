"""
Data fetcher module for ticker health checking.

This module provides a robust interface for fetching financial data
using yfinance, with proper error handling and data validation.
"""

import logging
from typing import Optional, Dict, List, Union
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)


class DataFetchError(Exception):
    """Custom exception for data fetching errors."""
    pass


class DataFetcher:
    """
    Fetch and process financial data for ticker health analysis.
    
    This class provides methods to retrieve historical price data and
    ticker information using the yfinance library, with built-in retry
    logic and data validation.
    
    Attributes:
        cache_ttl (int): Time-to-live for cached data in seconds
        yf_params (dict): Default parameters for yfinance downloads
    """
    
    # Valid periods for yfinance
    VALID_PERIODS = ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"]
    VALID_INTERVALS = ["1d", "5d", "1wk", "1mo", "3mo"]
    
    def __init__(self, cache_ttl: int = 3600):
        """
        Initialize the DataFetcher.
        
        Args:
            cache_ttl: Cache time-to-live in seconds (default: 1 hour)
        """
        self.cache_ttl = cache_ttl
        self.yf_params = {
            'progress': False,
            'auto_adjust': True,
            'prepost': False,
            'threads': True,
        }
        self._cache: Dict[str, tuple] = {}
    
    def _validate_period(self, period: str) -> None:
        """Validate that the period is supported by yfinance."""
        if period not in self.VALID_PERIODS:
            raise ValueError(
                f"Invalid period '{period}'. Must be one of {self.VALID_PERIODS}"
            )
    
    def _validate_interval(self, interval: str) -> None:
        """Validate that the interval is supported by yfinance."""
        if interval not in self.VALID_INTERVALS:
            raise ValueError(
                f"Invalid interval '{interval}'. Must be one of {self.VALID_INTERVALS}"
            )
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True
    )
    def fetch_historical_data(
        self,
        ticker: str,
        period: str = "2y",
        interval: str = "1d",
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Fetch historical price data for a ticker.
        
        Args:
            ticker: Ticker symbol (e.g., "VWCE.DE")
            period: Time period for data ("1mo", "6mo", "1y", "2y", "5y")
            interval: Data interval ("1d", "1wk", "1mo")
            use_cache: Whether to use cached data if available
        
        Returns:
            DataFrame with columns: Open, High, Low, Close, Volume
        
        Raises:
            DataFetchError: If data cannot be fetched
            ValueError: If period or interval is invalid
        """
        self._validate_period(period)
        self._validate_interval(interval)
        
        cache_key = f"{ticker}_{period}_{interval}"
        
        # Check cache
        if use_cache and cache_key in self._cache:
            cached_data, timestamp = self._cache[cache_key]
            if (datetime.now() - timestamp).total_seconds() < self.cache_ttl:
                logger.info(f"Using cached data for {ticker}")
                return cached_data.copy()
        
        try:
            logger.info(f"Fetching data for {ticker} (period={period}, interval={interval})")
            
            data = yf.download(
                tickers=ticker,
                period=period,
                interval=interval,
                **self.yf_params
            )
            
            if data.empty:
                raise DataFetchError(f"No data returned for ticker '{ticker}'")
            
            # Handle MultiIndex columns (when multiple tickers)
            if isinstance(data.columns, pd.MultiIndex):
                # Check number of levels before dropping
                if data.columns.nlevels == 2:
                    data.columns = data.columns.droplevel(1)
                elif data.columns.nlevels > 2:
                    # Multiple tickers case
                    data = data.droplevel(0, axis=1)

            
            # Ensure we have the essential columns
            essential_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            missing_cols = [col for col in essential_cols if col not in data.columns]
            
            if missing_cols:
                logger.warning(f"Missing columns for {ticker}: {missing_cols}")
            
            # Keep only essential columns that exist
            available_cols = [col for col in essential_cols if col in data.columns]
            data = data[available_cols]
            
            # Remove rows with all NaN values
            data = data.dropna(how='all')

            # Yahoo a volte restituisce l'ultima riga con Volume valorizzato
            # ma OHLC ancora NaN (sessione non sincronizzata sul feed EU).
            # dropna(how='all') non la intercetta: va scartata separatamente.
            if 'Close' in data.columns:
                incomplete = data['Close'].isna()
                if incomplete.any():
                    logger.warning(
                        f"Scarto {incomplete.sum()} righe con Close mancante per "
                        f"{ticker}: {list(data.index[incomplete])}"
                    )
                    data = data[~incomplete]

            if data.empty:
                raise DataFetchError(f"All data for '{ticker}' contains NaN values")
            
            # Cache the result
            self._cache[cache_key] = (data.copy(), datetime.now())
            
            logger.info(f"Successfully fetched {len(data)} rows for {ticker}")
            return data
            
        except Exception as e:
            error_msg = f"Failed to fetch data for '{ticker}': {str(e)}"
            logger.error(error_msg)
            raise DataFetchError(error_msg) from e
    
    def get_ticker_info(self, ticker: str) -> Dict[str, Union[str, float, None]]:
        """
        Retrieve basic information about a ticker.
        
        Args:
            ticker: Ticker symbol
        
        Returns:
            Dictionary containing ticker information
        """
        try:
            logger.info(f"Fetching info for {ticker}")
            ticker_obj = yf.Ticker(ticker)
            info = ticker_obj.info
            
            return {
                'symbol': ticker,
                'name': info.get('longName', info.get('shortName', ticker)),
                'current_price': info.get('currentPrice') or info.get('regularMarketPrice'),
                'previous_close': info.get('previousClose'),
                'currency': info.get('currency', 'EUR'),
                'market_cap': info.get('marketCap'),
                'volume': info.get('volume'),
                'average_volume': info.get('averageVolume'),
                'fifty_two_week_high': info.get('fiftyTwoWeekHigh'),
                'fifty_two_week_low': info.get('fiftyTwoWeekLow'),
                'exchange': info.get('exchange'),
            }
            
        except Exception as e:
            logger.error(f"Failed to fetch info for {ticker}: {str(e)}")
            return {
                'symbol': ticker,
                'error': str(e),
                'name': ticker
            }
    
    def fetch_multiple_tickers(
        self,
        tickers: List[str],
        period: str = "2y",
        interval: str = "1d"
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch historical data for multiple tickers.
        
        Args:
            tickers: List of ticker symbols
            period: Time period for data
            interval: Data interval
        
        Returns:
            Dictionary mapping ticker symbols to DataFrames
        """
        results = {}
        
        for ticker in tickers:
            try:
                results[ticker] = self.fetch_historical_data(ticker, period, interval)
            except DataFetchError as e:
                logger.warning(f"Skipping {ticker}: {str(e)}")
                continue
        
        return results
    
    def clear_cache(self) -> None:
        """Clear the internal cache."""
        self._cache.clear()
        logger.info("Cache cleared")
    
    def get_latest_price(self, ticker: str) -> Optional[float]:
        """
        Get the most recent closing price for a ticker.
        
        Args:
            ticker: Ticker symbol
        
        Returns:
            Latest closing price or None if unavailable
        """
        try:
            data = self.fetch_historical_data(ticker, period="5d", interval="1d")
            if not data.empty and 'Close' in data.columns:
                return float(data['Close'].iloc[-1])
        except DataFetchError:
            pass
        
        return None

