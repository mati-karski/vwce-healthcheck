"""
Unit tests for the data_fetcher module.
"""

import pytest
import pandas as pd
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock

from lib.data_fetcher import DataFetcher, DataFetchError


class TestDataFetcher:
    """Test suite for DataFetcher class."""
    
    @pytest.fixture
    def fetcher(self):
        """Create a DataFetcher instance for testing."""
        return DataFetcher(cache_ttl=60)
    
    @pytest.fixture
    def mock_price_data(self):
        """Create mock price data."""
        dates = pd.date_range(start='2024-01-01', end='2024-01-10', freq='D')
        return pd.DataFrame({
            'Open': [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0],
            'High': [101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0, 110.0],
            'Low': [99.0, 100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0],
            'Close': [100.5, 101.5, 102.5, 103.5, 104.5, 105.5, 106.5, 107.5, 108.5, 109.5],
            'Volume': [1000000] * 10
        }, index=dates)
    
    def test_initialization(self, fetcher):
        """Test DataFetcher initialization."""
        assert fetcher.cache_ttl == 60
        assert fetcher.yf_params['progress'] is False
        assert fetcher.yf_params['auto_adjust'] is True
        assert isinstance(fetcher._cache, dict)
    
    def test_validate_period_valid(self, fetcher):
        """Test period validation with valid periods."""
        valid_periods = ["1mo", "6mo", "1y", "2y", "5y"]
        for period in valid_periods:
            fetcher._validate_period(period)  # Should not raise
    
    def test_validate_period_invalid(self, fetcher):
        """Test period validation with invalid period."""
        with pytest.raises(ValueError, match="Invalid period"):
            fetcher._validate_period("invalid_period")
    
    def test_validate_interval_valid(self, fetcher):
        """Test interval validation with valid intervals."""
        valid_intervals = ["1d", "1wk", "1mo"]
        for interval in valid_intervals:
            fetcher._validate_interval(interval)  # Should not raise
    
    def test_validate_interval_invalid(self, fetcher):
        """Test interval validation with invalid interval."""
        with pytest.raises(ValueError, match="Invalid interval"):
            fetcher._validate_interval("5min")
    
    @patch('lib.data_fetcher.yf.download')
    def test_fetch_historical_data_success(self, mock_download, fetcher, mock_price_data):
        """Test successful data fetching."""
        mock_download.return_value = mock_price_data
        
        result = fetcher.fetch_historical_data("VWCE.DE", period="1mo")
        
        assert isinstance(result, pd.DataFrame)
        assert not result.empty
        assert 'Close' in result.columns
        assert len(result) == 10
        mock_download.assert_called_once()
    
    @patch('lib.data_fetcher.yf.download')
    def test_fetch_historical_data_empty_response(self, mock_download, fetcher):
        """Test handling of empty data response."""
        mock_download.return_value = pd.DataFrame()
        
        with pytest.raises(DataFetchError, match="No data returned"):
            fetcher.fetch_historical_data("INVALID.TICKER")
    
    @patch('lib.data_fetcher.yf.download')
    def test_fetch_historical_data_multiindex(self, mock_download, fetcher, mock_price_data):
        """Test handling of MultiIndex columns."""
        multi_data = mock_price_data.copy()
        
        # yfinance REALE restituisce (Column, Ticker) come MultiIndex
        new_columns = pd.MultiIndex.from_tuples(
            [(col, 'VWCE.DE') for col in multi_data.columns],
            names=['Price', 'Ticker']
        )
        multi_data.columns = new_columns
        
        mock_download.return_value = multi_data
        
        result = fetcher.fetch_historical_data("VWCE.DE", use_cache=False)
        
        assert not isinstance(result.columns, pd.MultiIndex)
        assert 'Close' in result.columns
        assert len(result) == 10


    
    @patch('lib.data_fetcher.yf.download')
    def test_fetch_historical_data_drops_rows_with_nan_close(
        self, mock_download, fetcher, mock_price_data
    ):
        """Riga con Volume valorizzato ma OHLC NaN (sessione non sincronizzata
        sul feed EU): non deve arrivare al chiamante, ma non deve nemmeno far
        fallire l'intero fetch se le righe precedenti sono valide."""
        data_with_incomplete_row = mock_price_data.copy()
        data_with_incomplete_row.loc[
            data_with_incomplete_row.index[-1], ['Open', 'High', 'Low', 'Close']
        ] = float('nan')
        mock_download.return_value = data_with_incomplete_row

        result = fetcher.fetch_historical_data("VWCE.DE", use_cache=False)

        assert len(result) == len(mock_price_data) - 1
        assert not result['Close'].isna().any()

    @patch('lib.data_fetcher.yf.download')
    def test_fetch_historical_data_caching(self, mock_download, fetcher, mock_price_data):
        """Test that caching works correctly."""
        mock_download.return_value = mock_price_data
        
        # First call
        result1 = fetcher.fetch_historical_data("VWCE.DE", use_cache=True)
        # Second call (should use cache)
        result2 = fetcher.fetch_historical_data("VWCE.DE", use_cache=True)
        
        # Should only download once
        assert mock_download.call_count == 1
        pd.testing.assert_frame_equal(result1, result2)
    
    @patch('lib.data_fetcher.yf.download')
    def test_fetch_historical_data_no_cache(self, mock_download, fetcher, mock_price_data):
        """Test fetching without cache."""
        mock_download.return_value = mock_price_data
        
        result1 = fetcher.fetch_historical_data("VWCE.DE", use_cache=False)
        result2 = fetcher.fetch_historical_data("VWCE.DE", use_cache=False)
        
        # Should download twice
        assert mock_download.call_count == 2
    
    @patch('lib.data_fetcher.yf.Ticker')
    def test_get_ticker_info_success(self, mock_ticker, fetcher):
        """Test successful ticker info retrieval."""
        mock_info = {
            'longName': 'Vanguard FTSE All-World UCITS ETF',
            'currentPrice': 100.5,
            'currency': 'EUR',
            'previousClose': 99.8,
            'marketCap': 1000000000,
            'volume': 50000,
            'exchange': 'XETRA'
        }
        mock_ticker.return_value.info = mock_info
        
        result = fetcher.get_ticker_info("VWCE.DE")
        
        assert result['symbol'] == "VWCE.DE"
        assert result['name'] == 'Vanguard FTSE All-World UCITS ETF'
        assert result['current_price'] == 100.5
        assert result['currency'] == 'EUR'
    
    @patch('lib.data_fetcher.yf.Ticker')
    def test_get_ticker_info_error(self, mock_ticker, fetcher):
        """Test ticker info retrieval with error."""
        mock_ticker.side_effect = Exception("API Error")
        
        result = fetcher.get_ticker_info("INVALID.TICKER")
        
        assert 'error' in result
        assert result['symbol'] == "INVALID.TICKER"
    
    @patch('lib.data_fetcher.yf.download')
    def test_fetch_multiple_tickers(self, mock_download, fetcher, mock_price_data):
        """Test fetching data for multiple tickers."""
        mock_download.return_value = mock_price_data
        
        tickers = ["VWCE.DE", "VUSA.L"]
        results = fetcher.fetch_multiple_tickers(tickers)
        
        assert len(results) == 2
        assert "VWCE.DE" in results
        assert "VUSA.L" in results
        assert mock_download.call_count == 2
    
    @patch('lib.data_fetcher.yf.download')
    def test_get_latest_price(self, mock_download, fetcher, mock_price_data):
        """Test getting the latest price."""
        mock_download.return_value = mock_price_data
        
        price = fetcher.get_latest_price("VWCE.DE")
        
        assert price == 109.5  # Last close price in mock data
    
    def test_clear_cache(self, fetcher):
        """Test cache clearing."""
        fetcher._cache['test_key'] = ("data", datetime.now())
        assert len(fetcher._cache) > 0
        
        fetcher.clear_cache()
        
        assert len(fetcher._cache) == 0


class TestDataFetcherIntegration:
    """Integration tests (require internet connection)."""
    
    @pytest.mark.integration
    def test_real_ticker_fetch(self):
        """Test fetching real data from yfinance."""
        fetcher = DataFetcher()
        
        # Use a stable ticker for testing
        data = fetcher.fetch_historical_data("VWCE.DE", period="5d")
        
        assert isinstance(data, pd.DataFrame)
        assert not data.empty
        assert 'Close' in data.columns
    
    @pytest.mark.integration
    def test_real_ticker_info(self):
        """Test fetching real ticker info."""
        fetcher = DataFetcher()
        
        info = fetcher.get_ticker_info("VWCE.DE")
        
        assert info['symbol'] == "VWCE.DE"
        assert 'name' in info
        assert 'error' not in info

