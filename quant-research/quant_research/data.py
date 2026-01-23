"""
Data fetching module for financial market data.
"""

import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import Optional


class DataFetcher:
    """Fetches financial market data from Yahoo Finance."""

    def __init__(self):
        self._cache: dict[str, pd.DataFrame] = {}

    def get_stock_data(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        period: str = "1y",
    ) -> pd.DataFrame:
        """
        Fetch historical stock data.

        Args:
            symbol: Stock ticker symbol (e.g., 'AAPL', 'GOOGL')
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            period: Time period if dates not specified ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')

        Returns:
            DataFrame with OHLCV data
        """
        cache_key = f"{symbol}_{start_date}_{end_date}_{period}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        ticker = yf.Ticker(symbol)

        if start_date and end_date:
            data = ticker.history(start=start_date, end=end_date)
        else:
            data = ticker.history(period=period)

        if data.empty:
            raise ValueError(f"No data found for symbol: {symbol}")

        # Standardize column names
        data.columns = [col.lower().replace(" ", "_") for col in data.columns]

        self._cache[cache_key] = data
        return data

    def get_multiple_stocks(
        self,
        symbols: list[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        period: str = "1y",
    ) -> dict[str, pd.DataFrame]:
        """
        Fetch data for multiple stocks.

        Args:
            symbols: List of ticker symbols
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            period: Time period if dates not specified

        Returns:
            Dictionary mapping symbols to their DataFrames
        """
        result = {}
        for symbol in symbols:
            try:
                result[symbol] = self.get_stock_data(
                    symbol, start_date, end_date, period
                )
            except ValueError as e:
                print(f"Warning: {e}")
        return result

    def get_combined_close_prices(
        self,
        symbols: list[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        period: str = "1y",
    ) -> pd.DataFrame:
        """
        Get closing prices for multiple symbols in a single DataFrame.

        Args:
            symbols: List of ticker symbols
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            period: Time period if dates not specified

        Returns:
            DataFrame with close prices for all symbols
        """
        data = self.get_multiple_stocks(symbols, start_date, end_date, period)
        close_prices = pd.DataFrame()

        for symbol, df in data.items():
            if "close" in df.columns:
                close_prices[symbol] = df["close"]

        return close_prices.dropna()

    def get_stock_info(self, symbol: str) -> dict:
        """
        Get stock information and metadata.

        Args:
            symbol: Stock ticker symbol

        Returns:
            Dictionary with stock information
        """
        ticker = yf.Ticker(symbol)
        return ticker.info
