"""
Strategy module for backtesting engine.

This module defines the base Strategy class and example strategy implementations.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any
import pandas as pd
import numpy as np


class Strategy(ABC):
    """
    Abstract base class for trading strategies.

    All strategies must implement the generate_signals method which returns
    buy/sell/hold signals for each ticker based on market data and current positions.
    """

    @abstractmethod
    def generate_signals(
        self,
        current_date: pd.Timestamp,
        historical_data: Dict[str, pd.DataFrame],
        positions: Dict[str, int],
        cash: float
    ) -> Dict[str, str]:
        """
        Generate trading signals for each ticker.

        Args:
            current_date: Current date in the backtest
            historical_data: Dict mapping ticker -> DataFrame with historical price data
            positions: Dict mapping ticker -> number of shares held
            cash: Available cash in portfolio

        Returns:
            Dict mapping ticker -> signal ('buy', 'sell', or 'hold')
        """
        pass


class MovingAverageCrossoverStrategy(Strategy):
    """
    Moving Average Crossover Strategy.

    Buy signal: Short MA crosses above Long MA
    Sell signal: Short MA crosses below Long MA

    Allocates portfolio equally across all tickers when signals are generated.
    """

    def __init__(self, short_window: int = 20, long_window: int = 50):
        """
        Initialize the Moving Average Crossover Strategy.

        Args:
            short_window: Period for short-term moving average (default: 20)
            long_window: Period for long-term moving average (default: 50)
        """
        self.short_window = short_window
        self.long_window = long_window

    def generate_signals(
        self,
        current_date: pd.Timestamp,
        historical_data: Dict[str, pd.DataFrame],
        positions: Dict[str, int],
        cash: float
    ) -> Dict[str, str]:
        """
        Generate signals based on moving average crossover.

        Args:
            current_date: Current date in the backtest
            historical_data: Dict mapping ticker -> DataFrame with columns ['Close']
            positions: Dict mapping ticker -> number of shares held
            cash: Available cash in portfolio

        Returns:
            Dict mapping ticker -> signal ('buy', 'sell', or 'hold')
        """
        signals = {}

        for ticker, data in historical_data.items():
            # Get data up to current date
            data_to_date = data[data.index <= current_date].copy()

            # Need enough data to calculate long moving average
            if len(data_to_date) < self.long_window:
                signals[ticker] = 'hold'
                continue

            # Calculate moving averages
            data_to_date['SMA_short'] = data_to_date['Close'].rolling(
                window=self.short_window
            ).mean()
            data_to_date['SMA_long'] = data_to_date['Close'].rolling(
                window=self.long_window
            ).mean()

            # Get current and previous values
            current_short = data_to_date['SMA_short'].iloc[-1]
            current_long = data_to_date['SMA_long'].iloc[-1]

            # Check if we have previous data for crossover detection
            if len(data_to_date) < self.long_window + 1:
                signals[ticker] = 'hold'
                continue

            prev_short = data_to_date['SMA_short'].iloc[-2]
            prev_long = data_to_date['SMA_long'].iloc[-2]

            # Detect crossovers
            # Bullish crossover: short MA crosses above long MA
            if prev_short <= prev_long and current_short > current_long:
                # Only buy if we don't already have a position
                if positions.get(ticker, 0) == 0:
                    signals[ticker] = 'buy'
                else:
                    signals[ticker] = 'hold'

            # Bearish crossover: short MA crosses below long MA
            elif prev_short >= prev_long and current_short < current_long:
                # Only sell if we have a position
                if positions.get(ticker, 0) > 0:
                    signals[ticker] = 'sell'
                else:
                    signals[ticker] = 'hold'

            else:
                signals[ticker] = 'hold'

        return signals
