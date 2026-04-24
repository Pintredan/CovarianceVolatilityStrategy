"""
Backtesting Engine for Quantitative Trading Strategies.

This module provides a simple but professional backtesting framework for
evaluating trading strategies on historical stock data.
"""

from typing import List, Dict, Any
from datetime import datetime
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from strategy import Strategy, MovingAverageCrossoverStrategy


class Backtester:
    """
    A backtesting engine for evaluating trading strategies.

    The backtester downloads historical price data, executes a trading strategy
    day by day, tracks portfolio value, and provides performance analytics.
    """

    def __init__(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str,
        initial_cash: float,
        strategy: Strategy
    ):
        """
        Initialize the Backtester.

        Args:
            tickers: List of stock ticker symbols
            start_date: Start date for backtest (YYYY-MM-DD)
            end_date: End date for backtest (YYYY-MM-DD)
            initial_cash: Initial cash amount in portfolio
            strategy: Strategy object that generates trading signals
        """
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.initial_cash = initial_cash
        self.strategy = strategy

        # Portfolio state
        self.cash = initial_cash
        self.positions: Dict[str, int] = {ticker: 0 for ticker in tickers}
        self.trades: List[Dict[str, Any]] = []
        self.portfolio_history: List[Dict[str, Any]] = []

        # Market data
        self.historical_data: Dict[str, pd.DataFrame] = {}
        self.trading_dates: pd.DatetimeIndex = None

    def download_data(self) -> None:
        """
        Download historical adjusted close price data using yfinance.

        Data is stored in self.historical_data as a dict mapping ticker -> DataFrame.
        """
        print(f"Downloading data for {len(self.tickers)} tickers...")

        for ticker in self.tickers:
            try:
                # Download adjusted close prices
                data = yf.download(
                    ticker,
                    start=self.start_date,
                    end=self.end_date,
                    progress=False
                )

                if data.empty:
                    print(f"Warning: No data available for {ticker}")
                    continue

                # Extract adjusted close prices
                self.historical_data[ticker] = pd.DataFrame({
                    'Close': data['Adj Close']
                })

            except Exception as e:
                print(f"Error downloading {ticker}: {e}")

        # Get common trading dates across all tickers
        if self.historical_data:
            all_dates = set.intersection(*[
                set(df.index) for df in self.historical_data.values()
            ])
            self.trading_dates = pd.DatetimeIndex(sorted(all_dates))

        print(f"Downloaded {len(self.trading_dates)} trading days of data")

    def run(self) -> None:
        """
        Run the backtest day by day.

        For each trading day:
        1. Generate signals from the strategy
        2. Execute trades based on signals
        3. Update portfolio value
        """
        if not self.historical_data:
            self.download_data()

        if not self.trading_dates.any():
            print("No data to backtest")
            return

        print(f"\nRunning backtest from {self.start_date} to {self.end_date}...")

        for current_date in self.trading_dates:
            # Generate trading signals
            signals = self.strategy.generate_signals(
                current_date=current_date,
                historical_data=self.historical_data,
                positions=self.positions,
                cash=self.cash
            )

            # Execute trades based on signals
            self._execute_trades(current_date, signals)

            # Calculate and record portfolio value
            portfolio_value = self._calculate_portfolio_value(current_date)
            self.portfolio_history.append({
                'date': current_date,
                'cash': self.cash,
                'positions_value': portfolio_value - self.cash,
                'total_value': portfolio_value
            })

        print(f"Backtest complete. Executed {len(self.trades)} trades.")

    def _execute_trades(
        self,
        current_date: pd.Timestamp,
        signals: Dict[str, str]
    ) -> None:
        """
        Execute buy/sell trades based on strategy signals.

        Args:
            current_date: Current date in the backtest
            signals: Dict mapping ticker -> signal ('buy', 'sell', 'hold')
        """
        for ticker, signal in signals.items():
            if signal == 'hold':
                continue

            # Get current price
            price = self.historical_data[ticker].loc[current_date, 'Close']

            if signal == 'buy':
                # Calculate number of shares to buy (equal weight allocation)
                # Divide available cash by number of tickers for equal weighting
                allocation = self.cash / len(self.tickers)
                shares_to_buy = int(allocation / price)

                # Check if we have enough cash
                cost = shares_to_buy * price
                if shares_to_buy > 0 and cost <= self.cash:
                    self.cash -= cost
                    self.positions[ticker] += shares_to_buy

                    # Log the trade
                    self.trades.append({
                        'date': current_date,
                        'ticker': ticker,
                        'action': 'BUY',
                        'shares': shares_to_buy,
                        'price': price,
                        'value': cost
                    })

            elif signal == 'sell':
                # Sell all shares of this ticker
                shares_to_sell = self.positions[ticker]

                # Check if we have a position to sell
                if shares_to_sell > 0:
                    proceeds = shares_to_sell * price
                    self.cash += proceeds
                    self.positions[ticker] = 0

                    # Log the trade
                    self.trades.append({
                        'date': current_date,
                        'ticker': ticker,
                        'action': 'SELL',
                        'shares': shares_to_sell,
                        'price': price,
                        'value': proceeds
                    })

    def _calculate_portfolio_value(self, current_date: pd.Timestamp) -> float:
        """
        Calculate total portfolio value at a given date.

        Args:
            current_date: Date to calculate portfolio value

        Returns:
            Total portfolio value (cash + positions)
        """
        positions_value = 0.0

        for ticker, shares in self.positions.items():
            if shares > 0:
                price = self.historical_data[ticker].loc[current_date, 'Close']
                positions_value += shares * price

        return self.cash + positions_value

    def plot_results(self) -> None:
        """
        Plot the portfolio value / equity curve over time.
        """
        if not self.portfolio_history:
            print("No portfolio history to plot. Run the backtest first.")
            return

        # Convert portfolio history to DataFrame
        df = pd.DataFrame(self.portfolio_history)
        df.set_index('date', inplace=True)

        # Create the plot
        plt.figure(figsize=(12, 6))
        plt.plot(df.index, df['total_value'], label='Portfolio Value', linewidth=2)
        plt.axhline(
            y=self.initial_cash,
            color='r',
            linestyle='--',
            label='Initial Cash',
            alpha=0.7
        )

        plt.xlabel('Date')
        plt.ylabel('Portfolio Value ($)')
        plt.title('Backtesting Results: Portfolio Equity Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def print_summary(self) -> None:
        """
        Print a summary of backtest performance metrics.
        """
        if not self.portfolio_history:
            print("No results to summarize. Run the backtest first.")
            return

        final_value = self.portfolio_history[-1]['total_value']
        total_return = ((final_value - self.initial_cash) / self.initial_cash) * 100

        print("\n" + "="*60)
        print("BACKTEST PERFORMANCE SUMMARY")
        print("="*60)
        print(f"Initial Cash:          ${self.initial_cash:,.2f}")
        print(f"Final Portfolio Value: ${final_value:,.2f}")
        print(f"Total Return:          {total_return:+.2f}%")
        print(f"Number of Trades:      {len(self.trades)}")

        # Calculate per-ticker performance
        if self.trades:
            ticker_performance = self._calculate_ticker_performance()

            if ticker_performance:
                best_ticker = max(ticker_performance, key=ticker_performance.get)
                worst_ticker = min(ticker_performance, key=ticker_performance.get)

                print(f"\nBest Performer:        {best_ticker} ({ticker_performance[best_ticker]:+.2f}%)")
                print(f"Worst Performer:       {worst_ticker} ({ticker_performance[worst_ticker]:+.2f}%)")

        print("="*60 + "\n")

    def _calculate_ticker_performance(self) -> Dict[str, float]:
        """
        Calculate profit/loss percentage for each ticker.

        Returns:
            Dict mapping ticker -> return percentage
        """
        ticker_pnl = {}

        for ticker in self.tickers:
            ticker_trades = [t for t in self.trades if t['ticker'] == ticker]

            if not ticker_trades:
                continue

            total_cost = sum(t['value'] for t in ticker_trades if t['action'] == 'BUY')
            total_proceeds = sum(t['value'] for t in ticker_trades if t['action'] == 'SELL')

            # If still holding position, calculate current value
            if self.positions[ticker] > 0:
                last_price = self.historical_data[ticker].iloc[-1]['Close']
                total_proceeds += self.positions[ticker] * last_price

            if total_cost > 0:
                ticker_pnl[ticker] = ((total_proceeds - total_cost) / total_cost) * 100

        return ticker_pnl


if __name__ == "__main__":
    # Example usage
    tickers = ["AAPL", "MSFT", "GOOGL", "SPY"]
    strategy = MovingAverageCrossoverStrategy(short_window=20, long_window=50)

    backtester = Backtester(
        tickers=tickers,
        start_date="2020-01-01",
        end_date="2024-01-01",
        initial_cash=100000,
        strategy=strategy
    )

    backtester.run()
    backtester.plot_results()
    backtester.print_summary()
