"""
Portfolio Analytics Engine.

This module provides the core analytics for portfolio construction,
risk analysis, and optimization.
"""

from typing import List, Dict, Tuple
import pandas as pd
import numpy as np
import yfinance as yf
from scipy.optimize import minimize


class PortfolioAnalyzer:
    """
    Portfolio analytics engine for multi-asset portfolio analysis.

    Calculates returns, risk metrics, correlations, betas, and performs
    portfolio optimization using Modern Portfolio Theory.
    """

    def __init__(
        self,
        stock_tickers: List[str],
        etf_tickers: List[str],
        start_date: str,
        end_date: str,
        initial_capital: float = 100000,
        risk_free_rate: float = 0.04
    ):
        """
        Initialize the PortfolioAnalyzer.

        Args:
            stock_tickers: List of stock ticker symbols
            etf_tickers: List of ETF ticker symbols for benchmarking
            start_date: Start date for analysis (YYYY-MM-DD)
            end_date: End date for analysis (YYYY-MM-DD)
            initial_capital: Initial portfolio value (default: $100,000)
            risk_free_rate: Annual risk-free rate for Sharpe ratio (default: 0.04)
        """
        self.stock_tickers = stock_tickers
        self.etf_tickers = etf_tickers
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital
        self.risk_free_rate = risk_free_rate

        # Data containers
        self.prices: pd.DataFrame = None
        self.stock_prices: pd.DataFrame = None
        self.etf_prices: pd.DataFrame = None
        self.returns: pd.DataFrame = None
        self.stock_returns: pd.DataFrame = None
        self.etf_returns: pd.DataFrame = None

        # Analytics results
        self.optimal_weights: pd.Series = None
        self.optimal_metrics: Dict[str, float] = None

    def download_data(self) -> bool:
        """
        Download historical price data using yfinance.

        Returns:
            True if data download was successful, False otherwise
        """
        all_tickers = self.stock_tickers + self.etf_tickers

        try:
            # Download data with auto_adjust=True
            data = yf.download(
                all_tickers,
                start=self.start_date,
                end=self.end_date,
                auto_adjust=True,
                progress=False
            )

            # Extract Close prices
            if len(all_tickers) == 1:
                # yfinance returns different structure for single ticker
                prices = pd.DataFrame({all_tickers[0]: data['Close']})
            else:
                prices = data['Close']

            # Handle missing data
            # Drop rows where all values are NaN
            prices = prices.dropna(how='all')

            # Forward fill missing values
            prices = prices.ffill()

            # Drop any remaining NaN rows
            prices = prices.dropna()

            if prices.empty:
                return False

            self.prices = prices

            # Split into stocks and ETFs
            self.stock_prices = prices[self.stock_tickers]
            self.etf_prices = prices[self.etf_tickers]

            # Calculate returns
            self.returns = prices.pct_change().dropna()
            self.stock_returns = self.returns[self.stock_tickers]
            self.etf_returns = self.returns[self.etf_tickers]

            return True

        except Exception as e:
            print(f"Error downloading data: {e}")
            return False

    def calculate_annualized_metrics(self) -> pd.DataFrame:
        """
        Calculate annualized return, volatility, and Sharpe ratio for each stock.

        Returns:
            DataFrame with metrics for each stock
        """
        if self.stock_returns is None:
            raise ValueError("No return data available. Run download_data() first.")

        metrics = []

        for ticker in self.stock_tickers:
            returns = self.stock_returns[ticker]

            # Annualized return (252 trading days)
            annual_return = returns.mean() * 252

            # Annualized volatility
            annual_volatility = returns.std() * np.sqrt(252)

            # Sharpe ratio
            sharpe_ratio = (annual_return - self.risk_free_rate) / annual_volatility

            metrics.append({
                'Ticker': ticker,
                'Annual Return': annual_return,
                'Annual Volatility': annual_volatility,
                'Sharpe Ratio': sharpe_ratio
            })

        return pd.DataFrame(metrics)

    def calculate_covariance_matrix(self) -> pd.DataFrame:
        """
        Calculate annualized covariance matrix of stock returns.

        Returns:
            Covariance matrix DataFrame
        """
        if self.stock_returns is None:
            raise ValueError("No return data available. Run download_data() first.")

        # Annualized covariance (252 trading days)
        return self.stock_returns.cov() * 252

    def calculate_correlation_matrix(self) -> pd.DataFrame:
        """
        Calculate correlation matrix of stock returns.

        Returns:
            Correlation matrix DataFrame
        """
        if self.stock_returns is None:
            raise ValueError("No return data available. Run download_data() first.")

        return self.stock_returns.corr()

    def calculate_beta(
        self,
        asset_returns: pd.Series,
        benchmark_returns: pd.Series
    ) -> float:
        """
        Calculate beta of an asset relative to a benchmark.

        Args:
            asset_returns: Daily returns of the asset
            benchmark_returns: Daily returns of the benchmark

        Returns:
            Beta coefficient
        """
        covariance = np.cov(asset_returns, benchmark_returns)[0, 1]
        benchmark_variance = np.var(benchmark_returns)

        if benchmark_variance == 0:
            return 0.0

        return covariance / benchmark_variance

    def calculate_stock_etf_relationships(self) -> pd.DataFrame:
        """
        Calculate correlation and beta of each stock to each ETF.

        Returns:
            DataFrame with stock-ETF relationship metrics
        """
        if self.stock_returns is None or self.etf_returns is None:
            raise ValueError("No return data available. Run download_data() first.")

        results = []

        for stock in self.stock_tickers:
            stock_ret = self.stock_returns[stock]

            # Calculate metrics for each ETF
            etf_correlations = {}
            etf_betas = {}

            for etf in self.etf_tickers:
                etf_ret = self.etf_returns[etf]

                # Correlation
                correlation = stock_ret.corr(etf_ret)
                etf_correlations[etf] = correlation

                # Beta
                beta = self.calculate_beta(stock_ret, etf_ret)
                etf_betas[etf] = beta

            # Find highest correlated ETF
            max_corr_etf = max(etf_correlations, key=etf_correlations.get)
            max_correlation = etf_correlations[max_corr_etf]
            beta_to_max_etf = etf_betas[max_corr_etf]

            # Calculate SPY beta if SPY is in ETF list
            spy_beta = etf_betas.get('SPY', np.nan)

            # Get stock metrics
            annual_return = stock_ret.mean() * 252
            annual_volatility = stock_ret.std() * np.sqrt(252)

            results.append({
                'Stock': stock,
                'Highest Corr ETF': max_corr_etf,
                'Correlation': max_correlation,
                'Beta to ETF': beta_to_max_etf,
                'Beta to SPY': spy_beta,
                'Annual Volatility': annual_volatility,
                'Annual Return': annual_return
            })

        return pd.DataFrame(results)

    def optimize_portfolio(
        self,
        train_start_date: str = None,
        train_end_date: str = None
    ) -> Tuple[pd.Series, Dict[str, float]]:
        """
        Optimize portfolio to maximize Sharpe ratio.

        Constraints:
        - Long-only (weights >= 0)
        - Weights sum to 1
        - No single position > 35%

        Args:
            train_start_date: Optional start date for training period (YYYY-MM-DD)
            train_end_date: Optional end date for training period (YYYY-MM-DD)
                           If not provided, uses full dataset

        Returns:
            Tuple of (optimal_weights Series, metrics dict)
        """
        if self.stock_returns is None:
            raise ValueError("No return data available. Run download_data() first.")

        # Filter to training period if specified
        if train_start_date and train_end_date:
            train_returns = self.stock_returns[train_start_date:train_end_date]
            if train_returns.empty:
                raise ValueError("No data in specified training period")
        else:
            train_returns = self.stock_returns

        n_assets = len(self.stock_tickers)

        # Mean returns and covariance matrix (using ONLY training data)
        mean_returns = train_returns.mean() * 252
        cov_matrix = train_returns.cov() * 252

        # Objective function: negative Sharpe ratio (for minimization)
        def negative_sharpe(weights):
            portfolio_return = np.dot(weights, mean_returns)
            portfolio_volatility = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))

            if portfolio_volatility == 0:
                return 0

            sharpe = (portfolio_return - self.risk_free_rate) / portfolio_volatility
            return -sharpe

        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}  # weights sum to 1
        ]

        # Bounds: 0 <= weight <= 0.35
        bounds = tuple((0, 0.35) for _ in range(n_assets))

        # Initial guess: equal weight
        initial_weights = np.array([1.0 / n_assets] * n_assets)

        # Optimize
        result = minimize(
            negative_sharpe,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000}
        )

        if not result.success:
            raise ValueError("Optimization failed to converge")

        # Extract optimal weights
        optimal_weights = pd.Series(result.x, index=self.stock_tickers)

        # Calculate portfolio metrics
        portfolio_return = np.dot(optimal_weights, mean_returns)
        portfolio_volatility = np.sqrt(
            np.dot(optimal_weights, np.dot(cov_matrix, optimal_weights))
        )
        sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_volatility

        metrics = {
            'Expected Annual Return': portfolio_return,
            'Expected Annual Volatility': portfolio_volatility,
            'Sharpe Ratio': sharpe_ratio
        }

        # Store results
        self.optimal_weights = optimal_weights
        self.optimal_metrics = metrics

        return optimal_weights, metrics

    def backtest_portfolio(
        self,
        weights: pd.Series = None,
        test_start_date: str = None,
        test_end_date: str = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Backtest portfolio using given weights and compare to SPY benchmark.

        Args:
            weights: Portfolio weights (uses optimal weights if None)
            test_start_date: Optional start date for test period (YYYY-MM-DD)
            test_end_date: Optional end date for test period (YYYY-MM-DD)
                          If not provided, uses full dataset

        Returns:
            Tuple of (portfolio equity curve, SPY equity curve)
        """
        if weights is None:
            if self.optimal_weights is None:
                raise ValueError("No weights provided and no optimal weights calculated")
            weights = self.optimal_weights

        if self.stock_returns is None:
            raise ValueError("No return data available. Run download_data() first.")

        # Filter to test period if specified
        if test_start_date and test_end_date:
            test_stock_returns = self.stock_returns[test_start_date:test_end_date]
            test_etf_returns = self.etf_returns[test_start_date:test_end_date]

            if test_stock_returns.empty:
                raise ValueError("No data in specified test period")
        else:
            test_stock_returns = self.stock_returns
            test_etf_returns = self.etf_returns

        # Calculate portfolio daily returns
        portfolio_returns = (test_stock_returns * weights).sum(axis=1)

        # Calculate cumulative returns
        portfolio_cumulative = (1 + portfolio_returns).cumprod()

        # Create equity curve
        portfolio_equity = portfolio_cumulative * self.initial_capital

        # Calculate SPY benchmark if available
        if 'SPY' in self.etf_tickers:
            spy_returns = test_etf_returns['SPY']
            spy_cumulative = (1 + spy_returns).cumprod()
            spy_equity = spy_cumulative * self.initial_capital

            return (
                pd.DataFrame({'Portfolio Value': portfolio_equity}),
                pd.DataFrame({'SPY Value': spy_equity})
            )
        else:
            return (
                pd.DataFrame({'Portfolio Value': portfolio_equity}),
                None
            )

    def calculate_after_tax_returns(
        self,
        capital_gains_tax: float,
        dividend_tax: float
    ) -> pd.DataFrame:
        """
        Estimate after-tax annual returns for each stock.

        Assumptions:
        - All price appreciation is taxed at the long-term capital gains rate.
        - Dividends are taxed at the qualified dividend rate.
        - Dividend yield is fetched from yfinance; defaults to 0 if unavailable.

        Args:
            capital_gains_tax: Long-term capital gains tax rate (e.g. 0.20)
            dividend_tax: Qualified dividend tax rate (e.g. 0.15)

        Returns:
            DataFrame with per-asset return decomposition and tax drag
        """
        if self.stock_returns is None:
            raise ValueError("No return data available. Run download_data() first.")

        rows = []
        for ticker in self.stock_tickers:
            annual_return = self.stock_returns[ticker].mean() * 252

            # Fetch dividend yield from yfinance; default to 0 if missing or unavailable
            try:
                info = yf.Ticker(ticker).info
                dividend_yield = float(info.get("dividendYield") or 0.0)
            except Exception:
                dividend_yield = 0.0

            price_appreciation = annual_return - dividend_yield
            after_tax_return = (
                price_appreciation * (1 - capital_gains_tax)
                + dividend_yield * (1 - dividend_tax)
            )
            tax_drag = annual_return - after_tax_return

            rows.append({
                "Ticker": ticker,
                "Pre-Tax Return": annual_return,
                "Dividend Yield": dividend_yield,
                "Price Appreciation": price_appreciation,
                "After-Tax Return": after_tax_return,
                "Tax Drag": tax_drag,
            })

        return pd.DataFrame(rows)

    def estimate_turnover_cost(
        self,
        weights_old: pd.Series,
        weights_new: pd.Series,
        turnover_penalty: float
    ) -> float:
        """
        Estimate the transaction cost of rebalancing between two allocations.

        Turnover = sum(|new_weight - old_weight|)
        Turnover cost = turnover * turnover_penalty

        Args:
            weights_old: Current portfolio weights
            weights_new: Target portfolio weights
            turnover_penalty: Cost per unit of turnover (e.g. 0.0025 for 25bps)

        Returns:
            Estimated turnover cost as a fraction of portfolio value
        """
        turnover = np.abs(weights_new - weights_old).sum()
        return float(turnover * turnover_penalty)

    def optimize_after_tax_portfolio(
        self,
        capital_gains_tax: float,
        dividend_tax: float,
        turnover_penalty: float,
        train_start_date: str = None,
        train_end_date: str = None,
        current_weights: pd.Series = None,
    ) -> Tuple[pd.Series, Dict[str, float]]:
        """
        Optimize portfolio to maximize after-tax Sharpe ratio net of turnover cost.

        Objective: maximize (after_tax_sharpe - turnover_cost), i.e. minimize the negative.
        Constraints mirror the pre-tax optimizer: long-only, weights sum to 1, max 35% per asset.
        Uses equal-weight as the baseline allocation if no current weights are provided.

        Args:
            capital_gains_tax: Long-term capital gains tax rate
            dividend_tax: Qualified dividend tax rate
            turnover_penalty: Transaction cost proxy per unit of turnover
            train_start_date: Optional start date for training period (YYYY-MM-DD)
            train_end_date: Optional end date for training period (YYYY-MM-DD)
            current_weights: Existing allocation (uses equal-weight if None)

        Returns:
            Tuple of (after_tax_weights Series, metrics dict)
        """
        if self.stock_returns is None:
            raise ValueError("No return data available. Run download_data() first.")

        if train_start_date and train_end_date:
            train_returns = self.stock_returns[train_start_date:train_end_date]
            if train_returns.empty:
                raise ValueError("No data in specified training period")
        else:
            train_returns = self.stock_returns

        n_assets = len(self.stock_tickers)

        if current_weights is None:
            current_weights = pd.Series(
                [1.0 / n_assets] * n_assets, index=self.stock_tickers
            )

        # After-tax expected returns and training-period covariance
        after_tax_df = self.calculate_after_tax_returns(capital_gains_tax, dividend_tax)
        after_tax_means = after_tax_df.set_index("Ticker")["After-Tax Return"]
        cov_matrix = train_returns.cov() * 252

        def neg_after_tax_sharpe(weights: np.ndarray) -> float:
            w = pd.Series(weights, index=self.stock_tickers)
            port_return = float(np.dot(weights, after_tax_means))
            port_vol = float(np.sqrt(np.dot(weights, np.dot(cov_matrix, weights))))
            if port_vol == 0:
                return 0.0
            sharpe = (port_return - self.risk_free_rate) / port_vol
            tc = self.estimate_turnover_cost(current_weights, w, turnover_penalty)
            return -(sharpe - tc)

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
        bounds = tuple((0, 0.35) for _ in range(n_assets))
        initial_weights = current_weights.values.copy()

        result = minimize(
            neg_after_tax_sharpe,
            initial_weights,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 1000},
        )

        if not result.success:
            raise ValueError("After-tax optimization failed to converge")

        at_weights = pd.Series(result.x, index=self.stock_tickers)

        port_return = float(np.dot(at_weights, after_tax_means))
        port_vol = float(
            np.sqrt(np.dot(at_weights, np.dot(cov_matrix, at_weights)))
        )
        sharpe = (port_return - self.risk_free_rate) / port_vol
        turnover_cost = self.estimate_turnover_cost(current_weights, at_weights, turnover_penalty)

        metrics = {
            "After-Tax Annual Return": port_return,
            "Annual Volatility": port_vol,
            "After-Tax Sharpe Ratio": sharpe,
            "Estimated Turnover Cost": turnover_cost,
        }

        return at_weights, metrics
