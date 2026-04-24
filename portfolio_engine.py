from typing import List, Dict, Tuple, Optional
import pandas as pd
import numpy as np
import yfinance as yf
from scipy.optimize import minimize


class PortfolioAnalyzer:

    def __init__(
        self,
        stock_tickers: List[str],
        etf_tickers: List[str],
        start_date: str,
        end_date: str,
        initial_capital: float = 100000,
        risk_free_rate: float = 0.04
    ):
        self.stock_tickers = stock_tickers
        self.etf_tickers = etf_tickers
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital
        self.risk_free_rate = risk_free_rate

        self.prices: pd.DataFrame = None
        self.stock_prices: pd.DataFrame = None
        self.etf_prices: pd.DataFrame = None
        self.returns: pd.DataFrame = None
        self.stock_returns: pd.DataFrame = None
        self.etf_returns: pd.DataFrame = None

        self.optimal_weights: pd.Series = None
        self.optimal_metrics: Dict[str, float] = None

    def download_data(self) -> bool:
        all_tickers = self.stock_tickers + self.etf_tickers

        try:
            data = yf.download(
                all_tickers,
                start=self.start_date,
                end=self.end_date,
                auto_adjust=True,
                progress=False
            )

            # yfinance returns a different shape for single tickers
            if len(all_tickers) == 1:
                prices = pd.DataFrame({all_tickers[0]: data['Close']})
            else:
                prices = data['Close']

            prices = prices.dropna(how='all').ffill().dropna()

            if prices.empty:
                return False

            self.prices = prices
            self.stock_prices = prices[self.stock_tickers]
            self.etf_prices = prices[self.etf_tickers]

            self.returns = prices.pct_change().dropna()
            self.stock_returns = self.returns[self.stock_tickers]
            self.etf_returns = self.returns[self.etf_tickers]

            return True

        except Exception as e:
            print(f"Error downloading data: {e}")
            return False

    def calculate_annualized_metrics(self) -> pd.DataFrame:
        if self.stock_returns is None:
            raise ValueError("No return data available. Run download_data() first.")

        rows = []
        for ticker in self.stock_tickers:
            r = self.stock_returns[ticker]
            ann_ret = r.mean() * 252
            ann_vol = r.std() * np.sqrt(252)
            rows.append({
                'Ticker': ticker,
                'Annual Return': ann_ret,
                'Annual Volatility': ann_vol,
                'Sharpe Ratio': (ann_ret - self.risk_free_rate) / ann_vol
            })

        return pd.DataFrame(rows)

    def calculate_covariance_matrix(self) -> pd.DataFrame:
        if self.stock_returns is None:
            raise ValueError("No return data available. Run download_data() first.")
        return self.stock_returns.cov() * 252

    def calculate_correlation_matrix(self) -> pd.DataFrame:
        if self.stock_returns is None:
            raise ValueError("No return data available. Run download_data() first.")
        return self.stock_returns.corr()

    def calculate_beta(self, asset_returns: pd.Series, benchmark_returns: pd.Series) -> float:
        cov = np.cov(asset_returns, benchmark_returns)[0, 1]
        bench_var = np.var(benchmark_returns)
        return 0.0 if bench_var == 0 else cov / bench_var

    def calculate_stock_etf_relationships(self) -> pd.DataFrame:
        if self.stock_returns is None or self.etf_returns is None:
            raise ValueError("No return data available. Run download_data() first.")

        rows = []
        for stock in self.stock_tickers:
            sr = self.stock_returns[stock]
            corrs = {etf: sr.corr(self.etf_returns[etf]) for etf in self.etf_tickers}
            betas = {etf: self.calculate_beta(sr, self.etf_returns[etf]) for etf in self.etf_tickers}
            best = max(corrs, key=corrs.get)

            rows.append({
                'Stock': stock,
                'Highest Corr ETF': best,
                'Correlation': corrs[best],
                'Beta to ETF': betas[best],
                'Beta to SPY': betas.get('SPY', np.nan),
                'Annual Volatility': sr.std() * np.sqrt(252),
                'Annual Return': sr.mean() * 252
            })

        return pd.DataFrame(rows)

    def optimize_portfolio(
        self,
        train_start_date: str = None,
        train_end_date: str = None
    ) -> Tuple[pd.Series, Dict[str, float]]:
        """Maximize Sharpe ratio. Long-only, weights sum to 1, max 35% per asset."""
        if self.stock_returns is None:
            raise ValueError("No return data available. Run download_data() first.")

        if train_start_date and train_end_date:
            train_ret = self.stock_returns[train_start_date:train_end_date]
            if train_ret.empty:
                raise ValueError("No data in specified training period")
        else:
            train_ret = self.stock_returns

        n = len(self.stock_tickers)
        mean_ret = train_ret.mean() * 252
        cov = train_ret.cov() * 252

        def neg_sharpe(w):
            vol = np.sqrt(w @ cov.values @ w)
            return 0 if vol == 0 else -(np.dot(w, mean_ret) - self.risk_free_rate) / vol

        result = minimize(
            neg_sharpe,
            np.full(n, 1.0 / n),
            method='SLSQP',
            bounds=tuple((0, 0.35) for _ in range(n)),
            constraints=[{'type': 'eq', 'fun': lambda w: w.sum() - 1}],
            options={'maxiter': 1000}
        )

        if not result.success:
            raise ValueError("Optimization failed to converge")

        weights = pd.Series(result.x, index=self.stock_tickers)
        port_ret = float(weights @ mean_ret)
        port_vol = float(np.sqrt(weights.values @ cov.values @ weights.values))

        metrics = {
            'Expected Annual Return': port_ret,
            'Expected Annual Volatility': port_vol,
            'Sharpe Ratio': (port_ret - self.risk_free_rate) / port_vol
        }

        self.optimal_weights = weights
        self.optimal_metrics = metrics
        return weights, metrics

    def backtest_portfolio(
        self,
        weights: pd.Series = None,
        test_start_date: str = None,
        test_end_date: str = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if weights is None:
            if self.optimal_weights is None:
                raise ValueError("No weights provided and no optimal weights calculated")
            weights = self.optimal_weights

        if self.stock_returns is None:
            raise ValueError("No return data available. Run download_data() first.")

        if test_start_date and test_end_date:
            stock_ret = self.stock_returns[test_start_date:test_end_date]
            etf_ret = self.etf_returns[test_start_date:test_end_date]
            if stock_ret.empty:
                raise ValueError("No data in specified test period")
        else:
            stock_ret = self.stock_returns
            etf_ret = self.etf_returns

        equity = (1 + (stock_ret * weights).sum(axis=1)).cumprod() * self.initial_capital

        if 'SPY' not in self.etf_tickers:
            return pd.DataFrame({'Portfolio Value': equity}), None

        spy_equity = (1 + etf_ret['SPY']).cumprod() * self.initial_capital
        return pd.DataFrame({'Portfolio Value': equity}), pd.DataFrame({'SPY Value': spy_equity})

    def calculate_after_tax_returns(
        self,
        capital_gains_tax: float,
        dividend_tax: float,
        returns_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Estimate after-tax returns per asset.
        Pass returns_df to scope the calculation to the training period.
        """
        if self.stock_returns is None:
            raise ValueError("No return data available. Run download_data() first.")

        src = returns_df if returns_df is not None else self.stock_returns
        rows = []

        for ticker in self.stock_tickers:
            ann_ret = src[ticker].mean() * 252

            try:
                div_yield = float(yf.Ticker(ticker).info.get("dividendYield") or 0.0)
            except Exception:
                div_yield = 0.0

            price_app = ann_ret - div_yield
            after_tax = price_app * (1 - capital_gains_tax) + div_yield * (1 - dividend_tax)

            rows.append({
                "Ticker": ticker,
                "Pre-Tax Return": ann_ret,
                "Dividend Yield": div_yield,
                "Price Appreciation": price_app,
                "After-Tax Return": after_tax,
                "Tax Drag": ann_ret - after_tax,
            })

        return pd.DataFrame(rows)

    def estimate_turnover_cost(
        self,
        weights_old: pd.Series,
        weights_new: pd.Series,
        turnover_penalty: float
    ) -> float:
        return float(np.abs(weights_new - weights_old).sum() * turnover_penalty)

    def optimize_after_tax_portfolio(
        self,
        capital_gains_tax: float,
        dividend_tax: float,
        turnover_penalty: float,
        train_start_date: str = None,
        train_end_date: str = None,
        current_weights: pd.Series = None,
    ) -> Tuple[pd.Series, Dict[str, float]]:
        if self.stock_returns is None:
            raise ValueError("No return data available. Run download_data() first.")

        if train_start_date and train_end_date:
            train_ret = self.stock_returns[train_start_date:train_end_date]
            if train_ret.empty:
                raise ValueError("No data in specified training period")
        else:
            train_ret = self.stock_returns

        n = len(self.stock_tickers)

        if current_weights is None:
            current_weights = pd.Series([1.0 / n] * n, index=self.stock_tickers)

        at_means = self.calculate_after_tax_returns(
            capital_gains_tax, dividend_tax, returns_df=train_ret
        ).set_index("Ticker")["After-Tax Return"]
        cov = train_ret.cov() * 252

        def objective(w):
            ww = pd.Series(w, index=self.stock_tickers)
            vol = float(np.sqrt(w @ cov.values @ w))
            if vol == 0:
                return 0.0
            sharpe = (float(np.dot(w, at_means)) - self.risk_free_rate) / vol
            return -(sharpe - self.estimate_turnover_cost(current_weights, ww, turnover_penalty))

        result = minimize(
            objective,
            current_weights.values.copy(),
            method="SLSQP",
            bounds=tuple((0, 0.35) for _ in range(n)),
            constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
            options={"maxiter": 1000},
        )

        if not result.success:
            raise ValueError("After-tax optimization failed to converge")

        at_weights = pd.Series(result.x, index=self.stock_tickers)
        port_ret = float(np.dot(at_weights, at_means))
        port_vol = float(np.sqrt(at_weights.values @ cov.values @ at_weights.values))

        return at_weights, {
            "After-Tax Annual Return": port_ret,
            "Annual Volatility": port_vol,
            "After-Tax Sharpe Ratio": (port_ret - self.risk_free_rate) / port_vol,
            "Estimated Turnover Cost": self.estimate_turnover_cost(current_weights, at_weights, turnover_penalty),
        }

    def backtest_after_tax_portfolio(
        self,
        weights: pd.Series,
        capital_gains_tax: float,
        dividend_tax: float,
        turnover_penalty: float,
        test_start_date: str = None,
        test_end_date: str = None,
        current_weights: pd.Series = None,
    ) -> pd.DataFrame:
        if self.stock_returns is None:
            raise ValueError("No return data available. Run download_data() first.")

        if test_start_date and test_end_date:
            test_ret = self.stock_returns[test_start_date:test_end_date]
            if test_ret.empty:
                raise ValueError("No data in specified test period")
        else:
            test_ret = self.stock_returns

        gross = (test_ret * weights).sum(axis=1)
        cg_drag = gross.apply(lambda r: max(r, 0.0) * capital_gains_tax)

        weighted_div_yield = 0.0
        for ticker in self.stock_tickers:
            try:
                dy = float(yf.Ticker(ticker).info.get("dividendYield") or 0.0)
            except Exception:
                dy = 0.0
            weighted_div_yield += float(weights[ticker]) * dy

        net = gross - cg_drag - (weighted_div_yield * dividend_tax / 252)

        if current_weights is None:
            n = len(self.stock_tickers)
            current_weights = pd.Series([1.0 / n] * n, index=self.stock_tickers)

        tc = float(np.abs(weights - current_weights).sum()) * turnover_penalty
        equity = (1 + net).cumprod() * (self.initial_capital * (1 - tc))

        return pd.DataFrame({"Estimated After-Tax Portfolio Value": equity})

    def calculate_risk_contributions(
        self,
        weights: pd.Series,
        train_start_date: str = None,
        train_end_date: str = None,
    ) -> pd.DataFrame:
        """Component variance decomposition: risk_i = w_i * (Sigma @ w)_i."""
        if self.stock_returns is None:
            raise ValueError("No return data available. Run download_data() first.")

        if train_start_date and train_end_date:
            train_ret = self.stock_returns[train_start_date:train_end_date]
            if train_ret.empty:
                raise ValueError("No data in specified training period")
        else:
            train_ret = self.stock_returns

        sigma = train_ret.cov().values * 252
        w = weights.values
        port_var = float(w @ sigma @ w)
        marginal = sigma @ w
        contrib = w * marginal

        df = pd.DataFrame({
            "Ticker": weights.index,
            "Weight": w,
            "Marginal Risk Contribution": marginal,
            "Risk Contribution": contrib,
            "Percent Risk Contribution": contrib / port_var,
        })

        return df.sort_values("Percent Risk Contribution", ascending=False).reset_index(drop=True)

    def calculate_eigen_risk_decomposition(
        self,
        weights: pd.Series,
        train_start_date: str = None,
        train_end_date: str = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """PCA of the covariance matrix. Eigenvector signs are arbitrary."""
        if self.stock_returns is None:
            raise ValueError("No return data available. Run download_data() first.")

        if train_start_date and train_end_date:
            train_ret = self.stock_returns[train_start_date:train_end_date]
            if train_ret.empty:
                raise ValueError("No data in specified training period")
        else:
            train_ret = self.stock_returns

        sigma = train_ret.cov().values * 252
        vals, vecs = np.linalg.eigh(sigma)

        order = np.argsort(vals)[::-1]
        vals, vecs = vals[order], vecs[:, order]

        pct = vals / vals.sum()
        labels = [f"Eigenfactor {i + 1}" for i in range(len(vals))]

        summary = pd.DataFrame({
            "Eigenfactor": labels,
            "Eigenvalue": vals,
            "Percent Variance Explained": pct,
            "Cumulative Variance Explained": np.cumsum(pct),
            "Portfolio Exposure": vecs.T @ weights.values,
        })

        loadings = pd.DataFrame(vecs, index=self.stock_tickers, columns=labels)
        return summary, loadings
