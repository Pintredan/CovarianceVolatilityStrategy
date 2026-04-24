import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from portfolio_engine import PortfolioAnalyzer


st.set_page_config(
    page_title="Portfolio Analytics",
    page_icon="📊",
    layout="wide"
)


def main():
    st.title("📊 Portfolio Analytics & Optimization")
    st.markdown("""
    Quantitative portfolio analysis using Modern Portfolio Theory.
    Enter your stock tickers and ETF benchmarks to generate an optimized portfolio.

    **Train/Test Split:** Optimization is trained on historical data and tested on out-of-sample
    data to avoid look-ahead bias.

    **Disclaimer:** For educational purposes only. Not financial advice.
    """)

    st.markdown("---")

    with st.sidebar:
        st.header("Portfolio Configuration")

        stocks_input = st.text_area(
            "Stock Tickers (comma-separated)",
            value="AAPL, MSFT, NVDA, JPM, XOM",
            help="Enter stock ticker symbols separated by commas"
        )

        etf_input = st.text_area(
            "ETF Benchmarks (comma-separated)",
            value="SPY, XLK, XLE, XLF",
            help="Enter ETF ticker symbols for sector/market benchmarks"
        )

        st.subheader("Training Period")
        col1, col2 = st.columns(2)
        with col1:
            train_start_date = st.date_input("Train Start", value=pd.to_datetime("2020-01-01"))
        with col2:
            train_end_date = st.date_input("Train End", value=pd.to_datetime("2024-12-31"))

        st.subheader("Testing Period")
        col3, col4 = st.columns(2)
        with col3:
            test_start_date = st.date_input("Test Start", value=pd.to_datetime("2025-01-01"))
        with col4:
            test_end_date = st.date_input("Test End", value=pd.to_datetime("2026-04-23"))

        initial_capital = st.number_input(
            "Initial Portfolio Value ($)",
            min_value=1000, max_value=10000000, value=100000, step=10000
        )

        risk_free_rate = st.number_input(
            "Risk-Free Rate (annual)",
            min_value=0.0, max_value=0.20, value=0.04, step=0.01, format="%.4f"
        )

        st.subheader("Tax Assumptions")
        use_after_tax = st.checkbox("Enable After-Tax Optimization", value=False)
        capital_gains_tax = st.number_input(
            "Capital Gains Tax Rate",
            min_value=0.0, max_value=0.50, value=0.20, step=0.01, format="%.2f",
            disabled=not use_after_tax,
            help="Applied to price appreciation (long-term rate assumed)"
        )
        dividend_tax = st.number_input(
            "Dividend Tax Rate",
            min_value=0.0, max_value=0.50, value=0.15, step=0.01, format="%.2f",
            disabled=not use_after_tax,
            help="Applied to dividend income (qualified dividend rate assumed)"
        )
        turnover_penalty = st.number_input(
            "Turnover Penalty",
            min_value=0.0, max_value=0.05, value=0.0025, step=0.0025, format="%.4f",
            disabled=not use_after_tax,
            help="Transaction cost proxy per unit of portfolio turnover"
        )

        run_analysis = st.button("🚀 Run Analysis", type="primary")

    if run_analysis:
        stock_tickers = [s.strip().upper() for s in stocks_input.split(",") if s.strip()]
        etf_tickers = [e.strip().upper() for e in etf_input.split(",") if e.strip()]

        if not stock_tickers:
            st.error("Please enter at least one stock ticker")
            return
        if not etf_tickers:
            st.error("Please enter at least one ETF ticker")
            return

        with st.spinner("Downloading market data..."):
            analyzer = PortfolioAnalyzer(
                stock_tickers=stock_tickers,
                etf_tickers=etf_tickers,
                start_date=train_start_date.strftime("%Y-%m-%d"),
                end_date=test_end_date.strftime("%Y-%m-%d"),
                initial_capital=initial_capital,
                risk_free_rate=risk_free_rate
            )
            success = analyzer.download_data()

        if not success:
            st.error("Failed to download data. Please check your tickers and date range.")
            return

        st.success(f"Downloaded data for {len(stock_tickers)} stocks and {len(etf_tickers)} ETFs")

        st.header("📈 Historical Prices")
        with st.expander("View Price Data", expanded=False):
            st.dataframe(analyzer.prices.tail(10).style.format("${:.2f}"), width='stretch')

        st.header("📊 Stock Performance Metrics")
        st.dataframe(
            analyzer.calculate_annualized_metrics().style.format({
                'Annual Return': '{:.2%}',
                'Annual Volatility': '{:.2%}',
                'Sharpe Ratio': '{:.3f}'
            }),
            width='stretch',
            hide_index=True
        )

        st.header("🔗 Stock Covariance Matrix")
        st.dataframe(
            analyzer.calculate_covariance_matrix().style.format("{:.6f}").background_gradient(cmap='RdYlGn', axis=None),
            width='stretch'
        )

        st.header("🔗 Stock Correlation Matrix")
        st.dataframe(
            analyzer.calculate_correlation_matrix().style.format("{:.3f}").background_gradient(cmap='coolwarm', axis=None),
            width='stretch'
        )

        st.header("🎯 Stock-ETF Relationships")
        st.dataframe(
            analyzer.calculate_stock_etf_relationships().style.format({
                'Correlation': '{:.3f}',
                'Beta to ETF': '{:.3f}',
                'Beta to SPY': '{:.3f}',
                'Annual Volatility': '{:.2%}',
                'Annual Return': '{:.2%}'
            }),
            width='stretch',
            hide_index=True
        )

        n_assets = len(stock_tickers)
        if n_assets < 10:
            confidence_level = "Low (Small sample size)"
            portfolio_label = "Unconstrained MPT Portfolio (Low Confidence)"
        elif n_assets < 30:
            confidence_level = "Medium"
            portfolio_label = "Optimal Portfolio"
        else:
            confidence_level = "High"
            portfolio_label = "Optimal Portfolio"

        st.header("🎯 Optimal Portfolio")
        st.info(f"Training on data from {train_start_date.strftime('%Y-%m-%d')} to {train_end_date.strftime('%Y-%m-%d')}")

        if n_assets < 10:
            st.warning(
                "**⚠️ Small Asset Universe Detected (N < 10)**\n\n"
                "Mean-variance optimization may produce unstable and highly concentrated portfolios "
                "when the number of assets is small.\n\n"
                "Covariance estimates become noisy, and the optimizer may overweight individual assets, "
                "leading to inflated expected returns and misleading risk metrics.\n\n"
                "Consider increasing the number of assets or applying constraints for more robust results."
            )

        with st.spinner("Optimizing portfolio..."):
            try:
                optimal_weights, optimal_metrics = analyzer.optimize_portfolio(
                    train_start_date=train_start_date.strftime("%Y-%m-%d"),
                    train_end_date=train_end_date.strftime("%Y-%m-%d")
                )

                col1, col2 = st.columns([1, 1])

                with col1:
                    st.subheader(portfolio_label)
                    st.dataframe(
                        pd.DataFrame({'Ticker': optimal_weights.index, 'Weight': optimal_weights.values})
                          .style.format({'Weight': '{:.2%}'}),
                        width='stretch',
                        hide_index=True
                    )

                with col2:
                    st.subheader("Portfolio Metrics")
                    st.metric("Expected Annual Return", f"{optimal_metrics['Expected Annual Return']:.2%}")
                    st.metric("Expected Annual Volatility", f"{optimal_metrics['Expected Annual Volatility']:.2%}")
                    st.metric("Sharpe Ratio", f"{optimal_metrics['Sharpe Ratio']:.3f}")
                    st.metric("Confidence Level", confidence_level)

            except Exception as e:
                st.error(f"Portfolio optimization failed: {e}")
                return

        at_weights = None
        at_metrics = None
        if use_after_tax:
            st.header("🧾 After-Tax Portfolio Optimization")
            st.caption(
                "After-tax returns are estimated using simplified capital gains and dividend "
                "tax assumptions. See the README for methodology notes."
            )
            with st.spinner("Fetching dividend data and optimizing after-tax portfolio..."):
                try:
                    at_weights, at_metrics = analyzer.optimize_after_tax_portfolio(
                        capital_gains_tax=capital_gains_tax,
                        dividend_tax=dividend_tax,
                        turnover_penalty=turnover_penalty,
                        train_start_date=train_start_date.strftime("%Y-%m-%d"),
                        train_end_date=train_end_date.strftime("%Y-%m-%d"),
                        current_weights=optimal_weights,
                    )
                    at_return_df = analyzer.calculate_after_tax_returns(capital_gains_tax, dividend_tax)

                    col1, col2 = st.columns([1, 1])

                    with col1:
                        st.subheader("Weight Comparison")
                        st.dataframe(
                            pd.DataFrame({
                                "Ticker": optimal_weights.index,
                                "Pre-Tax Weight": optimal_weights.values,
                                "After-Tax Weight": at_weights.values,
                                "Δ Weight": at_weights.values - optimal_weights.values,
                            }).style.format({
                                "Pre-Tax Weight": "{:.2%}",
                                "After-Tax Weight": "{:.2%}",
                                "Δ Weight": "{:+.2%}",
                            }),
                            hide_index=True,
                            width='stretch',
                        )

                    with col2:
                        st.subheader("After-Tax Metrics")
                        pretax_return = optimal_metrics["Expected Annual Return"]
                        st.metric(
                            "After-Tax Annual Return",
                            f"{at_metrics['After-Tax Annual Return']:.2%}",
                            delta=f"{at_metrics['After-Tax Annual Return'] - pretax_return:.2%} vs pre-tax",
                        )
                        st.metric("Annual Volatility", f"{at_metrics['Annual Volatility']:.2%}")
                        st.metric("After-Tax Sharpe Ratio", f"{at_metrics['After-Tax Sharpe Ratio']:.3f}")
                        st.metric("Avg Tax Drag (per asset)", f"{at_return_df['Tax Drag'].mean():.2%}")
                        st.metric("Estimated Turnover Cost", f"{at_metrics['Estimated Turnover Cost']:.4%}")

                    st.subheader("After-Tax Return Breakdown by Stock")
                    st.dataframe(
                        at_return_df.style.format({
                            "Pre-Tax Return": "{:.2%}",
                            "Dividend Yield": "{:.2%}",
                            "Price Appreciation": "{:.2%}",
                            "After-Tax Return": "{:.2%}",
                            "Tax Drag": "{:.2%}",
                        }),
                        hide_index=True,
                        width='stretch',
                    )

                except Exception as e:
                    st.error(f"After-tax optimization failed: {e}")

        st.header("🧠 Risk Decomposition")
        st.markdown(
            "Risk contribution shows how much each position contributes to total portfolio variance. "
            "A stock can have a small weight but a large risk contribution if it is highly volatile "
            "or strongly correlated with other holdings."
        )

        with st.spinner("Computing risk decomposition..."):
            try:
                risk_contrib_df = analyzer.calculate_risk_contributions(
                    weights=optimal_weights,
                    train_start_date=train_start_date.strftime("%Y-%m-%d"),
                    train_end_date=train_end_date.strftime("%Y-%m-%d"),
                )

                if risk_contrib_df["Percent Risk Contribution"].max() > 0.40:
                    st.warning("Warning: Portfolio risk is highly concentrated in one asset.")

                col1, col2 = st.columns([1, 1])

                with col1:
                    st.subheader("Asset-Level Risk Contribution")
                    st.dataframe(
                        risk_contrib_df.style.format({
                            "Weight": "{:.2%}",
                            "Marginal Risk Contribution": "{:.6f}",
                            "Risk Contribution": "{:.6f}",
                            "Percent Risk Contribution": "{:.2%}",
                        }),
                        hide_index=True,
                        width='stretch',
                    )

                with col2:
                    st.subheader("Risk Contribution by Ticker")
                    fig_rc = go.Figure(go.Bar(
                        x=risk_contrib_df["Ticker"],
                        y=risk_contrib_df["Percent Risk Contribution"],
                        marker_color='#EF553B',
                        text=[f"{v:.1%}" for v in risk_contrib_df["Percent Risk Contribution"]],
                        textposition='outside',
                    ))
                    fig_rc.update_layout(
                        xaxis_title="Ticker",
                        yaxis_title="% of Portfolio Variance",
                        yaxis_tickformat=".0%",
                        height=400,
                        template='plotly_white',
                        margin=dict(t=30, b=40),
                    )
                    st.plotly_chart(fig_rc, width='stretch')

                st.markdown("---")
                st.markdown(
                    "Eigenfactor decomposition breaks portfolio risk into independent modes of movement. "
                    "The first eigenfactor often captures broad market or common-factor risk, while later "
                    "eigenfactors may reflect sector rotation or relative-value relationships."
                )
                st.caption(
                    "Note: Eigenvector signs are arbitrary. Positive and negative values indicate "
                    "direction within the factor, not 'good' or 'bad' exposure."
                )

                if len(stock_tickers) < 10:
                    st.info("Note: Eigenfactor analysis is more reliable with a larger asset universe.")

                eigen_summary, eigen_weights = analyzer.calculate_eigen_risk_decomposition(
                    weights=optimal_weights,
                    train_start_date=train_start_date.strftime("%Y-%m-%d"),
                    train_end_date=train_end_date.strftime("%Y-%m-%d"),
                )

                if eigen_summary["Percent Variance Explained"].iloc[0] > 0.60:
                    st.warning("Warning: Portfolio risk may be dominated by a single common factor.")

                col3, col4 = st.columns([1, 1])

                with col3:
                    st.subheader("Eigenfactor Summary")
                    st.dataframe(
                        eigen_summary.style.format({
                            "Eigenvalue": "{:.4f}",
                            "Percent Variance Explained": "{:.2%}",
                            "Cumulative Variance Explained": "{:.2%}",
                            "Portfolio Exposure": "{:.4f}",
                        }),
                        hide_index=True,
                        width='stretch',
                    )

                with col4:
                    st.subheader("Scree Plot: Variance Explained by Eigenfactor")
                    fig_scree = go.Figure(go.Bar(
                        x=eigen_summary["Eigenfactor"],
                        y=eigen_summary["Percent Variance Explained"],
                        marker_color='#636EFA',
                        text=[f"{v:.1%}" for v in eigen_summary["Percent Variance Explained"]],
                        textposition='outside',
                    ))
                    fig_scree.update_layout(
                        xaxis_title="Eigenfactor",
                        yaxis_title="% Variance Explained",
                        yaxis_tickformat=".0%",
                        xaxis_tickangle=-45,
                        height=400,
                        template='plotly_white',
                        margin=dict(t=30, b=80),
                    )
                    st.plotly_chart(fig_scree, width='stretch')

                with st.expander("Eigenportfolio Weights (Eigenvector Loadings)", expanded=False):
                    st.dataframe(
                        eigen_weights.style.format("{:.4f}").background_gradient(cmap='RdBu', axis=None),
                        width='stretch',
                    )

            except Exception as e:
                st.error(f"Risk decomposition failed: {e}")

        st.header("📈 Portfolio Performance vs. S&P 500")
        st.info(f"Testing on out-of-sample data from {test_start_date.strftime('%Y-%m-%d')} to {test_end_date.strftime('%Y-%m-%d')}")

        with st.spinner("Running backtest..."):
            try:
                portfolio_equity, spy_equity = analyzer.backtest_portfolio(
                    test_start_date=test_start_date.strftime("%Y-%m-%d"),
                    test_end_date=test_end_date.strftime("%Y-%m-%d")
                )

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=portfolio_equity.index,
                    y=portfolio_equity['Portfolio Value'],
                    mode='lines',
                    name='Optimal Portfolio',
                    line=dict(color='#00CC96', width=2)
                ))

                if spy_equity is not None:
                    fig.add_trace(go.Scatter(
                        x=spy_equity.index,
                        y=spy_equity['SPY Value'],
                        mode='lines',
                        name='S&P 500 (SPY)',
                        line=dict(color='#636EFA', width=2, dash='dash')
                    ))

                fig.update_layout(
                    title="Out-of-Sample Portfolio Performance",
                    xaxis_title="Date",
                    yaxis_title="Portfolio Value ($)",
                    hovermode='x unified',
                    height=500,
                    template='plotly_white'
                )
                st.plotly_chart(fig, width='stretch')

                final_val = portfolio_equity['Portfolio Value'].iloc[-1]
                port_ret = (final_val - initial_capital) / initial_capital

                if spy_equity is not None:
                    spy_ret = (spy_equity['SPY Value'].iloc[-1] - initial_capital) / initial_capital
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Portfolio Total Return", f"{port_ret:.2%}", delta=f"{port_ret:.2%}")
                    with col2:
                        st.metric("SPY Total Return", f"{spy_ret:.2%}", delta=f"{spy_ret:.2%}")
                    with col3:
                        outperformance = port_ret - spy_ret
                        st.metric("Outperformance vs SPY", f"{outperformance:.2%}", delta=f"{outperformance:.2%}")

                if use_after_tax and at_weights is not None:
                    st.subheader("Portfolio Comparison: Pre-Tax vs Estimated After-Tax vs SPY")
                    try:
                        at_equity = analyzer.backtest_after_tax_portfolio(
                            weights=at_weights,
                            capital_gains_tax=capital_gains_tax,
                            dividend_tax=dividend_tax,
                            turnover_penalty=turnover_penalty,
                            test_start_date=test_start_date.strftime("%Y-%m-%d"),
                            test_end_date=test_end_date.strftime("%Y-%m-%d"),
                            current_weights=optimal_weights,
                        )
                        fig2 = go.Figure()
                        fig2.add_trace(go.Scatter(
                            x=portfolio_equity.index,
                            y=portfolio_equity['Portfolio Value'],
                            mode='lines',
                            name='Pre-Tax Optimal',
                            line=dict(color='#00CC96', width=2)
                        ))
                        fig2.add_trace(go.Scatter(
                            x=at_equity.index,
                            y=at_equity['Estimated After-Tax Portfolio Value'],
                            mode='lines',
                            name='Estimated After-Tax Optimal',
                            line=dict(color='#FFA15A', width=2)
                        ))
                        if spy_equity is not None:
                            fig2.add_trace(go.Scatter(
                                x=spy_equity.index,
                                y=spy_equity['SPY Value'],
                                mode='lines',
                                name='S&P 500 (SPY)',
                                line=dict(color='#636EFA', width=2, dash='dash')
                            ))
                        fig2.update_layout(
                            title="Portfolio Comparison: Pre-Tax vs Estimated After-Tax vs SPY",
                            xaxis_title="Date",
                            yaxis_title="Portfolio Value ($)",
                            hovermode='x unified',
                            height=500,
                            template='plotly_white'
                        )
                        st.plotly_chart(fig2, width='stretch')
                        st.caption(
                            "After-tax performance is estimated using simplified tax-drag assumptions "
                            "and does not represent full tax-lot accounting."
                        )
                    except Exception as e:
                        st.warning(f"Could not render comparison chart: {e}")

            except Exception as e:
                st.error(f"Backtest failed: {e}")


if __name__ == "__main__":
    main()
