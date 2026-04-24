"""
Portfolio Analytics Web Application.

A Streamlit web app for portfolio construction, risk analysis,
and optimization using Modern Portfolio Theory.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from portfolio_engine import PortfolioAnalyzer


# Page configuration
st.set_page_config(
    page_title="Portfolio Analytics",
    page_icon="📊",
    layout="wide"
)


def main():
    """Main application logic."""

    # Title and description
    st.title("📊 Portfolio Analytics & Optimization")
    st.markdown("""
    This application performs quantitative portfolio analysis using Modern Portfolio Theory.
    Enter your stock tickers and ETF benchmarks to generate an optimized portfolio
    with maximum risk-adjusted returns.

    **Train/Test Split:** The optimization is trained on historical data (2020-2024) and tested
    on out-of-sample data (2025-2026) to avoid look-ahead bias and provide realistic performance metrics.

    **Disclaimer:** This tool is for educational and analytical purposes only.
    It is not financial advice. Always consult with a qualified financial advisor
    before making investment decisions.
    """)

    st.markdown("---")

    # Sidebar inputs
    with st.sidebar:
        st.header("Portfolio Configuration")

        # Stock tickers input
        stocks_input = st.text_area(
            "Stock Tickers (comma-separated)",
            value="AAPL, MSFT, NVDA, JPM, XOM",
            help="Enter stock ticker symbols separated by commas"
        )

        # ETF benchmarks input
        etf_input = st.text_area(
            "ETF Benchmarks (comma-separated)",
            value="SPY, XLK, XLE, XLF",
            help="Enter ETF ticker symbols for sector/market benchmarks"
        )

        # Training period
        st.subheader("Training Period")
        col1, col2 = st.columns(2)
        with col1:
            train_start_date = st.date_input(
                "Train Start",
                value=pd.to_datetime("2020-01-01"),
                help="Start date for training the optimization"
            )
        with col2:
            train_end_date = st.date_input(
                "Train End",
                value=pd.to_datetime("2024-12-31"),
                help="End date for training the optimization"
            )

        # Testing period
        st.subheader("Testing Period")
        col3, col4 = st.columns(2)
        with col3:
            test_start_date = st.date_input(
                "Test Start",
                value=pd.to_datetime("2025-01-01"),
                help="Start date for out-of-sample testing"
            )
        with col4:
            test_end_date = st.date_input(
                "Test End",
                value=pd.to_datetime("2026-04-23"),
                help="End date for out-of-sample testing"
            )

        # Portfolio parameters
        initial_capital = st.number_input(
            "Initial Portfolio Value ($)",
            min_value=1000,
            max_value=10000000,
            value=100000,
            step=10000
        )

        risk_free_rate = st.number_input(
            "Risk-Free Rate (annual)",
            min_value=0.0,
            max_value=0.20,
            value=0.04,
            step=0.01,
            format="%.4f"
        )

        # Tax assumptions
        st.subheader("Tax Assumptions")
        use_after_tax = st.checkbox("Enable After-Tax Optimization", value=False)
        capital_gains_tax = st.number_input(
            "Capital Gains Tax Rate",
            min_value=0.0,
            max_value=0.50,
            value=0.20,
            step=0.01,
            format="%.2f",
            disabled=not use_after_tax,
            help="Applied to price appreciation (long-term rate assumed)"
        )
        dividend_tax = st.number_input(
            "Dividend Tax Rate",
            min_value=0.0,
            max_value=0.50,
            value=0.15,
            step=0.01,
            format="%.2f",
            disabled=not use_after_tax,
            help="Applied to dividend income (qualified dividend rate assumed)"
        )
        turnover_penalty = st.number_input(
            "Turnover Penalty",
            min_value=0.0,
            max_value=0.05,
            value=0.0025,
            step=0.0025,
            format="%.4f",
            disabled=not use_after_tax,
            help="Transaction cost proxy per unit of portfolio turnover"
        )

        # Run analysis button
        run_analysis = st.button("🚀 Run Analysis", type="primary")

    # Main content area
    if run_analysis:
        # Parse ticker inputs
        stock_tickers = [s.strip().upper() for s in stocks_input.split(",") if s.strip()]
        etf_tickers = [e.strip().upper() for e in etf_input.split(",") if e.strip()]

        if not stock_tickers:
            st.error("Please enter at least one stock ticker")
            return

        if not etf_tickers:
            st.error("Please enter at least one ETF ticker")
            return

        # Initialize portfolio analyzer with full date range
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

        st.success(f"Successfully downloaded data for {len(stock_tickers)} stocks and {len(etf_tickers)} ETFs")

        # Display price preview
        st.header("📈 Historical Prices")
        with st.expander("View Price Data", expanded=False):
            st.dataframe(
                analyzer.prices.tail(10).style.format("${:.2f}"),
                width='stretch'
            )

        # Calculate and display stock metrics
        st.header("📊 Stock Performance Metrics")
        metrics_df = analyzer.calculate_annualized_metrics()

        st.dataframe(
            metrics_df.style.format({
                'Annual Return': '{:.2%}',
                'Annual Volatility': '{:.2%}',
                'Sharpe Ratio': '{:.3f}'
            }),
            width='stretch',
            hide_index=True
        )

        # Display covariance matrix
        st.header("🔗 Stock Covariance Matrix")
        cov_matrix = analyzer.calculate_covariance_matrix()

        st.dataframe(
            cov_matrix.style.format("{:.6f}").background_gradient(cmap='RdYlGn', axis=None),
            width='stretch'
        )

        # Display correlation matrix
        st.header("🔗 Stock Correlation Matrix")
        corr_matrix = analyzer.calculate_correlation_matrix()

        st.dataframe(
            corr_matrix.style.format("{:.3f}").background_gradient(cmap='coolwarm', axis=None),
            width='stretch'
        )

        # Display ETF relationships
        st.header("🎯 Stock-ETF Relationships")
        etf_relationships = analyzer.calculate_stock_etf_relationships()

        st.dataframe(
            etf_relationships.style.format({
                'Correlation': '{:.3f}',
                'Beta to ETF': '{:.3f}',
                'Beta to SPY': '{:.3f}',
                'Annual Volatility': '{:.2%}',
                'Annual Return': '{:.2%}'
            }),
            width='stretch',
            hide_index=True
        )

        # Portfolio optimization
        st.header("🎯 Optimal Portfolio")
        st.info(f"Training on data from {train_start_date.strftime('%Y-%m-%d')} to {train_end_date.strftime('%Y-%m-%d')}")

        with st.spinner("Optimizing portfolio..."):
            try:
                optimal_weights, optimal_metrics = analyzer.optimize_portfolio(
                    train_start_date=train_start_date.strftime("%Y-%m-%d"),
                    train_end_date=train_end_date.strftime("%Y-%m-%d")
                )

                # Display optimal weights
                col1, col2 = st.columns([1, 1])

                with col1:
                    st.subheader("Optimal Weights")
                    weights_df = pd.DataFrame({
                        'Ticker': optimal_weights.index,
                        'Weight': optimal_weights.values
                    })

                    st.dataframe(
                        weights_df.style.format({'Weight': '{:.2%}'}),
                        width='stretch',
                        hide_index=True
                    )

                with col2:
                    st.subheader("Portfolio Metrics")
                    st.metric(
                        "Expected Annual Return",
                        f"{optimal_metrics['Expected Annual Return']:.2%}"
                    )
                    st.metric(
                        "Expected Annual Volatility",
                        f"{optimal_metrics['Expected Annual Volatility']:.2%}"
                    )
                    st.metric(
                        "Sharpe Ratio",
                        f"{optimal_metrics['Sharpe Ratio']:.3f}"
                    )

            except Exception as e:
                st.error(f"Portfolio optimization failed: {e}")
                return

        # After-tax optimization (optional)
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
                    at_return_df = analyzer.calculate_after_tax_returns(
                        capital_gains_tax, dividend_tax
                    )

                    col1, col2 = st.columns([1, 1])

                    with col1:
                        st.subheader("Weight Comparison")
                        comparison_df = pd.DataFrame({
                            "Ticker": optimal_weights.index,
                            "Pre-Tax Weight": optimal_weights.values,
                            "After-Tax Weight": at_weights.values,
                            "Δ Weight": at_weights.values - optimal_weights.values,
                        })
                        st.dataframe(
                            comparison_df.style.format({
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
                        avg_tax_drag = at_return_df["Tax Drag"].mean()
                        st.metric(
                            "After-Tax Annual Return",
                            f"{at_metrics['After-Tax Annual Return']:.2%}",
                            delta=f"{at_metrics['After-Tax Annual Return'] - pretax_return:.2%} vs pre-tax",
                        )
                        st.metric(
                            "Annual Volatility",
                            f"{at_metrics['Annual Volatility']:.2%}"
                        )
                        st.metric(
                            "After-Tax Sharpe Ratio",
                            f"{at_metrics['After-Tax Sharpe Ratio']:.3f}"
                        )
                        st.metric(
                            "Avg Tax Drag (per asset)",
                            f"{avg_tax_drag:.2%}"
                        )
                        st.metric(
                            "Estimated Turnover Cost",
                            f"{at_metrics['Estimated Turnover Cost']:.4%}"
                        )

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

        # Backtest portfolio
        st.header("📈 Portfolio Performance vs. S&P 500")
        st.info(f"Testing on out-of-sample data from {test_start_date.strftime('%Y-%m-%d')} to {test_end_date.strftime('%Y-%m-%d')}")

        with st.spinner("Running backtest..."):
            try:
                portfolio_equity, spy_equity = analyzer.backtest_portfolio(
                    test_start_date=test_start_date.strftime("%Y-%m-%d"),
                    test_end_date=test_end_date.strftime("%Y-%m-%d")
                )

                # Create plotly figure
                fig = go.Figure()

                # Add portfolio line
                fig.add_trace(go.Scatter(
                    x=portfolio_equity.index,
                    y=portfolio_equity['Portfolio Value'],
                    mode='lines',
                    name='Optimal Portfolio',
                    line=dict(color='#00CC96', width=2)
                ))

                # Add SPY benchmark line if available
                if spy_equity is not None:
                    fig.add_trace(go.Scatter(
                        x=spy_equity.index,
                        y=spy_equity['SPY Value'],
                        mode='lines',
                        name='S&P 500 (SPY)',
                        line=dict(color='#636EFA', width=2, dash='dash')
                    ))

                # Update layout
                fig.update_layout(
                    title="Out-of-Sample Portfolio Performance",
                    xaxis_title="Date",
                    yaxis_title="Portfolio Value ($)",
                    hovermode='x unified',
                    height=500,
                    template='plotly_white'
                )

                st.plotly_chart(fig, width='stretch')

                # Calculate final performance
                initial_value = initial_capital
                final_portfolio_value = portfolio_equity['Portfolio Value'].iloc[-1]
                portfolio_return = (final_portfolio_value - initial_value) / initial_value

                if spy_equity is not None:
                    final_spy_value = spy_equity['SPY Value'].iloc[-1]
                    spy_return = (final_spy_value - initial_value) / initial_value

                    # Display comparison metrics
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric(
                            "Portfolio Total Return",
                            f"{portfolio_return:.2%}",
                            delta=f"{portfolio_return:.2%}"
                        )

                    with col2:
                        st.metric(
                            "SPY Total Return",
                            f"{spy_return:.2%}",
                            delta=f"{spy_return:.2%}"
                        )

                    with col3:
                        outperformance = portfolio_return - spy_return
                        st.metric(
                            "Outperformance vs SPY",
                            f"{outperformance:.2%}",
                            delta=f"{outperformance:.2%}"
                        )

                # Optional: pre-tax vs after-tax vs SPY comparison chart
                if use_after_tax and at_weights is not None:
                    st.subheader("Portfolio Comparison: Pre-Tax vs After-Tax vs SPY")
                    try:
                        at_equity, _ = analyzer.backtest_portfolio(
                            weights=at_weights,
                            test_start_date=test_start_date.strftime("%Y-%m-%d"),
                            test_end_date=test_end_date.strftime("%Y-%m-%d"),
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
                            y=at_equity['Portfolio Value'],
                            mode='lines',
                            name='After-Tax Optimal',
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
                            title="Pre-Tax vs After-Tax Portfolio vs S&P 500",
                            xaxis_title="Date",
                            yaxis_title="Portfolio Value ($)",
                            hovermode='x unified',
                            height=500,
                            template='plotly_white'
                        )
                        st.plotly_chart(fig2, width='stretch')
                    except Exception as e:
                        st.warning(f"Could not render comparison chart: {e}")

            except Exception as e:
                st.error(f"Backtest failed: {e}")


if __name__ == "__main__":
    main()
