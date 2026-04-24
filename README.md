# CovarianceVolatilityStrategy

Built in a few hours for a job application, this project showcases how I approach quantitative analysis and financial tool development. It emphasizes clean structure, readability, and translating analytical ideas into practical, usable tools. With more time, I would extend testing, improve data robustness, and expand the modeling framework.

## Investment Strategy

This project implements a data-driven portfolio construction approach combining covariance-based risk modeling with ETF-based factor exposure analysis.

The strategy begins by downloading historical price data for a set of user-defined equities and benchmark ETFs. Prices are converted into daily returns, which are used to compute covariance and correlation matrices, providing insight into how assets co-move and contribute to portfolio risk.

Each stock is then evaluated relative to a set of ETFs to determine its strongest relationship (via correlation) and sensitivity (via beta). This effectively maps individual equities to their underlying sector or market drivers, allowing for a more structured understanding of factor exposure.

Using these inputs, the model calculates key metrics including annualized return, volatility, and Sharpe ratio. It then constructs an optimal portfolio by maximizing the Sharpe ratio under realistic constraints (long-only, fully invested, and position size limits), balancing return and risk efficiently.

The resulting portfolio is backtested over time and compared against the S&P 500 (SPY), generating an equity curve to evaluate performance relative to the broader market.

## Notes

This project was developed under tight time constraints, so I leveraged AI tools to accelerate parts of the implementation. I used them as productivity aids while maintaining ownership over the system design, quantitative logic, and validation of results. In a full development setting, I would further refine the architecture, expand testing, and enhance data handling.
