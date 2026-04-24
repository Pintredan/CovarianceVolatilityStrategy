# CovarianceVolatilityStrategy
Built in a few hours for a job application, this project showcases how I approach quantitative analysis and financial tool development. It emphasizes clean structure, readability, and practical implementation. With more time, I would expand testing and add robustness, but this provides a concise snapshot of my development style.

## Investment Strategy

This project implements a data-driven portfolio construction approach, with an ETF-based factor exposure analysis to account for market variance. 

The strategy begins by downloading historical price data for a set of user-defined stocks and benchmark ETFs between 2020 and 2024. Prices are converted into daily returns, which are used to compute covariance and correlation matrices, allowing the model to understand how assets move relative to one another.

Each stock is then evaluated against a set of ETFs to determine its strongest relationship (via correlation) and sensitivity (via beta). This effectively maps individual equities to their underlying sector or market drivers, providing insight into factor exposure.

Using these inputs, the model calculates risk and return metrics including annualized return, volatility, and Sharpe ratio. Following these calculations, it constructs an optimal portfolio by maximizing the Sharpe ratio under realistic constraints under a long-only, fully invested portfolio.

Finally, the optimized portfolio is backtested over time and compared against the S&P 500 (SPY) between 2025-current, generating an equity curve to evaluate performance relative to the broader market.

While not a sophisticated model by any means, I hope it demonstrates key traits (both technical and character-wise) valuable to the team. I'm looking forward to connecting!

Note: This project was built under a tight time constraint, so I leveraged AI tools to accelerate development. I treated them as productivity aids rather than decision-makers, and focused on structuring the logic, validating outputs, and ensuring the overall design reflects how I approach building quantitative systems. With more time, I would further refine, extend the implementation, and restructure the database.
