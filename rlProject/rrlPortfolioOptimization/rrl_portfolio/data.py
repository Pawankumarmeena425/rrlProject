from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf

ASSET_UNIVERSE = ["SPY", "IWD", "IWC", "DEM", "CLY"]
FALLBACK_TICKER_MAP = {
    "CLY": "VCLT"  # Proxy for iShares 10+ Year Credit Bond ETF (delisted CLY)
}


def download_etf_prices(tickers: list[str], start: str, end: str, interval: str = "1wk") -> pd.DataFrame:
    """Download weekly adjusted close prices for ETFs from Yahoo Finance.

    This function tries each ticker individually to avoid database lock
    errors, and maps delisted tickers to a proxy instrument when needed.
    """
    prices_by_ticker: dict[str, pd.Series] = {}
    for ticker in tickers:
        source_ticker = FALLBACK_TICKER_MAP.get(ticker, ticker)
        last_exception = None
        for attempt in range(3):
            try:
                raw = yf.download(
                    source_ticker,
                    start=start,
                    end=end,
                    interval=interval,
                    progress=False,
                    auto_adjust=True,
                )
                if raw is None or raw.empty:
                    continue
                if isinstance(raw.columns, pd.MultiIndex):
                    if "Close" in raw.columns.levels[0]:
                        data = raw["Close"]
                    elif "Adj Close" in raw.columns.levels[0]:
                        data = raw["Adj Close"]
                    else:
                        data = raw.iloc[:, raw.columns.get_level_values(1) == "Close"]
                else:
                    data = raw
                if source_ticker in data.columns:
                    series = data[source_ticker]
                else:
                    series = data.iloc[:, 0]
                series = series.dropna()
                if series.empty:
                    continue
                prices_by_ticker[ticker] = series.rename(ticker)
                break
            except Exception as exc:
                last_exception = exc
        else:
            raise ValueError(
                f"Failed to download prices for {ticker} (mapped to {source_ticker}). "
                f"Last error: {last_exception}"
            )
    prices = pd.concat(prices_by_ticker.values(), axis=1)
    prices = prices.sort_index()
    prices = prices.ffill().bfill()
    prices = prices.loc[~prices.isna().all(axis=1)]
    missing = [ticker for ticker in tickers if ticker not in prices.columns]
    if missing:
        raise ValueError(f"Missing tickers in downloaded data: {missing}")
    return prices


def prepare_weekly_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Convert aligned weekly prices to log returns."""
    returns = prices.sort_index().astype(float)
    returns = returns.replace([pd.NA], pd.NA)
    returns = returns.ffill().bfill()
    returns = returns.loc[~returns.isna().all(axis=1)]
    returns = returns.pct_change().apply(lambda x: np.log1p(x)).dropna(how="all")
    returns.index = pd.to_datetime(returns.index)
    return returns


def download_etf_returns(tickers: list[str], start: str, end: str, interval: str = "1wk") -> pd.DataFrame:
    prices = download_etf_prices(tickers, start, end, interval=interval)
    returns = prepare_weekly_returns(prices)
    return returns
