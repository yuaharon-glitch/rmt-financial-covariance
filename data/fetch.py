"""yfinance download + parquet cache for S&P 100 daily log returns."""

import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

CACHE_DIR = Path(__file__).parent / "cache"
CACHE_MAX_AGE_DAYS = 7

DEFAULT_START = "2018-01-01"
DEFAULT_END = "2023-12-31"

# ~100 liquid S&P 500 names as of end-2023
SP100_TICKERS = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "META", "BRK-B", "JPM", "JNJ",
    "V", "PG", "UNH", "HD", "MA", "DIS", "BAC", "NVDA", "PYPL", "ADBE",
    "NFLX", "CMCSA", "VZ", "KO", "NKE", "MRK", "T", "PFE", "INTC",
    "CSCO", "WMT", "ABT", "CVX", "XOM", "LLY", "TMO", "ACN", "DHR",
    "AVGO", "TXN", "COST", "NEE", "MDT", "HON", "UNP", "QCOM", "LIN",
    "BMY", "AMGN", "PM", "IBM", "MMM", "RTX", "GS", "CAT", "SPGI",
    "SBUX", "BLK", "AXP", "DE", "GILD", "ISRG", "MO", "SYK", "CI",
    "ZTS", "CL", "USB", "ADP", "TJX", "GE", "PLD", "SCHW", "MU",
    "SO", "D", "NSC", "EMR", "ITW", "COF", "PNC", "WBA", "ETN",
    "AON", "MCO", "EW", "EXC", "APD", "ROP", "ILMN", "ICE", "FIS",
    "FISV", "WM", "ADI", "KLAC", "LRCX", "SHW", "HUM", "CCI", "REGN",
    "BIIB", "ECL", "CTSH",
]


def _cache_path(tickers, start, end):
    return CACHE_DIR / ("returns_%s_%s_%d.parquet" % (start, end, len(tickers)))


def _is_fresh(path):
    if not path.exists():
        return False
    age = datetime.datetime.now() - datetime.datetime.fromtimestamp(path.stat().st_mtime)
    return age.days < CACHE_MAX_AGE_DAYS


def fetch_returns(
    tickers=None,
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
) -> pd.DataFrame:
    """
    Download daily log returns and cache locally as parquet.

    Drops tickers with >5% missing data; forward-fills the rest.
    Returns (T, p) DataFrame indexed by date.  Serves from cache if the
    parquet is less than CACHE_MAX_AGE_DAYS old.
    """
    if tickers is None:
        tickers = SP100_TICKERS

    cache_file = _cache_path(tickers, start, end)
    if _is_fresh(cache_file):
        return pd.read_parquet(cache_file)

    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)

    prices = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw

    prices = prices.loc[:, prices.isna().mean() < 0.05].ffill()
    returns = np.log(prices / prices.shift(1)).dropna()

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    returns.to_parquet(cache_file)
    return returns
