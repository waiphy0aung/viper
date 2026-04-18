"""
Historical data fetcher for backtesting.

Fetches and saves long-history data:
- XAUUSD: from OKX (180+ days of 15m)
- NAS100: from yfinance (60 days 15m, 2 years daily/1h)

Usage:
    python fetch_data.py
"""

from __future__ import annotations

import os
import time

import ccxt
import pandas as pd
import yfinance as yf


def fetch_xauusd_15m(days: int = 180) -> pd.DataFrame:
    """Fetch XAUUSD 15m from OKX."""
    csv = "xauusd_15m.csv"
    if os.path.exists(csv):
        df = pd.read_csv(csv, index_col=0, parse_dates=True)
        print(f"  XAUUSD 15m: loaded {len(df)} bars from cache")
        return df

    ex = ccxt.okx({"enableRateLimit": True})
    symbol = "XAU/USDT:USDT"
    all_candles = []
    total = (days * 24 * 60) // 15
    since = ex.parse8601(
        (pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=days)).isoformat()
    )

    print(f"  Fetching XAUUSD 15m from OKX ({days} days)...")
    while len(all_candles) < total:
        batch = ex.fetch_ohlcv(symbol, "15m", since=since, limit=300)
        if not batch:
            break
        all_candles.extend(batch)
        since = batch[-1][0] + 1
        time.sleep(ex.rateLimit / 1000 + 0.1)
        if len(all_candles) % 3000 == 0:
            print(f"    {len(all_candles)} candles...")

    df = pd.DataFrame(all_candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.drop_duplicates(subset="timestamp")
    df.set_index("timestamp", inplace=True)
    df.to_csv(csv)
    print(f"  XAUUSD 15m: {len(df)} bars saved")
    return df


def fetch_nas100_15m() -> pd.DataFrame:
    """Fetch NAS100 15m from yfinance (max 60 days)."""
    csv = "nas100_15m.csv"
    if os.path.exists(csv):
        df = pd.read_csv(csv, index_col=0, parse_dates=True)
        print(f"  NAS100 15m: loaded {len(df)} bars from cache")
        return df

    print("  Fetching NAS100 15m from yfinance (60 days)...")
    data = yf.download("NQ=F", period="60d", interval="15m", progress=False)
    data.columns = [c[0].lower() for c in data.columns]
    if data.index.tz is None:
        data.index = data.index.tz_localize("UTC")
    data.to_csv(csv)
    print(f"  NAS100 15m: {len(data)} bars saved")
    return data


def fetch_nas100_1h() -> pd.DataFrame:
    """Fetch NAS100 1h from yfinance (up to 730 days)."""
    csv = "nas100_1h.csv"
    if os.path.exists(csv):
        df = pd.read_csv(csv, index_col=0, parse_dates=True)
        print(f"  NAS100 1h: loaded {len(df)} bars from cache")
        return df

    print("  Fetching NAS100 1h from yfinance (730 days)...")
    data = yf.download("NQ=F", period="730d", interval="1h", progress=False)
    data.columns = [c[0].lower() for c in data.columns]
    if data.index.tz is None:
        data.index = data.index.tz_localize("UTC")
    data.to_csv(csv)
    print(f"  NAS100 1h: {len(data)} bars saved")
    return data


def fetch_daily(ticker: str, name: str) -> pd.DataFrame:
    """Fetch daily data for daily bias (2 years)."""
    csv = f"{name.lower()}_daily.csv"
    if os.path.exists(csv):
        df = pd.read_csv(csv, index_col=0, parse_dates=True)
        print(f"  {name} daily: loaded {len(df)} bars from cache")
        return df

    print(f"  Fetching {name} daily from yfinance (2 years)...")
    data = yf.download(ticker, period="2y", interval="1d", progress=False)
    data.columns = [c[0].lower() for c in data.columns]
    if data.index.tz is None:
        data.index = data.index.tz_localize("UTC")
    data.to_csv(csv)
    print(f"  {name} daily: {len(data)} bars saved")
    return data


def main():
    print("\n=== Fetching Historical Data ===\n")
    fetch_xauusd_15m(180)
    fetch_nas100_15m()
    fetch_nas100_1h()
    fetch_daily("GC=F", "XAUUSD")
    fetch_daily("NQ=F", "NAS100")
    print("\nDone. All data saved to CSV.\n")


if __name__ == "__main__":
    main()
