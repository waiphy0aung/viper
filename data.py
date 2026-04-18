"""
Market data layer — multi-source.
XAUUSD via Bybit, NAS100 via yfinance.
"""

from __future__ import annotations

import logging

import ccxt
import pandas as pd
import yfinance as yf

import config

logger = logging.getLogger(__name__)

CCXT_SYMBOLS = {"XAUUSD": "XAU/USDT:USDT"}
YF_TICKERS = {"NAS100": "NQ=F"}


class MarketData:
    def __init__(self):
        self.ccxt_client = ccxt.bybit({"enableRateLimit": True})
        self._yf_cache: dict[str, pd.DataFrame] = {}
        logger.info("Market data initialized (Bybit + yfinance)")

    def fetch_15m(self, symbol: str) -> pd.DataFrame:
        if symbol in CCXT_SYMBOLS:
            return self._fetch_ccxt(symbol, config.TIMEFRAME, config.CANDLE_LIMIT)
        elif symbol in YF_TICKERS:
            return self._fetch_yf(symbol, "15m", config.CANDLE_LIMIT)
        raise ValueError(f"Unknown symbol: {symbol}")

    def fetch_1h(self, symbol: str) -> pd.DataFrame:
        if symbol in CCXT_SYMBOLS:
            return self._fetch_ccxt(symbol, config.HTF_TIMEFRAME, config.HTF_CANDLE_LIMIT)
        elif symbol in YF_TICKERS:
            return self._fetch_yf(symbol, "1h", config.HTF_CANDLE_LIMIT)
        raise ValueError(f"Unknown symbol: {symbol}")

    def fetch_daily(self, symbol: str, limit: int = 250) -> pd.DataFrame:
        """Fetch daily candles for daily bias (EMA 50/200)."""
        if symbol in CCXT_SYMBOLS:
            return self._fetch_ccxt(symbol, "1d", limit)
        elif symbol in YF_TICKERS:
            return self._fetch_yf(symbol, "1d", limit)
        raise ValueError(f"Unknown symbol: {symbol}")

    def get_price(self, symbol: str) -> float:
        if symbol in CCXT_SYMBOLS:
            ticker = self.ccxt_client.fetch_ticker(CCXT_SYMBOLS[symbol])
            return float(ticker["last"])
        elif symbol in YF_TICKERS:
            df = self._fetch_yf(symbol, "15m", 5)
            return float(df["close"].iloc[-1])
        raise ValueError(f"Unknown symbol: {symbol}")

    def _fetch_ccxt(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
        raw = self.ccxt_client.fetch_ohlcv(CCXT_SYMBOLS[symbol], timeframe=timeframe, limit=limit)
        df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df.set_index("timestamp", inplace=True)
        return df

    def _fetch_yf(self, symbol: str, interval: str, limit: int) -> pd.DataFrame:
        ticker = YF_TICKERS[symbol]
        period = "60d" if interval in ("15m", "1h") else "2y"
        data = yf.download(ticker, period=period, interval=interval, progress=False)
        data.columns = [c[0].lower() for c in data.columns]
        if data.index.tz is None:
            data.index = data.index.tz_localize("UTC")
        return data.tail(limit)
