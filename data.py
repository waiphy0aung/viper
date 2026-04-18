"""
Market data layer — fetches XAUUSD via Bybit public API.
"""

from __future__ import annotations

import logging

import ccxt
import pandas as pd

import config

logger = logging.getLogger(__name__)

# Map our symbol names to CCXT symbols
CCXT_SYMBOLS = {
    "XAUUSD": "XAU/USDT:USDT",
}


class MarketData:
    def __init__(self):
        self.client = ccxt.bybit({"enableRateLimit": True})
        logger.info("Market data initialized (Bybit public API)")

    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
        ccxt_sym = CCXT_SYMBOLS.get(symbol, symbol)
        raw = self.client.fetch_ohlcv(ccxt_sym, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df.set_index("timestamp", inplace=True)
        return df

    def fetch_15m(self, symbol: str) -> pd.DataFrame:
        return self.fetch_ohlcv(symbol, config.TIMEFRAME, config.CANDLE_LIMIT)

    def fetch_1h(self, symbol: str) -> pd.DataFrame:
        return self.fetch_ohlcv(symbol, config.HTF_TIMEFRAME, config.HTF_CANDLE_LIMIT)

    def get_price(self, symbol: str) -> float:
        ccxt_sym = CCXT_SYMBOLS.get(symbol, symbol)
        ticker = self.client.fetch_ticker(ccxt_sym)
        return float(ticker["last"])
