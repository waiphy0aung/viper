"""
VIPER — Prop Firm Signal Engine v2
Configuration

Multi-instrument signal bot for Funding Pips $5k challenge.
XAUUSD + NAS100. Session-filtered, daily-bias, partial TP.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv
load_dotenv()

# =============================================================================
# PROP FIRM RULES — Funding Pips $5k
# =============================================================================
ACCOUNT_SIZE = 5000
DAILY_DD_LIMIT = 0.05
MAX_DD_LIMIT = 0.10
PROFIT_TARGET_PHASE1 = 0.08
PROFIT_TARGET_PHASE2 = 0.05
EQUITY_FLOOR = ACCOUNT_SIZE * (1 - MAX_DD_LIMIT)

# =============================================================================
# INSTRUMENTS
# =============================================================================
SYMBOLS = ["XAUUSD", "NAS100"]
SYMBOL_DISPLAY = {"XAUUSD": "XAUUSD", "NAS100": "NAS100"}

SYMBOL_DATA_SOURCE = {
    "XAUUSD": {"type": "ccxt", "exchange": "bybit", "ccxt_symbol": "XAU/USDT:USDT"},
    "NAS100": {"type": "yfinance", "ticker": "NQ=F"},
}

SPREAD = {
    "XAUUSD": 2.5,
    "NAS100": 1.5,
}

TIMEFRAME = "15m"
HTF_TIMEFRAME = "1h"
DAILY_TIMEFRAME = "1d"
CANDLE_LIMIT = 200
HTF_CANDLE_LIMIT = 100

# =============================================================================
# SESSION FILTER — only trade during liquid sessions
# =============================================================================
# Gold: London 07-11 UTC, NY 13-17 UTC
# NAS100: NY pre-market 12-13 UTC, NY session 13:30-20 UTC
SESSION_FILTER_ENABLED = True
SESSION_WINDOWS = {
    "XAUUSD": [(7, 11), (13, 17)],       # London + NY overlap
    "NAS100": [(13, 20)],                  # NY session
}

# =============================================================================
# DAILY BIAS — higher timeframe directional filter
# =============================================================================
# Check daily EMA 50/200. Only long when daily bullish, short when bearish.
# Per-asset: NAS100 respects daily trend. Gold trades intraday against daily.
DAILY_BIAS_ENABLED = True
DAILY_BIAS_SYMBOLS = ["NAS100"]  # only apply daily bias to these instruments
DAILY_EMA_FAST = 50
DAILY_EMA_SLOW = 200

# =============================================================================
# REGIME DETECTION
# =============================================================================
ADX_PERIOD = 14
CHOP_PERIOD = 14
ATR_FAST = 14
ATR_SLOW = 100

REGIME_THRESHOLDS = {
    "XAUUSD": {"trending_adx": 25, "choppy_adx": 18, "trending_chop": 48, "choppy_chop": 58},
    "NAS100": {"trending_adx": 22, "choppy_adx": 16, "trending_chop": 50, "choppy_chop": 58},
}
REGIME_THRESHOLDS_DEFAULT = {"trending_adx": 25, "choppy_adx": 18, "trending_chop": 48, "choppy_chop": 58}

DAILY_ATR_FILTER = True
DAILY_ATR_PERIOD = 14
DAILY_ATR_MIN_RATIO = 0.9
DAILY_ATR_MIN_ABSOLUTE = {"XAUUSD": 8.0, "NAS100": 30.0}

SUPERTREND_ATR_PERIOD = 10
SUPERTREND_MULTIPLIER = 3.0

# =============================================================================
# TREND MODULE
# =============================================================================
KC_PERIOD = 20
KC_ATR_PERIOD = 10
KC_MULTIPLIER = 1.5

BB_PERIOD = 20
BB_STD = 2.0

HMA_FAST = 9
HMA_SLOW = 21

TREND_VOLUME_MULTIPLIER = 1.3

# =============================================================================
# MEAN REVERSION MODULE
# =============================================================================
VWAP_PERIOD = 96
ZSCORE_ENTRY = 1.5
ZSCORE_STOP = 3.5
ZSCORE_EXIT = 0.0

MR_RSI_PERIOD = 9
MR_RSI_LONG = 30
MR_RSI_SHORT = 70

MR_VOLUME_MULTIPLIER = 1.5
MR_TIME_STOP_DEFAULT = 20

# =============================================================================
# STRATEGY MODE
# =============================================================================
ASSET_STRATEGY_MODE = {
    "XAUUSD": ["trend", "mean_reversion"],
    "NAS100": ["trend"],
}

# =============================================================================
# RISK MANAGEMENT
# =============================================================================
MAX_RISK_PER_TRADE = 0.0075
MAX_CONCURRENT_SIGNALS = 1

CHANDELIER_ATR_PERIOD = 14
CHANDELIER_MULTIPLIER = {
    "XAUUSD": 2.5,
    "NAS100": 3.5,      # NAS is noisier, needs wider trailing stop
}
CHANDELIER_MULTIPLIER_DEFAULT = 2.5

# =============================================================================
# TAKE PROFIT — Dynamic + Partial
# =============================================================================
# Primary TP: next 1H S/R level (dynamic)
# Fallback: fixed R:R if no clear S/R found
TARGET_RR = 2.0
MIN_RISK_REWARD = 1.5

# Partial TP: close 50% at 1:1, move SL to breakeven, trail rest
PARTIAL_TP_ENABLED = True
PARTIAL_TP_RATIO = 0.5         # close 50% of position
PARTIAL_TP_RR = 1.0            # at 1:1 R:R
MOVE_SL_TO_BE = True           # move SL to breakeven after partial TP

# S/R detection for dynamic TP
SR_LOOKBACK = 50               # 1H bars to scan for S/R levels
SR_TOUCH_COUNT = 2             # minimum touches to qualify as S/R

# =============================================================================
# SPREAD
# =============================================================================
SPREAD_POINTS = {"XAUUSD": 2.5, "NAS100": 1.5}

# =============================================================================
# TELEGRAM
# =============================================================================
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

# =============================================================================
# LOOP
# =============================================================================
LOOP_INTERVAL_SECONDS = 60
MAX_RETRIES_ON_ERROR = 5
RETRY_DELAY_SECONDS = 30

# =============================================================================
# LOGGING
# =============================================================================
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = "signal_bot.log"
