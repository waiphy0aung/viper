"""
VIPER — Prop Firm Signal Engine
Configuration

Sends Telegram signals for manual execution on Funding Pips.
Designed for XAUUSD (Gold) on 15m timeframe.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv
load_dotenv()

# =============================================================================
# PROP FIRM RULES — Funding Pips $5k
# =============================================================================
ACCOUNT_SIZE = 5000
DAILY_DD_LIMIT = 0.05           # 5% = $250
MAX_DD_LIMIT = 0.10             # 10% = $500 (static from initial balance)
PROFIT_TARGET_PHASE1 = 0.08     # 8% = $400
PROFIT_TARGET_PHASE2 = 0.05     # 5% = $250
EQUITY_FLOOR = ACCOUNT_SIZE * (1 - MAX_DD_LIMIT)  # $4,500

# =============================================================================
# INSTRUMENTS
# =============================================================================
# Multi-instrument: trade what's trending, skip what's chopping
# Gold via Bybit/OKX, NAS100 and EURUSD via yfinance
SYMBOLS = ["XAUUSD"]
SYMBOL_DISPLAY = {"XAUUSD": "XAUUSD", "NAS100": "NAS100", "EURUSD": "EURUSD"}

# Data source mapping
SYMBOL_DATA_SOURCE = {
    "XAUUSD": {"type": "ccxt", "exchange": "bybit", "ccxt_symbol": "XAU/USDT:USDT"},
    "NAS100": {"type": "yfinance", "ticker": "NQ=F"},
    "EURUSD": {"type": "yfinance", "ticker": "EURUSD=X"},
}

# Spread per instrument (points)
SPREAD = {
    "XAUUSD": 2.5,
    "NAS100": 1.5,    # typical index spread on prop firms
    "EURUSD": 0.00012, # ~1.2 pip spread on EURUSD
}

# Pip/point value per 0.01 lot for lot size calculation
POINT_VALUE = {
    "XAUUSD": 1.0,     # $1 per point per 0.01 lot
    "NAS100": 0.10,    # $0.10 per point per 0.01 lot (micro)
    "EURUSD": 1.0,     # $1 per pip per 0.1 lot (micro)
}

TIMEFRAME = "15m"
HTF_TIMEFRAME = "1h"
CANDLE_LIMIT = 200
HTF_CANDLE_LIMIT = 100

# =============================================================================
# REGIME DETECTION — tuned for Gold
# =============================================================================
ADX_PERIOD = 14                 # gold trends smoother than crypto, standard ADX works
CHOP_PERIOD = 14
ATR_FAST = 14
ATR_SLOW = 100

# Gold regime — tightened from backtests. ADX 20 was too loose, let in chop.
# ADX 28 only fires on real trends. Choppiness thresholds widen the MR zone.
REGIME_THRESHOLDS = {
    "XAUUSD": {"trending_adx": 25, "choppy_adx": 18, "trending_chop": 48, "choppy_chop": 58},
    "NAS100": {"trending_adx": 25, "choppy_adx": 18, "trending_chop": 48, "choppy_chop": 55},
    "EURUSD": {"trending_adx": 22, "choppy_adx": 16, "trending_chop": 50, "choppy_chop": 58},
}
REGIME_THRESHOLDS_DEFAULT = {"trending_adx": 25, "choppy_adx": 18, "trending_chop": 48, "choppy_chop": 58}

# Daily ATR filter — skip trading when daily volatility is below average
# This is the primary defense against Oct-Dec type chop periods
DAILY_ATR_FILTER = True
DAILY_ATR_PERIOD = 14
DAILY_ATR_MIN_RATIO = 0.9      # current ATR must be >= 90% of its 20-bar average
DAILY_ATR_MIN_ABSOLUTE = 8.0   # minimum ATR in points — below this, spread eats the edge

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

SUPERTREND_ATR_PERIOD = 10
SUPERTREND_MULTIPLIER = 3.0

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
# STRATEGY MODE — Gold does both trend and MR well
# =============================================================================
ASSET_STRATEGY_MODE = {
    "XAUUSD": ["trend", "mean_reversion"],
    "NAS100": ["trend"],              # NAS trends hard, MR is weak on indices
    "EURUSD": ["trend", "mean_reversion"],
}

# =============================================================================
# RISK MANAGEMENT — Prop firm style
# =============================================================================
# Base risk 0.75% — keeps max DD under 10% over 180 days.
# Dynamically throttled down further when approaching DD limits.
MAX_RISK_PER_TRADE = 0.0075     # 0.75% = $37.50 risk per trade
MAX_CONCURRENT_SIGNALS = 1      # one signal at a time — no stacking risk

# Gold Chandelier Exit — tighter than crypto
CHANDELIER_ATR_PERIOD = 14
CHANDELIER_MULTIPLIER = {
    "XAUUSD": 2.5,
    "NAS100": 3.0,              # indices are noisier
    "EURUSD": 2.5,
}
CHANDELIER_MULTIPLIER_DEFAULT = 2.5

# Spread simulation (points) — applied on entry
# Funding Pips raw spread XAUUSD: ~1.5-3.0 points during liquid hours
# Using 2.5 as conservative average (covers spread + slippage)
SPREAD_POINTS = 2.5

# R:R target for TP calculation
TARGET_RR = 2.0                 # 2.5 killed win rate. Keep 2.0, let ATR filter handle spread periods

# Risk:Reward minimum — don't signal unless R:R >= this
MIN_RISK_REWARD = 1.5

# Lot size reference for Funding Pips (gold)
# 1 lot = 100 oz, pip value ~$1 per 0.1 lot on gold
GOLD_POINT_VALUE = 1.0          # $1 per point per 0.01 lot

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
