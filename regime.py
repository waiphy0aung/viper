"""
Market regime detection for VIPER v2.

Three layers:
1. ADX + Choppiness → TRENDING / RANGING / UNCERTAIN
2. 1H Supertrend → directional bias
3. Daily EMA 50/200 → weekly bias (only trade with the daily trend)
"""

from __future__ import annotations

import logging
from enum import Enum

import pandas as pd

import config
from indicators import adx, choppiness_index, atr, supertrend, ema

logger = logging.getLogger(__name__)


class Regime(Enum):
    TRENDING = "TRENDING"
    RANGING = "RANGING"
    UNCERTAIN = "UNCERTAIN"


class RegimeState:
    def __init__(self, regime: Regime, adx_value: float, chop_value: float,
                 atr_ratio: float, htf_direction: int, daily_bias: int,
                 adx_slope: float):
        self.regime = regime
        self.adx_value = adx_value
        self.chop_value = chop_value
        self.atr_ratio = atr_ratio
        self.htf_direction = htf_direction  # 1=bull, -1=bear
        self.daily_bias = daily_bias        # 1=bull, -1=bear, 0=neutral
        self.adx_slope = adx_slope

    def __repr__(self):
        htf = "BULL" if self.htf_direction == 1 else "BEAR"
        bias = "BULL" if self.daily_bias == 1 else "BEAR" if self.daily_bias == -1 else "NEUTRAL"
        return (
            f"Regime({self.regime.value} | ADX={self.adx_value:.1f} "
            f"CHOP={self.chop_value:.1f} | 1H={htf} Daily={bias})"
        )


def detect_regime(df_15m: pd.DataFrame, df_1h: pd.DataFrame,
                  symbol: str, df_daily: pd.DataFrame | None = None) -> RegimeState:
    if len(df_15m) < config.ATR_SLOW + 10:
        return RegimeState(Regime.UNCERTAIN, 0, 50, 1.0, 1, 0, 0)

    # 15m indicators
    adx_data = adx(df_15m["high"], df_15m["low"], df_15m["close"], config.ADX_PERIOD)
    chop = choppiness_index(df_15m["high"], df_15m["low"], df_15m["close"], config.CHOP_PERIOD)
    atr_fast = atr(df_15m["high"], df_15m["low"], df_15m["close"], config.ATR_FAST)
    atr_slow = atr(df_15m["high"], df_15m["low"], df_15m["close"], config.ATR_SLOW)

    adx_now = adx_data["adx"].iloc[-1]
    adx_prev = adx_data["adx"].iloc[-4]
    adx_slope = adx_now - adx_prev
    chop_now = chop.iloc[-1]
    atr_ratio = atr_fast.iloc[-1] / atr_slow.iloc[-1] if atr_slow.iloc[-1] > 0 else 1.0

    # 1H Supertrend
    htf_direction = 1
    if len(df_1h) >= config.SUPERTREND_ATR_PERIOD + 5:
        st = supertrend(df_1h["high"], df_1h["low"], df_1h["close"],
                        config.SUPERTREND_ATR_PERIOD, config.SUPERTREND_MULTIPLIER)
        htf_direction = int(st["direction"].iloc[-1])

    # Daily bias — EMA 50/200
    daily_bias = 0
    if config.DAILY_BIAS_ENABLED and df_daily is not None and len(df_daily) >= config.DAILY_EMA_SLOW + 5:
        ema_fast = ema(df_daily["close"], config.DAILY_EMA_FAST)
        ema_slow_val = ema(df_daily["close"], config.DAILY_EMA_SLOW)
        if ema_fast.iloc[-1] > ema_slow_val.iloc[-1]:
            daily_bias = 1
        elif ema_fast.iloc[-1] < ema_slow_val.iloc[-1]:
            daily_bias = -1

    # Classify regime
    thresholds = config.REGIME_THRESHOLDS.get(symbol, config.REGIME_THRESHOLDS_DEFAULT)
    is_trending = adx_now >= thresholds["trending_adx"] and chop_now <= thresholds["trending_chop"]
    is_choppy = adx_now <= thresholds["choppy_adx"] or chop_now >= thresholds["choppy_chop"]

    if is_trending:
        regime = Regime.TRENDING
    elif is_choppy:
        regime = Regime.RANGING
    else:
        regime = Regime.UNCERTAIN

    state = RegimeState(regime, adx_now, chop_now, atr_ratio, htf_direction,
                        daily_bias, adx_slope)
    logger.info(f"{config.SYMBOL_DISPLAY.get(symbol, symbol)} | {state}")
    return state
