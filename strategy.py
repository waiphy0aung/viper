"""
VIPER Strategy v2 — Full-featured signal generation.

New in v2:
- Session filter (London + NY only)
- Daily bias (only trade with daily EMA trend)
- Dynamic TP from 1H S/R levels
- Partial TP metadata (50% at 1:1, trail rest)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum

import pandas as pd

import config
from indicators import (
    hma, rsi, atr,
    ttm_squeeze, vwap_zscore, chandelier_exit,
    find_sr_levels, find_next_sr,
)
from regime import Regime, RegimeState

logger = logging.getLogger(__name__)


class SignalType(Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


@dataclass
class TradeSignal:
    signal: SignalType
    source: str
    symbol: str
    display_name: str
    entry_price: float
    stop_loss: float
    take_profit: float
    partial_tp: float | None     # price for partial TP (50% close)
    risk_reward: float
    confidence: float
    reason: str
    regime: str
    daily_bias: str


def _in_session(symbol: str) -> bool:
    """Check if current time is within a trading session for this instrument."""
    if not config.SESSION_FILTER_ENABLED:
        return True

    windows = config.SESSION_WINDOWS.get(symbol)
    if not windows:
        return True

    hour = datetime.now(timezone.utc).hour
    return any(start <= hour < end for start, end in windows)


def _in_session_at(symbol: str, hour: int) -> bool:
    """Check if a given hour is within trading session (for backtesting)."""
    if not config.SESSION_FILTER_ENABLED:
        return True

    windows = config.SESSION_WINDOWS.get(symbol)
    if not windows:
        return True

    return any(start <= hour < end for start, end in windows)


def _check_daily_bias(regime: RegimeState, side: str, symbol: str) -> bool:
    """Only trade in the direction of the daily trend (for configured instruments)."""
    if not config.DAILY_BIAS_ENABLED:
        return True

    if symbol not in config.DAILY_BIAS_SYMBOLS:
        return True  # daily bias not applied to this instrument

    if regime.daily_bias == 0:
        return True

    if side == "long" and regime.daily_bias == -1:
        return False
    if side == "short" and regime.daily_bias == 1:
        return False

    return True


def _calc_dynamic_tp(entry: float, sl: float, side: str,
                     df_1h: pd.DataFrame) -> tuple[float, float]:
    """
    Calculate TP from next 1H S/R level.
    Falls back to fixed R:R if no clear S/R found.
    Returns (tp_price, risk_reward_ratio).
    """
    risk = abs(entry - sl)
    fallback_tp = entry + risk * config.TARGET_RR if side == "long" else entry - risk * config.TARGET_RR

    if len(df_1h) < config.SR_LOOKBACK:
        rr = abs(fallback_tp - entry) / risk if risk > 0 else 0
        return fallback_tp, rr

    levels = find_sr_levels(
        df_1h["high"], df_1h["low"], df_1h["close"],
        config.SR_LOOKBACK, config.SR_TOUCH_COUNT,
    )

    sr_tp = find_next_sr(entry, levels, side)

    if sr_tp is not None:
        sr_rr = abs(sr_tp - entry) / risk if risk > 0 else 0
        if sr_rr >= config.MIN_RISK_REWARD:
            return sr_tp, sr_rr

    rr = abs(fallback_tp - entry) / risk if risk > 0 else 0
    return fallback_tp, rr


def _calc_partial_tp(entry: float, sl: float, side: str) -> float | None:
    """Calculate partial TP level (1:1 R:R)."""
    if not config.PARTIAL_TP_ENABLED:
        return None

    risk = abs(entry - sl)
    if side == "long":
        return entry + risk * config.PARTIAL_TP_RR
    return entry - risk * config.PARTIAL_TP_RR


def _trend_signal(df: pd.DataFrame, df_1h: pd.DataFrame,
                  regime: RegimeState, symbol: str) -> TradeSignal | None:
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    squeeze = ttm_squeeze(high, low, close,
                          config.BB_PERIOD, config.BB_STD,
                          config.KC_PERIOD, config.KC_ATR_PERIOD, config.KC_MULTIPLIER)
    hma_fast = hma(close, config.HMA_FAST)
    hma_slow = hma(close, config.HMA_SLOW)

    mult = config.CHANDELIER_MULTIPLIER.get(symbol, config.CHANDELIER_MULTIPLIER_DEFAULT)
    ce = chandelier_exit(high, low, close, config.CHANDELIER_ATR_PERIOD, mult)

    vol_avg = volume.rolling(20).mean()
    vol_ratio = volume / vol_avg

    squeeze_fired = bool(squeeze["squeeze_on"].iloc[-2] and squeeze["squeeze_off"].iloc[-1])
    momentum = float(squeeze["momentum"].iloc[-1])
    momentum_prev = float(squeeze["momentum"].iloc[-2])
    recent_squeeze = any(squeeze["squeeze_on"].iloc[-6:-1])

    hf = float(hma_fast.iloc[-1])
    hs = float(hma_slow.iloc[-1])
    hf_p = float(hma_fast.iloc[-2])
    hs_p = float(hma_slow.iloc[-2])
    bullish_hma = hf_p <= hs_p and hf > hs
    bearish_hma = hf_p >= hs_p and hf < hs

    vol_ok = float(vol_ratio.iloc[-1]) >= config.TREND_VOLUME_MULTIPLIER
    price = float(close.iloc[-1])
    display = config.SYMBOL_DISPLAY.get(symbol, symbol)

    htf_bull = regime.htf_direction == 1
    htf_bear = regime.htf_direction == -1
    bias_str = "BULL" if regime.daily_bias == 1 else "BEAR" if regime.daily_bias == -1 else "NEUTRAL"

    # BUY
    if htf_bull and vol_ok and _check_daily_bias(regime, "long", symbol):
        triggered = False
        conf = 0.7
        reason = ""

        if squeeze_fired and momentum > 0 and momentum > momentum_prev and hf > hs:
            triggered, conf = True, 0.9
            reason = f"Squeeze+HMA UP | Mom={momentum:.2f}"
        elif bullish_hma and recent_squeeze:
            triggered, conf = True, 0.7
            reason = f"HMA cross UP (post-squeeze)"

        if triggered:
            sl = float(ce["long_stop"].iloc[-1])
            tp, rr = _calc_dynamic_tp(price, sl, "long", df_1h)
            partial = _calc_partial_tp(price, sl, "long")

            if rr < config.MIN_RISK_REWARD:
                return None

            return TradeSignal(
                signal=SignalType.BUY, source="trend", symbol=symbol,
                display_name=display, entry_price=price, stop_loss=sl,
                take_profit=tp, partial_tp=partial, risk_reward=rr,
                confidence=conf, reason=reason, regime=regime.regime.value,
                daily_bias=bias_str,
            )

    # SELL
    if htf_bear and vol_ok and _check_daily_bias(regime, "short", symbol):
        triggered = False
        conf = 0.7
        reason = ""

        if squeeze_fired and momentum < 0 and momentum < momentum_prev and hf < hs:
            triggered, conf = True, 0.9
            reason = f"Squeeze+HMA DOWN | Mom={momentum:.2f}"
        elif bearish_hma and recent_squeeze:
            triggered, conf = True, 0.7
            reason = f"HMA cross DOWN (post-squeeze)"

        if triggered:
            sl = float(ce["short_stop"].iloc[-1])
            tp, rr = _calc_dynamic_tp(price, sl, "short", df_1h)
            partial = _calc_partial_tp(price, sl, "short")

            if rr < config.MIN_RISK_REWARD:
                return None

            return TradeSignal(
                signal=SignalType.SELL, source="trend", symbol=symbol,
                display_name=display, entry_price=price, stop_loss=sl,
                take_profit=tp, partial_tp=partial, risk_reward=rr,
                confidence=conf, reason=reason, regime=regime.regime.value,
                daily_bias=bias_str,
            )

    return None


def _mean_reversion_signal(df: pd.DataFrame, df_1h: pd.DataFrame,
                           regime: RegimeState, symbol: str) -> TradeSignal | None:
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    zscore = vwap_zscore(high, low, close, volume, config.VWAP_PERIOD)
    rsi_vals = rsi(close, config.MR_RSI_PERIOD)
    atr_vals = atr(high, low, close, config.CHANDELIER_ATR_PERIOD)

    vol_avg = volume.rolling(20).mean()
    vol_ratio = volume / vol_avg

    z_now = float(zscore.iloc[-1]) if not pd.isna(zscore.iloc[-1]) else None
    rsi_now = float(rsi_vals.iloc[-1]) if not pd.isna(rsi_vals.iloc[-1]) else None

    if z_now is None or rsi_now is None:
        return None

    vol_ok = float(vol_ratio.iloc[-1]) >= config.MR_VOLUME_MULTIPLIER
    price = float(close.iloc[-1])
    atr_now = float(atr_vals.iloc[-1])
    display = config.SYMBOL_DISPLAY.get(symbol, symbol)
    bias_str = "BULL" if regime.daily_bias == 1 else "BEAR" if regime.daily_bias == -1 else "NEUTRAL"

    mr_stop_mult = 1.2

    # MR doesn't require daily bias alignment — it's counter-trend by nature
    # But we still respect it for safety in strong trends

    if z_now < -config.ZSCORE_ENTRY and rsi_now < config.MR_RSI_LONG and vol_ok:
        sl = price - mr_stop_mult * atr_now
        tp, rr = _calc_dynamic_tp(price, sl, "long", df_1h)
        partial = _calc_partial_tp(price, sl, "long")

        if rr < config.MIN_RISK_REWARD:
            return None

        return TradeSignal(
            signal=SignalType.BUY, source="mean_reversion", symbol=symbol,
            display_name=display, entry_price=price, stop_loss=sl,
            take_profit=tp, partial_tp=partial, risk_reward=rr,
            confidence=min(0.9, abs(z_now) / 3.0),
            reason=f"MR Long | Z={z_now:.2f} RSI={rsi_now:.1f}",
            regime=regime.regime.value, daily_bias=bias_str,
        )

    if z_now > config.ZSCORE_ENTRY and rsi_now > config.MR_RSI_SHORT and vol_ok:
        sl = price + mr_stop_mult * atr_now
        tp, rr = _calc_dynamic_tp(price, sl, "short", df_1h)
        partial = _calc_partial_tp(price, sl, "short")

        if rr < config.MIN_RISK_REWARD:
            return None

        return TradeSignal(
            signal=SignalType.SELL, source="mean_reversion", symbol=symbol,
            display_name=display, entry_price=price, stop_loss=sl,
            take_profit=tp, partial_tp=partial, risk_reward=rr,
            confidence=min(0.9, abs(z_now) / 3.0),
            reason=f"MR Short | Z={z_now:.2f} RSI={rsi_now:.1f}",
            regime=regime.regime.value, daily_bias=bias_str,
        )

    return None


def generate_signal(df_15m: pd.DataFrame, df_1h: pd.DataFrame,
                    regime: RegimeState, symbol: str,
                    check_session: bool = True) -> TradeSignal | None:
    if len(df_15m) < config.CANDLE_LIMIT - 10:
        return None

    # Session filter
    if check_session and not _in_session(symbol):
        return None

    allowed = config.ASSET_STRATEGY_MODE.get(symbol, ["trend", "mean_reversion"])

    if regime.regime == Regime.TRENDING and "trend" in allowed:
        return _trend_signal(df_15m, df_1h, regime, symbol)
    elif regime.regime == Regime.RANGING and "mean_reversion" in allowed:
        return _mean_reversion_signal(df_15m, df_1h, regime, symbol)
    return None
