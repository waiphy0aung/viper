"""
VIPER Strategy — Signal generation for Gold (XAUUSD).

Same regime-adaptive approach as PHANTOM:
  TRENDING → Keltner squeeze + HMA crossover
  RANGING → VWAP Z-Score mean reversion
  UNCERTAIN → no signal

Outputs a complete signal with entry, SL, TP for manual execution.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum

import pandas as pd

import config
from indicators import (
    hma, rsi, atr,
    ttm_squeeze, vwap_zscore, chandelier_exit,
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
    source: str              # "trend" or "mean_reversion"
    symbol: str
    display_name: str
    entry_price: float
    stop_loss: float
    take_profit: float
    risk_reward: float
    confidence: float        # 0-1
    reason: str
    regime: str


def _calc_take_profit(entry: float, sl: float, rr_ratio: float = None) -> float:
    """Calculate TP from entry, SL, and desired R:R ratio."""
    if rr_ratio is None:
        rr_ratio = config.TARGET_RR
    risk = abs(entry - sl)
    if entry > sl:  # long
        return entry + risk * rr_ratio
    else:  # short
        return entry - risk * rr_ratio


def _trend_signal(df: pd.DataFrame, regime: RegimeState, symbol: str) -> TradeSignal | None:
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

    hma_f = float(hma_fast.iloc[-1])
    hma_s = float(hma_slow.iloc[-1])
    hma_f_prev = float(hma_fast.iloc[-2])
    hma_s_prev = float(hma_slow.iloc[-2])
    bullish_hma = hma_f_prev <= hma_s_prev and hma_f > hma_s
    bearish_hma = hma_f_prev >= hma_s_prev and hma_f < hma_s

    vol_ok = float(vol_ratio.iloc[-1]) >= config.TREND_VOLUME_MULTIPLIER
    price = float(close.iloc[-1])
    display = config.SYMBOL_DISPLAY.get(symbol, symbol)

    htf_bull = regime.htf_direction == 1
    htf_bear = regime.htf_direction == -1

    # BUY signals
    if htf_bull and vol_ok:
        triggered = False
        conf = 0.7
        reason = ""

        if squeeze_fired and momentum > 0 and momentum > momentum_prev and hma_f > hma_s:
            triggered = True
            conf = 0.9
            reason = f"Squeeze breakout UP + HMA aligned | Mom={momentum:.2f}"
        elif bullish_hma and recent_squeeze:
            triggered = True
            conf = 0.7
            reason = f"HMA cross UP (post-squeeze) | HMA9={hma_f:.2f}"

        if triggered:
            sl = float(ce["long_stop"].iloc[-1])
            tp = _calc_take_profit(price, sl, 2.0)
            rr = abs(tp - price) / abs(price - sl) if abs(price - sl) > 0 else 0

            if rr < config.MIN_RISK_REWARD:
                return None

            return TradeSignal(
                signal=SignalType.BUY, source="trend", symbol=symbol,
                display_name=display, entry_price=price, stop_loss=sl,
                take_profit=tp, risk_reward=rr, confidence=conf,
                reason=reason, regime=regime.regime.value,
            )

    # SELL signals
    if htf_bear and vol_ok:
        triggered = False
        conf = 0.7
        reason = ""

        if squeeze_fired and momentum < 0 and momentum < momentum_prev and hma_f < hma_s:
            triggered = True
            conf = 0.9
            reason = f"Squeeze breakout DOWN + HMA aligned | Mom={momentum:.2f}"
        elif bearish_hma and recent_squeeze:
            triggered = True
            conf = 0.7
            reason = f"HMA cross DOWN (post-squeeze) | HMA9={hma_f:.2f}"

        if triggered:
            sl = float(ce["short_stop"].iloc[-1])
            tp = _calc_take_profit(price, sl, 2.0)
            rr = abs(tp - price) / abs(price - sl) if abs(price - sl) > 0 else 0

            if rr < config.MIN_RISK_REWARD:
                return None

            return TradeSignal(
                signal=SignalType.SELL, source="trend", symbol=symbol,
                display_name=display, entry_price=price, stop_loss=sl,
                take_profit=tp, risk_reward=rr, confidence=conf,
                reason=reason, regime=regime.regime.value,
            )

    return None


def _mean_reversion_signal(df: pd.DataFrame, regime: RegimeState, symbol: str) -> TradeSignal | None:
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

    mr_stop_mult = 1.2

    # LONG — price below VWAP, oversold, volume climax
    if z_now < -config.ZSCORE_ENTRY and rsi_now < config.MR_RSI_LONG and vol_ok:
        sl = price - mr_stop_mult * atr_now
        tp = _calc_take_profit(price, sl, 2.0)
        rr = abs(tp - price) / abs(price - sl) if abs(price - sl) > 0 else 0

        if rr < config.MIN_RISK_REWARD:
            return None

        return TradeSignal(
            signal=SignalType.BUY, source="mean_reversion", symbol=symbol,
            display_name=display, entry_price=price, stop_loss=sl,
            take_profit=tp, risk_reward=rr,
            confidence=min(0.9, abs(z_now) / 3.0),
            reason=f"MR Long | Z={z_now:.2f} RSI={rsi_now:.1f} Vol={vol_ratio.iloc[-1]:.1f}x",
            regime=regime.regime.value,
        )

    # SHORT — price above VWAP, overbought, volume climax
    if z_now > config.ZSCORE_ENTRY and rsi_now > config.MR_RSI_SHORT and vol_ok:
        sl = price + mr_stop_mult * atr_now
        tp = _calc_take_profit(price, sl, 2.0)
        rr = abs(tp - price) / abs(price - sl) if abs(price - sl) > 0 else 0

        if rr < config.MIN_RISK_REWARD:
            return None

        return TradeSignal(
            signal=SignalType.SELL, source="mean_reversion", symbol=symbol,
            display_name=display, entry_price=price, stop_loss=sl,
            take_profit=tp, risk_reward=rr,
            confidence=min(0.9, abs(z_now) / 3.0),
            reason=f"MR Short | Z={z_now:.2f} RSI={rsi_now:.1f} Vol={vol_ratio.iloc[-1]:.1f}x",
            regime=regime.regime.value,
        )

    return None


def generate_signal(df_15m: pd.DataFrame, df_1h: pd.DataFrame,
                    regime: RegimeState, symbol: str) -> TradeSignal | None:
    if len(df_15m) < config.CANDLE_LIMIT - 10:
        return None

    allowed = config.ASSET_STRATEGY_MODE.get(symbol, ["trend", "mean_reversion"])

    if regime.regime == Regime.TRENDING and "trend" in allowed:
        return _trend_signal(df_15m, regime, symbol)
    elif regime.regime == Regime.RANGING and "mean_reversion" in allowed:
        return _mean_reversion_signal(df_15m, regime, symbol)
    return None
