"""
VIPER Backtester v2 — all features.

Session filter, daily bias, dynamic TP, partial TP, multi-instrument.
Phased Funding Pips simulation.

Usage:
    python backtest.py --days 90
    python backtest.py --days 180 --phases
"""

from __future__ import annotations

import argparse
import logging
import os

import numpy as np
import pandas as pd

import config
from indicators import (
    atr, adx, choppiness_index, supertrend, ema,
    hma, rsi, ttm_squeeze, vwap_zscore, chandelier_exit,
    find_sr_levels, find_next_sr,
)
from strategy import _in_session_at

logging.basicConfig(level=logging.WARNING)

COMMISSION_PER_LOT = {"XAUUSD": 7.0, "NAS100": 3.0}


def load_data():
    """Load all instrument data from CSVs."""
    data = {}

    files = {
        "XAUUSD": ("xauusd_15m.csv", "xauusd_daily.csv"),
        "NAS100": ("nas100_15m.csv", "nas100_daily.csv"),
    }

    for symbol in config.SYMBOLS:
        if symbol not in files:
            continue
        f15m, fdaily = files[symbol]

        if not os.path.exists(f15m):
            print(f"  {symbol}: no data file. Run fetch_data.py first.")
            continue

        df = pd.read_csv(f15m, index_col=0, parse_dates=True)
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")

        df_1h = df.resample("1h").agg({
            "open": "first", "high": "max", "low": "min",
            "close": "last", "volume": "sum",
        }).dropna()

        df_daily = None
        if os.path.exists(fdaily):
            df_daily = pd.read_csv(fdaily, index_col=0, parse_dates=True)
            if df_daily.index.tz is None:
                df_daily.index = df_daily.index.tz_localize("UTC")

        data[symbol] = (df, df_1h, df_daily)
        print(f"  {symbol}: {len(df)} bars ({df.index[0].date()} to {df.index[-1].date()})")

    return data


def run_phased(data: dict, max_days: int = 180):
    # Find common range
    starts = [df.index[0] for df, _, _ in data.values()]
    ends = [df.index[-1] for df, _, _ in data.values()]
    start = max(starts)
    end = min(ends)

    master = list(data.keys())[0]
    all_idx = data[master][0].loc[start:end].index
    warmup = config.CANDLE_LIMIT
    if len(all_idx) <= warmup:
        print("Not enough data.")
        return

    # Trim to max_days
    cutoff = end - pd.Timedelta(days=max_days)
    all_idx = all_idx[all_idx >= cutoff]
    if len(all_idx) <= warmup:
        print("Not enough data after trim.")
        return

    tradeable = all_idx[warmup:]
    total_bars = len(tradeable)

    phases = [
        {"name": "Phase 1", "target_pct": config.PROFIT_TARGET_PHASE1},
        {"name": "Phase 2", "target_pct": config.PROFIT_TARGET_PHASE2},
        {"name": "Funded",  "target_pct": None},
    ]

    print(f"\n{'='*65}")
    print(f"  VIPER v2 PHASED BACKTEST")
    print(f"{'='*65}")
    print(f"  Instruments: {', '.join(data.keys())}")
    print(f"  Period: {tradeable[0].date()} to {tradeable[-1].date()} ({total_bars} bars)")
    print(f"  Features: session filter, daily bias, dynamic TP, partial TP")

    current_bar = 0

    for phase in phases:
        if current_bar >= total_bars:
            print(f"\n  {phase['name']}: No data remaining.")
            break

        name = phase["name"]
        account = config.ACCOUNT_SIZE
        equity = account
        target = phase["target_pct"] * account if phase["target_pct"] else None
        floor = account * (1 - config.MAX_DD_LIMIT)

        print(f"\n  --- {name} ---")
        print(f"  Account: ${account:,} | Target: {'${:,.0f}'.format(target) if target else 'None'} | Floor: ${floor:,.0f}")

        pos = {}
        trades = []
        equity_curve = [equity]
        daily_start = equity
        current_date = None
        total_comm = 0.0
        phase_start = current_bar
        blown = False
        partial_tp_done = {}  # track which positions had partial TP

        while current_bar < total_bars:
            ts = tradeable[current_bar]
            hour = ts.hour

            day = ts.date()
            if day != current_date:
                current_date = day
                daily_start = equity

            # --- Manage positions ---
            for sym in list(pos.keys()):
                df_15m, df_1h, _ = data[sym]
                if ts not in df_15m.index:
                    continue

                loc = df_15m.index.get_loc(ts)
                w = df_15m.iloc[max(0, loc - 199):loc + 1]
                if len(w) < 100:
                    continue

                price = float(w["close"].iloc[-1])
                p = pos[sym]
                bars_held = current_bar - p["bar"]

                if p["side"] == "long":
                    p["highest"] = max(p["highest"], price)
                else:
                    p["lowest"] = min(p["lowest"], price)

                # --- Partial TP check ---
                if config.PARTIAL_TP_ENABLED and sym not in partial_tp_done and p.get("partial_tp"):
                    hit_partial = False
                    if p["side"] == "long" and price >= p["partial_tp"]:
                        hit_partial = True
                    elif p["side"] == "short" and price <= p["partial_tp"]:
                        hit_partial = True

                    if hit_partial:
                        # Close 50%
                        close_lots = p["lots"] * config.PARTIAL_TP_RATIO
                        remain_lots = p["lots"] - close_lots

                        if p["side"] == "long":
                            raw = (price - p["entry"]) * close_lots * 100
                        else:
                            raw = (p["entry"] - price) * close_lots * 100

                        comm = COMMISSION_PER_LOT.get(sym, 5.0) * close_lots
                        net = raw - comm
                        equity += net
                        total_comm += comm

                        trades.append({
                            "sym": sym, "pnl": net, "side": p["side"],
                            "bars": bars_held, "source": p["source"],
                            "reason": f"Partial TP (50%)",
                        })

                        p["lots"] = remain_lots
                        if config.MOVE_SL_TO_BE:
                            p["sl"] = p["entry"]  # move SL to breakeven

                        partial_tp_done[sym] = True

                close_it = False
                reason = ""

                # SL
                if p["side"] == "long" and price <= p["sl"]:
                    close_it, reason = True, "SL"
                elif p["side"] == "short" and price >= p["sl"]:
                    close_it, reason = True, "SL"

                # TP
                if not close_it:
                    if p["side"] == "long" and price >= p["tp"]:
                        close_it, reason = True, "TP"
                    elif p["side"] == "short" and price <= p["tp"]:
                        close_it, reason = True, "TP"

                # Trailing
                if not close_it:
                    mult = config.CHANDELIER_MULTIPLIER.get(sym, config.CHANDELIER_MULTIPLIER_DEFAULT)
                    atr_v = atr(w["high"], w["low"], w["close"], config.CHANDELIER_ATR_PERIOD)
                    ca = float(atr_v.iloc[-1])
                    if p["side"] == "long":
                        trail = p["highest"] - mult * ca
                        if trail > p["sl"]: p["sl"] = trail
                        if price <= p["sl"]: close_it, reason = True, "Trail"
                    else:
                        trail = p["lowest"] + mult * ca
                        if trail < p["sl"]: p["sl"] = trail
                        if price >= p["sl"]: close_it, reason = True, "Trail"

                # MR exits
                if not close_it and p["source"] == "mean_reversion":
                    if len(w) >= config.VWAP_PERIOD:
                        zs = vwap_zscore(w["high"], w["low"], w["close"], w["volume"], config.VWAP_PERIOD)
                        z = float(zs.iloc[-1]) if not pd.isna(zs.iloc[-1]) else None
                        if z is not None:
                            if p["side"] == "long" and z >= 0: close_it, reason = True, "MR Z=0"
                            elif p["side"] == "short" and z <= 0: close_it, reason = True, "MR Z=0"
                    if not close_it and bars_held >= config.MR_TIME_STOP_DEFAULT:
                        close_it, reason = True, "Time"

                if close_it:
                    lots = p["lots"]
                    raw = ((price - p["entry"]) if p["side"] == "long" else (p["entry"] - price)) * lots * 100
                    comm = COMMISSION_PER_LOT.get(sym, 5.0) * lots
                    net = raw - comm
                    equity += net
                    total_comm += comm
                    trades.append({"sym": sym, "pnl": net, "side": p["side"],
                                   "bars": bars_held, "source": p["source"], "reason": reason})
                    del pos[sym]
                    partial_tp_done.pop(sym, None)

                    if equity <= floor:
                        blown = True
                        break
                    if target and (equity - account) >= target:
                        d = (current_bar - phase_start) * 15 / 60 / 24
                        print(f"  >>> {name} PASSED in {d:.0f} days | ${equity:,.2f} | {len(trades)} trades")
                        current_bar += 1
                        break

            if blown or (target and (equity - account) >= target):
                break

            # --- New entries ---
            if len(pos) == 0 and not blown:
                daily_dd = (daily_start - equity) / daily_start if equity < daily_start else 0
                if daily_dd >= config.DAILY_DD_LIMIT * 0.8:
                    equity_curve.append(equity)
                    current_bar += 1
                    continue

                dd_pct = 1.0 - (equity / account) if equity < account else 0
                dd_util = dd_pct / config.MAX_DD_LIMIT
                throttle = 0.25 if dd_util > 0.8 else 0.5 if dd_util > 0.5 else 1.0

                for sym in config.SYMBOLS:
                    if sym not in data:
                        continue

                    # Session filter
                    if not _in_session_at(sym, hour):
                        continue

                    df_15m, df_1h, df_daily = data[sym]
                    if ts not in df_15m.index:
                        continue

                    loc = df_15m.index.get_loc(ts)
                    w = df_15m.iloc[max(0, loc - 199):loc + 1]
                    w1h = df_1h.loc[:ts].iloc[-100:]

                    if len(w) < 150:
                        continue

                    price = float(w["close"].iloc[-1])

                    # ATR filter
                    atr_v = atr(w["high"], w["low"], w["close"], config.DAILY_ATR_PERIOD)
                    atr_a = atr_v.rolling(20).mean()
                    if len(atr_a.dropna()) > 0:
                        ca = float(atr_v.iloc[-1])
                        aa = float(atr_a.iloc[-1])
                        min_abs = config.DAILY_ATR_MIN_ABSOLUTE.get(sym, 0)
                        if (aa > 0 and ca / aa < config.DAILY_ATR_MIN_RATIO) or ca < min_abs:
                            continue

                    # Daily bias
                    daily_bias = 0
                    if config.DAILY_BIAS_ENABLED and df_daily is not None and len(df_daily) >= config.DAILY_EMA_SLOW + 5:
                        ema_f = ema(df_daily["close"], config.DAILY_EMA_FAST)
                        ema_s = ema(df_daily["close"], config.DAILY_EMA_SLOW)
                        daily_bias = 1 if ema_f.iloc[-1] > ema_s.iloc[-1] else -1

                    # Regime
                    adx_d = adx(w["high"], w["low"], w["close"], config.ADX_PERIOD)
                    chop_d = choppiness_index(w["high"], w["low"], w["close"], config.CHOP_PERIOD)
                    adx_now = float(adx_d["adx"].iloc[-1])
                    chop_now = float(chop_d.iloc[-1])
                    th = config.REGIME_THRESHOLDS.get(sym, config.REGIME_THRESHOLDS_DEFAULT)

                    is_trending = adx_now >= th["trending_adx"] and chop_now <= th["trending_chop"]
                    is_ranging = adx_now <= th["choppy_adx"] or chop_now >= th["choppy_chop"]

                    allowed = config.ASSET_STRATEGY_MODE.get(sym, ["trend", "mean_reversion"])
                    signal_type = None
                    sl = 0.0
                    conf = 0.7
                    source = "trend"

                    if is_trending and "trend" in allowed:
                        sq = ttm_squeeze(w["high"], w["low"], w["close"],
                                         config.BB_PERIOD, config.BB_STD,
                                         config.KC_PERIOD, config.KC_ATR_PERIOD, config.KC_MULTIPLIER)
                        hf = hma(w["close"], config.HMA_FAST)
                        hs_v = hma(w["close"], config.HMA_SLOW)
                        va = w["volume"].rolling(20).mean()
                        vr = w["volume"] / va
                        vok = float(vr.iloc[-1]) >= config.TREND_VOLUME_MULTIPLIER

                        sq_fired = bool(sq["squeeze_on"].iloc[-2] and sq["squeeze_off"].iloc[-1])
                        mom = float(sq["momentum"].iloc[-1])
                        mom_p = float(sq["momentum"].iloc[-2])
                        rsq = any(sq["squeeze_on"].iloc[-6:-1])
                        hfv = float(hf.iloc[-1])
                        hsv = float(hs_v.iloc[-1])
                        hfp = float(hf.iloc[-2])
                        hsp = float(hs_v.iloc[-2])

                        htf_dir = 1
                        if len(w1h) >= 15:
                            st = supertrend(w1h["high"], w1h["low"], w1h["close"],
                                            config.SUPERTREND_ATR_PERIOD, config.SUPERTREND_MULTIPLIER)
                            htf_dir = int(st["direction"].iloc[-1])

                        mult = config.CHANDELIER_MULTIPLIER.get(sym, config.CHANDELIER_MULTIPLIER_DEFAULT)
                        ce = chandelier_exit(w["high"], w["low"], w["close"], config.CHANDELIER_ATR_PERIOD, mult)

                        # Daily bias: only restrict configured symbols
                        use_bias = sym in config.DAILY_BIAS_SYMBOLS
                        long_ok = (not use_bias) or (daily_bias >= 0)
                        short_ok = (not use_bias) or (daily_bias <= 0)

                        if htf_dir == 1 and vok and long_ok:
                            if sq_fired and mom > 0 and mom > mom_p and hfv > hsv:
                                signal_type, sl, conf = "long", float(ce["long_stop"].iloc[-1]), 0.9
                            elif (hfp <= hsp and hfv > hsv) and rsq:
                                signal_type, sl, conf = "long", float(ce["long_stop"].iloc[-1]), 0.7
                        if htf_dir == -1 and vok and short_ok:
                            if sq_fired and mom < 0 and mom < mom_p and hfv < hsv:
                                signal_type, sl, conf = "short", float(ce["short_stop"].iloc[-1]), 0.9
                            elif (hfp >= hsp and hfv < hsv) and rsq:
                                signal_type, sl, conf = "short", float(ce["short_stop"].iloc[-1]), 0.7

                    elif is_ranging and "mean_reversion" in allowed:
                        source = "mean_reversion"
                        if len(w) >= config.VWAP_PERIOD:
                            zs = vwap_zscore(w["high"], w["low"], w["close"], w["volume"], config.VWAP_PERIOD)
                            rv = rsi(w["close"], config.MR_RSI_PERIOD)
                            av = atr(w["high"], w["low"], w["close"], config.CHANDELIER_ATR_PERIOD)
                            va2 = w["volume"].rolling(20).mean()
                            vr2 = w["volume"] / va2
                            z = float(zs.iloc[-1]) if not pd.isna(zs.iloc[-1]) else None
                            r = float(rv.iloc[-1]) if not pd.isna(rv.iloc[-1]) else None
                            vok2 = float(vr2.iloc[-1]) >= config.MR_VOLUME_MULTIPLIER
                            a = float(av.iloc[-1])
                            if z is not None and r is not None and vok2:
                                if z < -config.ZSCORE_ENTRY and r < config.MR_RSI_LONG:
                                    signal_type, sl, conf = "long", price - 1.2 * a, min(0.9, abs(z) / 3.0)
                                elif z > config.ZSCORE_ENTRY and r > config.MR_RSI_SHORT:
                                    signal_type, sl, conf = "short", price + 1.2 * a, min(0.9, abs(z) / 3.0)

                    if signal_type and sl > 0:
                        risk_dist = abs(price - sl)

                        # Dynamic TP from 1H S/R
                        sr_levels = find_sr_levels(w1h["high"], w1h["low"], w1h["close"],
                                                   config.SR_LOOKBACK, config.SR_TOUCH_COUNT) if len(w1h) >= config.SR_LOOKBACK else []
                        sr_tp = find_next_sr(price, sr_levels, signal_type)

                        if sr_tp and abs(sr_tp - price) / risk_dist >= config.MIN_RISK_REWARD:
                            tp = sr_tp
                        else:
                            tp = price + risk_dist * config.TARGET_RR if signal_type == "long" else price - risk_dist * config.TARGET_RR

                        rr = abs(tp - price) / risk_dist if risk_dist > 0 else 0
                        if rr < config.MIN_RISK_REWARD:
                            continue

                        # Partial TP level
                        partial_tp = None
                        if config.PARTIAL_TP_ENABLED:
                            if signal_type == "long":
                                partial_tp = price + risk_dist * config.PARTIAL_TP_RR
                            else:
                                partial_tp = price - risk_dist * config.PARTIAL_TP_RR

                        # Lot sizing
                        risk_d = equity * config.MAX_RISK_PER_TRADE * conf * throttle
                        lots = round(risk_d / (risk_dist * 100), 2)
                        lots = max(lots, 0.01)

                        spread = config.SPREAD_POINTS.get(sym, 2.5)
                        fill = price + spread if signal_type == "long" else price - spread

                        pos[sym] = {
                            "side": signal_type, "entry": fill, "sl": sl, "tp": tp,
                            "partial_tp": partial_tp,
                            "lots": lots, "source": source, "bar": current_bar,
                            "highest": fill, "lowest": fill,
                        }
                        break

            eq = equity
            for sym, p in pos.items():
                df_15m = data[sym][0]
                if ts in df_15m.index:
                    px = float(df_15m.loc[ts, "close"])
                    eq += ((px - p["entry"]) if p["side"] == "long" else (p["entry"] - px)) * p["lots"] * 100
            equity_curve.append(eq)
            current_bar += 1

        # Phase summary
        phase_pnl = equity - account
        phase_bars = current_bar - phase_start
        phase_days = phase_bars * 15 / 60 / 24
        wins = sum(1 for t in trades if t["pnl"] > 0)
        wr = wins / len(trades) * 100 if trades else 0

        peak = equity_curve[0]
        mdd = 0
        for eq in equity_curve:
            peak = max(peak, eq)
            dd = (peak - eq) / peak * 100 if peak > 0 else 0
            mdd = max(mdd, dd)

        status = "BLOWN" if blown else ("PASSED" if target and phase_pnl >= target else "RUNNING" if not target else "NOT YET")

        print(f"\n  {name}: {status}")
        print(f"  PnL: ${phase_pnl:,.2f} ({phase_pnl/account*100:+.2f}%) | Trades: {len(trades)} ({wins}W/{len(trades)-wins}L) WR: {wr:.1f}%")
        print(f"  Max DD: {mdd:.2f}% | Commission: ${total_comm:,.2f} | Days: {phase_days:.0f}")

        syms = set(t["sym"] for t in trades)
        for s in sorted(syms):
            st = [t for t in trades if t["sym"] == s]
            sw = sum(1 for t in st if t["pnl"] > 0)
            sp = sum(t["pnl"] for t in st)
            print(f"    {s:8s} {len(st):3d}T  {sw/len(st)*100 if st else 0:5.1f}%WR  ${sp:>8,.2f}")

        if blown:
            print(f"\n  FAILED — account blown.")
            break

    print(f"\n{'='*65}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=90)
    parser.add_argument("--phases", action="store_true")
    args = parser.parse_args()

    print("\nLoading data...")
    data = load_data()

    if args.phases:
        run_phased(data, args.days)
    else:
        run_phased(data, args.days)  # always phased now


if __name__ == "__main__":
    main()
