"""
VIPER Backtester — test Gold signal strategy against historical data.

Simulates the exact same strategy with prop firm rules:
- 1% risk per trade, lot sizing for gold
- 5% daily DD, 10% max DD
- Fees modeled (commission per lot)

Usage:
    python backtest.py                # default 90 days
    python backtest.py --days 180     # 6 months
"""

from __future__ import annotations

import argparse
import logging
import time
from dataclasses import dataclass, field

import ccxt
import numpy as np
import pandas as pd

import config
from indicators import (
    atr, adx, choppiness_index, supertrend,
    hma, rsi, ttm_squeeze, vwap_zscore, chandelier_exit,
)
from regime import Regime

logging.basicConfig(level=logging.WARNING)


# Commission: ~$7 round-trip per 1.0 lot on gold (Funding Pips raw spread)
COMMISSION_PER_LOT = 7.0
SPREAD = config.SPREAD_POINTS  # applied on entry


CSV_FILE = "xauusd_15m_180d.csv"


def fetch_gold_data(days: int):
    """Load XAUUSD data from CSV (pre-fetched) or fetch from OKX."""
    import os

    if os.path.exists(CSV_FILE):
        print(f"Loading from {CSV_FILE}...", end=" ", flush=True)
        df = pd.read_csv(CSV_FILE, index_col="timestamp", parse_dates=True)
    else:
        print(f"CSV not found, fetching from OKX ({days} days)...", end=" ", flush=True)
        exchange = ccxt.okx({"enableRateLimit": True})
        symbol = "XAU/USDT:USDT"
        all_candles = []
        total = (days * 24 * 60) // 15
        since = exchange.parse8601(
            (pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=days)).isoformat()
        )
        while len(all_candles) < total:
            batch = exchange.fetch_ohlcv(symbol, "15m", since=since, limit=300)
            if not batch:
                break
            all_candles.extend(batch)
            since = batch[-1][0] + 1
            time.sleep(exchange.rateLimit / 1000 + 0.1)

        df = pd.DataFrame(all_candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df.set_index("timestamp", inplace=True)
        df = df[~df.index.duplicated(keep="first")]
        df.to_csv(CSV_FILE)

    # Trim to requested days
    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=days)
    df = df[df.index >= cutoff]

    df_1h = df.resample("1h").agg({
        "open": "first", "high": "max", "low": "min",
        "close": "last", "volume": "sum",
    }).dropna()

    print(f"{len(df)} candles ({df.index[0].date()} to {df.index[-1].date()})")
    return df, df_1h


@dataclass
class Trade:
    side: str
    source: str
    entry: float
    exit: float
    sl: float
    tp: float
    lots: float
    pnl: float
    commission: float
    net_pnl: float
    bars_held: int
    exit_reason: str


@dataclass
class Result:
    trades: list[Trade] = field(default_factory=list)
    equity_curve: list[float] = field(default_factory=list)
    account_size: float = 5000.0
    total_commission: float = 0.0
    daily_dd_breaches: int = 0
    max_dd_hit: bool = False

    @property
    def total_trades(self): return len(self.trades)
    @property
    def wins(self): return sum(1 for t in self.trades if t.net_pnl > 0)
    @property
    def losses(self): return sum(1 for t in self.trades if t.net_pnl <= 0)
    @property
    def win_rate(self): return self.wins / self.total_trades * 100 if self.trades else 0
    @property
    def total_pnl(self): return sum(t.net_pnl for t in self.trades)
    @property
    def total_return_pct(self): return self.total_pnl / self.account_size * 100
    @property
    def avg_win(self):
        w = [t.net_pnl for t in self.trades if t.net_pnl > 0]
        return sum(w) / len(w) if w else 0
    @property
    def avg_loss(self):
        l = [t.net_pnl for t in self.trades if t.net_pnl < 0]
        return sum(l) / len(l) if l else 0
    @property
    def profit_factor(self):
        gw = sum(t.net_pnl for t in self.trades if t.net_pnl > 0)
        gl = abs(sum(t.net_pnl for t in self.trades if t.net_pnl < 0))
        return gw / gl if gl > 0 else (float("inf") if gw > 0 else 0)
    @property
    def max_drawdown_pct(self):
        if not self.equity_curve: return 0
        peak = self.equity_curve[0]
        mdd = 0
        for eq in self.equity_curve:
            peak = max(peak, eq)
            dd = (peak - eq) / peak * 100
            mdd = max(mdd, dd)
        return mdd
    @property
    def sharpe(self):
        if len(self.equity_curve) < 2: return 0
        rets = pd.Series(self.equity_curve).pct_change().dropna()
        if rets.std() == 0: return 0
        return float(rets.mean() / rets.std() * np.sqrt(35040))


def run_backtest(days: int = 90) -> Result:
    df_15m, df_1h = fetch_gold_data(days)
    symbol = "XAU/USDT:USDT"

    warmup = config.CANDLE_LIMIT
    tradeable = df_15m.index[warmup:]
    total_bars = len(tradeable)
    period_days = total_bars * 15 / 60 / 24

    print(f"\nBacktest: {df_15m.index[warmup].date()} to {df_15m.index[-1].date()}")
    print(f"Bars: {total_bars} | Account: ${config.ACCOUNT_SIZE:,}\n")

    equity = config.ACCOUNT_SIZE
    result = Result(account_size=config.ACCOUNT_SIZE)
    result.equity_curve.append(equity)

    pos = None
    pos_bar = 0
    pos_highest = 0.0
    pos_lowest = float("inf")

    # Daily DD tracking
    daily_start_equity = equity
    current_date = None

    for bar_idx, ts in enumerate(tradeable):
        loc = df_15m.index.get_loc(ts)
        w = df_15m.iloc[max(0, loc - config.CANDLE_LIMIT + 1):loc + 1]
        w1h = df_1h.loc[:ts].iloc[-config.HTF_CANDLE_LIMIT:]

        if len(w) < 150:
            result.equity_curve.append(equity)
            continue

        price = float(w["close"].iloc[-1])

        # Daily reset
        day = ts.date()
        if day != current_date:
            current_date = day
            daily_start_equity = equity

        # --- Manage position ---
        if pos is not None:
            bars_held = bar_idx - pos_bar

            if pos["side"] == "long":
                pos_highest = max(pos_highest, price)
            else:
                pos_lowest = min(pos_lowest, price)

            close_it = False
            reason = ""

            # Check SL
            if pos["side"] == "long" and price <= pos["sl"]:
                close_it = True
                reason = f"SL hit: {price:.2f} <= {pos['sl']:.2f}"
            elif pos["side"] == "short" and price >= pos["sl"]:
                close_it = True
                reason = f"SL hit: {price:.2f} >= {pos['sl']:.2f}"

            # Check TP
            if not close_it:
                if pos["side"] == "long" and price >= pos["tp"]:
                    close_it = True
                    reason = f"TP hit: {price:.2f} >= {pos['tp']:.2f}"
                elif pos["side"] == "short" and price <= pos["tp"]:
                    close_it = True
                    reason = f"TP hit: {price:.2f} <= {pos['tp']:.2f}"

            # Chandelier trailing stop (moves SL up)
            if not close_it:
                mult = config.CHANDELIER_MULTIPLIER.get(symbol, config.CHANDELIER_MULTIPLIER_DEFAULT)
                atr_vals = atr(w["high"], w["low"], w["close"], config.CHANDELIER_ATR_PERIOD)
                cur_atr = float(atr_vals.iloc[-1])

                if pos["side"] == "long":
                    trail = pos_highest - mult * cur_atr
                    if trail > pos["sl"]:  # only tighten, never widen
                        pos["sl"] = trail
                    if price <= pos["sl"]:
                        close_it = True
                        reason = f"Trail SL: {price:.2f} <= {pos['sl']:.2f}"
                else:
                    trail = pos_lowest + mult * cur_atr
                    if trail < pos["sl"]:
                        pos["sl"] = trail
                    if price >= pos["sl"]:
                        close_it = True
                        reason = f"Trail SL: {price:.2f} >= {pos['sl']:.2f}"

            # MR Z-score exit
            if not close_it and pos["source"] == "mean_reversion" and len(w) >= config.VWAP_PERIOD:
                zscore = vwap_zscore(w["high"], w["low"], w["close"], w["volume"], config.VWAP_PERIOD)
                z = float(zscore.iloc[-1]) if not pd.isna(zscore.iloc[-1]) else None
                if z is not None:
                    if pos["side"] == "long" and z >= 0:
                        close_it = True
                        reason = f"MR target: Z={z:.2f}"
                    elif pos["side"] == "short" and z <= 0:
                        close_it = True
                        reason = f"MR target: Z={z:.2f}"

            # MR time stop
            if not close_it and pos["source"] == "mean_reversion" and bars_held >= config.MR_TIME_STOP_DEFAULT:
                close_it = True
                reason = f"Time stop: {bars_held} bars"

            if close_it:
                lots = pos["lots"]
                if pos["side"] == "long":
                    raw_pnl = (price - pos["entry"]) * lots * 100
                else:
                    raw_pnl = (pos["entry"] - price) * lots * 100

                commission = COMMISSION_PER_LOT * lots
                net_pnl = raw_pnl - commission
                equity += net_pnl
                result.total_commission += commission

                result.trades.append(Trade(
                    side=pos["side"], source=pos["source"],
                    entry=pos["entry"], exit=price,
                    sl=pos["sl"], tp=pos["tp"],
                    lots=lots, pnl=raw_pnl, commission=commission,
                    net_pnl=net_pnl, bars_held=bars_held, exit_reason=reason,
                ))

                # Check daily DD
                daily_dd = (daily_start_equity - equity) / daily_start_equity if equity < daily_start_equity else 0
                if daily_dd >= config.DAILY_DD_LIMIT:
                    result.daily_dd_breaches += 1

                # Check max DD
                if equity <= config.EQUITY_FLOOR:
                    result.max_dd_hit = True

                pos = None

        # --- Check for entry ---
        if pos is None:
            # Skip if daily DD would be breached
            daily_dd = (daily_start_equity - equity) / daily_start_equity if equity < daily_start_equity else 0
            if daily_dd >= config.DAILY_DD_LIMIT * 0.8:  # stop at 80% of limit
                result.equity_curve.append(equity)
                continue

            if equity <= config.EQUITY_FLOOR:
                result.equity_curve.append(equity)
                continue

            if len(w) < config.CANDLE_LIMIT - 10:
                result.equity_curve.append(equity)
                continue

            # Daily ATR filter — skip low-volatility periods where spread kills
            if config.DAILY_ATR_FILTER:
                atr_vals_daily = atr(w["high"], w["low"], w["close"], config.DAILY_ATR_PERIOD)
                atr_avg = atr_vals_daily.rolling(20).mean()
                if len(atr_avg.dropna()) > 0:
                    current_atr = float(atr_vals_daily.iloc[-1])
                    avg_atr = float(atr_avg.iloc[-1])
                    if (avg_atr > 0 and current_atr / avg_atr < config.DAILY_ATR_MIN_RATIO) or current_atr < config.DAILY_ATR_MIN_ABSOLUTE:
                        result.equity_curve.append(equity)
                        continue

            # Regime
            adx_data = adx(w["high"], w["low"], w["close"], config.ADX_PERIOD)
            chop = choppiness_index(w["high"], w["low"], w["close"], config.CHOP_PERIOD)
            adx_now = float(adx_data["adx"].iloc[-1])
            chop_now = float(chop.iloc[-1])

            thresholds = config.REGIME_THRESHOLDS.get(symbol, config.REGIME_THRESHOLDS_DEFAULT)
            is_trending = adx_now >= thresholds["trending_adx"] and chop_now <= thresholds["trending_chop"]
            is_ranging = adx_now <= thresholds["choppy_adx"] or chop_now >= thresholds["choppy_chop"]

            allowed = config.ASSET_STRATEGY_MODE.get(symbol, ["trend", "mean_reversion"])

            signal_type = None
            sl = 0.0
            tp = 0.0
            conf = 0.7
            source = "trend"

            if is_trending and "trend" in allowed:
                # Trend signal logic
                squeeze = ttm_squeeze(w["high"], w["low"], w["close"],
                                      config.BB_PERIOD, config.BB_STD,
                                      config.KC_PERIOD, config.KC_ATR_PERIOD, config.KC_MULTIPLIER)
                hma_f = hma(w["close"], config.HMA_FAST)
                hma_s = hma(w["close"], config.HMA_SLOW)

                vol_avg = w["volume"].rolling(20).mean()
                vol_ratio = w["volume"] / vol_avg
                vol_ok = float(vol_ratio.iloc[-1]) >= config.TREND_VOLUME_MULTIPLIER

                sq_fired = bool(squeeze["squeeze_on"].iloc[-2] and squeeze["squeeze_off"].iloc[-1])
                mom = float(squeeze["momentum"].iloc[-1])
                mom_prev = float(squeeze["momentum"].iloc[-2])
                recent_sq = any(squeeze["squeeze_on"].iloc[-6:-1])

                hf = float(hma_f.iloc[-1])
                hs = float(hma_s.iloc[-1])
                hf_p = float(hma_f.iloc[-2])
                hs_p = float(hma_s.iloc[-2])

                # HTF direction
                htf_dir = 1
                if len(w1h) >= 15:
                    st = supertrend(w1h["high"], w1h["low"], w1h["close"],
                                    config.SUPERTREND_ATR_PERIOD, config.SUPERTREND_MULTIPLIER)
                    htf_dir = int(st["direction"].iloc[-1])

                mult = config.CHANDELIER_MULTIPLIER.get(symbol, config.CHANDELIER_MULTIPLIER_DEFAULT)
                ce = chandelier_exit(w["high"], w["low"], w["close"], config.CHANDELIER_ATR_PERIOD, mult)

                if htf_dir == 1 and vol_ok:
                    if sq_fired and mom > 0 and mom > mom_prev and hf > hs:
                        signal_type = "long"
                        sl = float(ce["long_stop"].iloc[-1])
                        conf = 0.9
                    elif (hf_p <= hs_p and hf > hs) and recent_sq:
                        signal_type = "long"
                        sl = float(ce["long_stop"].iloc[-1])
                        conf = 0.7

                if htf_dir == -1 and vol_ok:
                    if sq_fired and mom < 0 and mom < mom_prev and hf < hs:
                        signal_type = "short"
                        sl = float(ce["short_stop"].iloc[-1])
                        conf = 0.9
                    elif (hf_p >= hs_p and hf < hs) and recent_sq:
                        signal_type = "short"
                        sl = float(ce["short_stop"].iloc[-1])
                        conf = 0.7

            elif is_ranging and "mean_reversion" in allowed:
                source = "mean_reversion"
                if len(w) >= config.VWAP_PERIOD:
                    zscore = vwap_zscore(w["high"], w["low"], w["close"], w["volume"], config.VWAP_PERIOD)
                    rsi_vals = rsi(w["close"], config.MR_RSI_PERIOD)
                    atr_vals = atr(w["high"], w["low"], w["close"], config.CHANDELIER_ATR_PERIOD)

                    vol_avg = w["volume"].rolling(20).mean()
                    vol_ratio = w["volume"] / vol_avg

                    z = float(zscore.iloc[-1]) if not pd.isna(zscore.iloc[-1]) else None
                    r = float(rsi_vals.iloc[-1]) if not pd.isna(rsi_vals.iloc[-1]) else None
                    vol_ok = float(vol_ratio.iloc[-1]) >= config.MR_VOLUME_MULTIPLIER
                    a = float(atr_vals.iloc[-1])

                    if z is not None and r is not None and vol_ok:
                        if z < -config.ZSCORE_ENTRY and r < config.MR_RSI_LONG:
                            signal_type = "long"
                            sl = price - 1.2 * a
                            conf = min(0.9, abs(z) / 3.0)
                        elif z > config.ZSCORE_ENTRY and r > config.MR_RSI_SHORT:
                            signal_type = "short"
                            sl = price + 1.2 * a
                            conf = min(0.9, abs(z) / 3.0)

            if signal_type and sl > 0:
                # Calculate TP (2:1 R:R)
                risk_dist = abs(price - sl)
                if signal_type == "long":
                    tp = price + risk_dist * config.TARGET_RR
                else:
                    tp = price - risk_dist * config.TARGET_RR

                rr = abs(tp - price) / risk_dist if risk_dist > 0 else 0
                if rr < config.MIN_RISK_REWARD:
                    result.equity_curve.append(equity)
                    continue

                # Drawdown throttle — scale risk down as we approach limits
                # Never fully stop — that traps you in the drawdown
                dd_from_peak = 1.0 - (equity / config.ACCOUNT_SIZE) if equity < config.ACCOUNT_SIZE else 0
                max_dd_budget = config.MAX_DD_LIMIT  # 0.10
                dd_utilization = dd_from_peak / max_dd_budget  # 0-1, 1 = at the floor

                if dd_utilization > 0.8:
                    throttle = 0.25  # survival mode — tiny trades to claw back
                elif dd_utilization > 0.5:
                    throttle = 0.5   # half risk
                else:
                    throttle = 1.0

                # Lot sizing
                risk_dollars = equity * config.MAX_RISK_PER_TRADE * conf * throttle
                lots = risk_dollars / (risk_dist * 100)
                lots = round(lots, 2)
                lots = max(lots, 0.01)

                # Apply spread to entry — longs fill higher, shorts fill lower
                if signal_type == "long":
                    fill_price = price + SPREAD
                else:
                    fill_price = price - SPREAD

                pos = {
                    "side": signal_type, "entry": fill_price, "sl": sl, "tp": tp,
                    "lots": lots, "source": source,
                }
                pos_bar = bar_idx
                pos_highest = fill_price
                pos_lowest = fill_price

        # Equity tracking (with unrealized PnL)
        eq = equity
        if pos:
            if pos["side"] == "long":
                eq += (price - pos["entry"]) * pos["lots"] * 100
            else:
                eq += (pos["entry"] - price) * pos["lots"] * 100
        result.equity_curve.append(eq)

        if bar_idx % 1000 == 0 and bar_idx > 0:
            pct = bar_idx / total_bars * 100
            print(f"  [{pct:5.1f}%] Bar {bar_idx}/{total_bars} | "
                  f"Equity: ${eq:,.2f} | Trades: {len(result.trades)} | "
                  f"{'IN' if pos else 'OUT'}")

    # Close remaining
    if pos:
        last_p = float(df_15m["close"].iloc[-1])
        if pos["side"] == "long":
            raw = (last_p - pos["entry"]) * pos["lots"] * 100
        else:
            raw = (pos["entry"] - last_p) * pos["lots"] * 100
        comm = COMMISSION_PER_LOT * pos["lots"]
        equity += raw - comm
        result.total_commission += comm
        result.trades.append(Trade(
            side=pos["side"], source=pos["source"],
            entry=pos["entry"], exit=last_p, sl=pos["sl"], tp=pos["tp"],
            lots=pos["lots"], pnl=raw, commission=comm, net_pnl=raw - comm,
            bars_held=total_bars - pos_bar, exit_reason="End of backtest",
        ))
        result.equity_curve.append(equity)

    return result


def print_report(result: Result, days: int):
    print("\n" + "=" * 60)
    print("  VIPER BACKTEST — XAUUSD (Gold)")
    print("=" * 60)

    if not result.trades:
        print("\n  No trades.\n")
        return

    pf = result.profit_factor
    pf_str = f"{pf:.2f}" if pf != float("inf") else "INF"
    final = result.equity_curve[-1]
    months = days / 30
    monthly = result.total_return_pct / months if months > 0 else 0

    # Prop firm pass/fail
    target = config.ACCOUNT_SIZE * config.PROFIT_TARGET_PHASE1
    passed = result.total_pnl >= target and not result.max_dd_hit
    phase1_status = "PASSED" if passed else "FAILED"
    dd_status = "BLOWN" if result.max_dd_hit else "OK"

    print(f"""
  Period:           {days} days
  Account:          ${config.ACCOUNT_SIZE:,}
  Final Equity:     ${final:,.2f}
  Total PnL:        ${result.total_pnl:,.2f} ({result.total_return_pct:+.2f}%)
  Monthly Return:   {monthly:+.2f}%

  Phase 1 Target:   ${target:,.0f} (8%) — {phase1_status}
  Max DD:           {result.max_drawdown_pct:.2f}% — {dd_status}
  Daily DD Breaches: {result.daily_dd_breaches}

  Trades:           {result.total_trades}
  Wins:             {result.wins}
  Losses:           {result.losses}
  Win Rate:         {result.win_rate:.1f}%

  Avg Win:          ${result.avg_win:,.2f}
  Avg Loss:         ${result.avg_loss:,.2f}
  Profit Factor:    {pf_str}
  Sharpe Ratio:     {result.sharpe:.2f}
  Commission:       ${result.total_commission:,.2f}
""")

    # By source
    for source in set(t.source for t in result.trades):
        trades = [t for t in result.trades if t.source == source]
        wins = sum(1 for t in trades if t.net_pnl > 0)
        pnl = sum(t.net_pnl for t in trades)
        label = "Trend" if source == "trend" else "MeanRev"
        print(f"  {label:8s}  {len(trades):3d} trades  {wins/len(trades)*100:5.1f}% WR  ${pnl:>10,.2f}")
    print()

    # Best/worst
    sorted_t = sorted(result.trades, key=lambda t: t.net_pnl)
    print("  --- Worst ---")
    for t in sorted_t[:3]:
        print(f"  {t.side:5s} ${t.net_pnl:>8,.2f} | entry={t.entry:.2f} exit={t.exit:.2f} "
              f"lots={t.lots:.2f} | {t.exit_reason}")
    print("\n  --- Best ---")
    for t in sorted_t[-3:]:
        print(f"  {t.side:5s} ${t.net_pnl:>8,.2f} | entry={t.entry:.2f} exit={t.exit:.2f} "
              f"lots={t.lots:.2f} | {t.exit_reason}")

    print("\n" + "=" * 60)


def run_phased_backtest(days: int = 180):
    """
    Simulate all Funding Pips phases:
      Phase 1: 8% target, account resets to $5k
      Phase 2: 5% target, account resets to $5k
      Funded:  No target, run remaining data
    """
    df_15m, df_1h = fetch_gold_data(days)
    symbol = "XAU/USDT:USDT"
    warmup = config.CANDLE_LIMIT
    tradeable = df_15m.index[warmup:]
    total_bars = len(tradeable)

    phases = [
        {"name": "Phase 1", "target_pct": config.PROFIT_TARGET_PHASE1, "account": config.ACCOUNT_SIZE},
        {"name": "Phase 2", "target_pct": config.PROFIT_TARGET_PHASE2, "account": config.ACCOUNT_SIZE},
        {"name": "Funded",  "target_pct": None,                        "account": config.ACCOUNT_SIZE},
    ]

    print(f"\n{'='*60}")
    print(f"  VIPER PHASED BACKTEST — Funding Pips Simulation")
    print(f"{'='*60}")
    print(f"  Data: {df_15m.index[warmup].date()} to {df_15m.index[-1].date()}")
    print(f"  Bars: {total_bars}\n")

    current_bar = 0  # where we are in the tradeable index

    for phase in phases:
        if current_bar >= total_bars:
            print(f"\n  {phase['name']}: No data remaining.\n")
            break

        name = phase["name"]
        equity = phase["account"]
        target = phase["target_pct"] * equity if phase["target_pct"] else None
        floor = equity * (1 - config.MAX_DD_LIMIT)

        print(f"\n  --- {name} ---")
        print(f"  Account: ${equity:,.0f} | Target: {'${:,.0f}'.format(target) if target else 'None (funded)'} | Floor: ${floor:,.0f}")

        trades = []
        equity_curve = [equity]
        pos = None
        pos_bar = 0
        pos_highest = 0.0
        pos_lowest = float("inf")
        daily_start_equity = equity
        current_date = None
        total_commission = 0.0
        phase_start_bar = current_bar
        blown = False

        while current_bar < total_bars:
            ts = tradeable[current_bar]
            loc = df_15m.index.get_loc(ts)
            w = df_15m.iloc[max(0, loc - config.CANDLE_LIMIT + 1):loc + 1]
            w1h = df_1h.loc[:ts].iloc[-config.HTF_CANDLE_LIMIT:]

            if len(w) < 150:
                equity_curve.append(equity)
                current_bar += 1
                continue

            price = float(w["close"].iloc[-1])

            # Daily reset
            day = ts.date()
            if day != current_date:
                current_date = day
                daily_start_equity = equity

            # --- Manage position ---
            if pos is not None:
                bars_held = current_bar - pos_bar
                if pos["side"] == "long":
                    pos_highest = max(pos_highest, price)
                else:
                    pos_lowest = min(pos_lowest, price)

                close_it = False
                reason = ""

                # SL check
                if pos["side"] == "long" and price <= pos["sl"]:
                    close_it, reason = True, f"SL hit: {price:.2f}"
                elif pos["side"] == "short" and price >= pos["sl"]:
                    close_it, reason = True, f"SL hit: {price:.2f}"

                # TP check
                if not close_it:
                    if pos["side"] == "long" and price >= pos["tp"]:
                        close_it, reason = True, f"TP hit: {price:.2f}"
                    elif pos["side"] == "short" and price <= pos["tp"]:
                        close_it, reason = True, f"TP hit: {price:.2f}"

                # Trailing stop
                if not close_it:
                    mult = config.CHANDELIER_MULTIPLIER.get(symbol, config.CHANDELIER_MULTIPLIER_DEFAULT)
                    atr_vals = atr(w["high"], w["low"], w["close"], config.CHANDELIER_ATR_PERIOD)
                    cur_atr = float(atr_vals.iloc[-1])
                    if pos["side"] == "long":
                        trail = pos_highest - mult * cur_atr
                        if trail > pos["sl"]: pos["sl"] = trail
                        if price <= pos["sl"]:
                            close_it, reason = True, f"Trail: {price:.2f}"
                    else:
                        trail = pos_lowest + mult * cur_atr
                        if trail < pos["sl"]: pos["sl"] = trail
                        if price >= pos["sl"]:
                            close_it, reason = True, f"Trail: {price:.2f}"

                # MR exits
                if not close_it and pos["source"] == "mean_reversion" and len(w) >= config.VWAP_PERIOD:
                    zscore = vwap_zscore(w["high"], w["low"], w["close"], w["volume"], config.VWAP_PERIOD)
                    z = float(zscore.iloc[-1]) if not pd.isna(zscore.iloc[-1]) else None
                    if z is not None:
                        if pos["side"] == "long" and z >= 0:
                            close_it, reason = True, f"MR target Z={z:.2f}"
                        elif pos["side"] == "short" and z <= 0:
                            close_it, reason = True, f"MR target Z={z:.2f}"

                if not close_it and pos["source"] == "mean_reversion" and bars_held >= config.MR_TIME_STOP_DEFAULT:
                    close_it, reason = True, f"Time stop {bars_held} bars"

                if close_it:
                    lots = pos["lots"]
                    raw_pnl = ((price - pos["entry"]) if pos["side"] == "long" else (pos["entry"] - price)) * lots * 100
                    comm = COMMISSION_PER_LOT * lots
                    net = raw_pnl - comm
                    equity += net
                    total_commission += comm
                    trades.append(Trade(
                        side=pos["side"], source=pos["source"], entry=pos["entry"], exit=price,
                        sl=pos["sl"], tp=pos["tp"], lots=lots, pnl=raw_pnl,
                        commission=comm, net_pnl=net, bars_held=bars_held, exit_reason=reason,
                    ))
                    pos = None

                    # Check blown
                    if equity <= floor:
                        blown = True
                        print(f"  !!! BLOWN at bar {current_bar - phase_start_bar} — equity ${equity:,.2f} <= ${floor:,.0f}")
                        break

                    # Check phase target reached
                    if target and (equity - phase["account"]) >= target:
                        phase_days = (current_bar - phase_start_bar) * 15 / 60 / 24
                        print(f"  >>> {name} PASSED in {phase_days:.0f} days ({current_bar - phase_start_bar} bars)")
                        print(f"      Equity: ${equity:,.2f} | PnL: ${equity - phase['account']:,.2f} | Trades: {len(trades)}")
                        current_bar += 1
                        break

            # --- Entry ---
            if pos is None and not blown:
                daily_dd = (daily_start_equity - equity) / daily_start_equity if equity < daily_start_equity else 0
                if daily_dd < config.DAILY_DD_LIMIT * 0.8 and equity > floor and len(w) >= config.CANDLE_LIMIT - 10:

                    # ATR filter
                    skip_atr = False
                    if config.DAILY_ATR_FILTER:
                        atr_v = atr(w["high"], w["low"], w["close"], config.DAILY_ATR_PERIOD)
                        atr_a = atr_v.rolling(20).mean()
                        if len(atr_a.dropna()) > 0:
                            ca = float(atr_v.iloc[-1])
                            aa = float(atr_a.iloc[-1])
                            if aa > 0 and ca / aa < config.DAILY_ATR_MIN_RATIO:
                                skip_atr = True

                    if not skip_atr:
                        # Regime
                        adx_data = adx(w["high"], w["low"], w["close"], config.ADX_PERIOD)
                        chop_data = choppiness_index(w["high"], w["low"], w["close"], config.CHOP_PERIOD)
                        adx_now = float(adx_data["adx"].iloc[-1])
                        chop_now = float(chop_data.iloc[-1])
                        th = config.REGIME_THRESHOLDS.get(symbol, config.REGIME_THRESHOLDS_DEFAULT)

                        is_trending = adx_now >= th["trending_adx"] and chop_now <= th["trending_chop"]
                        is_ranging = adx_now <= th["choppy_adx"] or chop_now >= th["choppy_chop"]

                        allowed = config.ASSET_STRATEGY_MODE.get(symbol, ["trend", "mean_reversion"])
                        signal_type = None
                        sl = 0.0
                        conf = 0.7
                        source = "trend"

                        if is_trending and "trend" in allowed:
                            squeeze = ttm_squeeze(w["high"], w["low"], w["close"],
                                                  config.BB_PERIOD, config.BB_STD,
                                                  config.KC_PERIOD, config.KC_ATR_PERIOD, config.KC_MULTIPLIER)
                            hma_f = hma(w["close"], config.HMA_FAST)
                            hma_s = hma(w["close"], config.HMA_SLOW)
                            vol_avg = w["volume"].rolling(20).mean()
                            vol_ratio = w["volume"] / vol_avg
                            vol_ok = float(vol_ratio.iloc[-1]) >= config.TREND_VOLUME_MULTIPLIER

                            sq_fired = bool(squeeze["squeeze_on"].iloc[-2] and squeeze["squeeze_off"].iloc[-1])
                            mom = float(squeeze["momentum"].iloc[-1])
                            mom_prev = float(squeeze["momentum"].iloc[-2])
                            recent_sq = any(squeeze["squeeze_on"].iloc[-6:-1])
                            hf = float(hma_f.iloc[-1])
                            hs = float(hma_s.iloc[-1])
                            hf_p = float(hma_f.iloc[-2])
                            hs_p = float(hma_s.iloc[-2])

                            htf_dir = 1
                            if len(w1h) >= 15:
                                st = supertrend(w1h["high"], w1h["low"], w1h["close"],
                                                config.SUPERTREND_ATR_PERIOD, config.SUPERTREND_MULTIPLIER)
                                htf_dir = int(st["direction"].iloc[-1])

                            mult = config.CHANDELIER_MULTIPLIER.get(symbol, config.CHANDELIER_MULTIPLIER_DEFAULT)
                            ce = chandelier_exit(w["high"], w["low"], w["close"], config.CHANDELIER_ATR_PERIOD, mult)

                            if htf_dir == 1 and vol_ok:
                                if sq_fired and mom > 0 and mom > mom_prev and hf > hs:
                                    signal_type, sl, conf = "long", float(ce["long_stop"].iloc[-1]), 0.9
                                elif (hf_p <= hs_p and hf > hs) and recent_sq:
                                    signal_type, sl, conf = "long", float(ce["long_stop"].iloc[-1]), 0.7
                            if htf_dir == -1 and vol_ok:
                                if sq_fired and mom < 0 and mom < mom_prev and hf < hs:
                                    signal_type, sl, conf = "short", float(ce["short_stop"].iloc[-1]), 0.9
                                elif (hf_p >= hs_p and hf < hs) and recent_sq:
                                    signal_type, sl, conf = "short", float(ce["short_stop"].iloc[-1]), 0.7

                        elif is_ranging and "mean_reversion" in allowed:
                            source = "mean_reversion"
                            if len(w) >= config.VWAP_PERIOD:
                                zs = vwap_zscore(w["high"], w["low"], w["close"], w["volume"], config.VWAP_PERIOD)
                                rv = rsi(w["close"], config.MR_RSI_PERIOD)
                                av = atr(w["high"], w["low"], w["close"], config.CHANDELIER_ATR_PERIOD)
                                va = w["volume"].rolling(20).mean()
                                vr = w["volume"] / va

                                z = float(zs.iloc[-1]) if not pd.isna(zs.iloc[-1]) else None
                                r = float(rv.iloc[-1]) if not pd.isna(rv.iloc[-1]) else None
                                vok = float(vr.iloc[-1]) >= config.MR_VOLUME_MULTIPLIER
                                a = float(av.iloc[-1])

                                if z is not None and r is not None and vok:
                                    if z < -config.ZSCORE_ENTRY and r < config.MR_RSI_LONG:
                                        signal_type, sl, conf = "long", price - 1.2 * a, min(0.9, abs(z) / 3.0)
                                    elif z > config.ZSCORE_ENTRY and r > config.MR_RSI_SHORT:
                                        signal_type, sl, conf = "short", price + 1.2 * a, min(0.9, abs(z) / 3.0)

                        if signal_type and sl > 0:
                            risk_dist = abs(price - sl)
                            tp = price + risk_dist * config.TARGET_RR if signal_type == "long" else price - risk_dist * config.TARGET_RR
                            rr = abs(tp - price) / risk_dist if risk_dist > 0 else 0

                            if rr >= config.MIN_RISK_REWARD:
                                # DD throttle
                                dd_pct = 1.0 - (equity / phase["account"]) if equity < phase["account"] else 0
                                dd_util = dd_pct / config.MAX_DD_LIMIT
                                throttle = 0.25 if dd_util > 0.8 else 0.5 if dd_util > 0.5 else 1.0

                                risk_dollars = equity * config.MAX_RISK_PER_TRADE * conf * throttle
                                lots = round(risk_dollars / (risk_dist * 100), 2)
                                lots = max(lots, 0.01)

                                fill = price + SPREAD if signal_type == "long" else price - SPREAD
                                pos = {"side": signal_type, "entry": fill, "sl": sl, "tp": tp,
                                       "lots": lots, "source": source}
                                pos_bar = current_bar
                                pos_highest = fill
                                pos_lowest = fill

            # Equity tracking
            eq = equity
            if pos:
                eq += ((price - pos["entry"]) if pos["side"] == "long" else (pos["entry"] - price)) * pos["lots"] * 100
            equity_curve.append(eq)
            current_bar += 1

        # Phase summary
        phase_pnl = equity - phase["account"]
        phase_bars = current_bar - phase_start_bar
        phase_days_total = phase_bars * 15 / 60 / 24
        wins = sum(1 for t in trades if t.net_pnl > 0)
        wr = wins / len(trades) * 100 if trades else 0

        peak = equity_curve[0]
        mdd = 0
        for eq in equity_curve:
            peak = max(peak, eq)
            dd = (peak - eq) / peak * 100 if peak > 0 else 0
            mdd = max(mdd, dd)

        status = "BLOWN" if blown else ("PASSED" if target and phase_pnl >= target else "IN PROGRESS" if not target else "NOT PASSED YET")

        print(f"\n  {name} Result: {status}")
        print(f"  PnL: ${phase_pnl:,.2f} ({phase_pnl/phase['account']*100:+.2f}%)")
        print(f"  Trades: {len(trades)} ({wins}W/{len(trades)-wins}L) WR: {wr:.1f}%")
        print(f"  Max DD: {mdd:.2f}% | Commission: ${total_commission:,.2f}")
        print(f"  Duration: {phase_days_total:.0f} days ({phase_bars} bars)")

        if blown:
            print(f"\n  Account blown in {name}. Challenge failed.")
            break


def main():
    parser = argparse.ArgumentParser(description="VIPER Backtester")
    parser.add_argument("--days", type=int, default=90)
    parser.add_argument("--phases", action="store_true", help="Simulate all Funding Pips phases")
    args = parser.parse_args()

    if args.phases:
        run_phased_backtest(args.days)
    else:
        result = run_backtest(args.days)
        print_report(result, args.days)


if __name__ == "__main__":
    main()
