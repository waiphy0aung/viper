"""
VIPER — Prop Firm Signal Engine
Main loop: scans XAUUSD, sends Telegram signals for manual execution.
"""

from __future__ import annotations

import logging
import sys
import time
import traceback
from datetime import datetime, timezone

import config
from data import MarketData
from regime import detect_regime
from strategy import generate_signal, SignalType
from risk import PropFirmRisk, calculate_lot_size
from notifier import (
    send_signal,
    send_startup,
    send_daily_summary,
    send_warning,
    send_dd_alert,
)


def setup_logging():
    fmt = "%(asctime)s | %(levelname)-8s | %(name)-12s | %(message)s"
    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(config.LOG_FILE),
    ]
    logging.basicConfig(
        level=getattr(logging, config.LOG_LEVEL),
        format=fmt,
        handlers=handlers,
    )


logger = logging.getLogger("main")


def run_cycle(data: MarketData, risk: PropFirmRisk, last_signal_time: dict):
    """One scan cycle."""
    risk.new_day_check()

    if not risk.can_signal:
        if risk.is_blown:
            logger.critical("Account blown — max DD breached. Stopping signals.")
        elif risk.is_daily_stopped:
            logger.warning("Daily DD limit — no signals until tomorrow.")
        return

    for symbol in config.SYMBOLS:
        try:
            df_15m = data.fetch_15m(symbol)
            df_1h = data.fetch_1h(symbol)
            df_daily = data.fetch_daily(symbol)

            regime = detect_regime(df_15m, df_1h, symbol, df_daily)
            signal = generate_signal(df_15m, df_1h, regime, symbol)

            if signal is None or signal.signal == SignalType.HOLD:
                continue

            # Cooldown: don't spam signals — minimum 30 min between signals per symbol
            now = time.time()
            last = last_signal_time.get(symbol, 0)
            if now - last < 1800:
                logger.debug(f"Signal cooldown for {symbol} — skipping")
                continue

            # Calculate lot size (with drawdown throttle)
            lot_size = calculate_lot_size(signal.entry_price, signal.stop_loss, risk)
            risk_dollars = config.ACCOUNT_SIZE * config.MAX_RISK_PER_TRADE * risk.risk_throttle

            if lot_size < 0.01:
                continue

            # Send signal
            send_signal(signal, lot_size, risk_dollars)
            last_signal_time[symbol] = now

            logger.info(
                f"SIGNAL: {signal.signal.value} {signal.display_name} @ {signal.entry_price:.2f} "
                f"SL={signal.stop_loss:.2f} TP={signal.take_profit:.2f} "
                f"Lot={lot_size:.2f} R:R=1:{signal.risk_reward:.1f}"
            )

        except Exception as e:
            logger.error(f"Error scanning {symbol}: {e}")


def main():
    setup_logging()

    logger.info("=" * 50)
    logger.info("VIPER — Prop Firm Signal Engine")
    logger.info("=" * 50)
    logger.info(f"Account: ${config.ACCOUNT_SIZE:,}")
    logger.info(f"Instrument: {', '.join(config.SYMBOL_DISPLAY.values())}")
    logger.info(f"Timeframe: {config.TIMEFRAME}")
    logger.info(f"Risk/trade: {config.MAX_RISK_PER_TRADE*100:.0f}%")
    logger.info("=" * 50)

    data = MarketData()
    risk = PropFirmRisk()
    last_signal_time: dict[str, float] = {}

    send_startup(risk)

    consecutive_errors = 0
    last_summary_date = None

    while True:
        try:
            now = datetime.now(timezone.utc)

            # Daily summary at 21:00 UTC
            if now.hour == 21 and last_summary_date != now.date():
                send_daily_summary(risk)
                last_summary_date = now.date()

            run_cycle(data, risk, last_signal_time)

            consecutive_errors = 0
            time.sleep(config.LOOP_INTERVAL_SECONDS)

        except KeyboardInterrupt:
            logger.info("Shutting down...")
            break

        except Exception as e:
            consecutive_errors += 1
            logger.error(f"Cycle error #{consecutive_errors}: {e}\n{traceback.format_exc()}")

            if consecutive_errors >= config.MAX_RETRIES_ON_ERROR:
                send_warning(f"Bot stopped after {config.MAX_RETRIES_ON_ERROR} errors: {e}")
                break

            time.sleep(config.RETRY_DELAY_SECONDS * consecutive_errors)

    logger.info("VIPER stopped.")


if __name__ == "__main__":
    main()
