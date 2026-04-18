"""
Telegram signal sender for VIPER.

Sends clean, actionable signals you can execute on Funding Pips:
- Entry price, SL, TP, lot size
- Risk amount and R:R ratio
- Regime context
- Daily summary with P&L tracking
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import requests

import config
from strategy import TradeSignal, SignalType
from risk import PropFirmRisk

logger = logging.getLogger(__name__)


def _send(text: str):
    if not config.TELEGRAM_BOT_TOKEN or not config.TELEGRAM_CHAT_ID:
        logger.warning("Telegram not configured — signal not sent")
        return

    url = f"https://api.telegram.org/bot{config.TELEGRAM_BOT_TOKEN}/sendMessage"
    try:
        resp = requests.post(url, json={
            "chat_id": config.TELEGRAM_CHAT_ID,
            "text": text,
            "parse_mode": "HTML",
            "disable_web_page_preview": True,
        }, timeout=10)
        if resp.status_code != 200:
            logger.warning(f"Telegram error: {resp.status_code}")
    except Exception as e:
        logger.warning(f"Telegram send failed: {e}")


def send_signal(signal: TradeSignal, lot_size: float, risk_dollars: float):
    """Send a trading signal to Telegram."""
    if signal.signal == SignalType.BUY:
        emoji = "\U0001f7e2"
        direction = "BUY"
    else:
        emoji = "\U0001f534"
        direction = "SELL"

    now = datetime.now(timezone.utc).strftime("%H:%M UTC")

    text = (
        f"{emoji} <b>{direction} {signal.display_name}</b>\n"
        f"\n"
        f"\U0001f4cd Entry:  <code>{signal.entry_price:.2f}</code>\n"
        f"\U0001f6d1 SL:     <code>{signal.stop_loss:.2f}</code>\n"
        f"\U0001f3af TP:     <code>{signal.take_profit:.2f}</code>\n"
        f"\n"
        f"\U0001f4e6 Lot:    <code>{lot_size:.2f}</code>\n"
        f"\U0001f4b0 Risk:   <code>${risk_dollars:.0f}</code> ({config.MAX_RISK_PER_TRADE*100:.0f}%)\n"
        f"\U0001f4ca R:R:    <code>1:{signal.risk_reward:.1f}</code>\n"
        f"\n"
        f"\U0001f9e0 {signal.source.replace('_', ' ').title()} | {signal.regime}\n"
        f"\U0001f4ac {signal.reason}\n"
        f"\n"
        f"\u23f0 {now}"
    )
    _send(text)
    logger.info(f"Signal sent: {direction} {signal.display_name} @ {signal.entry_price:.2f}")


def send_startup(risk: PropFirmRisk):
    text = (
        f"\U0001f680 <b>VIPER Signal Bot Started</b>\n"
        f"\n"
        f"Account: ${config.ACCOUNT_SIZE:,}\n"
        f"Instrument: XAUUSD\n"
        f"Timeframe: {config.TIMEFRAME}\n"
        f"Risk/trade: {config.MAX_RISK_PER_TRADE*100:.0f}%\n"
        f"Daily DD limit: {config.DAILY_DD_LIMIT*100:.0f}%\n"
        f"Max DD floor: ${config.EQUITY_FLOOR:,.0f}\n"
        f"\n"
        f"Waiting for signals..."
    )
    _send(text)


def send_daily_summary(risk: PropFirmRisk):
    progress = risk.progress_pct
    bar_len = 20
    filled = int(progress / 100 * bar_len)
    bar = "\u2588" * filled + "\u2591" * (bar_len - filled)

    pnl_emoji = "\U0001f4b0" if risk.daily_pnl >= 0 else "\U0001f4a5"
    pnl_sign = "+" if risk.daily_pnl >= 0 else ""

    text = (
        f"\U0001f4ca <b>Daily Summary</b>\n"
        f"\n"
        f"Equity:    <code>${risk.equity:,.2f}</code>\n"
        f"Today:     <code>{pnl_sign}${risk.daily_pnl:,.2f}</code> {pnl_emoji}\n"
        f"Total PnL: <code>{'+' if risk.total_pnl >= 0 else ''}${risk.total_pnl:,.2f}</code>\n"
        f"DD remain: <code>${risk.daily_dd_remaining:,.2f}</code> daily | "
        f"<code>${risk.max_dd_remaining:,.2f}</code> max\n"
        f"\n"
        f"Target: [{bar}] {progress:.1f}%"
    )
    _send(text)


def send_warning(message: str):
    text = f"\u26a0\ufe0f <b>WARNING</b>\n\n{message}"
    _send(text)


def send_dd_alert(risk: PropFirmRisk):
    text = (
        f"\U0001f6a8 <b>DRAWDOWN ALERT</b>\n"
        f"\n"
        f"Daily DD remaining: <code>${risk.daily_dd_remaining:,.2f}</code>\n"
        f"Max DD remaining: <code>${risk.max_dd_remaining:,.2f}</code>\n"
        f"Equity: <code>${risk.equity:,.2f}</code>\n"
        f"\n"
        f"<b>Signals paused until conditions improve.</b>"
    )
    _send(text)
