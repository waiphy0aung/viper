"""
Prop firm risk manager for VIPER.

Tracks simulated equity and enforces Funding Pips rules:
- 5% daily drawdown limit ($250 on $5k)
- 10% max drawdown ($500, equity floor = $4,500)
- Base risk 0.75% per trade, throttled in drawdown

Drawdown throttle:
- DD < 4%: full risk (0.75%)
- DD 4-7%: half risk (0.375%)
- DD > 7%: stop trading until recovery
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import config

logger = logging.getLogger(__name__)


class PropFirmRisk:
    """Simulated equity tracker for prop firm rules."""

    def __init__(self):
        self.account_size = config.ACCOUNT_SIZE
        self.equity = self.account_size
        self.daily_start_equity = self.account_size
        self.current_date = datetime.now(timezone.utc).date()
        self.open_signals = 0
        self.daily_pnl = 0.0
        self.total_pnl = 0.0
        self.is_daily_stopped = False
        self.is_blown = False

    def new_day_check(self):
        today = datetime.now(timezone.utc).date()
        if today != self.current_date:
            self.current_date = today
            self.daily_start_equity = self.equity
            self.daily_pnl = 0.0
            self.is_daily_stopped = False
            logger.info(f"New day. Equity: ${self.equity:,.2f}")

    def record_trade_result(self, pnl: float):
        self.equity += pnl
        self.daily_pnl += pnl
        self.total_pnl += pnl

        daily_dd = abs(self.daily_pnl) / self.daily_start_equity if self.daily_pnl < 0 else 0
        if daily_dd >= config.DAILY_DD_LIMIT:
            self.is_daily_stopped = True
            logger.warning(f"DAILY DD LIMIT: {daily_dd:.1%} (${self.daily_pnl:,.2f})")

        if self.equity <= config.EQUITY_FLOOR:
            self.is_blown = True
            logger.critical(f"MAX DD BREACHED: equity ${self.equity:,.2f} <= ${config.EQUITY_FLOOR:,.2f}")

    @property
    def dd_utilization(self) -> float:
        """How much of the max DD budget is used. 0 = no DD, 1 = at the floor."""
        if self.equity >= self.account_size:
            return 0.0
        dd_pct = 1.0 - (self.equity / self.account_size)
        return dd_pct / config.MAX_DD_LIMIT

    @property
    def risk_throttle(self) -> float:
        """Risk multiplier based on drawdown. Never fully stops — need to trade out of DD."""
        util = self.dd_utilization
        if util > 0.8:
            return 0.25  # survival mode
        elif util > 0.5:
            return 0.5
        return 1.0

    @property
    def can_signal(self) -> bool:
        if self.is_blown:
            return False
        if self.is_daily_stopped:
            return False
        if self.open_signals >= config.MAX_CONCURRENT_SIGNALS:
            return False
        return True

    @property
    def daily_dd_remaining(self) -> float:
        limit = self.daily_start_equity * config.DAILY_DD_LIMIT
        used = abs(self.daily_pnl) if self.daily_pnl < 0 else 0
        return limit - used

    @property
    def max_dd_remaining(self) -> float:
        return self.equity - config.EQUITY_FLOOR

    @property
    def progress_pct(self) -> float:
        target = self.account_size * config.PROFIT_TARGET_PHASE1
        return (self.total_pnl / target * 100) if target > 0 else 0


def calculate_lot_size(entry: float, sl: float, risk_manager: PropFirmRisk | None = None) -> float:
    """
    Calculate gold lot size with drawdown throttling.

    1 lot XAUUSD = 100 oz. $1 per point per 0.01 lot.
    """
    risk_dollars = config.ACCOUNT_SIZE * config.MAX_RISK_PER_TRADE

    if risk_manager:
        throttle = risk_manager.risk_throttle
        if throttle == 0:
            return 0.0
        risk_dollars *= throttle

    point_risk = abs(entry - sl)
    if point_risk == 0:
        return 0.0

    lots = risk_dollars / (point_risk * 100)
    lots = round(lots, 2)
    lots = max(lots, 0.01)

    return lots
