"""
risk_manager.py — Pure-Python, rule-based pre-trade risk checks.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from models import AccountState, Action, ProposedTrade, TradeDecision

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RiskConfig:
    # Maximum fraction of total_assets a single position may represent
    max_position_pct: float = 0.10          # 10 %
    # Maximum cumulative realized loss allowed today (absolute dollar value)
    max_daily_loss: float = 500.0           # USD
    # Minimum cash cushion that must remain after any trade
    min_cash_reserve: float = 1_000.0       # USD
    # Hard cap on single-order quantity (contracts or shares)
    max_order_quantity: int = 100


class RiskManager:
    """
    Validates proposed trades against a fixed set of risk rules.
    Every check is pure Python — no network calls, no AI.
    """

    def __init__(self, config: RiskConfig | None = None) -> None:
        self._cfg = config or RiskConfig()

    # ------------------------------------------------------------------ #
    # Public API                                                            #
    # ------------------------------------------------------------------ #

    def validate_trade(
        self,
        trade: ProposedTrade,
        account: AccountState,
        market_price: float | None = None,
    ) -> TradeDecision:
        """
        Run all risk checks on *trade* against current *account* state.

        market_price — indicative price used for position-size check when
                       trade.limit_price is None.  If both are None the check
                       is skipped with a warning.
        Returns a TradeDecision (approved or rejected with reason).
        """
        checks = [
            self._check_quantity_cap,
            self._check_daily_loss,
            self._check_position_size,
            self._check_cash_reserve,
        ]

        for check in checks:
            decision = check(trade, account, market_price)
            if decision is not None:
                logger.warning(
                    "Trade REJECTED [%s %d %s] — %s",
                    trade.action, trade.quantity, trade.symbol, decision,
                )
                return TradeDecision(
                    approved=False,
                    reason=decision,
                    proposed=trade,
                    account_snapshot=account,
                )

        logger.info(
            "Trade APPROVED [%s %d %s]",
            trade.action, trade.quantity, trade.symbol,
        )
        return TradeDecision(
            approved=True,
            reason="All risk checks passed.",
            proposed=trade,
            account_snapshot=account,
        )

    # ------------------------------------------------------------------ #
    # Individual checks — return None (pass) or str (rejection reason)    #
    # ------------------------------------------------------------------ #

    def _check_quantity_cap(
        self,
        trade: ProposedTrade,
        account: AccountState,
        _price: float | None,
    ) -> str | None:
        if trade.quantity > self._cfg.max_order_quantity:
            return (
                f"Order quantity {trade.quantity} exceeds hard cap "
                f"{self._cfg.max_order_quantity}."
            )
        return None

    def _check_daily_loss(
        self,
        trade: ProposedTrade,
        account: AccountState,
        _price: float | None,
    ) -> str | None:
        # Only gate new BUY orders when daily loss threshold is breached.
        if trade.action != Action.BUY:
            return None
        loss = -min(account.realized_pnl_today, 0.0)  # positive number
        if loss >= self._cfg.max_daily_loss:
            return (
                f"Daily loss ${loss:,.2f} has reached the threshold "
                f"${self._cfg.max_daily_loss:,.2f}. No new BUY orders permitted."
            )
        return None

    def _check_position_size(
        self,
        trade: ProposedTrade,
        account: AccountState,
        market_price: float | None,
    ) -> str | None:
        price = trade.limit_price or market_price
        if price is None:
            logger.warning(
                "Position-size check skipped for %s — no price available.",
                trade.symbol,
            )
            return None

        existing_val = 0.0
        existing = account.positions_by_symbol.get(trade.symbol)
        if existing is not None:
            existing_val = abs(existing.market_value)

        additional_val = trade.quantity * price
        total_exposure = existing_val + additional_val

        if account.total_assets <= 0:
            return "total_assets is zero or negative; cannot compute position size."

        exposure_pct = total_exposure / account.total_assets
        if exposure_pct > self._cfg.max_position_pct:
            return (
                f"Position in {trade.symbol} would be "
                f"{exposure_pct:.1%} of total assets, exceeding the "
                f"{self._cfg.max_position_pct:.0%} limit."
            )
        return None

    def _check_cash_reserve(
        self,
        trade: ProposedTrade,
        account: AccountState,
        market_price: float | None,
    ) -> str | None:
        if trade.action != Action.BUY:
            return None
        price = trade.limit_price or market_price
        if price is None:
            return None

        cost = trade.quantity * price
        cash_after = account.cash - cost
        if cash_after < self._cfg.min_cash_reserve:
            return (
                f"Trade would leave ${cash_after:,.2f} cash, below the "
                f"${self._cfg.min_cash_reserve:,.2f} reserve requirement."
            )
        return None
