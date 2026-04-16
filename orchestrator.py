"""
orchestrator.py — Lightweight event-driven trading bot orchestrator.

Architecture
────────────
EventEngine          — central dispatcher; owns the 5-minute candle clock
TradingOrchestrator  — subscriber that owns all domain modules and handles
                       every event in the pipeline sequentially

Event flow per candle
─────────────────────
  NEW_CANDLE
      │
      ▼
  fetch_market_data   (BrokerGateway)
      │
      ▼
  calculate_indicators (QuantitativeDataAgent)
      │
      ▼
  generate_signal      (StrategyAgent)
      │
    HOLD ──► loop back
      │
   BUY/SELL
      │
      ▼
  check_risk           (RiskManager)  ──► log decision (TradeLogger)
      │
  REJECTED ──► loop back
      │
  APPROVED
      │
      ▼
  execute_order        (BrokerGateway)  ──► log execution (TradeLogger)
"""

from __future__ import annotations

import logging
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from enum import Enum, auto
from typing import Any, Callable, Optional

import pandas as pd
from moomoo import KLType, OrderType, RET_OK, TrdEnv, TrdSide

from broker_gateway import AccountInfo, BrokerGateway, BrokerGatewayError, RateLimitError
from models import (
    AccountState,
    Action,
    ExecutedOrder,
    MarketContext,
    MarketState,
    OptionQuote,
    ProposedTrade,
    SignalDirection,
    TradeDecision,
    TradeSignal,
)
from quant_agent import InsufficientDataError, QuantitativeDataAgent
from risk_manager import RiskConfig, RiskManager
from strategy_agent import StrategyAgent, StrategyConfig
from trade_logger import TradeLogger

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Events
# ──────────────────────────────────────────────────────────────────────────────

class EventType(Enum):
    NEW_CANDLE     = auto()   # fired by the engine clock each 5-min boundary
    MARKET_DATA    = auto()   # fired after a successful data fetch
    INDICATORS     = auto()   # fired after indicator calculation
    SIGNAL         = auto()   # fired after strategy evaluation
    RISK_APPROVED  = auto()   # fired when risk manager approves
    RISK_REJECTED  = auto()   # fired when risk manager rejects
    ORDER_EXECUTED = auto()   # fired after a successful order placement
    ERROR          = auto()   # fired on recoverable errors; engine skips to next candle


@dataclass
class Event:
    type: EventType
    payload: Any = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(tz=timezone.utc))


Handler = Callable[[Event], None]


# ──────────────────────────────────────────────────────────────────────────────
# EventEngine — dispatcher + 5-minute candle clock
# ──────────────────────────────────────────────────────────────────────────────

class EventEngine:
    """
    Central event dispatcher.

    • Handlers are registered per EventType via subscribe().
    • dispatch() calls all handlers for a given event synchronously in
      registration order, stopping early only on unhandled exceptions.
    • run() blocks, sleeping precisely until the next 5-minute candle
      boundary (e.g. 09:35:00, 09:40:00 …), then fires NEW_CANDLE.
    """

    _CANDLE_SECONDS = 300   # 5-minute candle

    def __init__(self) -> None:
        self._handlers: dict[EventType, list[Handler]] = defaultdict(list)
        self._running = False

    def subscribe(self, event_type: EventType, handler: Handler) -> None:
        self._handlers[event_type].append(handler)
        logger.debug("Subscribed %s → %s", event_type.name, handler.__qualname__)

    def dispatch(self, event: Event) -> None:
        handlers = self._handlers.get(event.type, [])
        if not handlers:
            logger.debug("No handlers for %s.", event.type.name)
            return
        logger.debug("Dispatching %s to %d handler(s).", event.type.name, len(handlers))
        for handler in handlers:
            try:
                handler(event)
            except Exception as exc:
                logger.error(
                    "Handler %s raised on %s: %s",
                    handler.__qualname__, event.type.name, exc,
                    exc_info=True,
                )

    def run(self, max_cycles: int = 0) -> None:
        """
        Block and fire NEW_CANDLE at every 5-minute UTC boundary.

        max_cycles=0 means run forever; any positive integer caps iterations.
        """
        self._running = True
        cycle = 0
        logger.info(
            "EventEngine started — firing NEW_CANDLE every %ds (max_cycles=%s).",
            self._CANDLE_SECONDS,
            max_cycles if max_cycles else "∞",
        )
        try:
            while self._running:
                sleep_secs = self._seconds_to_next_candle()
                logger.info(
                    "Next candle in %.1fs (at %s UTC).",
                    sleep_secs,
                    datetime.fromtimestamp(
                        time.time() + sleep_secs, tz=timezone.utc
                    ).strftime("%H:%M:%S"),
                )
                time.sleep(sleep_secs)

                cycle += 1
                logger.info("══════ Candle %d ══════", cycle)
                self.dispatch(Event(EventType.NEW_CANDLE, payload={"cycle": cycle}))

                if max_cycles and cycle >= max_cycles:
                    logger.info("Reached max_cycles=%d — stopping.", max_cycles)
                    break
        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt — engine shutting down.")
        finally:
            self._running = False

    def stop(self) -> None:
        self._running = False

    @classmethod
    def _seconds_to_next_candle(cls) -> float:
        """Return seconds until the next multiple of _CANDLE_SECONDS from epoch."""
        now = time.time()
        remainder = now % cls._CANDLE_SECONDS
        wait = cls._CANDLE_SECONDS - remainder
        # Add a small buffer so we land just after the boundary, not just before
        return wait + 0.1


# ──────────────────────────────────────────────────────────────────────────────
# Bot configuration
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class BotConfig:
    spy_options_contract: str
    trade_password: str = ""
    trade_password_md5: str = ""
    trd_env: TrdEnv = TrdEnv.SIMULATE
    candle_limit: int = 100          # OHLCV bars fetched per cycle
    order_quantity: int = 1          # contracts/shares per signal
    db_path: str = "trades.db"
    risk: RiskConfig = field(default_factory=RiskConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    max_cycles: int = 0              # 0 = run forever


# ──────────────────────────────────────────────────────────────────────────────
# TradingOrchestrator — domain logic, wired to the engine as a subscriber
# ──────────────────────────────────────────────────────────────────────────────

class TradingOrchestrator:
    """
    Owns all domain modules (gateway, quant, strategy, risk, logger) and
    subscribes handler methods to the EventEngine for each pipeline stage.
    """

    def __init__(self, engine: EventEngine, config: BotConfig) -> None:
        self._engine = engine
        self._cfg = config

        self._gateway = BrokerGateway(
            trade_password=config.trade_password or None,
            trade_password_md5=config.trade_password_md5 or None,
            trd_env=config.trd_env,
        )
        self._quant    = QuantitativeDataAgent()
        self._strategy = StrategyAgent(config=config.strategy)
        self._risk     = RiskManager(config=config.risk)
        self._db       = TradeLogger(db_path=config.db_path)

        self._wire_handlers()

    # ------------------------------------------------------------------ #
    # Lifecycle                                                             #
    # ------------------------------------------------------------------ #

    def start(self, max_cycles: int = 0) -> None:
        """Connect to the broker and start the engine."""
        logger.info("Connecting to broker (env=%s)…", self._cfg.trd_env)
        with self._gateway, self._db:
            self._engine.run(max_cycles=max_cycles)

    # ------------------------------------------------------------------ #
    # Handler wiring                                                        #
    # ------------------------------------------------------------------ #

    def _wire_handlers(self) -> None:
        sub = self._engine.subscribe
        sub(EventType.NEW_CANDLE,    self._handle_new_candle)
        sub(EventType.MARKET_DATA,   self._handle_market_data)
        sub(EventType.INDICATORS,    self._handle_indicators)
        sub(EventType.SIGNAL,        self._handle_signal)
        sub(EventType.RISK_APPROVED, self._handle_risk_approved)
        sub(EventType.RISK_REJECTED, self._handle_risk_rejected)
        sub(EventType.ORDER_EXECUTED,self._handle_order_executed)
        sub(EventType.ERROR,         self._handle_error)

    # ------------------------------------------------------------------ #
    # Handlers — each one dispatches the next event in the chain          #
    # ------------------------------------------------------------------ #

    def _handle_new_candle(self, event: Event) -> None:
        logger.info("[handler] new_candle — fetching market data…")
        try:
            spy_quote, opt_quote = self._gateway.fetch_spy_quotes(
                self._cfg.spy_options_contract
            )
            account_info: AccountInfo = self._gateway.fetch_account_info()
            ohlcv = self._fetch_ohlcv("US.SPY", limit=self._cfg.candle_limit)

            today = date.today().isoformat()
            realized_pnl = self._db.realized_pnl_today(today)

            market_state = MarketState(
                spy_price=spy_quote.last_price,
                spy_volume=spy_quote.volume,
                option=OptionQuote(
                    symbol=opt_quote.code,
                    last_price=opt_quote.last_price,
                    volume=opt_quote.volume,
                ),
            )
            account_state = AccountState(
                cash=account_info.cash,
                total_assets=account_info.total_assets,
                buying_power=account_info.power,
                realized_pnl_today=realized_pnl,
            )

        except RateLimitError as exc:
            logger.warning("Rate limit during data fetch — skipping cycle: %s", exc)
            self._engine.dispatch(Event(EventType.ERROR, payload=str(exc)))
            return
        except BrokerGatewayError as exc:
            logger.error("Broker error during data fetch: %s", exc)
            self._engine.dispatch(Event(EventType.ERROR, payload=str(exc)))
            return

        self._engine.dispatch(Event(
            EventType.MARKET_DATA,
            payload={"market_state": market_state, "account_state": account_state, "ohlcv": ohlcv},
        ))

    def _handle_market_data(self, event: Event) -> None:
        logger.info("[handler] market_data — calculating indicators…")
        ohlcv: pd.DataFrame = event.payload["ohlcv"]

        try:
            market_ctx = self._quant.get_market_context(ohlcv, symbol="US.SPY")
        except InsufficientDataError as exc:
            logger.warning("Insufficient OHLCV data: %s", exc)
            self._engine.dispatch(Event(EventType.ERROR, payload=str(exc)))
            return

        self._engine.dispatch(Event(
            EventType.INDICATORS,
            payload={**event.payload, "market_ctx": market_ctx},
        ))

    def _handle_indicators(self, event: Event) -> None:
        logger.info("[handler] indicators — generating signal…")
        market_ctx: MarketContext    = event.payload["market_ctx"]
        market_state: MarketState   = event.payload["market_state"]
        account_state: AccountState = event.payload["account_state"]

        signal = self._strategy.evaluate_context(market_ctx, market_state)

        self._engine.dispatch(Event(
            EventType.SIGNAL,
            payload={
                "signal":        signal,
                "market_ctx":    market_ctx,
                "market_state":  market_state,
                "account_state": account_state,
            },
        ))

    def _handle_signal(self, event: Event) -> None:
        signal: TradeSignal         = event.payload["signal"]
        account_state: AccountState = event.payload["account_state"]

        if signal.direction == SignalDirection.HOLD:
            logger.info("[handler] signal=HOLD — no action this candle.")
            return

        logger.info("[handler] signal=%s (conf=%.2f) — running risk check…",
                    signal.direction.value, signal.confidence)

        action = Action.BUY if signal.direction == SignalDirection.BUY else Action.SELL
        proposed = ProposedTrade(
            symbol=signal.symbol,
            action=action,
            quantity=self._cfg.order_quantity,
            limit_price=signal.entry_price,
            note=signal.rationale[:200],
        )

        decision = self._risk.validate_trade(
            trade=proposed,
            account=account_state,
            market_price=signal.entry_price,
        )
        self._db.log_decision(decision)

        next_event_type = EventType.RISK_APPROVED if decision.approved else EventType.RISK_REJECTED
        self._engine.dispatch(Event(
            next_event_type,
            payload={**event.payload, "proposed_trade": proposed, "trade_decision": decision},
        ))

    def _handle_risk_approved(self, event: Event) -> None:
        proposed: ProposedTrade = event.payload["proposed_trade"]
        signal: TradeSignal     = event.payload["signal"]
        logger.info("[handler] risk_approved — executing order for %s x%d…",
                    proposed.symbol, proposed.quantity)

        trd_side = TrdSide.BUY if proposed.action == Action.BUY else TrdSide.SELL
        try:
            order = self._place_order(
                symbol=proposed.symbol,
                trd_side=trd_side,
                quantity=proposed.quantity,
                price=proposed.limit_price or signal.entry_price,
            )
        except BrokerGatewayError as exc:
            logger.error("[execute_order] Order placement failed: %s", exc)
            self._engine.dispatch(Event(EventType.ERROR, payload=str(exc)))
            return

        self._engine.dispatch(Event(
            EventType.ORDER_EXECUTED,
            payload={**event.payload, "executed_order": order},
        ))

    def _handle_risk_rejected(self, event: Event) -> None:
        decision: TradeDecision = event.payload["trade_decision"]
        logger.info("[handler] risk_rejected — %s", decision.reason)

    def _handle_order_executed(self, event: Event) -> None:
        order: ExecutedOrder    = event.payload["executed_order"]
        account_state: AccountState = event.payload["account_state"]

        self._db.log_execution(order)
        logger.info(
            "[handler] order_executed — %s %d %s @ %.4f (order_id=%s)",
            order.action.value, order.quantity, order.symbol,
            order.fill_price, order.order_id,
        )
        logger.info(
            "Account snapshot — cash=%.2f  total_assets=%.2f  pnl_today=%.2f",
            account_state.cash, account_state.total_assets,
            account_state.realized_pnl_today,
        )

    def _handle_error(self, event: Event) -> None:
        logger.warning("[handler] error — %s — will retry next candle.", event.payload)

    # ------------------------------------------------------------------ #
    # Broker helpers                                                        #
    # ------------------------------------------------------------------ #

    def _fetch_ohlcv(self, code: str, limit: int) -> pd.DataFrame:
        ctx = self._gateway._require_quote_ctx()
        ret, df, _ = ctx.request_history_kline(
            code=code,
            ktype=KLType.K_5M,
            max_count=limit,
        )
        if ret != RET_OK:
            raise BrokerGatewayError(f"request_history_kline failed: {df}")
        return df

    def _place_order(
        self,
        symbol: str,
        trd_side: TrdSide,
        quantity: int,
        price: float,
    ) -> ExecutedOrder:
        ctx = self._gateway._require_trade_ctx()
        ret, data = ctx.place_order(
            price=price,
            qty=quantity,
            code=symbol,
            trd_side=trd_side,
            order_type=OrderType.NORMAL,
            trd_env=self._cfg.trd_env,
        )
        if ret != RET_OK:
            raise BrokerGatewayError(f"place_order failed: {data}")

        row = data.iloc[0]
        order_id   = str(row.get("order_id", f"local-{int(time.time())}"))
        fill_price = float(row.get("dealt_avg_price", price))
        commission = float(row.get("commission", 0.0))

        action = Action.BUY if trd_side == TrdSide.BUY else Action.SELL
        return ExecutedOrder(
            order_id=order_id,
            symbol=symbol,
            action=action,
            quantity=quantity,
            fill_price=fill_price,
            commission=commission,
        )


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    for noisy in ("httpx", "httpcore", "moomoo"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


if __name__ == "__main__":
    _configure_logging()

    cfg = BotConfig(
        spy_options_contract=os.environ.get(
            "SPY_OPTIONS_CONTRACT", "US.SPY240620C00530000"
        ),
        trade_password=os.environ.get("MOOMOO_TRADE_PASSWORD", ""),
        trd_env=TrdEnv.SIMULATE,            # switch to TrdEnv.REAL for live trading
        candle_limit=100,
        order_quantity=1,
        db_path="trades.db",
        risk=RiskConfig(
            max_position_pct=0.10,
            max_daily_loss=500.0,
            min_cash_reserve=1_000.0,
        ),
        max_cycles=0,                       # 0 = run forever
    )

    engine = EventEngine()
    orchestrator = TradingOrchestrator(engine, cfg)
    orchestrator.start(max_cycles=cfg.max_cycles)
