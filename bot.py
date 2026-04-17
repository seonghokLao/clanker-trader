"""
bot.py — Risk manager, SQLite logger, EventEngine, and TradingBot entry point.

Run:  python bot.py
Env:  MOOMOO_TRADE_PASSWORD, SPY_OPTIONS_CONTRACT, ES_CONTRACT
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
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
    AccountState, Direction, ExecutedOrder, HTFContext,
    OptionType, Position, StrategyStep, TradeSetup, TradeSignal,
)
from strategy import SMCEngine, build_htf_context

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Risk Manager
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class RiskConfig:
    max_risk_pct: float   = 0.01     # max 1% of account per trade (options premium)
    max_daily_loss: float = 500.0    # halt new entries if daily loss exceeds this
    min_rr: float         = 2.0      # minimum risk:reward to take the trade
    max_contracts: int    = 5        # hard cap on contracts per signal
    min_cash_reserve: float = 1_000.0


class RiskManager:
    def __init__(self, config: RiskConfig | None = None) -> None:
        self._cfg = config or RiskConfig()

    def approve(
        self,
        setup: TradeSetup,
        account: AccountState,
        option_price: float,
    ) -> tuple[bool, str, int]:
        """
        Returns (approved, reason, quantity).
        quantity=0 when rejected.
        """
        cfg = self._cfg

        # Daily loss gate
        loss = -min(account.realized_pnl_today, 0.0)
        if loss >= cfg.max_daily_loss:
            return False, f"Daily loss ${loss:,.2f} ≥ limit ${cfg.max_daily_loss:,.2f}", 0

        # Minimum R:R
        if setup.entry_price and setup.target_price and setup.stop_loss:
            risk   = abs(setup.entry_price - setup.stop_loss)
            reward = abs(setup.target_price - setup.entry_price)
            rr = reward / risk if risk > 0 else 0
            if rr < cfg.min_rr:
                return False, f"R:R {rr:.2f} < minimum {cfg.min_rr:.2f}", 0

        # Position sizing: risk 1% of account on premium paid
        if option_price <= 0:
            return False, "Option price is zero or negative", 0

        risk_dollars = account.total_assets * cfg.max_risk_pct
        contracts = int(risk_dollars / (option_price * 100))
        contracts = max(1, min(contracts, cfg.max_contracts))

        cost = contracts * option_price * 100
        if account.cash - cost < cfg.min_cash_reserve:
            return False, f"Insufficient cash after trade (need ${cfg.min_cash_reserve:,.0f} reserve)", 0

        return True, f"{contracts} contract(s) @ ${option_price:.2f} premium", contracts


# ──────────────────────────────────────────────────────────────────────────────
# Trade Logger (SQLite)
# ──────────────────────────────────────────────────────────────────────────────

_DDL = """
CREATE TABLE IF NOT EXISTS setups (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp   TEXT    NOT NULL,
    direction   TEXT    NOT NULL,
    step2_tag   TEXT,
    step2b      INTEGER NOT NULL DEFAULT 0,
    step3_tag   TEXT,
    step4_tag   TEXT,
    entry_price REAL,
    stop_loss   REAL,
    target_price REAL,
    approved    INTEGER NOT NULL,
    reject_reason TEXT,
    account_json  TEXT
);

CREATE TABLE IF NOT EXISTS executions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    order_id    TEXT    NOT NULL UNIQUE,
    timestamp   TEXT    NOT NULL,
    symbol      TEXT    NOT NULL,
    option_type TEXT    NOT NULL,
    quantity    INTEGER NOT NULL,
    fill_price  REAL    NOT NULL,
    commission  REAL    NOT NULL DEFAULT 0,
    is_entry    INTEGER NOT NULL DEFAULT 1
);
"""


class TradeLogger:
    def __init__(self, db_path: str = "trades.db") -> None:
        self._lock = threading.Lock()
        self._conn = self._open(db_path)

    def _open(self, path: str) -> sqlite3.Connection:
        conn = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.executescript(_DDL)
        logger.info("Trade database: %s", os.path.abspath(path))
        return conn

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __enter__(self): return self
    def __exit__(self, *_): self.close()

    def log_setup(
        self,
        setup: TradeSetup,
        approved: bool,
        reject_reason: Optional[str],
        account: AccountState,
    ) -> None:
        sql = """INSERT INTO setups
            (timestamp, direction, step2_tag, step2b, step3_tag, step4_tag,
             entry_price, stop_loss, target_price, approved, reject_reason, account_json)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?)"""
        params = (
            setup.timestamp.isoformat(),
            setup.direction.value,
            setup.step2_tag.value if setup.step2_tag else None,
            int(setup.step2b_triggered),
            setup.step3_tag.value if setup.step3_tag else None,
            setup.step4_tag.value if setup.step4_tag else None,
            setup.entry_price, setup.stop_loss, setup.target_price,
            int(approved), reject_reason,
            account.model_dump_json(),
        )
        self._exec(sql, params)

    def log_execution(self, order: ExecutedOrder) -> None:
        sql = """INSERT OR IGNORE INTO executions
            (order_id, timestamp, symbol, option_type, quantity, fill_price, commission, is_entry)
            VALUES (?,?,?,?,?,?,?,?)"""
        self._exec(sql, (
            order.order_id, order.timestamp.isoformat(), order.symbol,
            order.option_type.value, order.quantity, order.fill_price,
            order.commission, int(order.is_entry),
        ))
        logger.info("Logged execution: %s %s x%d @ %.2f",
                    order.symbol, order.option_type.value, order.quantity, order.fill_price)

    def realized_pnl_today(self, date_str: str) -> float:
        sql = """SELECT option_type, SUM(fill_price * quantity * 100)
                 FROM executions WHERE timestamp LIKE ? GROUP BY option_type"""
        with self._lock:
            rows = self._conn.execute(sql, (f"{date_str}%",)).fetchall()
        # Simplified: entries are costs (negative), exits are proceeds (positive)
        totals = {r[0]: r[1] for r in rows}
        return totals.get("exit", 0.0) - totals.get("entry", 0.0)

    def _exec(self, sql: str, params: tuple) -> None:
        with self._lock:
            try:
                self._conn.execute("BEGIN")
                self._conn.execute(sql, params)
                self._conn.execute("COMMIT")
            except sqlite3.Error as exc:
                self._conn.execute("ROLLBACK")
                logger.error("DB write error: %s", exc)
                raise


# ──────────────────────────────────────────────────────────────────────────────
# EventEngine — 1-minute candle clock
# ──────────────────────────────────────────────────────────────────────────────

class EventType(Enum):
    NEW_1M_CANDLE  = auto()
    NEW_5M_CANDLE  = auto()
    SETUP_COMPLETE = auto()
    ERROR          = auto()


@dataclass
class Event:
    type: EventType
    payload: Any = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(tz=timezone.utc))


Handler = Callable[[Event], None]


class EventEngine:
    _1M = 60
    _5M = 300

    def __init__(self) -> None:
        self._handlers: dict[EventType, list[Handler]] = defaultdict(list)
        self._running = False

    def subscribe(self, event_type: EventType, handler: Handler) -> None:
        self._handlers[event_type].append(handler)

    def dispatch(self, event: Event) -> None:
        for handler in self._handlers.get(event.type, []):
            try:
                handler(event)
            except Exception as exc:
                logger.error("Handler %s raised: %s", handler.__qualname__, exc, exc_info=True)

    def run(self, max_cycles: int = 0) -> None:
        self._running = True
        cycle = 0
        logger.info("EventEngine running (1-min clock, max_cycles=%s)",
                    max_cycles if max_cycles else "∞")
        try:
            while self._running:
                sleep = self._until_next(self._1M)
                logger.debug("Next 1-min candle in %.1fs", sleep)
                time.sleep(sleep)

                cycle += 1
                is_5m = (int(time.time()) % self._5M) < self._1M

                self.dispatch(Event(EventType.NEW_1M_CANDLE, {"cycle": cycle}))
                if is_5m:
                    self.dispatch(Event(EventType.NEW_5M_CANDLE, {"cycle": cycle}))

                if max_cycles and cycle >= max_cycles:
                    break
        except KeyboardInterrupt:
            logger.info("Interrupted — shutting down.")
        finally:
            self._running = False

    @staticmethod
    def _until_next(interval: int) -> float:
        now = time.time()
        return interval - (now % interval) + 0.05

    def stop(self) -> None:
        self._running = False


# ──────────────────────────────────────────────────────────────────────────────
# Bot configuration
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class BotConfig:
    spy_options_contract: str           # e.g. "US.SPY260516C00530000"
    es_contract: str = "US.ESmain"      # continuous ES futures
    trade_password: str = ""
    trd_env: TrdEnv = TrdEnv.SIMULATE
    db_path: str = "trades.db"
    candle_limit: int = 200
    risk: RiskConfig = field(default_factory=RiskConfig)
    max_cycles: int = 0                 # 0 = run forever


# ──────────────────────────────────────────────────────────────────────────────
# TradingBot — wires everything together
# ──────────────────────────────────────────────────────────────────────────────

class TradingBot:
    def __init__(self, engine: EventEngine, config: BotConfig) -> None:
        self._cfg     = config
        self._engine  = engine
        self._gateway = BrokerGateway(
            trade_password=config.trade_password or None,
            trd_env=config.trd_env,
        )
        self._risk    = RiskManager(config.risk)
        self._db      = TradeLogger(config.db_path)
        self._smc     = SMCEngine()

        engine.subscribe(EventType.NEW_5M_CANDLE,  self._on_5m_candle)
        engine.subscribe(EventType.NEW_1M_CANDLE,  self._on_1m_candle)
        engine.subscribe(EventType.SETUP_COMPLETE, self._on_setup_complete)
        engine.subscribe(EventType.ERROR,          self._on_error)

    def start(self, max_cycles: int = 0) -> None:
        with self._gateway, self._db:
            self._engine.run(max_cycles=max_cycles)

    # ── Handlers ───────────────────────────────────────────────────────

    def _on_5m_candle(self, event: Event) -> None:
        """Refresh HTF context and advance SMC engine on every 5-min bar."""
        logger.info("[5m] Fetching multi-timeframe data…")
        try:
            df_1h  = self._gateway.fetch_ohlcv("US.SPY", KLType.K_60M,  limit=self._cfg.candle_limit)
            df_4h  = self._gateway.fetch_ohlcv("US.SPY", KLType.K_4H,   limit=self._cfg.candle_limit)
            df_5m  = self._gateway.fetch_ohlcv("US.SPY", KLType.K_5M,   limit=self._cfg.candle_limit)
            df_5m_es = self._gateway.fetch_ohlcv(self._cfg.es_contract, KLType.K_5M, limit=self._cfg.candle_limit)
            spy_price = float(df_5m["close"].iloc[-1])
        except (BrokerGatewayError, RateLimitError) as exc:
            logger.warning("[5m] Data fetch failed: %s", exc)
            self._engine.dispatch(Event(EventType.ERROR, str(exc)))
            return

        today = date.today().isoformat()
        htf_ctx = build_htf_context(df_1h, df_4h, spy_price, today)

        # Store for 1-min handler to use
        self._last_5m_spy  = df_5m
        self._last_5m_es   = df_5m_es
        self._last_htf_ctx = htf_ctx

        logger.info("[5m] bias=%s  draw=%s @ %.2f  price=%.2f",
                    htf_ctx.bias.value, htf_ctx.draw_target.label,
                    htf_ctx.draw_target.price, spy_price)

    def _on_1m_candle(self, event: Event) -> None:
        """Advance the SMC state machine on every 1-min bar."""
        htf_ctx: HTFContext | None = getattr(self, "_last_htf_ctx", None)
        if htf_ctx is None or htf_ctx.bias == Direction.NEUTRAL:
            return
        if self._smc.current_step in (StrategyStep.ENTERED, StrategyStep.INVALIDATED):
            return

        try:
            df_1m     = self._gateway.fetch_ohlcv("US.SPY", KLType.K_1M, limit=100)
            df_1m_es  = self._gateway.fetch_ohlcv(self._cfg.es_contract, KLType.K_1M, limit=100)
            spy_price = float(df_1m["close"].iloc[-1])
        except (BrokerGatewayError, RateLimitError) as exc:
            logger.warning("[1m] Data fetch failed: %s", exc)
            return

        df_5m    = getattr(self, "_last_5m_spy", df_1m)
        df_5m_es = getattr(self, "_last_5m_es",  df_1m_es)

        completed_setup = self._smc.advance(
            htf_ctx=htf_ctx,
            df_5m_spy=df_5m,
            df_5m_es=df_5m_es,
            df_1m_spy=df_1m,
            df_1m_es=df_1m_es,
            current_price=spy_price,
        )

        if completed_setup:
            self._engine.dispatch(Event(EventType.SETUP_COMPLETE, {
                "setup": completed_setup,
                "spy_price": spy_price,
            }))

    def _on_setup_complete(self, event: Event) -> None:
        """Risk check → execute → log."""
        setup: TradeSetup = event.payload["setup"]
        spy_price: float  = event.payload["spy_price"]

        # Fetch live option quote for premium-based position sizing
        try:
            spy_q, opt_q = self._gateway.fetch_spy_quotes(self._cfg.spy_options_contract)
            option_price  = opt_q.last_price
        except BrokerGatewayError as exc:
            logger.error("Cannot fetch option quote: %s", exc)
            return

        # Build account state
        try:
            info = self._gateway.fetch_account_info()
        except BrokerGatewayError as exc:
            logger.error("Cannot fetch account info: %s", exc)
            return

        today = date.today().isoformat()
        account = AccountState(
            cash=info.cash,
            total_assets=info.total_assets,
            buying_power=info.power,
            realized_pnl_today=self._db.realized_pnl_today(today),
        )

        approved, reason, contracts = self._risk.approve(setup, account, option_price)
        self._db.log_setup(setup, approved, None if approved else reason, account)

        if not approved:
            logger.warning("Setup REJECTED: %s", reason)
            self._smc.reset()
            return

        logger.info("Setup APPROVED: %s — entering %d contract(s)", reason, contracts)
        option_type = OptionType.CALL if setup.direction == Direction.BULLISH else OptionType.PUT
        trd_side    = TrdSide.BUY

        try:
            order = self._place_order(
                symbol=self._cfg.spy_options_contract,
                trd_side=trd_side,
                quantity=contracts,
                price=option_price,
                option_type=option_type,
            )
        except BrokerGatewayError as exc:
            logger.error("Order placement failed: %s", exc)
            self._smc.reset()
            return

        self._db.log_execution(order)
        setup.step = StrategyStep.ENTERED
        logger.info(
            "ENTERED %s x%d @ %.2f  |  stop=%.2f  target=%.2f  R:R=%.2f",
            option_type.value, contracts, option_price,
            setup.stop_loss or 0, setup.target_price or 0,
            abs((setup.target_price or 0) - spy_price) /
            max(abs(spy_price - (setup.stop_loss or spy_price)), 0.01),
        )

    def _on_error(self, event: Event) -> None:
        logger.warning("[error] %s — will retry next candle.", event.payload)

    # ── Broker helper ──────────────────────────────────────────────────

    def _place_order(
        self,
        symbol: str,
        trd_side: TrdSide,
        quantity: int,
        price: float,
        option_type: OptionType,
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

        row        = data.iloc[0]
        order_id   = str(row.get("order_id", f"local-{int(time.time())}"))
        fill_price = float(row.get("dealt_avg_price", price))
        commission = float(row.get("commission", 0.0))

        return ExecutedOrder(
            order_id=order_id,
            symbol=symbol,
            option_type=option_type,
            quantity=quantity,
            fill_price=fill_price,
            commission=commission,
            is_entry=True,
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
        spy_options_contract=os.environ.get("SPY_OPTIONS_CONTRACT", "US.SPY260516C00530000"),
        es_contract=os.environ.get("ES_CONTRACT", "US.ESmain"),
        trade_password=os.environ.get("MOOMOO_TRADE_PASSWORD", ""),
        trd_env=TrdEnv.SIMULATE,        # change to TrdEnv.REAL for live trading
        candle_limit=200,
        db_path="trades.db",
        risk=RiskConfig(
            max_risk_pct=0.01,
            max_daily_loss=500.0,
            min_rr=2.0,
            max_contracts=5,
        ),
        max_cycles=0,                   # 0 = run forever
    )

    engine = EventEngine()
    bot    = TradingBot(engine, cfg)
    bot.start(max_cycles=cfg.max_cycles)
