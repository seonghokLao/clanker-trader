"""
bot.py — Risk manager, SQLite logger, EventEngine, TradingBot entry point.

Run:  python bot.py
Env:  MOOMOO_TRADE_PASSWORD, SPY_OPTIONS_CONTRACT
"""
from __future__ import annotations

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

from moomoo import KLType, OrderType, RET_OK, TrdEnv, TrdSide

from broker_gateway import AccountInfo, BrokerGateway, BrokerGatewayError, RateLimitError
from models import (
    AccountState, Direction, ExecutedOrder, KeyLevel,
    OptionType, SetupStep, TradeSetup,
)
from strategy import SetupEngine, build_key_levels, get_daily_bias

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Risk Manager
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class RiskConfig:
    max_risk_pct: float    = 0.01    # 1 % of account per trade (options premium)
    max_daily_loss: float  = 500.0
    min_rr: float          = 2.0
    max_contracts: int     = 5
    min_cash_reserve: float = 1_000.0


class RiskManager:
    def __init__(self, cfg: RiskConfig | None = None) -> None:
        self._cfg = cfg or RiskConfig()

    def approve(
        self,
        setup: TradeSetup,
        account: AccountState,
        option_price: float,
    ) -> tuple[bool, str, int]:
        """Returns (approved, reason, quantity)."""
        cfg = self._cfg

        loss = -min(account.realized_pnl_today, 0.0)
        if loss >= cfg.max_daily_loss:
            return False, f"Daily loss ${loss:,.2f} ≥ limit ${cfg.max_daily_loss:,.2f}", 0

        if setup.entry_price and setup.target_price and setup.stop_loss:
            risk   = abs(setup.entry_price - setup.stop_loss)
            reward = abs(setup.target_price - setup.entry_price)
            rr = reward / risk if risk > 0 else 0.0
            if rr < cfg.min_rr:
                return False, f"R:R {rr:.2f} below minimum {cfg.min_rr:.2f}", 0

        if option_price <= 0:
            return False, "Option price is zero", 0

        risk_dollars = account.total_assets * cfg.max_risk_pct
        contracts    = max(1, min(int(risk_dollars / (option_price * 100)), cfg.max_contracts))

        cost = contracts * option_price * 100
        if account.cash - cost < cfg.min_cash_reserve:
            return False, f"Insufficient cash (need ${cfg.min_cash_reserve:,.0f} reserve)", 0

        return True, f"{contracts} contract(s) @ ${option_price:.2f}", contracts


# ──────────────────────────────────────────────────────────────────────────────
# Trade Logger
# ──────────────────────────────────────────────────────────────────────────────

_DDL = """
CREATE TABLE IF NOT EXISTS setups (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp    TEXT NOT NULL,
    direction    TEXT NOT NULL,
    swept_level  TEXT,
    bos_level    REAL,
    entry_price  REAL,
    stop_loss    REAL,
    target_price REAL,
    approved     INTEGER NOT NULL,
    reject_reason TEXT,
    account_json TEXT
);
CREATE TABLE IF NOT EXISTS executions (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    order_id     TEXT NOT NULL UNIQUE,
    timestamp    TEXT NOT NULL,
    symbol       TEXT NOT NULL,
    option_type  TEXT NOT NULL,
    quantity     INTEGER NOT NULL,
    fill_price   REAL NOT NULL,
    commission   REAL NOT NULL DEFAULT 0,
    is_entry     INTEGER NOT NULL DEFAULT 1
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
        logger.info("Trade DB: %s", os.path.abspath(path))
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
            (timestamp, direction, swept_level, bos_level, entry_price,
             stop_loss, target_price, approved, reject_reason, account_json)
            VALUES (?,?,?,?,?,?,?,?,?,?)"""
        self._exec(sql, (
            setup.timestamp.isoformat(),
            setup.direction.value,
            setup.sweep.level.label if setup.sweep else None,
            setup.bos.broken_level  if setup.bos   else None,
            setup.entry_price,
            setup.stop_loss,
            setup.target_price,
            int(approved),
            reject_reason,
            account.model_dump_json(),
        ))

    def log_execution(self, order: ExecutedOrder) -> None:
        sql = """INSERT OR IGNORE INTO executions
            (order_id, timestamp, symbol, option_type, quantity,
             fill_price, commission, is_entry)
            VALUES (?,?,?,?,?,?,?,?)"""
        self._exec(sql, (
            order.order_id, order.timestamp.isoformat(), order.symbol,
            order.option_type.value, order.quantity, order.fill_price,
            order.commission, int(order.is_entry),
        ))
        logger.info("Execution logged: %s x%d @ %.2f", order.symbol, order.quantity, order.fill_price)

    def realized_pnl_today(self, date_str: str) -> float:
        sql = """SELECT is_entry, SUM(fill_price * quantity * 100)
                 FROM executions WHERE timestamp LIKE ? GROUP BY is_entry"""
        with self._lock:
            rows = self._conn.execute(sql, (f"{date_str}%",)).fetchall()
        totals = {r[0]: r[1] for r in rows}
        # exits (is_entry=0) are proceeds, entries (is_entry=1) are costs
        return totals.get(0, 0.0) - totals.get(1, 0.0)

    def _exec(self, sql: str, params: tuple) -> None:
        with self._lock:
            try:
                self._conn.execute("BEGIN")
                self._conn.execute(sql, params)
                self._conn.execute("COMMIT")
            except sqlite3.Error as exc:
                self._conn.execute("ROLLBACK")
                logger.error("DB error: %s", exc)
                raise


# ──────────────────────────────────────────────────────────────────────────────
# EventEngine — 1-minute candle clock
# ──────────────────────────────────────────────────────────────────────────────

class EventType(Enum):
    NEW_1M_CANDLE = auto()
    SETUP_READY   = auto()
    ERROR         = auto()


@dataclass
class Event:
    type: EventType
    payload: Any = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(tz=timezone.utc))


Handler = Callable[[Event], None]


class EventEngine:
    _1M = 60

    def __init__(self) -> None:
        self._handlers: dict[EventType, list[Handler]] = defaultdict(list)
        self._running = False

    def subscribe(self, et: EventType, h: Handler) -> None:
        self._handlers[et].append(h)

    def dispatch(self, event: Event) -> None:
        for h in self._handlers.get(event.type, []):
            try:
                h(event)
            except Exception as exc:
                logger.error("Handler %s: %s", h.__qualname__, exc, exc_info=True)

    def run(self, max_cycles: int = 0) -> None:
        self._running = True
        cycle = 0
        logger.info("EventEngine started (1-min clock, max=%s)",
                    max_cycles if max_cycles else "∞")
        try:
            while self._running:
                sleep = self._1M - (time.time() % self._1M) + 0.05
                logger.debug("Next candle in %.1fs", sleep)
                time.sleep(sleep)
                cycle += 1
                self.dispatch(Event(EventType.NEW_1M_CANDLE, {"cycle": cycle}))
                if max_cycles and cycle >= max_cycles:
                    break
        except KeyboardInterrupt:
            logger.info("Interrupted — shutting down.")
        finally:
            self._running = False

    def stop(self) -> None:
        self._running = False


# ──────────────────────────────────────────────────────────────────────────────
# Bot configuration
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class BotConfig:
    spy_options_contract: str
    trade_password: str  = ""
    trd_env: TrdEnv      = TrdEnv.SIMULATE
    db_path: str         = "trades.db"
    candle_limit: int    = 200
    risk: RiskConfig     = field(default_factory=RiskConfig)
    max_cycles: int      = 0


# ──────────────────────────────────────────────────────────────────────────────
# TradingBot
# ──────────────────────────────────────────────────────────────────────────────

class TradingBot:
    def __init__(self, engine: EventEngine, cfg: BotConfig) -> None:
        self._cfg     = cfg
        self._engine  = engine
        self._gateway = BrokerGateway(
            trade_password=cfg.trade_password or None,
            trd_env=cfg.trd_env,
        )
        self._risk    = RiskManager(cfg.risk)
        self._db      = TradeLogger(cfg.db_path)
        self._smc     = SetupEngine()

        # Cached state refreshed on each candle
        self._levels:  list[KeyLevel] = []
        self._last_level_date: str = ""

        engine.subscribe(EventType.NEW_1M_CANDLE, self._on_candle)
        engine.subscribe(EventType.SETUP_READY,   self._on_setup_ready)
        engine.subscribe(EventType.ERROR,         lambda e: logger.warning("[error] %s", e.payload))

    def start(self, max_cycles: int = 0) -> None:
        with self._gateway, self._db:
            self._engine.run(max_cycles=max_cycles)

    # ── Main candle handler ────────────────────────────────────────────

    def _on_candle(self, event: Event) -> None:
        today = date.today().isoformat()

        try:
            # Refresh HTF data and key levels once per day
            if today != self._last_level_date:
                self._refresh_levels(today)

            # Always fetch latest 1-min and 5-min candles
            df_1m = self._gateway.fetch_ohlcv("US.SPY", KLType.K_1M,  limit=100)
            df_5m = self._gateway.fetch_ohlcv("US.SPY", KLType.K_5M,  limit=100)

        except (BrokerGatewayError, RateLimitError) as exc:
            logger.warning("Data fetch failed: %s", exc)
            self._engine.dispatch(Event(EventType.ERROR, str(exc)))
            return

        # Re-evaluate daily bias on each candle (cheap, uses cached 4h/1h data)
        bias = get_daily_bias(self._df_4h, self._df_1h)

        # Reset engine if bias flips
        if self._smc.step != SetupStep.IDLE and bias.direction != self._smc._setup.direction:
            logger.info("Bias flipped to %s — resetting setup.", bias.direction.value)
            self._smc.reset()

        completed = self._smc.advance(
            bias=bias,
            levels=self._levels,
            df_5m=df_5m,
            df_1m=df_1m,
        )

        if completed:
            self._engine.dispatch(Event(EventType.SETUP_READY, completed))

    # ── Entry handler ──────────────────────────────────────────────────

    def _on_setup_ready(self, event: Event) -> None:
        setup: TradeSetup = event.payload

        try:
            spy_q, opt_q = self._gateway.fetch_spy_quotes(self._cfg.spy_options_contract)
            info: AccountInfo = self._gateway.fetch_account_info()
        except BrokerGatewayError as exc:
            logger.error("Cannot fetch quotes/account: %s", exc)
            self._smc.reset()
            return

        account = AccountState(
            cash=info.cash,
            total_assets=info.total_assets,
            buying_power=info.power,
            realized_pnl_today=self._db.realized_pnl_today(date.today().isoformat()),
        )
        option_price = opt_q.last_price

        approved, reason, contracts = self._risk.approve(setup, account, option_price)
        self._db.log_setup(setup, approved, None if approved else reason, account)

        if not approved:
            logger.warning("Setup rejected: %s", reason)
            self._smc.reset()
            return

        option_type = OptionType.CALL if setup.direction == Direction.BULLISH else OptionType.PUT
        logger.info("Entering: %s %s x%d @ %.2f  stop=%.2f  target=%.2f",
                    option_type.value, self._cfg.spy_options_contract, contracts,
                    option_price, setup.stop_loss or 0, setup.target_price or 0)

        try:
            order = self._place_order(
                symbol=self._cfg.spy_options_contract,
                trd_side=TrdSide.BUY,
                quantity=contracts,
                price=option_price,
                option_type=option_type,
            )
        except BrokerGatewayError as exc:
            logger.error("Order failed: %s", exc)
            self._smc.reset()
            return

        self._db.log_execution(order)
        rr = abs((setup.target_price or 0) - (setup.entry_price or 0)) / \
             max(abs((setup.entry_price or 0) - (setup.stop_loss or 0)), 0.01)
        logger.info("ENTERED — R:R %.2f | order_id=%s", rr, order.order_id)

    # ── Helpers ────────────────────────────────────────────────────────

    def _refresh_levels(self, today: str) -> None:
        """Fetch 4h, 1h, daily, and 30-min candles; rebuild key levels."""
        logger.info("Refreshing HTF data and key levels for %s…", today)
        self._df_4h    = self._gateway.fetch_ohlcv("US.SPY", KLType.K_4H,   limit=self._cfg.candle_limit)
        self._df_1h    = self._gateway.fetch_ohlcv("US.SPY", KLType.K_60M,  limit=self._cfg.candle_limit)
        df_daily       = self._gateway.fetch_ohlcv("US.SPY", KLType.K_DAY,  limit=5)
        df_30m         = self._gateway.fetch_ohlcv("US.SPY", KLType.K_30M,  limit=self._cfg.candle_limit)

        self._levels         = build_key_levels(df_daily, df_30m)
        self._last_level_date = today
        self._smc.reset()   # new day = fresh setup
        logger.info("Key levels rebuilt: %s",
                    ", ".join(f"{l.label}={l.price:.2f}" for l in self._levels))

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
            price=price, qty=quantity, code=symbol,
            trd_side=trd_side, order_type=OrderType.NORMAL,
            trd_env=self._cfg.trd_env,
        )
        if ret != RET_OK:
            raise BrokerGatewayError(f"place_order failed: {data}")

        row        = data.iloc[0]
        order_id   = str(row.get("order_id", f"local-{int(time.time())}"))
        fill_price = float(row.get("dealt_avg_price", price))

        return ExecutedOrder(
            order_id=order_id, symbol=symbol, option_type=option_type,
            quantity=quantity, fill_price=fill_price,
            commission=float(row.get("commission", 0.0)), is_entry=True,
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
        max_cycles=0,
    )

    engine = EventEngine()
    bot    = TradingBot(engine, cfg)
    bot.start(max_cycles=cfg.max_cycles)
