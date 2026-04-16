"""
trade_logger.py — SQLite persistence for proposed trades, risk decisions, and
                  executed orders.  Thread-safe via a single serialised connection.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from pathlib import Path

from models import ExecutedOrder, TradeDecision

logger = logging.getLogger(__name__)

_DEFAULT_DB = Path("trades.db")

_DDL = """
CREATE TABLE IF NOT EXISTS proposed_trades (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       TEXT    NOT NULL,
    symbol          TEXT    NOT NULL,
    action          TEXT    NOT NULL,
    quantity        INTEGER NOT NULL,
    limit_price     REAL,
    note            TEXT,
    approved        INTEGER NOT NULL,          -- 0 / 1
    rejection_reason TEXT,
    account_snapshot TEXT   NOT NULL           -- JSON
);

CREATE TABLE IF NOT EXISTS executed_orders (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    order_id    TEXT    NOT NULL UNIQUE,
    timestamp   TEXT    NOT NULL,
    symbol      TEXT    NOT NULL,
    action      TEXT    NOT NULL,
    quantity    INTEGER NOT NULL,
    fill_price  REAL    NOT NULL,
    commission  REAL    NOT NULL DEFAULT 0.0
);
"""


class TradeLogger:
    """
    Persists trade decisions and executions to a local SQLite database.
    A single connection is reused across calls; a threading.Lock serialises
    concurrent writes from multiple threads.
    """

    def __init__(self, db_path: Path | str = _DEFAULT_DB) -> None:
        self._db_path = Path(db_path)
        self._lock = threading.Lock()
        self._conn = self._open()

    # ------------------------------------------------------------------ #
    # Lifecycle                                                             #
    # ------------------------------------------------------------------ #

    def _open(self) -> sqlite3.Connection:
        logger.info("Opening trade database: %s", self._db_path.resolve())
        conn = sqlite3.connect(
            str(self._db_path),
            check_same_thread=False,   # we manage thread safety ourselves
            isolation_level=None,      # autocommit; we issue explicit transactions
        )
        conn.execute("PRAGMA journal_mode=WAL;")  # concurrent reads during writes
        conn.execute("PRAGMA foreign_keys=ON;")
        conn.executescript(_DDL)
        return conn

    def close(self) -> None:
        with self._lock:
            try:
                self._conn.close()
                logger.info("Trade database closed.")
            except Exception as exc:
                logger.warning("Error closing database: %s", exc)

    def __enter__(self) -> "TradeLogger":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    # ------------------------------------------------------------------ #
    # Writes                                                                #
    # ------------------------------------------------------------------ #

    def log_decision(self, decision: TradeDecision) -> None:
        """Persist a risk-manager decision (approved or rejected)."""
        trade = decision.proposed
        account_json = decision.account_snapshot.model_dump_json()

        sql = """
            INSERT INTO proposed_trades
                (timestamp, symbol, action, quantity, limit_price, note,
                 approved, rejection_reason, account_snapshot)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        params = (
            decision.timestamp.isoformat(),
            trade.symbol,
            trade.action.value,
            trade.quantity,
            trade.limit_price,
            trade.note,
            int(decision.approved),
            None if decision.approved else decision.reason,
            account_json,
        )
        self._execute(sql, params)
        logger.debug(
            "Logged %s decision for %s %d %s.",
            "APPROVED" if decision.approved else "REJECTED",
            trade.action, trade.quantity, trade.symbol,
        )

    def log_execution(self, order: ExecutedOrder) -> None:
        """Persist a filled order returned by the broker."""
        sql = """
            INSERT OR IGNORE INTO executed_orders
                (order_id, timestamp, symbol, action, quantity, fill_price, commission)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """
        params = (
            order.order_id,
            order.timestamp.isoformat(),
            order.symbol,
            order.action.value,
            order.quantity,
            order.fill_price,
            order.commission,
        )
        self._execute(sql, params)
        logger.info(
            "Logged execution order_id=%s %s %d %s @ %.4f.",
            order.order_id, order.action, order.quantity,
            order.symbol, order.fill_price,
        )

    # ------------------------------------------------------------------ #
    # Reads                                                                 #
    # ------------------------------------------------------------------ #

    def realized_pnl_today(self, date_str: str) -> float:
        """
        Compute realized P&L for *date_str* (ISO date, e.g. "2024-06-20")
        from executed_orders.  Sells are treated as closing positions.

        This is a simple approximation: sum of (fill_price * quantity) for
        SELLs minus (fill_price * quantity) for BUYs on the same day.
        """
        sql = """
            SELECT action, SUM(fill_price * quantity) as notional
            FROM executed_orders
            WHERE timestamp LIKE ?
            GROUP BY action
        """
        rows = self._fetchall(sql, (f"{date_str}%",))
        totals = {row[0]: row[1] for row in rows}
        return totals.get("SELL", 0.0) - totals.get("BUY", 0.0)

    def recent_decisions(self, limit: int = 50) -> list[dict]:
        sql = """
            SELECT timestamp, symbol, action, quantity, approved, rejection_reason
            FROM proposed_trades
            ORDER BY id DESC
            LIMIT ?
        """
        rows = self._fetchall(sql, (limit,))
        keys = ["timestamp", "symbol", "action", "quantity", "approved", "rejection_reason"]
        return [dict(zip(keys, row)) for row in rows]

    def recent_executions(self, limit: int = 50) -> list[dict]:
        sql = """
            SELECT order_id, timestamp, symbol, action, quantity, fill_price, commission
            FROM executed_orders
            ORDER BY id DESC
            LIMIT ?
        """
        rows = self._fetchall(sql, (limit,))
        keys = ["order_id", "timestamp", "symbol", "action", "quantity", "fill_price", "commission"]
        return [dict(zip(keys, row)) for row in rows]

    # ------------------------------------------------------------------ #
    # Internal helpers                                                      #
    # ------------------------------------------------------------------ #

    def _execute(self, sql: str, params: tuple) -> None:
        with self._lock:
            try:
                self._conn.execute("BEGIN")
                self._conn.execute(sql, params)
                self._conn.execute("COMMIT")
            except sqlite3.Error as exc:
                self._conn.execute("ROLLBACK")
                logger.error("DB write failed: %s | SQL: %s", exc, sql.strip())
                raise

    def _fetchall(self, sql: str, params: tuple) -> list[tuple]:
        with self._lock:
            try:
                cur = self._conn.execute(sql, params)
                return cur.fetchall()
            except sqlite3.Error as exc:
                logger.error("DB read failed: %s | SQL: %s", exc, sql.strip())
                raise
