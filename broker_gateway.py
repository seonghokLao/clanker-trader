"""
broker_gateway.py — Moomoo OpenD connection and market data interface.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import moomoo as ft
from moomoo import (
    OpenQuoteContext,
    OpenSecTradeContext,
    RET_OK,
    SubType,
    TrdEnv,
    TrdMarket,
)

logger = logging.getLogger(__name__)

_OPEND_HOST = "127.0.0.1"
_OPEND_PORT = 11111

# Moomoo rate-limit windows (requests / 30 s)
_RATE_LIMIT_QUOTE = 60
_RATE_LIMIT_TRADE = 10

_MAX_RETRIES = 3
_RETRY_BACKOFF = 2.0  # seconds, doubles each attempt


@dataclass
class AccountInfo:
    power: float
    total_assets: float
    cash: float
    market_val: float
    risk_status: str
    margin_call: float = 0.0
    raw: dict = field(default_factory=dict)


@dataclass
class QuoteData:
    code: str
    last_price: float
    open_price: float
    high_price: float
    low_price: float
    prev_close_price: float
    volume: int
    turnover: float
    raw: dict = field(default_factory=dict)


class BrokerGatewayError(Exception):
    """Raised for non-retryable gateway errors."""


class RateLimitError(BrokerGatewayError):
    """Raised when Moomoo rate limit is detected."""


def _retry(fn, retries: int = _MAX_RETRIES, backoff: float = _RETRY_BACKOFF):
    """Call *fn* up to *retries* times, with exponential back-off on failure."""
    delay = backoff
    for attempt in range(1, retries + 1):
        ret, data = fn()
        if ret == RET_OK:
            return data
        # Detect rate-limit errors by message content
        if isinstance(data, str) and "limit" in data.lower():
            logger.warning("Rate limit hit (attempt %d/%d): %s", attempt, retries, data)
            raise RateLimitError(data)
        if attempt < retries:
            logger.warning(
                "API call failed (attempt %d/%d): %s — retrying in %.1fs",
                attempt, retries, data, delay,
            )
            time.sleep(delay)
            delay *= 2
        else:
            raise BrokerGatewayError(f"API call failed after {retries} attempts: {data}")


class BrokerGateway:
    """
    Thin wrapper around Moomoo OpenD for quote and trade operations.

    Usage:
        with BrokerGateway(trade_password="secret") as gw:
            info = gw.fetch_account_info()
            spy_quote = gw.fetch_quotes(["US.SPY"])
    """

    def __init__(
        self,
        host: str = _OPEND_HOST,
        port: int = _OPEND_PORT,
        trade_password: Optional[str] = None,
        trade_password_md5: Optional[str] = None,
        trd_env: TrdEnv = TrdEnv.REAL,
        connection_timeout: float = 10.0,
    ) -> None:
        self._host = host
        self._port = port
        self._trade_password = trade_password
        self._trade_password_md5 = trade_password_md5
        self._trd_env = trd_env
        self._connection_timeout = connection_timeout

        self._quote_ctx: Optional[OpenQuoteContext] = None
        self._trade_ctx: Optional[OpenSecTradeContext] = None

    # ------------------------------------------------------------------ #
    # Context manager                                                       #
    # ------------------------------------------------------------------ #

    def __enter__(self) -> "BrokerGateway":
        self.connect()
        return self

    def __exit__(self, *_) -> None:
        self.disconnect()

    # ------------------------------------------------------------------ #
    # Connection lifecycle                                                  #
    # ------------------------------------------------------------------ #

    def connect(self) -> None:
        """Open quote and trade contexts, then unlock trading."""
        logger.info("Connecting to Moomoo OpenD at %s:%d", self._host, self._port)
        try:
            self._quote_ctx = OpenQuoteContext(host=self._host, port=self._port)
            self._trade_ctx = OpenSecTradeContext(
                filter_trdmarket=TrdMarket.US,
                host=self._host,
                port=self._port,
            )
        except Exception as exc:
            raise BrokerGatewayError(f"Failed to open OpenD contexts: {exc}") from exc

        logger.info("OpenD contexts opened successfully.")
        self._unlock_trade()

    def disconnect(self) -> None:
        """Close both contexts gracefully."""
        for ctx_name, ctx in [("quote", self._quote_ctx), ("trade", self._trade_ctx)]:
            if ctx is not None:
                try:
                    ctx.close()
                    logger.info("Closed %s context.", ctx_name)
                except Exception as exc:
                    logger.warning("Error closing %s context: %s", ctx_name, exc)
        self._quote_ctx = None
        self._trade_ctx = None

    def _unlock_trade(self) -> None:
        if self._trd_env == TrdEnv.SIMULATE:
            logger.info("Paper trading — unlock not required.")
            return
        if not (self._trade_password or self._trade_password_md5):
            logger.warning("No trade password provided; skipping unlock.")
            return

        logger.info("Unlocking trading account…")
        ret, msg = self._trade_ctx.unlock_trade(
            password=self._trade_password,
            password_md5=self._trade_password_md5,
            is_unlock=True,
        )
        if ret != RET_OK:
            raise BrokerGatewayError(f"Failed to unlock trade account: {msg}")
        logger.info("Trade account unlocked.")

    # ------------------------------------------------------------------ #
    # Guards                                                                #
    # ------------------------------------------------------------------ #

    def _require_quote_ctx(self) -> OpenQuoteContext:
        if self._quote_ctx is None:
            raise BrokerGatewayError("Quote context is not open. Call connect() first.")
        return self._quote_ctx

    def _require_trade_ctx(self) -> OpenSecTradeContext:
        if self._trade_ctx is None:
            raise BrokerGatewayError("Trade context is not open. Call connect() first.")
        return self._trade_ctx

    # ------------------------------------------------------------------ #
    # Account info                                                          #
    # ------------------------------------------------------------------ #

    def fetch_account_info(self, acc_index: int = 0) -> AccountInfo:
        """Return balance and margin data for the specified account index."""
        ctx = self._require_trade_ctx()
        logger.debug("Fetching account info (acc_index=%d)…", acc_index)

        def _call():
            return ctx.accinfo_query(
                trd_env=self._trd_env,
                acc_index=acc_index,
                refresh_cache=True,
            )

        df = _retry(_call)
        row = df.iloc[0].to_dict()

        info = AccountInfo(
            power=float(row.get("power", 0.0)),
            total_assets=float(row.get("total_assets", 0.0)),
            cash=float(row.get("cash", 0.0)),
            market_val=float(row.get("market_val", 0.0)),
            risk_status=str(row.get("risk_status", "UNKNOWN")),
            margin_call=float(row.get("margin_call_margin", 0.0)),
            raw=row,
        )
        logger.info(
            "Account info — cash: %.2f, total_assets: %.2f, power: %.2f, risk: %s",
            info.cash, info.total_assets, info.power, info.risk_status,
        )
        return info

    # ------------------------------------------------------------------ #
    # Real-time Level 1 quotes                                             #
    # ------------------------------------------------------------------ #

    def fetch_quotes(self, codes: list[str]) -> list[QuoteData]:
        """
        Return Level 1 snapshot quotes for *codes*.

        codes — list of Moomoo security codes, e.g. ["US.SPY", "US.SPY240620C00530000"]
        Securities are subscribed automatically before querying.
        """
        ctx = self._require_quote_ctx()
        logger.debug("Subscribing and fetching quotes for: %s", codes)

        # Subscribe (idempotent; required before get_stock_quote)
        ret_sub, err_sub = ctx.subscribe(codes, [SubType.QUOTE], subscribe_push=False)
        if ret_sub != RET_OK:
            raise BrokerGatewayError(f"Subscription failed for {codes}: {err_sub}")

        def _call():
            return ctx.get_stock_quote(codes)

        df = _retry(_call)
        results: list[QuoteData] = []
        for _, row in df.iterrows():
            row_dict = row.to_dict()
            quote = QuoteData(
                code=str(row_dict.get("code", "")),
                last_price=float(row_dict.get("last_price", 0.0)),
                open_price=float(row_dict.get("open_price", 0.0)),
                high_price=float(row_dict.get("high_price", 0.0)),
                low_price=float(row_dict.get("low_price", 0.0)),
                prev_close_price=float(row_dict.get("prev_close_price", 0.0)),
                volume=int(row_dict.get("volume", 0)),
                turnover=float(row_dict.get("turnover", 0.0)),
                raw=row_dict,
            )
            logger.info(
                "Quote [%s] last=%.4f open=%.4f high=%.4f low=%.4f vol=%d",
                quote.code, quote.last_price, quote.open_price,
                quote.high_price, quote.low_price, quote.volume,
            )
            results.append(quote)
        return results

    def fetch_spy_quotes(self, options_contract: str) -> tuple[QuoteData, QuoteData]:
        """
        Convenience method: fetch SPY ETF and one SPY options contract.

        options_contract — full Moomoo code, e.g. "US.SPY240620C00530000"
        Returns (spy_quote, option_quote).
        """
        codes = ["US.SPY", options_contract]
        quotes = self.fetch_quotes(codes)
        by_code = {q.code: q for q in quotes}

        spy = by_code.get("US.SPY")
        opt = by_code.get(options_contract)
        if spy is None or opt is None:
            missing = [c for c in codes if c not in by_code]
            raise BrokerGatewayError(f"Missing quotes for: {missing}")
        return spy, opt
