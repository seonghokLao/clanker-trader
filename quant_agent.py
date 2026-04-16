"""
quant_agent.py — Vectorized technical indicator engine and market context builder.

Expected OHLCV DataFrame columns (case-insensitive after normalisation):
    time_key | open | high | low | close | volume

All indicator calculations are fully vectorized — no Python-level loops over rows.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from models import EmaTrend, MarketContext

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Column name normalisation map (Moomoo OpenD field names → internal names)
# ──────────────────────────────────────────────────────────────────────────────
_COL_MAP = {
    "time_key": "time",
    "open": "open",
    "high": "high",
    "low": "low",
    "close": "close",
    "volume": "volume",
}

_MIN_RSI_PERIODS = 15   # need at least rsi_period+1 rows for a meaningful RSI
_MIN_EMA_PERIODS = 22   # need at least ema_slow+1 rows


class InsufficientDataError(ValueError):
    """Raised when the OHLCV DataFrame is too short for a reliable calculation."""


# ──────────────────────────────────────────────────────────────────────────────
# Module-level vectorized functions
# ──────────────────────────────────────────────────────────────────────────────

def compute_ema(series: pd.Series, period: int) -> pd.Series:
    """
    Exponential Moving Average using pandas ewm (span convention).

    adjust=False matches the standard EMA recurrence:
        EMA_t = α * price_t + (1 − α) * EMA_{t-1},  α = 2 / (span + 1)
    min_periods=period ensures leading values are NaN until enough data exists.
    """
    if len(series) < period:
        raise InsufficientDataError(
            f"Need at least {period} rows for EMA-{period}; got {len(series)}."
        )
    return series.ewm(span=period, adjust=False, min_periods=period).mean()


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    Wilder's RSI — fully vectorized via ewm (com = period - 1).

    Uses the standard two-step approach:
      1. Separate up-moves and down-moves.
      2. Smooth each with Wilder's EMA (equivalent to ewm com=period-1).
      3. RS = avg_gain / avg_loss;  RSI = 100 − 100 / (1 + RS)
    """
    if len(series) < period + 1:
        raise InsufficientDataError(
            f"Need at least {period + 1} rows for RSI-{period}; got {len(series)}."
        )

    delta = series.diff()                         # price change; first element is NaN

    gains = delta.clip(lower=0.0)                 # zero-out losses
    losses = (-delta).clip(lower=0.0)             # zero-out gains, flip sign

    # Wilder's smoothing = ewm with com = period - 1
    avg_gain = gains.ewm(com=period - 1, adjust=False, min_periods=period).mean()
    avg_loss = losses.ewm(com=period - 1, adjust=False, min_periods=period).mean()

    rs = avg_gain / avg_loss.replace(0.0, np.nan)  # avoid divide-by-zero
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(100.0)   # loss=0 means perfectly bullish → RSI=100


# ──────────────────────────────────────────────────────────────────────────────
# Agent class
# ──────────────────────────────────────────────────────────────────────────────

class QuantitativeDataAgent:
    """
    Consumes raw OHLCV candle data and produces technical indicators
    and a structured MarketContext.

    Parameters
    ----------
    ema_fast    : fast EMA period (default 9)
    ema_slow    : slow EMA period (default 21)
    rsi_period  : RSI look-back period (default 14)
    """

    def __init__(
        self,
        ema_fast: int = 9,
        ema_slow: int = 21,
        rsi_period: int = 14,
    ) -> None:
        if ema_fast >= ema_slow:
            raise ValueError(
                f"ema_fast ({ema_fast}) must be less than ema_slow ({ema_slow})."
            )
        self._ema_fast = ema_fast
        self._ema_slow = ema_slow
        self._rsi_period = rsi_period

    # ------------------------------------------------------------------ #
    # Public API                                                            #
    # ------------------------------------------------------------------ #

    def enrich(self, raw: pd.DataFrame, symbol: str = "UNKNOWN") -> pd.DataFrame:
        """
        Return a copy of *raw* with three additional columns appended:
            ema_fast | ema_slow | rsi

        The input DataFrame is never mutated.
        """
        df = self._normalise(raw)
        self._validate_length(df, symbol)

        close = df["close"]
        df = df.copy()
        df[f"ema_{self._ema_fast}"] = compute_ema(close, self._ema_fast)
        df[f"ema_{self._ema_slow}"] = compute_ema(close, self._ema_slow)
        df["rsi"] = compute_rsi(close, self._rsi_period)

        logger.debug(
            "[%s] Enriched %d rows — last close=%.4f  EMA%d=%.4f  EMA%d=%.4f  RSI=%.2f",
            symbol, len(df),
            df["close"].iat[-1],
            self._ema_fast, df[f"ema_{self._ema_fast}"].iat[-1],
            self._ema_slow, df[f"ema_{self._ema_slow}"].iat[-1],
            df["rsi"].iat[-1],
        )
        return df

    def get_market_context(self, raw: pd.DataFrame, symbol: str = "UNKNOWN") -> MarketContext:
        """
        Compute indicators on *raw* and return the latest bar as a MarketContext.

        Raises InsufficientDataError if the DataFrame is too short.
        """
        df = self.enrich(raw, symbol)
        last = df.iloc[-1]

        ema_f: float = last[f"ema_{self._ema_fast}"]
        ema_s: float = last[f"ema_{self._ema_slow}"]

        if ema_f > ema_s:
            trend = EmaTrend.BULLISH
        elif ema_f < ema_s:
            trend = EmaTrend.BEARISH
        else:
            trend = EmaTrend.NEUTRAL

        ts = (
            last["time"].to_pydatetime().replace(tzinfo=timezone.utc)
            if isinstance(last.get("time"), pd.Timestamp)
            else datetime.now(tz=timezone.utc)
        )

        ctx = MarketContext(
            symbol=symbol,
            latest_price=float(last["close"]),
            ema_9=round(float(ema_f), 6),
            ema_21=round(float(ema_s), 6),
            ema_trend=trend,
            rsi=round(float(last["rsi"]), 4),
            timestamp=ts,
        )
        logger.info(
            "[%s] MarketContext — price=%.4f  EMA%d=%.4f  EMA%d=%.4f  "
            "trend=%s  RSI=%.2f (%s)",
            symbol, ctx.latest_price,
            self._ema_fast, ctx.ema_9,
            self._ema_slow, ctx.ema_21,
            ctx.ema_trend.value, ctx.rsi, ctx.rsi_zone,
        )
        return ctx

    # ------------------------------------------------------------------ #
    # Internal helpers                                                      #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _normalise(df: pd.DataFrame) -> pd.DataFrame:
        """Lowercase column names and apply the Moomoo → internal name map."""
        renamed = df.rename(columns=lambda c: c.strip().lower())
        renamed = renamed.rename(columns=_COL_MAP)
        missing = {"close", "volume"} - set(renamed.columns)
        if missing:
            raise ValueError(f"OHLCV DataFrame is missing required columns: {missing}")
        for col in ("open", "high", "low", "close", "volume"):
            if col in renamed.columns:
                renamed[col] = pd.to_numeric(renamed[col], errors="coerce")
        return renamed

    def _validate_length(self, df: pd.DataFrame, symbol: str) -> None:
        min_rows = max(self._ema_slow + 1, self._rsi_period + 1)
        if len(df) < min_rows:
            raise InsufficientDataError(
                f"[{symbol}] Need at least {min_rows} candles; received {len(df)}. "
                "Fetch more history before computing indicators."
            )
