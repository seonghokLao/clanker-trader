"""
strategy.py — Trend detection, key level building, and the 3-step setup engine.

Flow:
  1. Daily bias  — 4h AND 1h must both show HH/HL (bullish) or LH/LL (bearish)
  2. Key levels  — previous day H/L (daily TF) + Asia/London H/L (30-min TF)
  3. Liq sweep   — wick through a key level, candle closes back the other side
  4. 5-min BOS   — after the sweep, 5-min structure breaks in bias direction
  5. 1-min entry — FVG (or inverse FVG) with engulfing candle in bias direction
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

import pandas as pd

from models import (
    BOS, FVG, Direction, DailyBias, KeyLevel,
    LiquiditySweep, SetupStep, SwingPoint, TradeSetup,
)

logger = logging.getLogger(__name__)

# Session hour boundaries in UTC (approximate; covers EST and EDT)
# Asia:   8pm–2am ET  →  00:00–07:00 UTC
# London: 3am–8am ET  →  08:00–13:00 UTC
_ASIA_UTC_START   = 0
_ASIA_UTC_END     = 7
_LONDON_UTC_START = 8
_LONDON_UTC_END   = 13

_SWING_LOOKBACK = 3   # bars each side for swing detection


# ──────────────────────────────────────────────────────────────────────────────
# Trend detection
# ──────────────────────────────────────────────────────────────────────────────

def detect_swings(df: pd.DataFrame, lookback: int = _SWING_LOOKBACK) -> list[SwingPoint]:
    """
    Vectorized swing high/low detection.
    A swing high: highest high in a window of (2*lookback+1) bars centred on that bar.
    A swing low:  lowest  low  in the same window.
    """
    highs = df["high"]
    lows  = df["low"]
    w     = 2 * lookback + 1

    swing_high_mask = highs == highs.rolling(w, center=True).max()
    swing_low_mask  = lows  == lows.rolling(w, center=True).min()

    swings: list[SwingPoint] = []
    for idx in df.index[swing_high_mask.fillna(False)]:
        swings.append(SwingPoint(
            price=float(highs[idx]),
            direction=Direction.BEARISH,
            timestamp=_ts(df.at[idx, "time"]),
        ))
    for idx in df.index[swing_low_mask.fillna(False)]:
        swings.append(SwingPoint(
            price=float(lows[idx]),
            direction=Direction.BULLISH,
            timestamp=_ts(df.at[idx, "time"]),
        ))
    return sorted(swings, key=lambda s: s.timestamp)


def detect_trend(df: pd.DataFrame, lookback: int = _SWING_LOOKBACK) -> Direction:
    """
    Determine trend from the last two swing highs and two swing lows.
    Bullish: latest swing high > previous swing high AND latest swing low > previous swing low.
    Bearish: latest swing high < previous swing high AND latest swing low < previous swing low.
    """
    swings = detect_swings(df, lookback)

    highs = [s.price for s in swings if s.direction == Direction.BEARISH]
    lows  = [s.price for s in swings if s.direction == Direction.BULLISH]

    if len(highs) < 2 or len(lows) < 2:
        return Direction.NEUTRAL

    hh = highs[-1] > highs[-2]   # higher high
    hl = lows[-1]  > lows[-2]    # higher low
    lh = highs[-1] < highs[-2]   # lower high
    ll = lows[-1]  < lows[-2]    # lower low

    if hh and hl:
        return Direction.BULLISH
    if lh and ll:
        return Direction.BEARISH
    return Direction.NEUTRAL


def get_daily_bias(df_4h: pd.DataFrame, df_1h: pd.DataFrame) -> DailyBias:
    """
    Both 4h and 1h must agree for a confirmed bias.
    Uses the last 20 bars of each timeframe.
    """
    trend_4h = detect_trend(df_4h.tail(20))
    trend_1h = detect_trend(df_1h.tail(20))

    if trend_4h == trend_1h and trend_4h != Direction.NEUTRAL:
        direction = trend_4h
        reason = f"4h {trend_4h.value} + 1h {trend_1h.value}"
    else:
        direction = Direction.NEUTRAL
        reason = f"4h {trend_4h.value} vs 1h {trend_1h.value} — no confluence"

    logger.info("Daily bias: %s (%s)", direction.value, reason)
    return DailyBias(direction=direction, reason=reason)


# ──────────────────────────────────────────────────────────────────────────────
# Key level building
# ──────────────────────────────────────────────────────────────────────────────

def build_key_levels(
    df_daily: pd.DataFrame,
    df_30m: pd.DataFrame,
) -> list[KeyLevel]:
    """
    Mark the levels we watch for liquidity sweeps:
      - Previous day high and low  (from daily candles)
      - Asian session high and low (from 30-min candles, UTC 00–07)
      - London session high and low(from 30-min candles, UTC 08–13)
    """
    levels: list[KeyLevel] = []
    now = datetime.now(tz=timezone.utc)

    # ── Previous day H/L ──────────────────────────────────────────────
    if len(df_daily) >= 2:
        prev = df_daily.iloc[-2]
        levels += [
            KeyLevel(price=float(prev["high"]), label="PREV_DAY_HIGH",
                     direction=Direction.BEARISH, timestamp=_ts(prev["time"])),
            KeyLevel(price=float(prev["low"]),  label="PREV_DAY_LOW",
                     direction=Direction.BULLISH, timestamp=_ts(prev["time"])),
        ]

    # ── Session H/L from 30-min data ──────────────────────────────────
    if not df_30m.empty:
        df_30m = df_30m.copy()
        df_30m["_hour"] = pd.to_datetime(df_30m["time"], utc=True).dt.hour

        asia   = df_30m[df_30m["_hour"].between(_ASIA_UTC_START,   _ASIA_UTC_END   - 1)]
        london = df_30m[df_30m["_hour"].between(_LONDON_UTC_START, _LONDON_UTC_END - 1)]

        for name, subset in [("ASIA", asia), ("LONDON", london)]:
            if subset.empty:
                continue
            levels += [
                KeyLevel(price=float(subset["high"].max()), label=f"{name}_HIGH",
                         direction=Direction.BEARISH, timestamp=now),
                KeyLevel(price=float(subset["low"].min()),  label=f"{name}_LOW",
                         direction=Direction.BULLISH, timestamp=now),
            ]

    for lv in levels:
        logger.debug("Key level: %s @ %.2f", lv.label, lv.price)

    return levels


# ──────────────────────────────────────────────────────────────────────────────
# Step 1 — Liquidity sweep detection
# ──────────────────────────────────────────────────────────────────────────────

def detect_liquidity_sweep(
    df: pd.DataFrame,
    levels: list[KeyLevel],
) -> Optional[LiquiditySweep]:
    """
    Check the last candle for a sweep of any unswept key level.
    Bullish sweep: low wicks BELOW a BULLISH level (sell-stops taken), close ABOVE it.
    Bearish sweep: high wicks ABOVE a BEARISH level (buy-stops taken), close BELOW it.
    Returns the most recent sweep found, or None.
    """
    last = df.iloc[-1]
    low   = float(last["low"])
    high  = float(last["high"])
    close = float(last["close"])
    ts    = _ts(last["time"])

    for level in levels:
        if level.swept:
            continue

        # Bullish sweep: wick below sell-stop pool, closes back above
        if level.direction == Direction.BULLISH and low < level.price < close:
            level.swept = True
            sweep = LiquiditySweep(
                level=level, sweep_low=low, close_price=close,
                direction=Direction.BULLISH, timestamp=ts,
            )
            logger.info("Bullish liq sweep: %s @ %.2f (wick=%.2f, close=%.2f)",
                        level.label, level.price, low, close)
            return sweep

        # Bearish sweep: wick above buy-stop pool, closes back below
        if level.direction == Direction.BEARISH and close < level.price < high:
            level.swept = True
            sweep = LiquiditySweep(
                level=level, sweep_low=high, close_price=close,
                direction=Direction.BEARISH, timestamp=ts,
            )
            logger.info("Bearish liq sweep: %s @ %.2f (wick=%.2f, close=%.2f)",
                        level.label, level.price, high, close)
            return sweep

    return None


# ──────────────────────────────────────────────────────────────────────────────
# Step 2 — 5-min Break of Structure
# ──────────────────────────────────────────────────────────────────────────────

def detect_bos_5m(df: pd.DataFrame, bias: Direction) -> Optional[BOS]:
    """
    After a liquidity sweep, look for a 5-min BOS in the bias direction.
    Bullish BOS: a candle closes above the most recent 5-min swing high.
    Bearish BOS: a candle closes below the most recent 5-min swing low.
    """
    swings = detect_swings(df)
    closes = df["close"]
    times  = df["time"]

    if bias == Direction.BULLISH:
        # Find the most recent swing high that hasn't been broken
        swing_highs = [s for s in swings if s.direction == Direction.BEARISH and not s.broken]
        if not swing_highs:
            return None
        level = swing_highs[-1]
        # Check if any close is above it (after the swing formed)
        mask = (closes > level.price) & (pd.to_datetime(df["time"], utc=True) > level.timestamp)
        if mask.any():
            first = int(mask.values.argmax())
            level.broken = True
            logger.info("5-min BOS BULLISH: broke %.2f", level.price)
            return BOS(direction=Direction.BULLISH, broken_level=level.price,
                       timestamp=_ts(times.iloc[first]))

    else:  # BEARISH
        swing_lows = [s for s in swings if s.direction == Direction.BULLISH and not s.broken]
        if not swing_lows:
            return None
        level = swing_lows[-1]
        mask = (closes < level.price) & (pd.to_datetime(df["time"], utc=True) > level.timestamp)
        if mask.any():
            first = int(mask.values.argmax())
            level.broken = True
            logger.info("5-min BOS BEARISH: broke %.2f", level.price)
            return BOS(direction=Direction.BEARISH, broken_level=level.price,
                       timestamp=_ts(times.iloc[first]))

    return None


# ──────────────────────────────────────────────────────────────────────────────
# Step 3 — 1-min FVG + engulfing entry
# ──────────────────────────────────────────────────────────────────────────────

def detect_fvgs(df: pd.DataFrame) -> list[FVG]:
    """
    3-candle FVG pattern (vectorized):
      Bullish: candle[i-1].high < candle[i+1].low   — gap = imbalance buyers must fill
      Bearish: candle[i-1].low  > candle[i+1].high  — gap = imbalance sellers must fill
    """
    prev_high = df["high"].shift(1)
    prev_low  = df["low"].shift(1)
    next_low  = df["low"].shift(-1)
    next_high = df["high"].shift(-1)

    bull_mask = prev_high < next_low
    bear_mask = prev_low  > next_high

    fvgs: list[FVG] = []
    for idx in df.index[bull_mask.fillna(False)]:
        fvgs.append(FVG(
            top=float(next_low[idx]), bottom=float(prev_high[idx]),
            direction=Direction.BULLISH, timestamp=_ts(df.at[idx, "time"]),
        ))
    for idx in df.index[bear_mask.fillna(False)]:
        fvgs.append(FVG(
            top=float(prev_low[idx]), bottom=float(next_high[idx]),
            direction=Direction.BEARISH, timestamp=_ts(df.at[idx, "time"]),
        ))
    return sorted(fvgs, key=lambda f: f.timestamp)


def detect_fvg_entry(
    df_1m: pd.DataFrame,
    bias: Direction,
) -> Optional[tuple[FVG, float]]:
    """
    Look for a 1-min FVG entry signal:
      - A FVG aligned with bias (or an inverse FVG) where price is currently reacting.
      - Confirmed by an engulfing candle in the bias direction on the last closed bar.

    Returns (triggering FVG, entry_price) or None.
    An inverse FVG means price traded through a counter-bias FVG and it now
    acts as support/resistance flipped in the bias direction.
    """
    if len(df_1m) < 3:
        return None

    fvgs         = detect_fvgs(df_1m)
    current_price = float(df_1m["close"].iloc[-1])

    # Find the most relevant untouched FVG:
    # 1. Bias-aligned FVG price is currently inside (rejection)
    # 2. Counter-bias FVG price is currently inside (inverse/flip)
    candidate: Optional[FVG] = None
    for fvg in reversed(fvgs):
        if fvg.filled:
            continue
        if fvg.contains(current_price):
            if fvg.direction == bias:
                candidate = fvg      # direct FVG rejection
                break
            else:
                candidate = fvg      # inverse FVG
                break

    if candidate is None:
        return None

    # Confirm with engulfing candle in bias direction
    if not _is_engulfing(df_1m, bias):
        return None

    entry = current_price
    logger.info(
        "1-min FVG entry: %s FVG [%.2f–%.2f] + %s engulf @ %.2f",
        candidate.direction.value, candidate.bottom, candidate.top,
        bias.value, entry,
    )
    return candidate, entry


def _is_engulfing(df: pd.DataFrame, bias: Direction) -> bool:
    """
    True if the last closed candle is a full-body engulfing in the bias direction.
    Bullish engulf: current close > previous open AND current open < previous close
                   AND current candle is green.
    Bearish engulf: mirror.
    """
    if len(df) < 2:
        return False

    curr = df.iloc[-1]
    prev = df.iloc[-2]

    curr_open  = float(curr["open"])
    curr_close = float(curr["close"])
    prev_open  = float(prev["open"])
    prev_close = float(prev["close"])

    if bias == Direction.BULLISH:
        return (curr_close > curr_open              # green candle
                and curr_close > prev_open          # closes above prev open
                and curr_open  < prev_close)        # opens below prev close

    else:  # BEARISH
        return (curr_close < curr_open              # red candle
                and curr_close < prev_open          # closes below prev open
                and curr_open  > prev_close)        # opens above prev close


# ──────────────────────────────────────────────────────────────────────────────
# Stop loss and target helpers
# ──────────────────────────────────────────────────────────────────────────────

def calc_stop(df_1m: pd.DataFrame, bias: Direction, buffer: float = 0.10) -> float:
    """Stop = just beyond the most recent 1-min swing extreme."""
    swings = detect_swings(df_1m, lookback=2)
    if bias == Direction.BULLISH:
        lows = [s.price for s in swings if s.direction == Direction.BULLISH]
        base = min(lows) if lows else float(df_1m["low"].iloc[-3:].min())
        return round(base - buffer, 2)
    else:
        highs = [s.price for s in swings if s.direction == Direction.BEARISH]
        base  = max(highs) if highs else float(df_1m["high"].iloc[-3:].max())
        return round(base + buffer, 2)


def calc_target(
    current_price: float,
    bias: Direction,
    levels: list[KeyLevel],
    stop: float,
    min_rr: float = 2.0,
) -> float:
    """
    Target = nearest opposing key level in the bias direction beyond min_rr.
    Falls back to a calculated R:R multiple if no level qualifies.
    """
    risk = abs(current_price - stop)

    if bias == Direction.BULLISH:
        # Target a BEARISH level (buy-stops) above current price
        candidates = [
            lv for lv in levels
            if lv.direction == Direction.BEARISH
            and lv.price > current_price + risk * min_rr
            and not lv.swept
        ]
        if candidates:
            return round(min(candidates, key=lambda l: l.price).price, 2)
        return round(current_price + risk * min_rr, 2)
    else:
        # Target a BULLISH level (sell-stops) below current price
        candidates = [
            lv for lv in levels
            if lv.direction == Direction.BULLISH
            and lv.price < current_price - risk * min_rr
            and not lv.swept
        ]
        if candidates:
            return round(max(candidates, key=lambda l: l.price).price, 2)
        return round(current_price - risk * min_rr, 2)


# ──────────────────────────────────────────────────────────────────────────────
# Stateful setup engine
# ──────────────────────────────────────────────────────────────────────────────

class SetupEngine:
    """
    Advances through SetupStep states on each candle tick.
    Call advance() every candle; it returns a completed TradeSetup on Step 3.
    Call reset() after entry or invalidation.
    """

    def __init__(self) -> None:
        self._setup: Optional[TradeSetup] = None

    @property
    def step(self) -> SetupStep:
        return self._setup.step if self._setup else SetupStep.IDLE

    def reset(self) -> None:
        logger.info("SetupEngine reset.")
        self._setup = None

    def advance(
        self,
        bias: DailyBias,
        levels: list[KeyLevel],
        df_5m: pd.DataFrame,
        df_1m: pd.DataFrame,
    ) -> Optional[TradeSetup]:
        """
        Call on every 1-min candle close.
        Returns completed TradeSetup when the 1-min FVG entry triggers, else None.
        """
        current_price = float(df_1m["close"].iloc[-1])

        # ── Step 0 → 1: Confirm bias ──────────────────────────────────
        if self.step == SetupStep.IDLE:
            if bias.direction == Direction.NEUTRAL:
                return None
            self._setup = TradeSetup(
                direction=bias.direction,
                step=SetupStep.BIAS_CONFIRMED,
                bias=bias,
            )
            logger.info("Step 1: Bias confirmed (%s)", bias.reason)
            return None

        # ── Step 1 → 2: Watch for liquidity sweep ─────────────────────
        if self.step == SetupStep.BIAS_CONFIRMED:
            sweep = detect_liquidity_sweep(df_5m, levels)
            if sweep is None:
                return None
            # Sweep direction must match bias (bullish sweep → expect bullish move)
            if sweep.direction != self._setup.direction:
                logger.debug("Sweep direction mismatch — ignoring.")
                return None
            self._setup.sweep = sweep
            self._setup.step  = SetupStep.SWEEP_FOUND
            logger.info("Step 2: Liq sweep on %s", sweep.level.label)
            return None

        # ── Step 2 → 3: 5-min BOS in bias direction ───────────────────
        if self.step == SetupStep.SWEEP_FOUND:
            bos = detect_bos_5m(df_5m, self._setup.direction)
            if bos is None:
                return None
            self._setup.bos  = bos
            self._setup.step = SetupStep.BOS_CONFIRMED
            logger.info("Step 3: 5-min BOS confirmed (broke %.2f)", bos.broken_level)
            return None

        # ── Step 3: 1-min FVG + engulfing entry ───────────────────────
        if self.step == SetupStep.BOS_CONFIRMED:
            result = detect_fvg_entry(df_1m, self._setup.direction)
            if result is None:
                return None

            _fvg, entry = result
            stop   = calc_stop(df_1m, self._setup.direction)
            target = calc_target(entry, self._setup.direction, levels, stop)

            self._setup.entry_price  = entry
            self._setup.stop_loss    = stop
            self._setup.target_price = target
            self._setup.step         = SetupStep.ENTERED

            logger.info(
                "Entry signal: %s @ %.2f  stop=%.2f  target=%.2f  R:R=%.2f",
                self._setup.direction.value, entry, stop, target,
                abs(target - entry) / max(abs(entry - stop), 0.01),
            )
            return self._setup

        return None


# ──────────────────────────────────────────────────────────────────────────────
# Utility
# ──────────────────────────────────────────────────────────────────────────────

def _ts(val) -> datetime:
    if isinstance(val, pd.Timestamp):
        dt = val.to_pydatetime()
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    if isinstance(val, datetime):
        return val if val.tzinfo else val.replace(tzinfo=timezone.utc)
    return datetime.now(tz=timezone.utc)
