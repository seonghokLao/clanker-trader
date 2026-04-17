"""
strategy.py — SMC indicator detection and multi-step confluence engine.

Implements the strategy:
  Step 1 : 1h/4h session highs/lows → bias + draw on liquidity
  Step 2 : 5-min confirmation — BOS, iFVG, 79% OTE, or SMT divergence
  Step 2b: if Step 1 formed before market open → wait for 5-min liq sweep first
  Step 3 : 5-min continuation — equal FVG, or SMT (if 2b triggered)
  Step 4 : 1-min entry confirmation — BOS, iFVG, 79% OTE, or SMT
  Enter  : SPY options (call=bullish, put=bearish)
  Target : next opposing liquidity draw
"""
from __future__ import annotations

import logging
from datetime import datetime, time as dtime, timezone
from typing import Optional

import numpy as np
import pandas as pd

from models import (
    BOS, FVG, OTELevel, SMTDivergence, SessionLevels, HTFContext,
    ConfluenceTag, Direction, LiquidityLevel, SessionName,
    StrategyStep, SwingPoint, TimeFrame, TradeSetup,
)

logger = logging.getLogger(__name__)

# ── Session time boundaries (US Eastern, converted offsets used at runtime) ───
# Times are compared against the candle's US/Eastern hour
_SESSION_HOURS: dict[SessionName, tuple[int, int]] = {
    SessionName.ASIA:        (20, 2),    # 8pm–2am ET (crosses midnight)
    SessionName.LONDON:      (3,  8),    # 3am–8am ET
    SessionName.PRE_MARKET:  (4,  9),    # 4am–9:30am ET (approx)
    SessionName.NEW_YORK:    (9, 16),    # 9:30am–4pm ET (use 9 for simplicity)
    SessionName.AFTER_HOURS: (16, 20),
}

_NY_OPEN_HOUR  = 9
_NY_OPEN_MIN   = 30
_SWING_LOOKBACK = 5   # bars each side to confirm a swing high/low


# ──────────────────────────────────────────────────────────────────────────────
# Low-level vectorized indicator functions
# ──────────────────────────────────────────────────────────────────────────────

def detect_swings(df: pd.DataFrame, lookback: int = _SWING_LOOKBACK) -> list[SwingPoint]:
    """
    Identify swing highs and lows using a rolling window comparison.
    A swing high is where the high is the highest within ±lookback bars.
    A swing low  is where the low  is the lowest  within ±lookback bars.
    Vectorized via rolling max/min — no row iteration.
    """
    highs = df["high"]
    lows  = df["low"]

    roll_max = highs.rolling(window=2 * lookback + 1, center=True).max()
    roll_min = lows.rolling(window=2 * lookback + 1, center=True).min()

    is_swing_high = highs == roll_max
    is_swing_low  = lows  == roll_min

    swings: list[SwingPoint] = []
    for idx in df.index[is_swing_high]:
        swings.append(SwingPoint(
            price=float(df.at[idx, "high"]),
            timestamp=_ts(df.at[idx, "time"]),
            direction=Direction.BEARISH,
        ))
    for idx in df.index[is_swing_low]:
        swings.append(SwingPoint(
            price=float(df.at[idx, "low"]),
            timestamp=_ts(df.at[idx, "time"]),
            direction=Direction.BULLISH,
        ))
    swings.sort(key=lambda s: s.timestamp)
    return swings


def detect_bos(df: pd.DataFrame, swings: list[SwingPoint], tf: TimeFrame) -> list[BOS]:
    """
    A BOS occurs when a candle closes beyond a swing high (bullish BOS)
    or swing low (bearish BOS).  Uses vectorized close comparisons.
    """
    if not swings:
        return []

    closes = df["close"].values
    times  = df["time"].values
    bos_list: list[BOS] = []

    swing_highs = [s for s in swings if s.direction == Direction.BEARISH and not s.broken]
    swing_lows  = [s for s in swings if s.direction == Direction.BULLISH and not s.broken]

    for swing in swing_highs:
        mask = closes > swing.price
        if mask.any():
            first_break_idx = int(np.argmax(mask))
            # Only count if the swing happened before the break
            if _ts(times[first_break_idx]) > swing.timestamp:
                swing.broken = True
                bos_list.append(BOS(
                    direction=Direction.BULLISH,
                    level=swing.price,
                    timestamp=_ts(times[first_break_idx]),
                    timeframe=tf,
                ))

    for swing in swing_lows:
        mask = closes < swing.price
        if mask.any():
            first_break_idx = int(np.argmax(mask))
            if _ts(times[first_break_idx]) > swing.timestamp:
                swing.broken = True
                bos_list.append(BOS(
                    direction=Direction.BEARISH,
                    level=swing.price,
                    timestamp=_ts(times[first_break_idx]),
                    timeframe=tf,
                ))

    bos_list.sort(key=lambda b: b.timestamp)
    return bos_list


def detect_fvgs(df: pd.DataFrame, tf: TimeFrame) -> list[FVG]:
    """
    Fair Value Gap — 3-candle pattern:
      Bullish FVG: df[i-1].high < df[i+1].low   (gap between candle i-1 and i+1)
      Bearish FVG: df[i-1].low  > df[i+1].high

    Fully vectorized using shifted series.
    """
    high  = df["high"]
    low   = df["low"]
    times = df["time"]

    # Shift without looping
    prev_high = high.shift(1)
    next_low  = low.shift(-1)
    prev_low  = low.shift(1)
    next_high = high.shift(-1)

    bullish_mask = prev_high < next_low
    bearish_mask = prev_low  > next_high

    fvgs: list[FVG] = []
    for idx in df.index[bullish_mask.fillna(False)]:
        fvgs.append(FVG(
            top=float(next_low[idx]),
            bottom=float(prev_high[idx]),
            direction=Direction.BULLISH,
            timestamp=_ts(times[idx]),
            timeframe=tf,
        ))
    for idx in df.index[bearish_mask.fillna(False)]:
        fvgs.append(FVG(
            top=float(prev_low[idx]),
            bottom=float(next_high[idx]),
            direction=Direction.BEARISH,
            timestamp=_ts(times[idx]),
            timeframe=tf,
        ))

    # Mark equal FVGs: two FVGs whose ranges overlap within 0.1%
    _mark_equal_fvgs(fvgs)
    return sorted(fvgs, key=lambda f: f.timestamp)


def detect_ifvg(
    fvgs: list[FVG], current_price: float, bias: Direction
) -> Optional[FVG]:
    """
    Inverse FVG: a previously bearish FVG that price has returned to from
    below (now support), or a bullish FVG retested from above (resistance).
    Returns the most recent untouched iFVG aligned with bias, or None.
    """
    if bias == Direction.BULLISH:
        # Price retracing into a bearish FVG from below = bullish iFVG
        candidates = [
            f for f in fvgs
            if f.direction == Direction.BEARISH and not f.filled and f.contains(current_price)
        ]
    else:
        candidates = [
            f for f in fvgs
            if f.direction == Direction.BULLISH and not f.filled and f.contains(current_price)
        ]
    return candidates[-1] if candidates else None


def calc_ote(swing_high: float, swing_low: float, direction: Direction) -> OTELevel:
    """
    Optimal Trade Entry zone (62%–79% retracement of the impulse move).
    direction = direction of the *original impulse* (BULLISH = low-to-high move).
    """
    rng = swing_high - swing_low
    return OTELevel(
        level_79=round(swing_high - 0.79 * rng, 4) if direction == Direction.BULLISH
                 else round(swing_low + 0.79 * rng, 4),
        level_62=round(swing_high - 0.62 * rng, 4) if direction == Direction.BULLISH
                 else round(swing_low + 0.62 * rng, 4),
        swing_high=swing_high,
        swing_low=swing_low,
        direction=direction,
    )


def price_in_ote(ote: OTELevel, price: float) -> bool:
    lo = min(ote.level_62, ote.level_79)
    hi = max(ote.level_62, ote.level_79)
    return lo <= price <= hi


def detect_smt(
    spy_df: pd.DataFrame,
    es_df: pd.DataFrame,
    lookback: int = 5,
) -> Optional[SMTDivergence]:
    """
    SMT Divergence: SPY and ES should move together.
    Bullish SMT: SPY prints a new N-bar low but ES does NOT → SPY was swept.
    Bearish SMT: SPY prints a new N-bar high but ES does NOT → SPY was pumped.
    Compares the last `lookback` bars of each aligned DataFrame.
    """
    if len(spy_df) < lookback + 1 or len(es_df) < lookback + 1:
        return None

    spy_recent = spy_df.iloc[-(lookback + 1):]
    es_recent  = es_df.iloc[-(lookback + 1):]

    spy_low  = spy_recent["low"].iloc[-1]
    es_low   = es_recent["low"].iloc[-1]
    spy_high = spy_recent["high"].iloc[-1]
    es_high  = es_recent["high"].iloc[-1]

    spy_prev_low  = spy_recent["low"].iloc[:-1].min()
    es_prev_low   = es_recent["low"].iloc[:-1].min()
    spy_prev_high = spy_recent["high"].iloc[:-1].max()
    es_prev_high  = es_recent["high"].iloc[:-1].max()

    ts = _ts(spy_df["time"].iloc[-1])
    tf = TimeFrame.M5  # caller should specify; defaulting to M5

    # Bullish SMT: SPY makes new low, ES doesn't
    if spy_low < spy_prev_low and es_low >= es_prev_low:
        logger.info("Bullish SMT: SPY low=%.2f < prev=%.2f, ES held at %.2f",
                    spy_low, spy_prev_low, es_low)
        return SMTDivergence(
            direction=Direction.BULLISH,
            spy_extreme=spy_low,
            es_extreme=es_low,
            timestamp=ts,
            timeframe=tf,
        )

    # Bearish SMT: SPY makes new high, ES doesn't
    if spy_high > spy_prev_high and es_high <= es_prev_high:
        logger.info("Bearish SMT: SPY high=%.2f > prev=%.2f, ES held at %.2f",
                    spy_high, spy_prev_high, es_high)
        return SMTDivergence(
            direction=Direction.BEARISH,
            spy_extreme=spy_high,
            es_extreme=es_high,
            timestamp=ts,
            timeframe=tf,
        )

    return None


def build_session_levels(df_h1: pd.DataFrame, today_date: str) -> SessionLevels:
    """
    Build session high/low levels from 1-hour OHLCV data.
    Expects a 'time' column as pd.Timestamp (US Eastern aware or naive).
    """
    if df_h1.empty:
        return SessionLevels(date=today_date)

    def _hour(ts: pd.Timestamp) -> int:
        return ts.hour if hasattr(ts, "hour") else 0

    h = df_h1.copy()
    h["_hour"] = pd.to_datetime(h["time"]).dt.hour

    def _hi_lo(mask: pd.Series) -> tuple[Optional[float], Optional[float]]:
        sub = h[mask]
        if sub.empty:
            return None, None
        return float(sub["high"].max()), float(sub["low"].min())

    # Previous day: anything before today (use last 24h proxy)
    pd_mask = h["_hour"] < 0   # placeholder — caller should pre-filter for date
    prev_hi, prev_lo = None, None

    asia_hi,   asia_lo   = _hi_lo((h["_hour"] >= 20) | (h["_hour"] < 2))
    london_hi, london_lo = _hi_lo((h["_hour"] >= 3)  & (h["_hour"] < 8))
    ny_hi,     ny_lo     = _hi_lo((h["_hour"] >= 9)  & (h["_hour"] < 16))

    return SessionLevels(
        date=today_date,
        asia_high=asia_hi,   asia_low=asia_lo,
        london_high=london_hi, london_low=london_lo,
        ny_high=ny_hi,       ny_low=ny_lo,
    )


def nearest_draw(
    levels: list[LiquidityLevel], price: float, direction: Direction
) -> Optional[LiquidityLevel]:
    """
    Find the nearest unswept liquidity draw in the trade direction.
    Bullish bias → target the nearest BEARISH level above price (buy-stops).
    Bearish bias → target the nearest BULLISH level below price (sell-stops).
    """
    if direction == Direction.BULLISH:
        candidates = [l for l in levels if l.direction == Direction.BEARISH
                      and l.price > price and not l.swept]
        return min(candidates, key=lambda l: l.price) if candidates else None
    else:
        candidates = [l for l in levels if l.direction == Direction.BULLISH
                      and l.price < price and not l.swept]
        return max(candidates, key=lambda l: l.price) if candidates else None


def is_pre_market(ts: datetime) -> bool:
    """True if timestamp is before 09:30 ET (approximate UTC-based check)."""
    # Assuming ts is UTC: NY open ≈ 13:30 UTC (EST) or 14:30 UTC (EDT)
    # Use a simple hour check; for production use pytz/zoneinfo
    et_hour = (ts.hour - 5) % 24   # rough EST offset
    return et_hour < _NY_OPEN_HOUR or (et_hour == _NY_OPEN_HOUR and ts.minute < _NY_OPEN_MIN)


# ──────────────────────────────────────────────────────────────────────────────
# Multi-step Confluence Engine
# ──────────────────────────────────────────────────────────────────────────────

class SMCEngine:
    """
    Stateful multi-step confluence engine implementing the 4-step strategy.

    Call advance() on each new candle set.  When step reaches STEP4_CONF
    it returns a TradeSetup ready for execution.  Call reset() after entry
    or invalidation.
    """

    def __init__(self) -> None:
        self._setup: Optional[TradeSetup] = None

    @property
    def current_step(self) -> StrategyStep:
        return self._setup.step if self._setup else StrategyStep.IDLE

    def reset(self) -> None:
        self._setup = None

    def advance(
        self,
        htf_ctx: HTFContext,
        df_5m_spy: pd.DataFrame,
        df_5m_es: pd.DataFrame,
        df_1m_spy: pd.DataFrame,
        df_1m_es: pd.DataFrame,
        current_price: float,
    ) -> Optional[TradeSetup]:
        """
        Advance the state machine by one tick.
        Returns the completed TradeSetup when Step 4 is confirmed, else None.
        """
        step = self.current_step

        # ── Step 1: HTF bias ───────────────────────────────────────────
        if step == StrategyStep.IDLE:
            if htf_ctx.bias != Direction.NEUTRAL:
                self._setup = TradeSetup(
                    direction=htf_ctx.bias,
                    step=StrategyStep.HTF_CONFIRMED,
                    htf_context=htf_ctx,
                )
                logger.info("Step 1 confirmed: bias=%s, draw=%s @ %.2f",
                            htf_ctx.bias.value, htf_ctx.draw_target.label,
                            htf_ctx.draw_target.price)
            return None

        setup = self._setup
        assert setup is not None

        # ── Step 2: 5-min confluence ───────────────────────────────────
        if step == StrategyStep.HTF_CONFIRMED:
            tag = self._check_5m_confluence(
                df_5m_spy, df_5m_es, current_price, setup.direction
            )
            if tag is None:
                return None

            # 2b: if before market open, require liquidity sweep first
            if is_pre_market(datetime.now(tz=timezone.utc)):
                logger.info("Step 2 met (%s) but pre-market → waiting for liq sweep (2b)", tag.value)
                setup.step2_tag = tag
                setup.step2b_triggered = True
                setup.step = StrategyStep.STEP2B_WAITING
                return None

            logger.info("Step 2 confirmed: %s", tag.value)
            setup.step2_tag = tag
            setup.step = StrategyStep.STEP2_CONF
            return None

        # ── Step 2b: wait for 5-min liquidity sweep ───────────────────
        if step == StrategyStep.STEP2B_WAITING:
            if self._check_liq_sweep(df_5m_spy, setup.direction, htf_ctx.all_levels):
                logger.info("Step 2b: liquidity sweep confirmed")
                setup.step = StrategyStep.STEP2_CONF
            return None

        # ── Step 3: 5-min continuation ────────────────────────────────
        if step == StrategyStep.STEP2_CONF:
            tag = self._check_5m_continuation(
                df_5m_spy, df_5m_es, current_price, setup.direction, setup.step2b_triggered
            )
            if tag is None:
                return None
            logger.info("Step 3 confirmed: %s", tag.value)
            setup.step3_tag = tag
            setup.step = StrategyStep.STEP3_CONF
            return None

        # ── Step 4: 1-min entry confirmation ──────────────────────────
        if step == StrategyStep.STEP3_CONF:
            tag = self._check_1m_confluence(
                df_1m_spy, df_1m_es, current_price, setup.direction
            )
            if tag is None:
                return None

            target = nearest_draw(
                htf_ctx.all_levels, current_price, setup.direction
            )
            stop = self._calc_stop(df_1m_spy, setup.direction)

            setup.step4_tag    = tag
            setup.step         = StrategyStep.STEP4_CONF
            setup.entry_price  = current_price
            setup.target_price = target.price if target else None
            setup.stop_loss    = stop

            logger.info(
                "Step 4 confirmed: %s | entry=%.2f stop=%.2f target=%s",
                tag.value, current_price, stop or 0,
                f"{target.price:.2f} ({target.label})" if target else "—",
            )
            return setup

        return None

    # ── Internal confluence checkers ───────────────────────────────────

    def _check_5m_confluence(
        self,
        df: pd.DataFrame,
        df_es: pd.DataFrame,
        price: float,
        bias: Direction,
    ) -> Optional[ConfluenceTag]:
        """Return the first matching 5-min confluence tag, or None."""
        fvgs   = detect_fvgs(df, TimeFrame.M5)
        swings = detect_swings(df)
        bos    = detect_bos(df, swings, TimeFrame.M5)

        # BOS in bias direction
        if any(b.direction == bias for b in bos[-3:]):
            return ConfluenceTag.BOS

        # iFVG aligned with bias
        if detect_ifvg(fvgs, price, bias):
            return ConfluenceTag.IFVG

        # OTE 79% level
        if swings:
            sh = max((s.price for s in swings if s.direction == Direction.BEARISH), default=None)
            sl = min((s.price for s in swings if s.direction == Direction.BULLISH), default=None)
            if sh and sl:
                ote = calc_ote(sh, sl, bias)
                if price_in_ote(ote, price):
                    return ConfluenceTag.OTE_79

        # SMT divergence
        smt = detect_smt(df, df_es)
        if smt and smt.direction == bias:
            return ConfluenceTag.SMT

        return None

    def _check_liq_sweep(
        self,
        df: pd.DataFrame,
        bias: Direction,
        levels: list[LiquidityLevel],
    ) -> bool:
        """5-min candle swept through a liquidity level then reversed."""
        last = df.iloc[-1]
        last_prev = df.iloc[-2] if len(df) > 1 else last

        for lv in levels:
            if lv.swept:
                continue
            if bias == Direction.BULLISH and lv.direction == Direction.BULLISH:
                # A sell-stop sweep: wick below level then close above
                if float(last["low"]) < lv.price < float(last["close"]):
                    lv.swept = True
                    return True
            if bias == Direction.BEARISH and lv.direction == Direction.BEARISH:
                # A buy-stop sweep: wick above level then close below
                if float(last["high"]) > lv.price > float(last["close"]):
                    lv.swept = True
                    return True
        return False

    def _check_5m_continuation(
        self,
        df: pd.DataFrame,
        df_es: pd.DataFrame,
        price: float,
        bias: Direction,
        step2b_triggered: bool,
    ) -> Optional[ConfluenceTag]:
        """Step 3: equal FVG, or SMT if 2b triggered."""
        fvgs = detect_fvgs(df, TimeFrame.M5)

        # Equal FVG (two overlapping FVGs at same level)
        eq_fvgs = [f for f in fvgs if f.is_equal and f.direction == bias and not f.filled]
        if eq_fvgs and eq_fvgs[-1].contains(price):
            return ConfluenceTag.EQ_FVG

        # If 2b triggered, require SMT confirmation
        if step2b_triggered:
            smt = detect_smt(df, df_es)
            if smt and smt.direction == bias:
                return ConfluenceTag.SMT
            return None  # must be SMT after a 2b

        # Otherwise a regular iFVG also qualifies
        if detect_ifvg(fvgs, price, bias):
            return ConfluenceTag.IFVG

        return None

    def _check_1m_confluence(
        self,
        df_1m: pd.DataFrame,
        df_1m_es: pd.DataFrame,
        price: float,
        bias: Direction,
    ) -> Optional[ConfluenceTag]:
        """Same confluence menu as Step 2 but on the 1-min chart."""
        fvgs   = detect_fvgs(df_1m, TimeFrame.M1)
        swings = detect_swings(df_1m, lookback=3)
        bos    = detect_bos(df_1m, swings, TimeFrame.M1)

        if any(b.direction == bias for b in bos[-3:]):
            return ConfluenceTag.BOS
        if detect_ifvg(fvgs, price, bias):
            return ConfluenceTag.IFVG

        if swings:
            sh = max((s.price for s in swings if s.direction == Direction.BEARISH), default=None)
            sl = min((s.price for s in swings if s.direction == Direction.BULLISH), default=None)
            if sh and sl:
                ote = calc_ote(sh, sl, bias)
                if price_in_ote(ote, price):
                    return ConfluenceTag.OTE_79

        smt = detect_smt(df_1m, df_1m_es)
        if smt and smt.direction == bias:
            return ConfluenceTag.SMT

        return None

    @staticmethod
    def _calc_stop(df_1m: pd.DataFrame, bias: Direction) -> float:
        """Stop = beyond the most recent 1-min swing extreme."""
        swings = detect_swings(df_1m, lookback=3)
        buffer = 0.10  # 10-cent buffer beyond the swing
        if bias == Direction.BULLISH:
            lows = [s.price for s in swings if s.direction == Direction.BULLISH]
            return round(min(lows) - buffer, 2) if lows else round(df_1m["low"].iloc[-3:].min() - buffer, 2)
        else:
            highs = [s.price for s in swings if s.direction == Direction.BEARISH]
            return round(max(highs) + buffer, 2) if highs else round(df_1m["high"].iloc[-3:].max() + buffer, 2)


# ──────────────────────────────────────────────────────────────────────────────
# HTF bias builder
# ──────────────────────────────────────────────────────────────────────────────

def build_htf_context(
    df_1h: pd.DataFrame,
    df_4h: pd.DataFrame,
    current_price: float,
    today_date: str,
) -> HTFContext:
    """
    Determine directional bias and nearest liquidity draw from 1h/4h charts.
    Bias = direction of the most recent BOS on either timeframe.
    """
    swings_1h = detect_swings(df_1h, lookback=3)
    bos_1h    = detect_bos(df_1h, swings_1h, TimeFrame.H1)

    swings_4h = detect_swings(df_4h, lookback=3)
    bos_4h    = detect_bos(df_4h, swings_4h, TimeFrame.H4)

    all_bos = sorted(bos_1h + bos_4h, key=lambda b: b.timestamp)
    bias = all_bos[-1].direction if all_bos else Direction.NEUTRAL

    session_levels = build_session_levels(df_1h, today_date)
    all_levels = session_levels.as_liquidity_levels()

    draw = nearest_draw(all_levels, current_price, bias)
    if draw is None:
        # Fallback: pick the most extreme level in bias direction
        draw = LiquidityLevel(
            price=current_price * (1.01 if bias == Direction.BULLISH else 0.99),
            label="ESTIMATED_DRAW",
            direction=Direction.BEARISH if bias == Direction.BULLISH else Direction.BULLISH,
            timestamp=datetime.now(tz=timezone.utc),
        )

    logger.info("HTF bias=%s  draw=%s @ %.2f", bias.value, draw.label, draw.price)
    return HTFContext(bias=bias, draw_target=draw, all_levels=all_levels,
                     session_levels=session_levels)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _ts(val) -> datetime:
    if isinstance(val, pd.Timestamp):
        dt = val.to_pydatetime()
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    if isinstance(val, datetime):
        return val if val.tzinfo else val.replace(tzinfo=timezone.utc)
    return datetime.now(tz=timezone.utc)


def _mark_equal_fvgs(fvgs: list[FVG], tolerance: float = 0.001) -> None:
    """Mark FVGs whose midpoints are within tolerance of each other as equal."""
    for i, a in enumerate(fvgs):
        for b in fvgs[i + 1:]:
            if a.direction == b.direction:
                spread = abs(a.midpoint - b.midpoint) / max(a.midpoint, 0.01)
                if spread <= tolerance:
                    a.is_equal = True
                    b.is_equal = True
