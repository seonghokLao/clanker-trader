"""
models.py — SMC strategy data models.
"""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ── Enums ─────────────────────────────────────────────────────────────────────

class TimeFrame(str, Enum):
    M1  = "M1"
    M5  = "M5"
    H1  = "H1"
    H4  = "H4"

class Direction(str, Enum):
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"

class SessionName(str, Enum):
    ASIA        = "ASIA"
    LONDON      = "LONDON"
    NEW_YORK    = "NEW_YORK"
    PRE_MARKET  = "PRE_MARKET"
    AFTER_HOURS = "AFTER_HOURS"

class StrategyStep(int, Enum):
    IDLE           = 0   # waiting for HTF draw
    HTF_CONFIRMED  = 1   # bias and draw identified
    STEP2_CONF     = 2   # 5-min confluence confirmed
    STEP2B_WAITING = 3   # pre-mkt: waiting for liquidity sweep
    STEP3_CONF     = 4   # 5-min continuation confirmed
    STEP4_CONF     = 5   # 1-min entry confirmation → enter
    ENTERED        = 6
    INVALIDATED    = 7

class OptionType(str, Enum):
    CALL = "CALL"
    PUT  = "PUT"

class ConfluenceTag(str, Enum):
    BOS       = "BOS"
    IFVG      = "IFVG"
    OTE_79    = "OTE_79"
    SMT       = "SMT"
    EQ_FVG    = "EQ_FVG"
    LIQ_SWEEP = "LIQ_SWEEP"


# ── SMC Building Blocks ───────────────────────────────────────────────────────

class SwingPoint(BaseModel):
    price: float
    timestamp: datetime
    direction: Direction        # BULLISH = swing low, BEARISH = swing high
    broken: bool = False


class FVG(BaseModel):
    top: float
    bottom: float
    direction: Direction        # BULLISH = buy-side imbalance, BEARISH = sell-side
    timestamp: datetime
    timeframe: TimeFrame
    filled: bool = False
    is_equal: bool = False      # stacked with another FVG at same level

    @property
    def midpoint(self) -> float:
        return (self.top + self.bottom) / 2

    def contains(self, price: float) -> bool:
        return self.bottom <= price <= self.top


class BOS(BaseModel):
    direction: Direction        # BULLISH = broke above swing high
    level: float
    timestamp: datetime
    timeframe: TimeFrame
    is_choch: bool = False      # Change of Character (first counter-trend BOS)


class LiquidityLevel(BaseModel):
    price: float
    label: str                  # e.g. "NY_HIGH", "PREV_DAY_LOW"
    direction: Direction        # BEARISH = buy-stops resting above, BULLISH = sell-stops below
    session: Optional[SessionName] = None
    swept: bool = False
    timestamp: datetime


class SMTDivergence(BaseModel):
    direction: Direction        # BULLISH = SPY swept lower but ES held (manipulation)
    spy_extreme: float
    es_extreme: float
    timestamp: datetime
    timeframe: TimeFrame


class OTELevel(BaseModel):
    level_79: float             # 79% retracement — primary OTE entry zone
    level_62: float             # 62% retracement — OTE zone start
    swing_high: float
    swing_low: float
    direction: Direction        # direction of the original impulse being retraced


# ── Session / HTF Context ─────────────────────────────────────────────────────

class SessionLevels(BaseModel):
    date: str
    prev_day_high: Optional[float] = None
    prev_day_low:  Optional[float] = None
    asia_high:     Optional[float] = None
    asia_low:      Optional[float] = None
    london_high:   Optional[float] = None
    london_low:    Optional[float] = None
    ny_high:       Optional[float] = None
    ny_low:        Optional[float] = None

    def as_liquidity_levels(self) -> list[LiquidityLevel]:
        now = datetime.now(tz=timezone.utc)
        mapping = [
            ("prev_day_high", "PREV_DAY_HIGH", Direction.BEARISH, None),
            ("prev_day_low",  "PREV_DAY_LOW",  Direction.BULLISH, None),
            ("asia_high",     "ASIA_HIGH",     Direction.BEARISH, SessionName.ASIA),
            ("asia_low",      "ASIA_LOW",      Direction.BULLISH, SessionName.ASIA),
            ("london_high",   "LONDON_HIGH",   Direction.BEARISH, SessionName.LONDON),
            ("london_low",    "LONDON_LOW",    Direction.BULLISH, SessionName.LONDON),
            ("ny_high",       "NY_HIGH",       Direction.BEARISH, SessionName.NEW_YORK),
            ("ny_low",        "NY_LOW",        Direction.BULLISH, SessionName.NEW_YORK),
        ]
        levels = []
        for attr, label, direction, session in mapping:
            price = getattr(self, attr)
            if price is not None:
                levels.append(LiquidityLevel(
                    price=price, label=label, direction=direction,
                    session=session, swept=False, timestamp=now,
                ))
        return levels


class HTFContext(BaseModel):
    bias: Direction
    draw_target: LiquidityLevel             # nearest liquidity draw in bias direction
    all_levels: list[LiquidityLevel]
    session_levels: SessionLevels
    timestamp: datetime = Field(default_factory=lambda: datetime.now(tz=timezone.utc))


# ── Trade Setup / Signal ──────────────────────────────────────────────────────

class TradeSetup(BaseModel):
    direction: Direction
    step: StrategyStep
    htf_context: HTFContext
    step2_tag: Optional[ConfluenceTag] = None
    step2b_triggered: bool = False
    step3_tag: Optional[ConfluenceTag] = None
    step4_tag: Optional[ConfluenceTag] = None
    entry_price: Optional[float] = None     # underlying (SPY) price
    stop_loss: Optional[float] = None
    target_price: Optional[float] = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(tz=timezone.utc))
    invalidation_reason: Optional[str] = None


class TradeSignal(BaseModel):
    symbol: str                             # options contract Moomoo code
    option_type: OptionType
    direction: Direction
    entry_price: float                      # options premium
    underlying_entry: float                 # SPY price at signal
    stop_loss: float                        # SPY price level
    target_price: float                     # next liquidity draw
    confidence: float = Field(ge=0.0, le=1.0)
    setup: TradeSetup
    rationale: str = ""
    timestamp: datetime = Field(default_factory=lambda: datetime.now(tz=timezone.utc))

    @property
    def risk_reward(self) -> float:
        risk   = abs(self.underlying_entry - self.stop_loss)
        reward = abs(self.target_price - self.underlying_entry)
        return round(reward / risk, 2) if risk > 0 else 0.0


# ── Account / Execution ───────────────────────────────────────────────────────

class Position(BaseModel):
    symbol: str
    option_type: OptionType
    quantity: int
    avg_cost: float
    current_price: float = 0.0
    underlying_entry: float = 0.0
    stop_loss: float = 0.0
    target_price: float = 0.0

    @property
    def market_value(self) -> float:
        return self.quantity * self.current_price * 100

    @property
    def unrealized_pnl(self) -> float:
        return self.quantity * (self.current_price - self.avg_cost) * 100


class AccountState(BaseModel):
    cash: float
    total_assets: float
    buying_power: float
    realized_pnl_today: float = 0.0
    open_position: Optional[Position] = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(tz=timezone.utc))


class ExecutedOrder(BaseModel):
    order_id: str
    symbol: str
    option_type: OptionType
    quantity: int
    fill_price: float
    commission: float = 0.0
    is_entry: bool = True
    timestamp: datetime = Field(default_factory=lambda: datetime.now(tz=timezone.utc))
