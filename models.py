"""
models.py — Data models for the simplified SMC strategy.
"""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class Direction(str, Enum):
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"

class OptionType(str, Enum):
    CALL = "CALL"
    PUT  = "PUT"

class SetupStep(int, Enum):
    IDLE           = 0
    BIAS_CONFIRMED = 1   # 4h + 1h trend agree
    SWEEP_FOUND    = 2   # liquidity level swept
    BOS_CONFIRMED  = 3   # 5-min structure broken in bias direction
    ENTERED        = 4


class KeyLevel(BaseModel):
    price: float
    label: str          # e.g. "PREV_DAY_HIGH", "ASIA_LOW", "LONDON_HIGH"
    direction: Direction  # BEARISH = buy-stops resting above, BULLISH = sell-stops below
    swept: bool = False
    timestamp: datetime


class DailyBias(BaseModel):
    direction: Direction
    reason: str          # e.g. "4h BULLISH + 1h BULLISH"
    timestamp: datetime = Field(default_factory=lambda: datetime.now(tz=timezone.utc))


class LiquiditySweep(BaseModel):
    level: KeyLevel
    sweep_low: float     # wick extreme that pierced the level
    close_price: float   # candle close that reversed back
    direction: Direction # BULLISH sweep = sell-stops taken, expect reversal up
    timestamp: datetime


class SwingPoint(BaseModel):
    price: float
    direction: Direction  # BEARISH = swing high, BULLISH = swing low
    timestamp: datetime
    broken: bool = False


class BOS(BaseModel):
    direction: Direction
    broken_level: float
    timestamp: datetime


class FVG(BaseModel):
    top: float
    bottom: float
    direction: Direction   # BULLISH = buy-side imbalance, BEARISH = sell-side
    timestamp: datetime
    filled: bool = False

    @property
    def midpoint(self) -> float:
        return (self.top + self.bottom) / 2

    def contains(self, price: float) -> bool:
        return self.bottom <= price <= self.top


class TradeSetup(BaseModel):
    direction: Direction
    step: SetupStep
    bias: DailyBias
    sweep: Optional[LiquiditySweep] = None
    bos: Optional[BOS] = None
    entry_price: Optional[float] = None    # SPY price
    stop_loss: Optional[float] = None      # SPY price
    target_price: Optional[float] = None   # SPY price (opposing level)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(tz=timezone.utc))


class TradeSignal(BaseModel):
    symbol: str                 # options contract
    option_type: OptionType
    entry_price: float          # option premium
    underlying_entry: float     # SPY price
    stop_loss: float            # SPY price level
    target_price: float         # SPY price level
    setup: TradeSetup
    timestamp: datetime = Field(default_factory=lambda: datetime.now(tz=timezone.utc))

    @property
    def risk_reward(self) -> float:
        risk   = abs(self.underlying_entry - self.stop_loss)
        reward = abs(self.target_price - self.underlying_entry)
        return round(reward / risk, 2) if risk > 0 else 0.0


class AccountState(BaseModel):
    cash: float
    total_assets: float
    buying_power: float
    realized_pnl_today: float = 0.0
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
