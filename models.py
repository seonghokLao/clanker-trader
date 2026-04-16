"""
models.py — Pydantic data models for market and account state.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class Action(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class Greeks(BaseModel):
    delta: float = 0.0
    gamma: float = 0.0
    theta: float = 0.0
    vega: float = 0.0
    rho: float = 0.0
    implied_volatility: float = 0.0


class OptionQuote(BaseModel):
    symbol: str
    last_price: float
    volume: int
    open_interest: int = 0
    greeks: Greeks = Field(default_factory=Greeks)
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class MarketState(BaseModel):
    spy_price: float
    spy_volume: int
    option: Optional[OptionQuote] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class Position(BaseModel):
    symbol: str
    quantity: int
    avg_cost: float
    current_price: float = 0.0

    @property
    def market_value(self) -> float:
        return self.quantity * self.current_price

    @property
    def unrealized_pnl(self) -> float:
        return self.quantity * (self.current_price - self.avg_cost)


class AccountState(BaseModel):
    cash: float
    total_assets: float
    buying_power: float
    realized_pnl_today: float = 0.0
    positions: list[Position] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    @field_validator("cash", "total_assets", "buying_power")
    @classmethod
    def must_be_finite(cls, v: float) -> float:
        if not isinstance(v, (int, float)) or v != v:  # NaN check
            raise ValueError("Financial fields must be finite numbers.")
        return v

    @property
    def positions_by_symbol(self) -> dict[str, Position]:
        return {p.symbol: p for p in self.positions}


class ProposedTrade(BaseModel):
    symbol: str
    action: Action
    quantity: int = Field(gt=0)
    limit_price: Optional[float] = None  # None => market order intent
    note: str = ""


class TradeDecision(BaseModel):
    approved: bool
    reason: str
    proposed: ProposedTrade
    account_snapshot: AccountState
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class EmaTrend(str, Enum):
    BULLISH = "BULLISH"   # EMA-9 > EMA-21
    BEARISH = "BEARISH"   # EMA-9 < EMA-21
    NEUTRAL = "NEUTRAL"   # equal (edge case)


class MarketContext(BaseModel):
    symbol: str
    latest_price: float
    ema_9: float
    ema_21: float
    ema_trend: EmaTrend
    rsi: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    @property
    def rsi_zone(self) -> str:
        if self.rsi >= 70:
            return "overbought"
        if self.rsi <= 30:
            return "oversold"
        return "neutral"


class SignalDirection(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class SignalStrength(str, Enum):
    STRONG = "STRONG"
    MODERATE = "MODERATE"
    WEAK = "WEAK"


class TradeSignal(BaseModel):
    symbol: str
    direction: SignalDirection
    strength: SignalStrength
    entry_price: float
    target_price: Optional[float] = None
    stop_loss: Optional[float] = None
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str = ""
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    @property
    def risk_reward(self) -> Optional[float]:
        if self.target_price is None or self.stop_loss is None or self.entry_price == 0:
            return None
        reward = abs(self.target_price - self.entry_price)
        risk = abs(self.entry_price - self.stop_loss)
        return round(reward / risk, 3) if risk > 0 else None


class ExecutedOrder(BaseModel):
    order_id: str
    symbol: str
    action: Action
    quantity: int
    fill_price: float
    commission: float = 0.0
    timestamp: datetime = Field(default_factory=datetime.utcnow)
