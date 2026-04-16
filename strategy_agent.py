"""
strategy_agent.py — Signal generation via rule-based mean-reversion logic and a
                    momentum tensor placeholder for future model integration.

This module only produces TradeSignal objects; it never submits orders.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import numpy as np

from broker_gateway import QuoteData
from models import (
    EmaTrend,
    MarketContext,
    MarketState,
    SignalDirection,
    SignalStrength,
    TradeSignal,
)

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Strategy configuration
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class StrategyConfig:
    # Mean-reversion thresholds
    rsi_oversold: float = 30.0
    rsi_overbought: float = 70.0

    # EMA cross confirmation: price must be within this fraction of EMA-9
    # for a cross to be considered "fresh" (avoids chasing extended moves)
    ema_cross_tolerance: float = 0.003       # 0.3 %

    # Momentum tensor weight in confidence blending (0 = pure rules, 1 = pure tensor)
    momentum_blend_weight: float = 0.30

    # Target and stop expressed as ATR multiples; if ATR unavailable, fallback
    # to fixed percentage offsets below
    target_atr_multiple: float = 2.0
    stop_atr_multiple: float = 1.0
    target_pct_fallback: float = 0.015      # 1.5 %
    stop_pct_fallback: float = 0.008        # 0.8 %

    # Minimum rule-based confidence required to emit a non-HOLD signal
    min_confidence_threshold: float = 0.40


# ──────────────────────────────────────────────────────────────────────────────
# Internal named-tuple for rule evaluation results
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class _RuleResult:
    direction: SignalDirection
    confidence: float               # 0.0 – 1.0 from rule layer alone
    rationale_parts: list[str] = field(default_factory=list)


# ──────────────────────────────────────────────────────────────────────────────
# Agent
# ──────────────────────────────────────────────────────────────────────────────

class StrategyAgent:
    """
    Generates TradeSignal objects from quantitative market context and
    live broker quote data.

    Inputs
    ------
    market_ctx  : MarketContext from QuantitativeDataAgent.get_market_context()
    market_state: MarketState from BrokerGateway (live L1 quote + option data)

    The agent never executes orders; callers route signals through RiskManager
    and BrokerGateway independently.
    """

    def __init__(self, config: StrategyConfig | None = None) -> None:
        self._cfg = config or StrategyConfig()

    # ------------------------------------------------------------------ #
    # Public API                                                            #
    # ------------------------------------------------------------------ #

    def evaluate_context(
        self,
        market_ctx: MarketContext,
        market_state: MarketState,
        atr: Optional[float] = None,
    ) -> TradeSignal:
        """
        Evaluate the combined context and emit a TradeSignal.

        Parameters
        ----------
        market_ctx   : structured technical context (EMA, RSI, price)
        market_state : live broker data (real-time price, volume, option chain)
        atr          : optional Average True Range for dynamic target/stop sizing
        """
        live_price = market_state.spy_price

        # 1. Rule-based mean-reversion evaluation
        rule_result = self._evaluate_mean_reversion(market_ctx, live_price)

        # 2. Momentum tensor score (placeholder — blended into confidence)
        tensor_score = self._calculate_momentum_tensor(market_ctx, market_state)

        # 3. Blend confidence
        blended_confidence = self._blend_confidence(
            rule_result.confidence, tensor_score
        )

        # 4. Suppress weak signals
        if blended_confidence < self._cfg.min_confidence_threshold:
            return self._hold_signal(
                symbol=market_ctx.symbol,
                price=live_price,
                rationale=(
                    f"Blended confidence {blended_confidence:.2f} below threshold "
                    f"{self._cfg.min_confidence_threshold:.2f}."
                ),
            )

        # 5. Compute target and stop
        target, stop = self._compute_levels(rule_result.direction, live_price, atr)

        # 6. Determine strength from confidence
        strength = self._grade_strength(blended_confidence)

        rationale = "; ".join(rule_result.rationale_parts)
        rationale += f" | momentum_score={tensor_score:.3f}"

        signal = TradeSignal(
            symbol=market_ctx.symbol,
            direction=rule_result.direction,
            strength=strength,
            entry_price=live_price,
            target_price=target,
            stop_loss=stop,
            confidence=round(blended_confidence, 4),
            rationale=rationale,
        )

        logger.info(
            "[%s] Signal: %s (%s) conf=%.2f  entry=%.4f  target=%s  stop=%s  R:R=%s",
            signal.symbol, signal.direction.value, signal.strength.value,
            signal.confidence, signal.entry_price,
            f"{signal.target_price:.4f}" if signal.target_price else "—",
            f"{signal.stop_loss:.4f}" if signal.stop_loss else "—",
            f"{signal.risk_reward:.2f}" if signal.risk_reward else "—",
        )
        return signal

    # ------------------------------------------------------------------ #
    # Mean-reversion rule evaluation                                        #
    # ------------------------------------------------------------------ #

    def _evaluate_mean_reversion(
        self, ctx: MarketContext, live_price: float
    ) -> _RuleResult:
        """
        Mean-reversion rules:

        BUY  — RSI < oversold threshold AND live price crosses above EMA-9
                (price ≥ EMA-9 and within ema_cross_tolerance of it)
        SELL — RSI > overbought threshold AND live price crosses below EMA-9
                (price ≤ EMA-9 and within ema_cross_tolerance of it)
        HOLD — neither condition met
        """
        rsi = ctx.rsi
        ema9 = ctx.ema_9
        cfg = self._cfg
        rationale: list[str] = []
        confidence = 0.0

        above_ema9 = live_price >= ema9
        below_ema9 = live_price <= ema9
        near_ema9 = abs(live_price - ema9) / ema9 <= cfg.ema_cross_tolerance

        # ── BUY condition ──────────────────────────────────────────────
        if rsi < cfg.rsi_oversold and above_ema9:
            rationale.append(f"RSI={rsi:.2f} < {cfg.rsi_oversold} (oversold)")
            rationale.append(f"price {live_price:.4f} crossed above EMA-9 {ema9:.4f}")
            if ctx.ema_trend == EmaTrend.BULLISH:
                rationale.append("EMA-9 > EMA-21 (trend confirmation)")
                confidence = 0.80
            else:
                confidence = 0.60   # counter-trend; lower confidence

            # Proximity bonus: fresh cross = higher confidence
            if near_ema9:
                confidence = min(1.0, confidence + 0.10)
                rationale.append("fresh EMA-9 cross (+0.10 confidence)")

            return _RuleResult(SignalDirection.BUY, confidence, rationale)

        # ── SELL condition ─────────────────────────────────────────────
        if rsi > cfg.rsi_overbought and below_ema9:
            rationale.append(f"RSI={rsi:.2f} > {cfg.rsi_overbought} (overbought)")
            rationale.append(f"price {live_price:.4f} crossed below EMA-9 {ema9:.4f}")
            if ctx.ema_trend == EmaTrend.BEARISH:
                rationale.append("EMA-9 < EMA-21 (trend confirmation)")
                confidence = 0.80
            else:
                confidence = 0.60

            if near_ema9:
                confidence = min(1.0, confidence + 0.10)
                rationale.append("fresh EMA-9 cross (+0.10 confidence)")

            return _RuleResult(SignalDirection.SELL, confidence, rationale)

        # ── HOLD ───────────────────────────────────────────────────────
        rationale.append(
            f"No trigger: RSI={rsi:.2f}, price={live_price:.4f}, EMA-9={ema9:.4f}"
        )
        return _RuleResult(SignalDirection.HOLD, 0.0, rationale)

    # ------------------------------------------------------------------ #
    # Momentum tensor (placeholder)                                         #
    # ------------------------------------------------------------------ #

    def _calculate_momentum_tensor(
        self,
        market_ctx: MarketContext,
        market_state: MarketState,
    ) -> float:
        """
        Placeholder for a future PyTorch / custom model that returns a
        scalar momentum score in [0, 1].

        Current implementation: a simple heuristic built from normalised
        NumPy operations so the interface contract is established and the
        downstream blending logic can be exercised end-to-end.

        Replace the body of this method with model inference when ready.
        Expected signature stays fixed:
            inputs  — MarketContext, MarketState
            returns — float in [0.0, 1.0]  (0=bearish, 0.5=neutral, 1=bullish)

        # ── Future PyTorch integration sketch ─────────────────────────
        # import torch
        # features = torch.tensor([
        #     market_ctx.rsi / 100.0,
        #     (market_ctx.ema_9 - market_ctx.ema_21) / market_ctx.ema_21,
        #     market_ctx.latest_price / market_ctx.ema_9 - 1.0,
        #     ... additional features ...
        # ], dtype=torch.float32).unsqueeze(0)
        # with torch.no_grad():
        #     score = self._model(features).squeeze().item()
        # return float(np.clip(score, 0.0, 1.0))
        # ──────────────────────────────────────────────────────────────
        """
        rsi_norm = market_ctx.rsi / 100.0                                    # [0, 1]
        ema_spread = (market_ctx.ema_9 - market_ctx.ema_21) / market_ctx.ema_21
        ema_norm = float(np.clip(ema_spread * 50 + 0.5, 0.0, 1.0))         # centred at 0.5

        price_vs_ema9 = (market_ctx.latest_price - market_ctx.ema_9) / market_ctx.ema_9
        price_norm = float(np.clip(price_vs_ema9 * 50 + 0.5, 0.0, 1.0))

        # Equal-weight combination — replace with learned weights later
        score = float(np.mean([rsi_norm, ema_norm, price_norm]))

        logger.debug(
            "[%s] Momentum tensor: rsi_norm=%.3f ema_norm=%.3f price_norm=%.3f → score=%.3f",
            market_ctx.symbol, rsi_norm, ema_norm, price_norm, score,
        )
        return score

    # ------------------------------------------------------------------ #
    # Internal helpers                                                      #
    # ------------------------------------------------------------------ #

    def _blend_confidence(self, rule_confidence: float, tensor_score: float) -> float:
        """
        Linear interpolation between rule-based confidence and the momentum
        tensor score.  tensor_score is first re-centred: values > 0.5 are
        bullish, < 0.5 bearish; we map it to a directional confidence delta.
        """
        w = self._cfg.momentum_blend_weight
        blended = (1.0 - w) * rule_confidence + w * tensor_score
        return float(np.clip(blended, 0.0, 1.0))

    def _compute_levels(
        self,
        direction: SignalDirection,
        price: float,
        atr: Optional[float],
    ) -> tuple[Optional[float], Optional[float]]:
        """Return (target_price, stop_loss) using ATR multiples or pct fallback."""
        if direction == SignalDirection.HOLD:
            return None, None

        if atr is not None and atr > 0:
            target_offset = atr * self._cfg.target_atr_multiple
            stop_offset = atr * self._cfg.stop_atr_multiple
        else:
            target_offset = price * self._cfg.target_pct_fallback
            stop_offset = price * self._cfg.stop_pct_fallback

        if direction == SignalDirection.BUY:
            return (
                round(price + target_offset, 4),
                round(price - stop_offset, 4),
            )
        # SELL
        return (
            round(price - target_offset, 4),
            round(price + stop_offset, 4),
        )

    @staticmethod
    def _grade_strength(confidence: float) -> SignalStrength:
        if confidence >= 0.75:
            return SignalStrength.STRONG
        if confidence >= 0.55:
            return SignalStrength.MODERATE
        return SignalStrength.WEAK

    @staticmethod
    def _hold_signal(symbol: str, price: float, rationale: str) -> TradeSignal:
        return TradeSignal(
            symbol=symbol,
            direction=SignalDirection.HOLD,
            strength=SignalStrength.WEAK,
            entry_price=price,
            target_price=None,
            stop_loss=None,
            confidence=0.0,
            rationale=rationale,
        )
