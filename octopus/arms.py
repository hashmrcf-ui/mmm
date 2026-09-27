"""The octopus's arms: independent strategies, each specialised for a market regime.

Every arm returns a DataFrame with `signal` (1 long / -1 short / 0) and `stop` for each bar.
A signal on bar t is executed after bar t closes (no look-ahead).
"""
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from whale_hunter.config import Config as HunterConfig
from whale_hunter.strategy import generate_signals as hunter_signals

from .indicators import atr, bollinger, ema, rsi, volume_zscore
from .regime import NEUTRAL, RANGE, TREND, classify


@dataclass(frozen=True)
class Arm:
    name: str
    fn: Callable[[pd.DataFrame, dict], pd.DataFrame]
    grid: tuple            # small parameter grid explored by walk-forward research
    regimes: frozenset     # regimes this arm is allowed to trade in
    target_r: float | None  # fixed take-profit in R (None = let the trailing stop run)


def _pack(df, long, short, stop_dist) -> pd.DataFrame:
    sig = pd.Series(0, index=df.index)
    sig[long.fillna(False)] = 1
    sig[short.fillna(False)] = -1
    stop = (df["close"] - sig * stop_dist).where(sig != 0)
    return pd.DataFrame({"signal": sig, "stop": stop})


def hunter(df: pd.DataFrame, p: dict) -> pd.DataFrame:
    """Liquidity sweep (stop hunt) confirmed by whale volume — the original Whale Hunter."""
    cfg = HunterConfig(swing_lookback=p["lookback"], volume_z=p["volume_z"], use_trend_filter=True)
    return hunter_signals(df, cfg)[["signal", "stop"]]


def trend(df: pd.DataFrame, p: dict) -> pd.DataFrame:
    """Donchian breakout in the direction of the 200 EMA; exits via trailing stop."""
    n, close = p["n"], df["close"]
    trend_line = ema(close, 200)
    long = (close > df["high"].rolling(n).max().shift(1)) & (close > trend_line)
    short = (close < df["low"].rolling(n).min().shift(1)) & (close < trend_line)
    return _pack(df, long, short, p["stop_atr"] * atr(df, 14))


def revert(df: pd.DataFrame, p: dict) -> pd.DataFrame:
    """Mean reversion: buy oversold dips in an uptrend / sell overbought rallies in a downtrend."""
    close = df["close"]
    lower, _, upper = bollinger(close, 20, p["k"])
    r = rsi(close, 14)
    trend_line = ema(close, 200)
    long = (close < lower) & (r < p["rsi"]) & (close > trend_line)
    short = (close > upper) & (r > 100 - p["rsi"]) & (close < trend_line)
    return _pack(df, long, short, 1.5 * atr(df, 14))


def squeeze(df: pd.DataFrame, p: dict) -> pd.DataFrame:
    """Volatility squeeze breakout: tight Bollinger bands, then an expansion with volume."""
    close = df["close"]
    lower, mid, upper = bollinger(close, 20, 2.0)
    width_rank = ((upper - lower) / mid).rolling(120).rank(pct=True)
    squeezed = width_rank.shift(1) <= p["pct"]
    volume_ok = volume_zscore(df["volume"], 50) > 1.0
    long = squeezed & (close > upper) & volume_ok
    short = squeezed & (close < lower) & volume_ok
    return _pack(df, long, short, 2.0 * atr(df, 14))


ALL_REGIMES = frozenset({TREND, RANGE, NEUTRAL})

ARMS = {
    a.name: a
    for a in (
        Arm("hunter", hunter, ({"lookback": 20, "volume_z": 2.0}, {"lookback": 40, "volume_z": 2.0},
                               {"lookback": 20, "volume_z": 2.5}), ALL_REGIMES, 2.0),
        Arm("trend", trend, ({"n": 20, "stop_atr": 3.0}, {"n": 55, "stop_atr": 3.0},
                             {"n": 100, "stop_atr": 3.0}), frozenset({TREND, NEUTRAL}), None),
        Arm("revert", revert, ({"k": 2.0, "rsi": 30}, {"k": 2.5, "rsi": 30},
                               {"k": 2.0, "rsi": 20}), frozenset({RANGE, NEUTRAL}), 1.0),
        Arm("squeeze", squeeze, ({"pct": 0.10}, {"pct": 0.20}), ALL_REGIMES, 2.5),
    )
}


def arm_signals(arm: Arm, df: pd.DataFrame, params: dict, regime: pd.Series | None = None,
                allow_shorts: bool = True) -> pd.DataFrame:
    """Arm signals, silenced outside the arm's regimes (never trades in `chaos`)."""
    out = arm.fn(df, params).copy()
    regime = classify(df) if regime is None else regime
    blocked = ~regime.isin(arm.regimes).to_numpy()
    if not allow_shorts:
        blocked |= out["signal"].to_numpy() < 0
    out.loc[blocked, "signal"] = 0
    out.loc[blocked, "stop"] = np.nan
    return out
