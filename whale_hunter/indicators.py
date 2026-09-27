import numpy as np
import pandas as pd


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def atr(df: pd.DataFrame, period: int) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def liquidity_levels(df: pd.DataFrame, lookback: int) -> tuple[pd.Series, pd.Series]:
    """Highest high / lowest low of the previous `lookback` bars (excluding the current bar).

    These are where stop-loss orders cluster: shorts' stops above the high, longs' stops below the low.
    """
    pool_high = df["high"].shift(1).rolling(lookback).max()
    pool_low = df["low"].shift(1).rolling(lookback).min()
    return pool_high, pool_low


def volume_zscore(volume: pd.Series, window: int) -> pd.Series:
    prev = volume.shift(1).rolling(window)
    std = prev.std().replace(0, np.nan)
    return (volume - prev.mean()) / std


def buy_ratio(df: pd.DataFrame) -> pd.Series:
    """Share of volume that was aggressive buying.

    Priority: real whale flow (live trades) > exchange taker-buy volume > candle-shape proxy.
    """
    if {"whale_buy", "whale_sell"}.issubset(df.columns):
        total = df["whale_buy"] + df["whale_sell"]
        real = df["whale_buy"] / total.replace(0, np.nan)
        return real.fillna(0.5)
    if "taker_buy_volume" in df.columns:
        return (df["taker_buy_volume"] / df["volume"].replace(0, np.nan)).fillna(0.5)
    rng = (df["high"] - df["low"]).replace(0, np.nan)
    return ((df["close"] - df["low"]) / rng).fillna(0.5)


def cumulative_delta(df: pd.DataFrame) -> pd.Series:
    """Cumulative volume delta (buy volume - sell volume) — tracks net whale/aggressor pressure."""
    ratio = buy_ratio(df)
    return (df["volume"] * (2 * ratio - 1)).cumsum()
