import numpy as np
import pandas as pd

from .config import Config
from .indicators import atr, buy_ratio, ema, liquidity_levels, volume_zscore


def generate_signals(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Annotate OHLCV data with Whale Hunter signals.

    Long setup (bullish stop hunt):
      1. Price wicks BELOW the recent liquidity pool low (sell stops get triggered).
      2. The candle CLOSES back above that level (the sweep failed -> trap).
      3. Long lower wick (rejection).
      4. Whale footprint: abnormal volume with buyers dominating (whales absorbed the stops).
      5. Optional: price above the trend EMA.
    Short setup is the mirror image above the liquidity pool high.

    A signal on bar t is meant to be executed at the OPEN of bar t+1 (no look-ahead).
    """
    out = df.copy()
    out["pool_high"], out["pool_low"] = liquidity_levels(out, cfg.swing_lookback)
    out["atr"] = atr(out, cfg.atr_period)
    out["vol_z"] = volume_zscore(out["volume"], cfg.volume_window)
    out["buy_ratio"] = buy_ratio(out)
    out["trend"] = ema(out["close"], cfg.trend_ema)

    rng = (out["high"] - out["low"]).replace(0, np.nan)
    body_low = out[["open", "close"]].min(axis=1)
    body_high = out[["open", "close"]].max(axis=1)
    lower_wick = (body_low - out["low"]) / rng
    upper_wick = (out["high"] - body_high) / rng

    whale_volume = out["vol_z"] >= cfg.volume_z

    long_sweep = (out["low"] < out["pool_low"]) & (out["close"] > out["pool_low"])
    long_sig = (
        long_sweep
        & (lower_wick >= cfg.min_wick_ratio)
        & whale_volume
        & (out["buy_ratio"] >= cfg.buy_ratio_long)
    )
    short_sweep = (out["high"] > out["pool_high"]) & (out["close"] < out["pool_high"])
    short_sig = (
        short_sweep
        & (upper_wick >= cfg.min_wick_ratio)
        & whale_volume
        & (out["buy_ratio"] <= cfg.buy_ratio_short)
    )
    if cfg.use_trend_filter:
        long_sig &= out["close"] > out["trend"]
        short_sig &= out["close"] < out["trend"]
    if not cfg.allow_shorts:
        short_sig &= False

    out["signal"] = 0
    out.loc[long_sig.fillna(False), "signal"] = 1
    out.loc[short_sig.fillna(False), "signal"] = -1

    # Stop sits just beyond the sweep wick — if price goes there again, the hunt thesis is wrong.
    out["stop"] = np.where(
        out["signal"] == 1,
        out["low"] - cfg.stop_atr_buffer * out["atr"],
        np.where(out["signal"] == -1, out["high"] + cfg.stop_atr_buffer * out["atr"], np.nan),
    )
    return out
