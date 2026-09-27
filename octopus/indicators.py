import numpy as np
import pandas as pd

from whale_hunter.indicators import atr, ema, volume_zscore  # noqa: F401  (re-exported)


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    down = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    return 100 - 100 / (1 + up / down.replace(0, np.nan))


def bollinger(close: pd.Series, period: int, k: float) -> tuple[pd.Series, pd.Series, pd.Series]:
    mid = close.rolling(period).mean()
    sd = close.rolling(period).std()
    return mid - k * sd, mid, mid + k * sd


def efficiency_ratio(close: pd.Series, period: int = 20) -> pd.Series:
    """Kaufman efficiency ratio: 1 = straight-line trend, ~0 = pure noise."""
    change = (close - close.shift(period)).abs()
    path = close.diff().abs().rolling(period).sum()
    return change / path.replace(0, np.nan)
