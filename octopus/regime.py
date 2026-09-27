import numpy as np
import pandas as pd

from .indicators import atr, efficiency_ratio

TREND, RANGE, NEUTRAL, CHAOS = "trend", "range", "neutral", "chaos"


def classify(df: pd.DataFrame, er_period: int = 20, vol_window: int = 250) -> pd.Series:
    """Label each bar with a market regime.

    chaos   : volatility in its top 3% of the last `vol_window` bars -> no new trades at all
    trend   : efficiency ratio >= 0.35
    range   : efficiency ratio <= 0.20
    neutral : in between
    """
    er = efficiency_ratio(df["close"], er_period)
    vol = (atr(df, 14) / df["close"]).rolling(vol_window, min_periods=50).rank(pct=True)
    labels = np.where(
        vol > 0.97, CHAOS, np.where(er >= 0.35, TREND, np.where(er <= 0.20, RANGE, NEUTRAL))
    )
    return pd.Series(labels, index=df.index)
