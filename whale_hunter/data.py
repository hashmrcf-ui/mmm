import numpy as np
import pandas as pd

TIMEFRAME_RULES = {"1m": "1min", "5m": "5min", "15m": "15min", "1h": "1h", "4h": "4h", "1d": "1D"}


def load_csv(path: str) -> pd.DataFrame:
    """Load OHLCV from CSV with columns: timestamp, open, high, low, close, volume[, taker_buy_volume]."""
    df = pd.read_csv(path)
    ts = df["timestamp"]
    df["timestamp"] = pd.to_datetime(ts, unit="ms" if np.issubdtype(ts.dtype, np.number) else None, utc=True)
    return df.set_index("timestamp").sort_index()


def fetch_ohlcv(exchange, symbol: str, timeframe: str, limit: int = 1000) -> pd.DataFrame:
    """Fetch candles through a ccxt exchange instance."""
    rows = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df.set_index("timestamp")


def whale_flow(trades: list[dict], timeframe: str, min_usd: float) -> pd.DataFrame:
    """Aggregate individual trades >= `min_usd` into per-candle whale buy/sell dollar volume.

    `trades` is a list of ccxt trade dicts (timestamp, side, cost/price/amount).
    """
    rows = []
    for t in trades:
        cost = t.get("cost") or (t["price"] * t["amount"])
        if cost >= min_usd and t.get("side") in ("buy", "sell"):
            rows.append((t["timestamp"], t["side"], cost))
    if not rows:
        return pd.DataFrame(columns=["whale_buy", "whale_sell"])
    df = pd.DataFrame(rows, columns=["timestamp", "side", "cost"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("timestamp")
    rule = TIMEFRAME_RULES[timeframe]
    buy = df[df["side"] == "buy"]["cost"].resample(rule).sum()
    sell = df[df["side"] == "sell"]["cost"].resample(rule).sum()
    return pd.concat({"whale_buy": buy, "whale_sell": sell}, axis=1).fillna(0.0)


def synthetic_ohlcv(n: int = 3000, seed: int = 42) -> pd.DataFrame:
    """Random-walk market with occasional injected stop hunts. For testing the pipeline only —
    results on this data say NOTHING about real-market profitability."""
    rng = np.random.default_rng(seed)
    close = 30_000 * np.exp(np.cumsum(rng.normal(0.0002, 0.006, n)))
    open_ = np.r_[close[0], close[:-1]]
    spread = np.abs(rng.normal(0, 0.004, n)) * close
    high = np.maximum(open_, close) + spread
    low = np.minimum(open_, close) - spread
    volume = rng.lognormal(3, 0.3, n)
    taker = volume * rng.uniform(0.4, 0.6, n)
    for i in range(60, n, 97):  # bullish stop hunt every ~97 bars
        pool_low = low[i - 20 : i].min()
        low[i] = pool_low * 0.99
        open_[i] = close[i - 1]
        close[i] = max(open_[i], pool_low * 1.004)
        high[i] = close[i] * 1.001
        volume[i] *= 4
        taker[i] = volume[i] * 0.7
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume, "taker_buy_volume": taker},
        index=idx,
    )
