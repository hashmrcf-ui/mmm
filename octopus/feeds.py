"""Market data. LiveFeed pulls from ccxt exchanges / Yahoo Finance; ReplayFeed replays history offline."""
import numpy as np
import pandas as pd

from .config import Market

TF_SECONDS = {"1m": 60, "5m": 300, "15m": 900, "30m": 1800, "1h": 3600, "4h": 14400, "1d": 86400}
YF_INTERVAL = {"1m": ("1m", 7), "5m": ("5m", 59), "15m": ("15m", 59), "30m": ("30m", 59),
               "1h": ("60m", 729), "1d": ("1d", 36500)}  # interval, max days of history


def tf_delta(tf: str) -> pd.Timedelta:
    return pd.Timedelta(seconds=TF_SECONDS[tf])


class LiveFeed:
    def __init__(self):
        self._exchanges = {}

    def now(self) -> pd.Timestamp:
        return pd.Timestamp.now(tz="UTC")

    def history(self, market: Market, bars: int) -> pd.DataFrame:
        return self.recent(market, bars)

    def recent(self, market: Market, bars: int) -> pd.DataFrame:
        df = self._ccxt(market, bars) if market.source == "ccxt" else self._yfinance(market, bars)
        df = df[~df.index.duplicated()].sort_index()
        closed = df.index + tf_delta(market.timeframe) <= self.now()  # drop the still-forming candle
        return df[closed].tail(bars)

    def price(self, market: Market) -> float:
        if market.source == "ccxt":
            return float(self._exchange(market.exchange).fetch_ticker(market.symbol)["last"])
        import yfinance as yf

        return float(yf.Ticker(market.symbol).fast_info["last_price"])

    # ---- ccxt ----
    def _exchange(self, exchange_id: str):
        if exchange_id not in self._exchanges:
            import ccxt

            ex = getattr(ccxt, exchange_id)({"enableRateLimit": True})
            ex.load_markets()
            self._exchanges[exchange_id] = ex
        return self._exchanges[exchange_id]

    def _ccxt(self, market: Market, bars: int) -> pd.DataFrame:
        ex = self._exchange(market.exchange)
        step = TF_SECONDS[market.timeframe] * 1000
        now_ms = int(self.now().timestamp() * 1000)
        # Binance spot exposes taker-buy volume per candle = real aggressive-buyer (whale) flow.
        use_raw = market.exchange == "binance" and ex.market(market.symbol).get("spot", False)
        for raw in ([True, False] if use_raw else [False]):
            try:
                rows, since = [], now_ms - (bars + 1) * step
                for _ in range(bars // 200 + 5):
                    batch = self._binance_batch(ex, market, since) if raw else ex.fetch_ohlcv(
                        market.symbol, market.timeframe, since=since, limit=1000)
                    if not batch:
                        break
                    rows.extend(batch)
                    since = batch[-1][0] + step
                    if since > now_ms:
                        break
                cols = ["timestamp", "open", "high", "low", "close", "volume"] + (["taker_buy_volume"] if raw else [])
                df = pd.DataFrame(rows, columns=cols)
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
                return df.set_index("timestamp").astype(float)
            except Exception:
                if not raw:
                    raise
        raise RuntimeError("unreachable")

    @staticmethod
    def _binance_batch(ex, market: Market, since: int) -> list:
        klines = ex.publicGetKlines({"symbol": ex.market_id(market.symbol), "interval": market.timeframe,
                                     "startTime": since, "limit": 1000})
        return [[int(k[0]), float(k[1]), float(k[2]), float(k[3]), float(k[4]), float(k[5]), float(k[9])]
                for k in klines]

    # ---- Yahoo Finance (stocks, ETFs, forex "EURUSD=X", gold "GC=F", indices "^GSPC") ----
    def _yfinance(self, market: Market, bars: int) -> pd.DataFrame:
        import yfinance as yf

        interval, max_days = YF_INTERVAL[market.timeframe]
        bars_per_day = 7 if market.timeframe == "1h" else 86400 / TF_SECONDS[market.timeframe]
        days = int(min(max_days, bars / bars_per_day * 1.6 + 5))
        df = yf.download(market.symbol, period=f"{days}d", interval=interval, progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]].dropna()
        df.index = df.index.tz_localize("UTC") if df.index.tz is None else df.index.tz_convert("UTC")
        return df.astype(float)


class ReplayFeed:
    """Replays historical data bar by bar — the same engine code runs offline for simulation/tests."""

    def __init__(self, markets: list[Market], data: dict[str, pd.DataFrame], start=None):
        self.data = data
        self.times = pd.DatetimeIndex(sorted(set().union(*[d.index for d in data.values()])))
        self.i = 0 if start is None else int(self.times.searchsorted(start))

    def now(self) -> pd.Timestamp:
        return self.times[self.i]

    def advance(self) -> bool:
        if self.i + 1 >= len(self.times):
            return False
        self.i += 1
        return True

    def recent(self, market: Market, bars: int) -> pd.DataFrame:
        df = self.data[market.key]
        end = int(df.index.searchsorted(self.now(), side="right"))
        return df.iloc[max(0, end - bars):end]

    history = recent

    def price(self, market: Market) -> float:
        return float(self.recent(market, 1)["close"].iloc[-1])


def synthetic_market(n: int, seed: int, kind: str = "random", start: float = 100.0,
                     freq: str = "1h") -> pd.DataFrame:
    """Synthetic OHLCV for testing. kind='random' has NO edge; kind='trend' has persistent trends."""
    rng = np.random.default_rng(seed)
    rets = rng.normal(0, 0.01, n)
    if kind == "trend":
        rets += np.repeat(rng.choice([-1.0, 1.0], n // 150 + 1) * 0.0025, 150)[:n]
    close = start * np.exp(np.cumsum(rets))
    open_ = np.r_[start, close[:-1]] * (1 + rng.normal(0, 0.0005, n))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.004, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.004, n)))
    volume = rng.lognormal(3, 0.4, n)
    idx = pd.date_range("2023-01-01", periods=n, freq=freq, tz="UTC")
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume,
                         "taker_buy_volume": volume * rng.uniform(0.4, 0.6, n)}, index=idx)
