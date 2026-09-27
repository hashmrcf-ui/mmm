import numpy as np
import pandas as pd

from whale_hunter.backtest import run_backtest
from whale_hunter.config import Config
from whale_hunter.data import synthetic_ohlcv, whale_flow
from whale_hunter.strategy import generate_signals


def flat_market(n=300):
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    close = np.full(n, 100.0) + np.sin(np.arange(n)) * 0.2
    return pd.DataFrame(
        {"open": close, "high": close + 0.5, "low": close - 0.5, "close": close,
         "volume": np.full(n, 10.0) + np.cos(np.arange(n)), "taker_buy_volume": np.full(n, 5.0)},
        index=idx,
    )


def inject_bull_sweep(df, i, buy_share=0.8):
    pool_low = df["low"].iloc[i - 20 : i].min()
    df.iloc[i, df.columns.get_loc("open")] = 100.0
    df.iloc[i, df.columns.get_loc("low")] = pool_low - 3
    df.iloc[i, df.columns.get_loc("close")] = 100.2
    df.iloc[i, df.columns.get_loc("high")] = 100.4
    df.iloc[i, df.columns.get_loc("volume")] = 60.0
    df.iloc[i, df.columns.get_loc("taker_buy_volume")] = 60.0 * buy_share
    return df


def cfg():
    return Config(use_trend_filter=False)


def test_bullish_stop_hunt_with_whale_buying_gives_long():
    df = inject_bull_sweep(flat_market(), 250)
    sig = generate_signals(df, cfg())
    assert sig["signal"].iloc[250] == 1
    assert sig["stop"].iloc[250] < df["low"].iloc[250]
    assert (sig["signal"].drop(sig.index[250]) == 0).all()


def test_sweep_without_whale_buying_is_ignored():
    df = inject_bull_sweep(flat_market(), 250, buy_share=0.3)
    assert generate_signals(df, cfg())["signal"].iloc[250] == 0


def test_sweep_without_volume_spike_is_ignored():
    df = inject_bull_sweep(flat_market(), 250)
    df.iloc[250, df.columns.get_loc("volume")] = 10.0
    df.iloc[250, df.columns.get_loc("taker_buy_volume")] = 8.0
    assert generate_signals(df, cfg())["signal"].iloc[250] == 0


def test_no_lookahead():
    df = synthetic_ohlcv(1500)
    full = generate_signals(df, cfg())["signal"]
    partial = generate_signals(df.iloc[:1000], cfg())["signal"]
    pd.testing.assert_series_equal(full.iloc[:1000], partial)


def test_backtest_enters_next_bar_and_hits_target():
    df = inject_bull_sweep(flat_market(), 250)
    # after the signal, price rallies strongly
    for j in range(251, 260):
        for col in ("open", "high", "low", "close"):
            df.iloc[j, df.columns.get_loc(col)] = 100.2 + (j - 250) * 2
    res = run_backtest(df, cfg(), 1000)
    assert len(res.trades) == 1
    t = res.trades[0]
    assert t.entry_time == df.index[251] and t.reason == "target" and t.pnl > 0
    # risk sized to ~1% of equity => profit >= ~2% (2R, more if price gaps past the target)
    assert t.pnl > 15


def test_backtest_stop_loss_limits_loss():
    df = inject_bull_sweep(flat_market(), 250)
    for col, v in (("open", 99.5), ("high", 99.6), ("low", 80.0), ("close", 85.0)):
        df.iloc[252, df.columns.get_loc(col)] = v
    res = run_backtest(df, cfg(), 1000)
    assert res.trades[0].reason == "stop"
    assert -12 < res.trades[0].pnl < -9  # ~1% risk + fees


def test_gap_through_stop_fills_at_open():
    df = inject_bull_sweep(flat_market(), 250)
    for col in ("open", "high", "low", "close"):
        df.iloc[252, df.columns.get_loc(col)] = 80.0
    t = run_backtest(df, cfg(), 1000).trades[0]
    assert t.reason == "stop" and t.exit == 80.0


def test_whale_flow_filters_small_trades():
    ts = int(pd.Timestamp("2024-01-01T00:10Z").value // 1_000_000)
    trades = [
        {"timestamp": ts, "side": "buy", "cost": 250_000},
        {"timestamp": ts, "side": "sell", "cost": 50_000},  # too small
        {"timestamp": ts, "side": "sell", "cost": 120_000},
    ]
    flow = whale_flow(trades, "1h", 100_000)
    assert flow["whale_buy"].iloc[0] == 250_000
    assert flow["whale_sell"].iloc[0] == 120_000


def test_synthetic_backtest_runs():
    res = run_backtest(synthetic_ohlcv(), Config(), 1000)
    assert res.stats["trades"] > 0
    assert res.equity.notna().all()
