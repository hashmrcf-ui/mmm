import math

import numpy as np
import pandas as pd
import pytest

from octopus.arms import ARMS, arm_signals
from octopus.brokers import PaperBroker
from octopus.config import ManageConfig, Market, ResearchConfig, RiskConfig, Settings, load_settings
from octopus.engine import Engine
from octopus.feeds import ReplayFeed, synthetic_market
from octopus.regime import classify
from octopus.research import research_market
from octopus.risk import RiskManager
from octopus.trade import Position, apply_exit, manage

M = ManageConfig(fee_rate=0.0, slippage=0.0, breakeven_buffer=0.0)
NOW = pd.Timestamp("2024-01-01T12:00Z")


def long_pos(entry=100.0, stop=90.0, target=None, qty=1.0):
    return Position("k", "crypto", "trend", 1, entry, stop, target, qty, entry - stop, str(NOW))


# ------------------------------------------------------------------ trade management
def test_stop_hit_exits_everything_at_stop():
    p = long_pos()
    ex = manage(p, 95, 96, 89, 90, 1.0, M)
    assert [(e.qty, e.price, e.reason) for e in ex] == [(1.0, 90.0, "stop")]


def test_gap_through_stop_fills_at_open():
    ex = manage(long_pos(), 85, 86, 84, 85, 1.0, M)
    assert ex[0].price == 85


def test_partial_profit_and_breakeven_make_trade_riskless():
    p = long_pos()
    ex = manage(p, 105, 111, 104, 110, 1.0, M)  # +1.1R
    assert ex[0].reason == "partial" and ex[0].qty == 0.5 and ex[0].price == 110
    apply_exit(p, ex[0].qty, ex[0].price, 0)
    assert p.stop == 100 and p.stop_reason == "breakeven"
    ex = manage(p, 101, 101, 99, 99.5, 1.0, M)  # falls back to entry
    apply_exit(p, ex[0].qty, ex[0].price, 0)
    assert p.closed and p.pnl == pytest.approx(5.0) and p.r_multiple == pytest.approx(0.5)


def test_trailing_stop_follows_price_and_never_loosens():
    p = long_pos()
    manage(p, 100, 120, 100, 118, 2.0, M)  # +2R -> trail at 120 - 2.5*2 = 115
    assert p.stop == pytest.approx(115) and p.stop_reason == "trail"
    manage(p, 118, 118, 116, 117, 5.0, M)  # wider ATR must not loosen the stop
    assert p.stop == pytest.approx(115)


def test_target_and_time_stop():
    p = long_pos(target=120)
    assert manage(p, 110, 121, 109, 119, 1.0, M)[0].reason == "target"
    p = long_pos()
    cfg = ManageConfig(max_bars=2, fee_rate=0, slippage=0)
    assert manage(p, 100, 101, 99, 100, 1.0, cfg) == []
    assert manage(p, 100, 101, 99, 100.5, 1.0, cfg)[0].reason == "time"


def test_short_position_management():
    p = Position("k", "crypto", "trend", -1, 100.0, 110.0, None, 1.0, 10.0, str(NOW))
    ex = manage(p, 95, 96, 89, 90, 1.0, M)  # +1.1R for a short
    assert ex[0].reason == "partial" and p.stop == 100
    assert manage(p, 101, 102, 100, 101, 1.0, M)[0].reason == "breakeven"


# ------------------------------------------------------------------ arms / regime
@pytest.mark.parametrize("name", list(ARMS))
def test_arms_do_not_look_ahead(name):
    df = synthetic_market(1500, 7, "trend")
    arm = ARMS[name]
    full = arm_signals(arm, df, arm.grid[0])
    part = arm_signals(arm, df.iloc[:1100], arm.grid[0])
    pd.testing.assert_frame_equal(full.iloc[:1100], part)


def test_regime_labels():
    df = synthetic_market(1500, 7, "trend")
    assert set(classify(df).unique()) <= {"trend", "range", "neutral", "chaos"}


# ------------------------------------------------------------------ research (the scientist)
def test_research_rejects_pure_noise():
    mk = Market("NOISE", source="synthetic")
    for seed in (101, 102, 103):
        evals = research_market(synthetic_market(3000, seed, "random"), mk, ResearchConfig(), ManageConfig())
        assert not [e for e in evals if e.approved]


def test_research_finds_real_trend_edge():
    mk = Market("TREND", source="synthetic")
    evals = research_market(synthetic_market(3000, 201, "trend"), mk, ResearchConfig(), ManageConfig())
    approved = {e.arm: e for e in evals if e.approved}
    assert "trend" in approved and approved["trend"].params in ARMS["trend"].grid
    assert 0.25 <= approved["trend"].weight <= 1.0


# ------------------------------------------------------------------ risk manager
def mk(sym="BTC/USDT", cls="crypto"):
    return Market(sym, asset_class=cls)


def test_kill_switch_on_max_drawdown():
    r = RiskManager(RiskConfig(max_drawdown=0.1), 1000)
    r.update(1000, NOW)
    assert r.update(950, NOW) == [] and not r.halted
    assert r.update(890, NOW) and r.halted
    assert not r.check_entry(mk(), 1, [], 890, NOW)[0]


def test_daily_loss_limit_resets_next_day():
    r = RiskManager(RiskConfig(daily_loss_limit=0.02), 1000)
    r.update(1000, NOW)
    r.update(975, NOW)
    assert r.check_entry(mk(), 1, [], 975, NOW) == (False, "daily loss limit reached")
    tomorrow = NOW + pd.Timedelta(days=1)
    r.update(975, tomorrow)
    assert r.check_entry(mk(), 1, [], 975, tomorrow)[0]


def test_losing_streak_pauses_trading():
    r = RiskManager(RiskConfig(loss_streak_pause=3, pause_hours=24), 1000)
    r.on_trade_closed(-5, NOW)
    r.on_trade_closed(-5, NOW)
    assert r.on_trade_closed(-5, NOW)
    assert not r.check_entry(mk(), 1, [], 1000, NOW + pd.Timedelta(hours=1))[0]
    assert r.check_entry(mk(), 1, [], 1000, NOW + pd.Timedelta(hours=25))[0]


def test_exposure_and_correlation_limits():
    r = RiskManager(RiskConfig(max_open_positions=3, max_per_asset_class=3, max_same_direction_per_class=2), 1000)
    open_ = [Position(f"m{i}", "crypto", "trend", 1, 100, 90, None, 1, 10, "") for i in range(2)]
    ok, why = r.check_entry(mk("SOL/USDT"), 1, open_, 1000, NOW)
    assert not ok and "correlated" in why
    assert r.check_entry(mk("SOL/USDT"), -1, open_, 1000, NOW)[0]
    same = [Position(mk("ETH/USDT").key, "crypto", "trend", 1, 100, 90, None, 1, 10, "")]
    assert r.check_entry(mk("ETH/USDT"), 1, same, 1000, NOW) == (False, "already positioned in this market")


def test_position_sizing_caps_and_drawdown_scaling():
    cfg = RiskConfig(risk_per_trade=0.01, max_position_notional=0.3, max_drawdown=0.2)
    r = RiskManager(cfg, 1000)
    assert r.size(1000, 100, 90, 1.0, 0) == pytest.approx(1.0)        # 10$ risk / 10$ per unit
    assert r.size(1000, 100, 99.9, 1.0, 0) == pytest.approx(3.0)      # capped at 30% notional
    assert r.size(1000, 100, 90, 1.0, 950) == pytest.approx(0.5)      # only 50$ gross room left
    r.peak = 1100
    assert r.size(1000, 100, 90, 1.0, 0) < 1.0                        # smaller while in drawdown


# ------------------------------------------------------------------ broker
def test_paper_broker_long_and_short_accounting():
    b = PaperBroker(1000, 0.0, 0.0)
    m = mk()
    b.execute(m, 1, 2, 100)
    b.execute(m, -1, 2, 110)
    assert b.cash == pytest.approx(1020)
    b.execute(m, -1, 1, 100)
    b.execute(m, 1, 1, 90)
    assert b.cash == pytest.approx(1030)


# ------------------------------------------------------------------ engine (full system, offline)
def make_engine(tmp_path=None, n=1400, risk=None):
    markets = [Market("TREND", source="synthetic", asset_class="a"),
               Market("NOISE", source="synthetic", asset_class="b")]
    s = Settings(markets=markets, window=400, risk=risk or RiskConfig(), manage=ManageConfig())
    data = {markets[0].key: synthetic_market(n, 201, "trend"), markets[1].key: synthetic_market(n, 101, "random")}
    approvals = [{"market": markets[0].key, "arm": "trend", "params": {"n": 20, "stop_atr": 3.0},
                  "weight": 1.0, "t_stat": 3.0}]
    feed = ReplayFeed(markets, data, start=data[markets[0].key].index[400])
    broker = PaperBroker(s.starting_equity, s.manage.fee_rate, s.manage.slippage)
    return Engine(s, feed, broker, approvals, str(tmp_path) if tmp_path else None, lambda *a, **k: None), feed


def run(engine, feed, steps=None):
    k = 0
    while True:
        engine.step()
        k += 1
        if (steps and k >= steps) or not feed.advance():
            break


def test_engine_trades_only_approved_arms_and_accounting_is_exact():
    engine, feed = make_engine()
    run(engine, feed)
    assert engine.closed, "expected some trades"
    assert {t["arm"] for t in engine.closed} == {"trend"}
    assert {t["market"] for t in engine.closed} == {"synthetic:TREND:1h"}
    engine._flatten(feed.now(), "end")
    realised = sum(t["pnl"] for t in engine.closed)
    assert engine.broker.cash == pytest.approx(1000 + realised)
    # never risked more than ~0.5% of equity on any single trade (+ fees/slippage)
    assert min(t["pnl"] for t in engine.closed) > -1000 * 0.012


def test_engine_kill_switch_flattens_and_halts():
    engine, feed = make_engine()
    run(engine, feed, steps=300)
    while not engine.positions and feed.advance():
        engine.step()
    assert engine.positions
    engine.risk.peak = engine.equity() * 2  # simulate a 50% drawdown
    feed.advance()
    engine.step()
    assert engine.risk.halted and not engine.positions
    n = len(engine.closed)
    run(engine, feed, steps=200)
    assert len(engine.closed) == n and not engine.positions


def test_engine_state_survives_restart(tmp_path):
    engine, feed = make_engine(tmp_path)
    run(engine, feed, steps=300)
    while not engine.positions and feed.advance():
        engine.step()
    restarted, _ = make_engine(tmp_path)
    assert [p.market for p in restarted.positions] == [p.market for p in engine.positions]
    assert restarted.broker.cash == pytest.approx(engine.broker.cash)
    assert (tmp_path / "trades.csv").exists() == bool(engine.closed)


def test_example_config_loads():
    s = load_settings("octopus.example.toml")
    assert s.mode == "paper" and len(s.markets) >= 4
    assert {m.source for m in s.markets} >= {"ccxt", "yfinance"}


def test_autonomous_loop_researches_trades_and_persists(tmp_path, monkeypatch):
    import octopus.engine as eng

    markets = [Market("TREND", source="synthetic", asset_class="a")]
    s = Settings(markets=markets, window=400, state_dir=str(tmp_path),
                 research=ResearchConfig(history_bars=3000, min_trades=10))
    data = {markets[0].key: synthetic_market(3000, 201, "trend")}
    monkeypatch.setattr(eng, "LiveFeed", lambda: ReplayFeed(markets, data, start=data[markets[0].key].index[-1]))
    ticks = []

    def fake_sleep(_):
        ticks.append(1)
        if len(ticks) >= 2:
            raise KeyboardInterrupt

    monkeypatch.setattr(eng.time, "sleep", fake_sleep)
    messages = []
    eng.run_forever(s, lambda msg, important=False: messages.append(msg))
    assert (tmp_path / "research.json").exists() and (tmp_path / "state.json").exists()
    assert any("approved" in m for m in messages) and messages[-1] == "stopped by user"
