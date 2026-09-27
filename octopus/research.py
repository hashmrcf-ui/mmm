"""The scientist: decides which arms have a real, statistically significant edge on which market.

Method (walk-forward, out-of-sample only):
  1. Run every parameter set of an arm over the full history (signals never look ahead).
  2. Split time into folds. For fold k, choose the parameter set with the best expectancy on folds
     0..k-1 only, then record how that choice performed on fold k (data it never saw).
  3. Pool the out-of-sample trades and test them: enough trades, profit factor, expectancy,
     t-statistic, bootstrap probability of a positive edge, and the most recent fold not losing.
  4. Only arms passing every test are approved. Everything else is ignored — not trading an
     unproven idea is the single most effective way to avoid losses.
"""
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from .arms import ARMS, Arm, arm_signals
from .config import ManageConfig, Market, ResearchConfig
from .indicators import atr
from .regime import classify
from .trade import Position, apply_exit, manage


def simulate_arm(df: pd.DataFrame, arm: Arm, params: dict, m: ManageConfig, allow_shorts: bool = True,
                 regime: pd.Series | None = None) -> list[tuple[pd.Timestamp, float]]:
    """Backtest one arm (one position at a time, 1 unit). Returns [(entry_time, R multiple)]."""
    sig = arm_signals(arm, df, params, regime, allow_shorts)
    signal, stops = sig["signal"].to_numpy(), sig["stop"].to_numpy()
    o, h, l, c = (df[k].to_numpy() for k in ("open", "high", "low", "close"))
    a = atr(df, 14).to_numpy()
    idx = df.index
    trades, pos = [], None

    for i in range(1, len(df)):
        if pos is None and signal[i - 1] != 0:
            side = int(signal[i - 1])
            entry = o[i] * (1 + side * m.slippage)
            risk = (entry - stops[i - 1]) * side
            if risk > entry * m.min_stop_pct:
                target = entry + side * arm.target_r * risk if arm.target_r else None
                pos = Position("", "", arm.name, side, entry, stops[i - 1], target, 1.0, risk, str(idx[i]))
                pos.pnl = -entry * m.fee_rate
                entry_i = i
        if pos is None:
            continue
        for ex in manage(pos, o[i], h[i], l[i], c[i], a[i], m):
            px = ex.price * (1 - pos.side * m.slippage)
            apply_exit(pos, ex.qty, px, ex.qty * px * m.fee_rate)
        if pos.closed:
            trades.append((idx[entry_i], pos.r_multiple))
            pos = None
    return trades


@dataclass
class Evaluation:
    market: str
    arm: str
    params: dict
    trades: int
    win_rate: float
    profit_factor: float
    expectancy_r: float
    t_stat: float
    p_positive: float
    worst_drawdown_r: float   # 95th percentile max drawdown (in R) from bootstrap
    last_fold_r: float
    approved: bool
    weight: float
    reason: str


def edge_stats(r: np.ndarray, seed: int = 0) -> dict:
    n = len(r)
    if n == 0:
        return dict(trades=0, win_rate=0.0, profit_factor=0.0, expectancy_r=0.0, t_stat=0.0,
                    p_positive=0.0, worst_drawdown_r=0.0)
    wins, losses = r[r > 0].sum(), -r[r < 0].sum()
    sd = r.std(ddof=1) if n > 1 else 0.0
    rng = np.random.default_rng(seed)
    samples = rng.choice(r, size=(2000, n), replace=True)
    equity = np.concatenate([np.zeros((2000, 1)), samples.cumsum(axis=1)], axis=1)
    drawdowns = (np.maximum.accumulate(equity, axis=1) - equity).max(axis=1)
    return dict(
        trades=n,
        win_rate=float((r > 0).mean()),
        profit_factor=float(wins / losses) if losses > 0 else float("inf"),
        expectancy_r=float(r.mean()),
        t_stat=float(r.mean() / (sd / np.sqrt(n))) if sd > 0 else 0.0,
        p_positive=float((samples.sum(axis=1) > 0).mean()),
        worst_drawdown_r=float(np.percentile(drawdowns, 95)),
    )


def walk_forward(df: pd.DataFrame, market: Market, arm: Arm, rc: ResearchConfig, m: ManageConfig,
                 allow_shorts: bool = True, regime: pd.Series | None = None) -> Evaluation:
    regime = classify(df) if regime is None else regime
    runs = [simulate_arm(df, arm, p, m, allow_shorts, regime) for p in arm.grid]
    bounds = [df.index[k] for k in np.linspace(0, len(df) - 1, rc.folds + 1).astype(int)]

    def pick(end) -> int | None:
        best, best_exp = None, 0.0
        for j, run in enumerate(runs):
            r = [x for t, x in run if t < end]
            if len(r) >= rc.min_train_trades and np.mean(r) > best_exp:
                best, best_exp = j, float(np.mean(r))
        return best

    oos, last_fold = [], []
    for k in range(1, rc.folds):
        j = pick(bounds[k])
        if j is None:
            continue  # nothing looked profitable in training -> the scientist stays out
        lo, hi = bounds[k], bounds[k + 1]
        fold_r = [x for t, x in runs[j] if lo <= t < hi or (k == rc.folds - 1 and t == hi)]
        oos.extend(fold_r)
        if k == rc.folds - 1:
            last_fold = fold_r

    st = edge_stats(np.array(oos))
    final = pick(df.index[-1] + pd.Timedelta(seconds=1))
    checks = [
        (st["trades"] >= rc.min_trades, f"only {st['trades']} out-of-sample trades"),
        (st["profit_factor"] >= rc.min_profit_factor, f"profit factor {st['profit_factor']:.2f}"),
        (st["expectancy_r"] >= rc.min_expectancy_r, f"expectancy {st['expectancy_r']:.3f}R"),
        (st["t_stat"] >= rc.min_t_stat, f"t-stat {st['t_stat']:.2f} not significant"),
        (st["p_positive"] >= rc.min_p_positive, f"P(edge>0) {st['p_positive']:.2f}"),
        (sum(last_fold) >= 0, "losing in the most recent period"),
        (final is not None, "no profitable parameters on full history"),
    ]
    failed = [msg for ok, msg in checks if not ok]
    approved = not failed
    return Evaluation(
        market=market.key, arm=arm.name, params=arm.grid[final] if final is not None else {},
        last_fold_r=float(sum(last_fold)), approved=approved,
        weight=float(np.clip(st["t_stat"] / 4, 0.25, 1.0)) if approved else 0.0,
        reason="approved" if approved else "; ".join(failed), **st,
    )


def research_market(df: pd.DataFrame, market: Market, rc: ResearchConfig, m: ManageConfig,
                    allow_shorts: bool = True) -> list[Evaluation]:
    regime = classify(df)
    return [walk_forward(df, market, arm, rc, m, allow_shorts, regime) for arm in ARMS.values()]


def research_all(settings, feed, log=print) -> dict:
    evaluations = []
    for market in settings.markets:
        try:
            df = feed.history(market, settings.research.history_bars)
        except Exception as e:
            log(f"research: could not load {market.key}: {e}")
            continue
        if len(df) < 1000:
            log(f"research: {market.key} has only {len(df)} bars, skipped")
            continue
        evaluations += research_market(df, market, settings.research, settings.manage,
                                       settings.shorts_allowed(market))
    return {
        "created": pd.Timestamp.now(tz="UTC").isoformat(),
        "approvals": [asdict(e) for e in evaluations if e.approved],
        "report": [asdict(e) for e in evaluations],
    }


def format_report(report: list[dict]) -> str:
    lines = [f"{'market':<28}{'arm':<9}{'trades':>7}{'win%':>7}{'PF':>7}{'E[R]':>8}{'t':>7}  verdict"]
    for e in report:
        lines.append(
            f"{e['market']:<28}{e['arm']:<9}{e['trades']:>7}{e['win_rate'] * 100:>6.1f}%"
            f"{min(e['profit_factor'], 99):>7.2f}{e['expectancy_r']:>8.3f}{e['t_stat']:>7.2f}  "
            f"{'✅ ' if e['approved'] else '❌ '}{e['reason']}"
        )
    return "\n".join(lines)
