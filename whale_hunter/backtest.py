from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import Config
from .strategy import generate_signals


@dataclass
class Trade:
    side: int
    entry_time: object
    entry: float
    stop: float
    target: float
    qty: float
    exit_time: object = None
    exit: float = np.nan
    reason: str = ""
    pnl: float = 0.0


@dataclass
class Result:
    trades: list = field(default_factory=list)
    equity: pd.Series = None
    stats: dict = field(default_factory=dict)


def run_backtest(df: pd.DataFrame, cfg: Config, initial_equity: float = 1000.0) -> Result:
    data = generate_signals(df, cfg)
    o, h, l, c = (data[k].to_numpy() for k in ("open", "high", "low", "close"))
    sig, stops = data["signal"].to_numpy(), data["stop"].to_numpy()
    idx = data.index

    equity = initial_equity
    curve = np.full(len(data), np.nan)
    trades: list[Trade] = []
    pos: Trade | None = None
    bars_in = 0

    for i in range(len(data)):
        if pos is not None:
            bars_in += 1
            exit_price, reason = None, ""
            # Conservative: if both stop and target are hit in the same bar, assume the stop hit first.
            if pos.side == 1:
                if l[i] <= pos.stop:
                    exit_price, reason = min(o[i], pos.stop), "stop"
                elif h[i] >= pos.target:
                    exit_price, reason = max(o[i], pos.target), "target"
            else:
                if h[i] >= pos.stop:
                    exit_price, reason = max(o[i], pos.stop), "stop"
                elif l[i] <= pos.target:
                    exit_price, reason = min(o[i], pos.target), "target"
            if exit_price is None and bars_in >= cfg.max_bars_in_trade:
                exit_price, reason = c[i], "time"
            if exit_price is not None:
                gross = (exit_price - pos.entry) * pos.qty * pos.side
                fees = (pos.entry + exit_price) * pos.qty * cfg.fee_rate
                pos.exit_time, pos.exit, pos.reason, pos.pnl = idx[i], exit_price, reason, gross - fees
                equity += pos.pnl
                trades.append(pos)
                pos = None

        # Enter at this bar's open on the previous bar's signal.
        if pos is None and i > 0 and sig[i - 1] != 0 and equity > 0:
            side, entry, stop = int(sig[i - 1]), o[i], stops[i - 1]
            risk = (entry - stop) * side
            if risk > 0:
                qty = min(equity * cfg.risk_per_trade / risk, equity / entry)  # no leverage
                target = entry + side * cfg.reward_risk * risk
                pos = Trade(side, idx[i], entry, stop, target, qty)
                bars_in = 0
                # Conservative: a stop hit inside the entry bar counts as a loss; targets are only
                # checked from the next bar onwards.
                if (side == 1 and l[i] <= stop) or (side == -1 and h[i] >= stop):
                    gross = (stop - entry) * qty * side
                    fees = (entry + stop) * qty * cfg.fee_rate
                    pos.exit_time, pos.exit, pos.reason, pos.pnl = idx[i], stop, "stop", gross - fees
                    equity += pos.pnl
                    trades.append(pos)
                    pos = None
        curve[i] = equity

    eq = pd.Series(curve, index=idx)
    return Result(trades, eq, compute_stats(trades, eq, initial_equity))


def compute_stats(trades: list, equity: pd.Series, initial_equity: float) -> dict:
    pnls = np.array([t.pnl for t in trades])
    wins, losses = pnls[pnls > 0], pnls[pnls <= 0]
    peak = equity.cummax()
    max_dd = ((equity - peak) / peak).min() if len(equity) else 0.0
    return {
        "trades": len(trades),
        "win_rate": float(len(wins) / len(pnls)) if len(pnls) else 0.0,
        "profit_factor": float(wins.sum() / -losses.sum()) if losses.sum() < 0 else float("inf") if len(wins) else 0.0,
        "total_return": float(equity.iloc[-1] / initial_equity - 1) if len(equity) else 0.0,
        "final_equity": float(equity.iloc[-1]) if len(equity) else initial_equity,
        "max_drawdown": float(max_dd),
        "expectancy": float(pnls.mean()) if len(pnls) else 0.0,
    }
