"""Position model and trade management rules shared by research, paper and live trading."""
import math
from dataclasses import dataclass

from .config import ManageConfig


@dataclass
class Position:
    market: str
    asset_class: str
    arm: str
    side: int              # 1 long, -1 short
    entry: float
    stop: float
    target: float | None
    qty: float
    risk: float            # initial risk per unit (price distance entry -> stop)
    entry_time: str
    initial_qty: float = 0.0
    initial_stop: float = math.nan
    best: float = math.nan
    bars: int = 0
    partial_done: bool = False
    pnl: float = 0.0       # realised pnl incl. fees
    stop_reason: str = "stop"

    def __post_init__(self):
        if not self.initial_qty:
            self.initial_qty = self.qty
        if math.isnan(self.initial_stop):
            self.initial_stop = self.stop
        if math.isnan(self.best):
            self.best = self.entry

    @property
    def closed(self) -> bool:
        return self.qty <= self.initial_qty * 1e-9

    @property
    def r_multiple(self) -> float:
        return self.pnl / (self.risk * self.initial_qty)


@dataclass
class ExitIntent:
    qty: float
    price: float
    reason: str


def manage(pos: Position, o: float, h: float, l: float, c: float, atr_value: float,
           m: ManageConfig) -> list[ExitIntent]:
    """Apply one bar to an open position and return the exits it triggers.

    Order (conservative): stop first, then target, then partial profit; afterwards the stop is
    ratcheted to breakeven / trailing for the next bar. Stops only ever move in the trade's favour.
    """
    s = pos.side
    if (s == 1 and l <= pos.stop) or (s == -1 and h >= pos.stop):
        return [ExitIntent(pos.qty, min(o, pos.stop) if s == 1 else max(o, pos.stop), pos.stop_reason)]
    if pos.target is not None and ((s == 1 and h >= pos.target) or (s == -1 and l <= pos.target)):
        return [ExitIntent(pos.qty, max(o, pos.target) if s == 1 else min(o, pos.target), "target")]

    exits, remaining = [], pos.qty
    if not pos.partial_done and m.partial_fraction > 0:
        level = pos.entry + s * m.partial_r * pos.risk
        if (s == 1 and h >= level) or (s == -1 and l <= level):
            q = min(remaining, pos.initial_qty * m.partial_fraction)
            exits.append(ExitIntent(q, max(o, level) if s == 1 else min(o, level), "partial"))
            pos.partial_done, remaining = True, remaining - q

    pos.best = max(pos.best, h) if s == 1 else min(pos.best, l)
    gain_r = (pos.best - pos.entry) * s / pos.risk
    moves = []
    if gain_r >= m.breakeven_r:
        moves.append((pos.entry * (1 + s * m.breakeven_buffer), "breakeven"))
    if gain_r >= m.trail_after_r and atr_value > 0:
        moves.append((pos.best - s * m.trail_atr * atr_value, "trail"))
    for level, reason in moves:
        if (level - pos.stop) * s > 0:
            pos.stop, pos.stop_reason = level, reason

    pos.bars += 1
    if pos.bars >= m.max_bars and remaining > 0:
        exits.append(ExitIntent(remaining, c, "time"))
    return exits


def apply_exit(pos: Position, qty: float, price: float, fee: float) -> float:
    """Book a (partial) exit; returns the pnl of this fill."""
    q = min(qty, pos.qty)
    pnl = q * (price - pos.entry) * pos.side - fee
    pos.pnl += pnl
    pos.qty -= q
    return pnl
