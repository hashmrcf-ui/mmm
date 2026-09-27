"""Portfolio-level risk manager: the layer that keeps losses small no matter what the arms do."""
import pandas as pd

from .config import Market, RiskConfig


class RiskManager:
    def __init__(self, cfg: RiskConfig, equity: float):
        self.cfg = cfg
        self.peak = equity
        self.day = None
        self.day_start = equity
        self.loss_streak = 0
        self.paused_until: pd.Timestamp | None = None
        self.halted = False
        self.halt_reason = ""

    # ---- state ----
    def update(self, equity: float, now: pd.Timestamp) -> list[str]:
        events = []
        if self.day != now.date():
            self.day, self.day_start = now.date(), equity
        self.peak = max(self.peak, equity)
        dd = self.drawdown(equity)
        if not self.halted and dd >= self.cfg.max_drawdown:
            self.halted = True
            self.halt_reason = f"max drawdown {dd:.1%} reached"
            events.append(f"KILL SWITCH: {self.halt_reason} — all positions closed, trading halted")
        return events

    def drawdown(self, equity: float) -> float:
        return 1 - equity / self.peak if self.peak > 0 else 0.0

    def daily_loss(self, equity: float) -> float:
        return 1 - equity / self.day_start if self.day_start > 0 else 0.0

    def on_trade_closed(self, pnl: float, now: pd.Timestamp) -> list[str]:
        if pnl >= 0:
            self.loss_streak = 0
            return []
        self.loss_streak += 1
        if self.loss_streak >= self.cfg.loss_streak_pause:
            self.loss_streak = 0
            self.paused_until = now + pd.Timedelta(hours=self.cfg.pause_hours)
            return [f"{self.cfg.loss_streak_pause} losses in a row — pausing new entries until {self.paused_until}"]
        return []

    # ---- decisions ----
    def check_entry(self, market: Market, side: int, positions: list, equity: float,
                    now: pd.Timestamp) -> tuple[bool, str]:
        c = self.cfg
        if self.halted:
            return False, f"halted ({self.halt_reason})"
        if self.paused_until is not None and now < self.paused_until:
            return False, "cooling down after a losing streak"
        if self.daily_loss(equity) >= c.daily_loss_limit:
            return False, "daily loss limit reached"
        if any(p.market == market.key for p in positions):
            return False, "already positioned in this market"
        if len(positions) >= c.max_open_positions:
            return False, "max open positions"
        same_class = [p for p in positions if p.asset_class == market.asset_class]
        if len(same_class) >= c.max_per_asset_class:
            return False, f"max positions in {market.asset_class}"
        if sum(p.side == side for p in same_class) >= c.max_same_direction_per_class:
            return False, f"too many correlated {market.asset_class} positions in the same direction"
        return True, ""

    def size(self, equity: float, entry: float, stop: float, weight: float, gross_exposure: float) -> float:
        """Quantity so that hitting the stop loses risk_per_trade × weight × drawdown-scale of equity."""
        c = self.cfg
        dist = abs(entry - stop)
        if dist <= 0 or entry <= 0 or equity <= 0:
            return 0.0
        dd_scale = max(0.25, 1 - self.drawdown(equity) / c.max_drawdown)  # shrink size while in drawdown
        qty = equity * c.risk_per_trade * weight * dd_scale / dist
        qty = min(qty, c.max_position_notional * equity / entry)
        room = c.max_gross_exposure * equity - gross_exposure
        return max(0.0, min(qty, room / entry))

    # ---- persistence ----
    def to_dict(self) -> dict:
        return {
            "peak": self.peak, "day": str(self.day) if self.day else None, "day_start": self.day_start,
            "loss_streak": self.loss_streak, "halted": self.halted, "halt_reason": self.halt_reason,
            "paused_until": self.paused_until.isoformat() if self.paused_until is not None else None,
        }

    def load(self, d: dict):
        self.peak, self.day_start = d["peak"] or 0.0, d["day_start"]
        self.day = pd.Timestamp(d["day"]).date() if d.get("day") else None
        self.loss_streak, self.halted, self.halt_reason = d["loss_streak"], d["halted"], d["halt_reason"]
        self.paused_until = pd.Timestamp(d["paused_until"]) if d.get("paused_until") else None
