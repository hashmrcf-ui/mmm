"""The brain: runs every approved arm on every market, manages positions, enforces risk, runs forever."""
import csv
import json
import os
import time
from dataclasses import asdict

import numpy as np
import pandas as pd

from .arms import ARMS, arm_signals
from .brokers import CcxtBroker, Fill, PaperBroker
from .feeds import LiveFeed, ReplayFeed
from .indicators import atr
from .regime import CHAOS, classify
from .research import format_report, research_all, research_market
from .risk import RiskManager
from .trade import Position, apply_exit, manage


class Engine:
    def __init__(self, settings, feed, broker, approvals: list[dict], state_dir: str | None = None,
                 notify=print, intrabar: bool = False):
        self.s, self.feed, self.broker, self.notify = settings, feed, broker, notify
        self.intrabar = intrabar  # live: check stops against the current price on every poll
        self.markets = {m.key: m for m in settings.markets}
        self.positions: list[Position] = []
        self.closed: list[dict] = []
        self.last_bar: dict[str, pd.Timestamp] = {}
        self.prices: dict[str, float] = {}
        self.equity_curve: list[tuple] = []
        self.state_dir = state_dir
        self.set_approvals(approvals)
        self.risk = RiskManager(settings.risk, broker.equity([], {}))
        if state_dir:
            os.makedirs(state_dir, exist_ok=True)
            self._load()

    def set_approvals(self, approvals: list[dict]):
        self.approvals: dict[str, list[dict]] = {}
        for a in approvals:
            self.approvals.setdefault(a["market"], []).append(a)

    def equity(self) -> float:
        return self.broker.equity(self.positions, self.prices)

    # ------------------------------------------------------------------ main step
    def step(self):
        now = self.feed.now()
        candidates = []
        for key, market in self.markets.items():
            try:
                candidates += self._process_market(market, now)
            except Exception as e:
                self.notify(f"{key}: error {e}")
        equity = self.equity()
        self.equity_curve.append((now, equity))
        for event in self.risk.update(equity, now):
            self.notify(event, important=True)
        if self.risk.halted:
            if self.positions:
                self._flatten(now, "kill switch")
        else:
            self._enter(candidates, equity, now)
        self._save()

    def _process_market(self, market, now) -> list[dict]:
        key = market.key
        df = self.feed.recent(market, self.s.window)
        if df.empty:
            return []
        self.prices[key] = float(df["close"].iloc[-1])

        if self.intrabar and any(p.market == key for p in self.positions):
            price = self.feed.price(market)
            self.prices[key] = price
            for p in [p for p in self.positions if p.market == key]:
                self._intrabar_exit(market, p, price, now)

        bar_time = df.index[-1]
        if self.last_bar.get(key) == bar_time:
            return []
        self.last_bar[key] = bar_time

        # 1) manage open positions with the newly closed bar
        bar = df.iloc[-1]
        atr_value = float(atr(df, 14).iloc[-1])
        for p in [p for p in self.positions if p.market == key]:
            for ex in manage(p, bar["open"], bar["high"], bar["low"], bar["close"], atr_value, self.s.manage):
                self._exit(market, p, ex.qty, ex.price, ex.reason, now)
                if p.closed:
                    break

        # 2) ask every approved arm for a signal
        if len(df) < 300 or not self.approvals.get(key):
            return []
        regime = classify(df)
        if regime.iloc[-1] == CHAOS:
            return []
        out = []
        for appr in self.approvals[key]:
            arm = ARMS[appr["arm"]]
            sig = arm_signals(arm, df, appr["params"], regime, self.s.shorts_allowed(market)).iloc[-1]
            if sig["signal"] != 0:
                out.append({"market": market, "arm": arm, "side": int(sig["signal"]), "stop": float(sig["stop"]),
                            "weight": appr["weight"], "score": appr["t_stat"], "regime": regime.iloc[-1]})
        return out

    # ------------------------------------------------------------------ entries
    def _enter(self, candidates: list[dict], equity: float, now):
        for c in sorted(candidates, key=lambda c: c["score"], reverse=True):
            m, side, stop, arm = c["market"], c["side"], c["stop"], c["arm"]
            ok, why = self.risk.check_entry(m, side, self.positions, equity, now)
            if not ok:
                self.notify(f"skip {m.symbol} {arm.name} {'LONG' if side > 0 else 'SHORT'}: {why}")
                continue
            price = self.feed.price(m) if self.intrabar else self.prices[m.key]
            if (price - stop) * side <= price * self.s.manage.min_stop_pct:
                continue  # price already too close to / beyond the stop
            gross = sum(p.qty * self.prices.get(p.market, p.entry) for p in self.positions)
            qty = self.risk.size(equity, price, stop, c["weight"], gross)
            if qty * price < self.s.risk.min_notional:
                self.notify(f"skip {m.symbol} {arm.name}: position too small ({qty * price:.2f})")
                continue
            fill = self.broker.execute(m, side, qty, price)
            if fill is None:
                continue
            risk_unit = (fill.price - stop) * side
            target = fill.price + side * arm.target_r * risk_unit if arm.target_r and risk_unit > 0 else None
            pos = Position(m.key, m.asset_class, arm.name, side, fill.price, stop, target, fill.qty,
                           max(risk_unit, fill.price * 1e-6), str(now))
            pos.pnl = -fill.fee
            self.positions.append(pos)
            self.notify(
                f"OPEN {'LONG' if side > 0 else 'SHORT'} {m.symbol} [{arm.name}/{c['regime']}] "
                f"qty={fill.qty:.6g} @ {fill.price:.6g} stop={stop:.6g} "
                f"target={'trail' if target is None else f'{target:.6g}'} risk={risk_unit * fill.qty:.2f}",
                important=True,
            )
            if risk_unit <= 0:  # slipped through the stop on entry
                self._exit(m, pos, pos.qty, fill.price, "stop", now)

    # ------------------------------------------------------------------ exits
    def _intrabar_exit(self, market, p: Position, price: float, now):
        if (price - p.stop) * p.side <= 0:
            self._exit(market, p, p.qty, price, p.stop_reason, now)
        elif p.target is not None and (price - p.target) * p.side >= 0:
            self._exit(market, p, p.qty, price, "target", now)

    def _exit(self, market, p: Position, qty: float, price: float, reason: str, now):
        full = qty >= p.qty * 0.999
        fill = self.broker.execute(market, -p.side, min(qty, p.qty), price, reduce=True)
        if fill is None:
            if not full:
                return  # partial below exchange minimum: skip it
            fill = Fill(p.qty, price, 0.0)  # dust: stop tracking the leftover
        if full:
            fill.qty = p.qty
        pnl = apply_exit(p, fill.qty, fill.price, fill.fee)
        if not p.closed:
            self.notify(f"PARTIAL {market.symbol} {reason} @ {fill.price:.6g} pnl={pnl:+.2f}")
            return
        self.positions.remove(p)
        record = {"market": p.market, "arm": p.arm, "side": p.side, "entry_time": p.entry_time,
                  "exit_time": str(now), "entry": p.entry, "exit": fill.price, "reason": reason,
                  "pnl": round(p.pnl, 4), "r": round(p.r_multiple, 3)}
        self.closed.append(record)
        self._journal(record)
        self.notify(f"CLOSE {market.symbol} [{p.arm}] {reason} @ {fill.price:.6g} "
                    f"pnl={p.pnl:+.2f} ({p.r_multiple:+.2f}R) equity={self.equity():.2f}", important=True)
        for event in self.risk.on_trade_closed(p.pnl, now):
            self.notify(event, important=True)

    def _flatten(self, now, reason: str):
        for p in list(self.positions):
            m = self.markets[p.market]
            price = self.feed.price(m) if self.intrabar else self.prices.get(p.market, p.entry)
            self._exit(m, p, p.qty, price, reason, now)

    # ------------------------------------------------------------------ reporting
    def summary(self) -> dict:
        r = np.array([t["r"] for t in self.closed])
        pnl = np.array([t["pnl"] for t in self.closed])
        eq = pd.Series([e for _, e in self.equity_curve]) if self.equity_curve else pd.Series([self.equity()])
        start = eq.iloc[0]
        return {
            "trades": len(r),
            "win_rate": float((pnl > 0).mean()) if len(r) else 0.0,
            "loss_rate": float((pnl < 0).mean()) if len(r) else 0.0,
            "avg_r": float(r.mean()) if len(r) else 0.0,
            "avg_loss_r": float(r[r < 0].mean()) if (r < 0).any() else 0.0,
            "net_pnl": float(pnl.sum()),
            "return": float(eq.iloc[-1] / start - 1),
            "max_drawdown": float((1 - eq / eq.cummax()).max()),
            "equity": float(eq.iloc[-1]),
            "open_positions": len(self.positions),
            "halted": self.risk.halted,
        }

    # ------------------------------------------------------------------ persistence
    def _save(self):
        if not self.state_dir:
            return
        state = {"positions": [asdict(p) for p in self.positions], "risk": self.risk.to_dict(),
                 "broker": self.broker.to_dict(), "last_bar": {k: str(v) for k, v in self.last_bar.items()}}
        path = os.path.join(self.state_dir, "state.json")
        with open(path + ".tmp", "w") as f:
            json.dump(state, f, indent=1)
        os.replace(path + ".tmp", path)

    def _load(self):
        path = os.path.join(self.state_dir, "state.json")
        if not os.path.exists(path):
            return
        with open(path) as f:
            state = json.load(f)
        self.positions = [Position(**p) for p in state["positions"]]
        self.risk.load(state["risk"])
        self.broker.load(state["broker"])
        self.last_bar = {k: pd.Timestamp(v) for k, v in state["last_bar"].items()}

    def _journal(self, record: dict):
        if not self.state_dir:
            return
        path = os.path.join(self.state_dir, "trades.csv")
        new = not os.path.exists(path)
        with open(path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(record))
            if new:
                w.writeheader()
            w.writerow(record)


# ====================================================================== runners
def validate_live(settings):
    bad = [m.key for m in settings.markets if m.source != "ccxt" or m.exchange != settings.live.exchange]
    if bad:
        raise ValueError(f"live mode trades only on '{settings.live.exchange}'; remove or change: {bad}")
    missing = [k for k in ("OCTOPUS_API_KEY", "OCTOPUS_API_SECRET") if not os.environ.get(k)]
    if missing:
        raise ValueError(f"live mode needs environment variables: {missing}")


def make_broker(settings):
    if settings.mode == "live":
        validate_live(settings)
        return CcxtBroker(settings.live.exchange, settings.live.market_type, settings.live.quote,
                          settings.manage.fee_rate)
    return PaperBroker(settings.starting_equity, settings.manage.fee_rate, settings.manage.slippage)


def load_research(state_dir: str) -> dict | None:
    path = os.path.join(state_dir, "research.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def save_research(state_dir: str, research: dict):
    os.makedirs(state_dir, exist_ok=True)
    with open(os.path.join(state_dir, "research.json"), "w") as f:
        json.dump(research, f, indent=1, default=str)


def run_forever(settings, notify):
    """Fully autonomous loop: re-validates strategies periodically and trades without supervision."""
    feed = LiveFeed()
    broker = make_broker(settings)
    research = load_research(settings.state_dir)
    engine = Engine(settings, feed, broker, research["approvals"] if research else [],
                    settings.state_dir, notify, intrabar=True)
    notify(f"Octopus started — mode={settings.mode} markets={len(settings.markets)} "
           f"equity={engine.equity():.2f}", important=True)
    if engine.risk.halted:
        notify(f"HALTED ({engine.risk.halt_reason}). Review, then run `python -m octopus resume`.", important=True)
    summary_day = None
    while True:
        try:
            age_h = ((pd.Timestamp.now(tz="UTC") - pd.Timestamp(research["created"])).total_seconds() / 3600
                     if research else float("inf"))
            if age_h >= settings.research.refresh_hours:
                notify("research: re-validating every arm on every market (walk-forward)...")
                research = research_all(settings, feed, notify)
                save_research(settings.state_dir, research)
                engine.set_approvals(research["approvals"])
                notify(format_report(research["report"]))
                names = [f"{a['market']}/{a['arm']}" for a in research["approvals"]]
                notify(f"research done — approved: {names or 'NONE (standing aside, no trades)'}", important=True)
            engine.step()
            today = feed.now().date()
            if summary_day is not None and today != summary_day:
                s = engine.summary()
                notify(f"daily summary: equity={s['equity']:.2f} trades={s['trades']} win={s['win_rate']:.0%} "
                       f"open={s['open_positions']} dd={engine.risk.drawdown(s['equity']):.1%}", important=True)
            summary_day = today
            time.sleep(settings.poll_seconds)
        except KeyboardInterrupt:
            notify("stopped by user", important=True)
            return
        except Exception as e:
            notify(f"loop error: {e}", important=True)
            time.sleep(min(300, settings.poll_seconds * 5))


def simulate(settings, data: dict[str, pd.DataFrame], split: float = 0.6, notify=None):
    """Honest offline test: research on the first `split` of each history, then let the engine trade
    the remaining part bar by bar, exactly as it would live."""
    evaluations, start = [], None
    for m in settings.markets:
        df = data[m.key]
        cut = int(len(df) * split)
        evaluations += research_market(df.iloc[:cut], m, settings.research, settings.manage,
                                       settings.shorts_allowed(m))
        start = df.index[cut] if start is None else min(start, df.index[cut])
    approvals = [asdict(e) for e in evaluations if e.approved]
    feed = ReplayFeed(settings.markets, data, start=start)
    broker = PaperBroker(settings.starting_equity, settings.manage.fee_rate, settings.manage.slippage)
    engine = Engine(settings, feed, broker, approvals, None, notify or (lambda *a, **k: None))
    while True:
        engine.step()
        if not feed.advance():
            break
    return engine, [asdict(e) for e in evaluations]
