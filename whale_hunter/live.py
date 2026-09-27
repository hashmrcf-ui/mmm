"""Paper trading: runs the strategy on live exchange data and simulates fills. No real orders are sent."""
import time
from datetime import datetime, timezone

from .config import Config
from .data import fetch_ohlcv, whale_flow
from .strategy import generate_signals


class PaperTrader:
    def __init__(self, exchange_id: str, symbol: str, timeframe: str, cfg: Config, equity: float):
        import ccxt

        self.ex = getattr(ccxt, exchange_id)({"enableRateLimit": True})
        self.symbol, self.timeframe, self.cfg = symbol, timeframe, cfg
        self.equity = equity
        self.pos = None
        self.bars_in = 0
        self.last_bar = None

    def log(self, msg: str):
        print(f"[{datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S}] {msg}", flush=True)

    def snapshot(self):
        df = fetch_ohlcv(self.ex, self.symbol, self.timeframe, limit=max(300, self.cfg.trend_ema + 50))
        try:
            flow = whale_flow(self.ex.fetch_trades(self.symbol, limit=1000), self.timeframe, self.cfg.whale_trade_usd)
            last_closed = df.index[-2]
            # Use real whale prints only if they cover the last closed candle; otherwise the
            # strategy falls back to candle-based buy pressure.
            if last_closed in flow.index and flow.loc[last_closed].sum() > 0:
                df = df.join(flow, how="left")
        except Exception as e:  # trades endpoint is optional
            self.log(f"whale flow unavailable: {e}")
        return df

    def step(self):
        df = self.snapshot()
        closed = df.iloc[:-1]  # last candle is still forming
        bar = closed.index[-1]
        if bar == self.last_bar:
            return
        self.last_bar = bar
        last = closed.iloc[-1]
        price = df["close"].iloc[-1]

        if self.pos:
            self.bars_in += 1
            side, entry, stop, target, qty = self.pos
            hit_stop = last["low"] <= stop if side == 1 else last["high"] >= stop
            hit_target = last["high"] >= target if side == 1 else last["low"] <= target
            exit_price = stop if hit_stop else target if hit_target else None
            if exit_price is None and self.bars_in >= self.cfg.max_bars_in_trade:
                exit_price = last["close"]
            if exit_price is not None:
                pnl = (exit_price - entry) * qty * side - (entry + exit_price) * qty * self.cfg.fee_rate
                self.equity += pnl
                self.log(f"EXIT @ {exit_price:.2f} pnl={pnl:+.2f} equity={self.equity:.2f}")
                self.pos = None

        sig = generate_signals(closed, self.cfg).iloc[-1]
        self.log(
            f"bar {bar} close={last['close']:.2f} vol_z={sig['vol_z']:.2f} "
            f"buy_ratio={sig['buy_ratio']:.2f} signal={int(sig['signal'])}"
        )
        if self.pos is None and sig["signal"] != 0:
            side = int(sig["signal"])
            risk = (price - sig["stop"]) * side
            if risk > 0:
                qty = min(self.equity * self.cfg.risk_per_trade / risk, self.equity / price)
                target = price + side * self.cfg.reward_risk * risk
                self.pos = (side, price, sig["stop"], target, qty)
                self.bars_in = 0
                self.log(
                    f"{'LONG' if side == 1 else 'SHORT'} @ {price:.2f} stop={sig['stop']:.2f} "
                    f"target={target:.2f} qty={qty:.6f}"
                )

    def run(self, poll_seconds: int = 30):
        self.log(f"Paper trading {self.symbol} {self.timeframe} on {self.ex.id} — equity {self.equity:.2f}")
        while True:
            try:
                self.step()
            except Exception as e:
                self.log(f"error: {e}")
            time.sleep(poll_seconds)
