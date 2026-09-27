"""Live-path tests against fake exchange objects (no network)."""
import pandas as pd
import pytest

from octopus.brokers import CcxtBroker
from octopus.config import Market
from octopus.feeds import LiveFeed

HOUR = 3_600_000


class FakeExchange:
    def __init__(self, start_ms, n, spot=True):
        self.start, self.n, self.spot, self.calls, self.orders = start_ms, n, spot, 0, []
        self.balance = {"total": {"USDT": 500.0}, "free": {"BTC": 0.3}}

    def market(self, sym):
        return {"spot": self.spot, "base": "BTC", "limits": {"amount": {"min": 0.001}, "cost": {"min": 5}}}

    def market_id(self, sym):
        return sym.replace("/", "")

    def _rows(self, since, limit):
        first = max(0, -(-(since - self.start) // HOUR))
        return [self.start + i * HOUR for i in range(first, min(self.n, first + limit))]

    def publicGetKlines(self, params):
        self.calls += 1
        return [[t, "1", "2", "0.5", "1.5", "10", t + HOUR - 1, "0", 5, "7", "0", "0"]
                for t in self._rows(params["startTime"], params["limit"])]

    def fetch_ohlcv(self, sym, tf, since=None, limit=None):
        self.calls += 1
        return [[t, 1.0, 2.0, 0.5, 1.5, 10.0] for t in self._rows(since, min(limit, 500))]

    def amount_to_precision(self, sym, qty):
        return f"{int(qty * 1000) / 1000:.3f}"

    def fetch_balance(self):
        return self.balance

    def create_order(self, sym, typ, side, amount, price, params):
        self.orders.append((sym, typ, side, amount, params))
        return {"filled": amount, "average": 100.0, "fee": {"cost": 0.001, "currency": "BTC"}}


def feed_with(fake, now):
    f = LiveFeed()
    f._exchanges["binance"] = fake
    f.now = lambda: now
    return f


def test_binance_feed_paginates_and_reads_taker_buy_volume():
    now = pd.Timestamp("2024-03-01T10:30Z")
    start = int((now - pd.Timedelta(hours=3000)).timestamp() * 1000) // HOUR * HOUR
    fake = FakeExchange(start, 3001)
    df = feed_with(fake, now).recent(Market("BTC/USDT"), 2500)
    assert len(df) == 2500 and fake.calls >= 3
    assert (df["taker_buy_volume"] == 7.0).all()
    assert df.index[-1] + pd.Timedelta(hours=1) <= now  # the forming candle is dropped


def test_generic_ccxt_feed_without_taker_volume():
    now = pd.Timestamp("2024-03-01T10:30Z")
    start = int((now - pd.Timedelta(hours=1200)).timestamp() * 1000) // HOUR * HOUR
    fake = FakeExchange(start, 1201, spot=False)
    df = feed_with(fake, now).recent(Market("BTC/USDT"), 1000)
    assert len(df) == 1000 and "taker_buy_volume" not in df.columns


def make_broker(market_type="spot"):
    b = object.__new__(CcxtBroker)
    b.ex, b.market_type, b.quote, b.fee_rate = FakeExchange(0, 0), market_type, "USDT", 0.001
    return b


def test_live_broker_buy_deducts_base_fee_and_respects_minimums():
    b = make_broker()
    fill = b.execute(Market("BTC/USDT"), 1, 0.12345, 100.0)
    assert b.ex.orders[-1][:4] == ("BTC/USDT", "market", "buy", 0.123)
    assert fill.qty == pytest.approx(0.122) and fill.fee == pytest.approx(0.1)
    assert b.execute(Market("BTC/USDT"), 1, 0.0004, 100.0) is None  # below exchange minimum


def test_live_spot_sell_never_exceeds_holdings():
    b = make_broker()
    b.execute(Market("BTC/USDT"), -1, 5.0, 100.0, reduce=True)
    assert b.ex.orders[-1][2:4] == ("sell", 0.3)


def test_live_futures_close_is_reduce_only():
    b = make_broker("future")
    b.execute(Market("BTC/USDT"), -1, 0.5, 100.0, reduce=True)
    assert b.ex.orders[-1][4] == {"reduceOnly": True}
