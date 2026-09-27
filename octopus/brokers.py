"""Order execution. PaperBroker simulates fills; CcxtBroker sends real orders."""
import os
from dataclasses import dataclass

from .config import Market


@dataclass
class Fill:
    qty: float
    price: float
    fee: float


class PaperBroker:
    live = False

    def __init__(self, cash: float, fee_rate: float, slippage: float):
        self.cash, self.fee_rate, self.slippage = cash, fee_rate, slippage

    def execute(self, market: Market, side: int, qty: float, price: float, reduce: bool = False) -> Fill | None:
        if qty <= 0:
            return None
        px = price * (1 + side * self.slippage)
        fee = px * qty * self.fee_rate
        self.cash -= side * qty * px + fee
        return Fill(qty, px, fee)

    def equity(self, positions: list, prices: dict) -> float:
        return self.cash + sum(p.side * p.qty * prices.get(p.market, p.entry) for p in positions)

    def to_dict(self) -> dict:
        return {"cash": self.cash}

    def load(self, d: dict):
        self.cash = d["cash"]


class CcxtBroker:
    """Real orders through any ccxt exchange (Binance, Bybit, OKX, Kraken, Alpaca stocks, ...).

    API keys are read from OCTOPUS_API_KEY / OCTOPUS_API_SECRET (and OCTOPUS_API_PASSWORD if the
    exchange needs one). Give the key TRADE permission only — never withdrawal permission.
    """
    live = True

    def __init__(self, exchange_id: str, market_type: str, quote: str, fee_rate: float):
        import ccxt

        creds = {"apiKey": os.environ["OCTOPUS_API_KEY"], "secret": os.environ["OCTOPUS_API_SECRET"]}
        if os.environ.get("OCTOPUS_API_PASSWORD"):
            creds["password"] = os.environ["OCTOPUS_API_PASSWORD"]
        self.ex = getattr(ccxt, exchange_id)({**creds, "enableRateLimit": True,
                                              "options": {"defaultType": market_type}})
        self.ex.load_markets()
        self.market_type, self.quote, self.fee_rate = market_type, quote, fee_rate

    def execute(self, market: Market, side: int, qty: float, price: float, reduce: bool = False) -> Fill | None:
        sym = market.symbol
        info = self.ex.market(sym)
        if self.market_type == "spot" and side < 0:  # never sell more than we actually hold
            qty = min(qty, float(self.ex.fetch_balance()["free"].get(info["base"], 0) or 0))
        amount = float(self.ex.amount_to_precision(sym, qty)) if qty > 0 else 0.0
        limits = info.get("limits") or {}
        min_amount = (limits.get("amount") or {}).get("min") or 0
        min_cost = (limits.get("cost") or {}).get("min") or 0
        if amount <= 0 or amount < min_amount or amount * price < min_cost:
            return None
        params = {"reduceOnly": True} if reduce and self.market_type != "spot" else {}
        order = self.ex.create_order(sym, "market", "buy" if side > 0 else "sell", amount, None, params)
        filled = float(order.get("filled") or amount)
        px = float(order.get("average") or order.get("price") or price)
        fee_info = order.get("fee") or {}
        fee = float(fee_info.get("cost") or 0)
        if fee and fee_info.get("currency") == info["base"]:
            if side > 0:
                filled -= fee      # fee was taken from the coins we bought
            fee *= px
        elif not fee:
            fee = filled * px * self.fee_rate
        return Fill(filled, px, fee)

    def equity(self, positions: list, prices: dict) -> float:
        cash = float(self.ex.fetch_balance()["total"].get(self.quote, 0) or 0)
        if self.market_type == "spot":
            return cash + sum(p.qty * prices.get(p.market, p.entry) for p in positions if p.side > 0)
        return cash + sum(p.side * p.qty * (prices.get(p.market, p.entry) - p.entry) for p in positions)

    def to_dict(self) -> dict:
        return {}

    def load(self, d: dict):
        pass
