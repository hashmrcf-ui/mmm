import tomllib
from dataclasses import dataclass, field


@dataclass
class Market:
    symbol: str
    source: str = "ccxt"          # "ccxt" (crypto exchanges, Alpaca stocks...) or "yfinance" (stocks, forex, commodities)
    exchange: str = "binance"     # ccxt exchange id (ignored for yfinance)
    asset_class: str = "crypto"   # used for correlation / exposure limits
    timeframe: str = "1h"
    allow_shorts: bool = True

    @property
    def key(self) -> str:
        return f"{self.source}:{self.symbol}:{self.timeframe}"


@dataclass
class RiskConfig:
    risk_per_trade: float = 0.005        # 0.5% of equity lost if the initial stop is hit
    max_open_positions: int = 5
    max_per_asset_class: int = 3
    max_same_direction_per_class: int = 2  # e.g. at most 2 crypto longs at once (they are highly correlated)
    max_position_notional: float = 0.3   # a single position is at most 30% of equity
    max_gross_exposure: float = 1.0      # total exposure <= equity (no leverage)
    daily_loss_limit: float = 0.02       # stop opening trades for the day after -2%
    max_drawdown: float = 0.12           # kill switch: halt + flatten after -12% from peak
    loss_streak_pause: int = 3           # pause after this many consecutive losing trades
    pause_hours: float = 24
    min_notional: float = 10.0           # skip dust orders


@dataclass
class ManageConfig:
    breakeven_r: float = 1.0      # once price moves 1R in favour, stop goes to entry (+ fees)
    breakeven_buffer: float = 0.003
    partial_r: float = 1.0        # take partial profit at 1R
    partial_fraction: float = 0.5
    trail_after_r: float = 1.5    # start ATR trailing stop after 1.5R
    trail_atr: float = 2.5
    max_bars: int = 72            # time stop
    fee_rate: float = 0.001
    slippage: float = 0.0005
    min_stop_pct: float = 0.002   # ignore setups whose stop is closer than 0.2% (noise)


@dataclass
class ResearchConfig:
    history_bars: int = 5000
    folds: int = 5
    min_trades: int = 30          # out-of-sample trades needed before an arm can be trusted
    min_train_trades: int = 5
    min_profit_factor: float = 1.2
    min_expectancy_r: float = 0.05
    min_t_stat: float = 2.5       # statistical significance of the out-of-sample edge
    min_p_positive: float = 0.97  # bootstrap probability that the edge is > 0
    refresh_hours: float = 24


@dataclass
class LiveConfig:
    exchange: str = "binance"
    market_type: str = "spot"     # "spot" (long only) or "future"
    quote: str = "USDT"


@dataclass
class Settings:
    mode: str = "paper"           # "paper" or "live"
    starting_equity: float = 1000.0
    poll_seconds: int = 20
    window: int = 1000            # bars loaded for live signal computation
    state_dir: str = "state"
    telegram: bool = False
    markets: list[Market] = field(default_factory=list)
    risk: RiskConfig = field(default_factory=RiskConfig)
    manage: ManageConfig = field(default_factory=ManageConfig)
    research: ResearchConfig = field(default_factory=ResearchConfig)
    live: LiveConfig = field(default_factory=LiveConfig)

    def shorts_allowed(self, market: Market) -> bool:
        return market.allow_shorts and not (self.mode == "live" and self.live.market_type == "spot")


def load_settings(path: str) -> Settings:
    with open(path, "rb") as f:
        data = tomllib.load(f)
    sections = {"risk": RiskConfig, "manage": ManageConfig, "research": ResearchConfig, "live": LiveConfig}
    kwargs = {k: v for k, v in data.items() if k not in sections and k not in ("markets", "notify")}
    for name, cls in sections.items():
        kwargs[name] = cls(**data.get(name, {}))
    kwargs["markets"] = [Market(**m) for m in data.get("markets", [])]
    kwargs["telegram"] = data.get("notify", {}).get("telegram", False)
    settings = Settings(**kwargs)
    if settings.mode not in ("paper", "live"):
        raise ValueError("mode must be 'paper' or 'live'")
    return settings
