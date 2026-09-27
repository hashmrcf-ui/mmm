from dataclasses import dataclass


@dataclass
class Config:
    # --- Liquidity sweep ("hunting") ---
    swing_lookback: int = 20        # bars used to find the liquidity pool (recent high/low where stops sit)
    min_wick_ratio: float = 0.5     # sweep wick must be >= this fraction of the candle range
    # --- Whale detection ---
    volume_window: int = 50         # window for volume z-score
    volume_z: float = 2.0           # volume must be this many std devs above mean
    buy_ratio_long: float = 0.55    # taker-buy share (or whale-buy share) needed to confirm a long
    buy_ratio_short: float = 0.45   # taker-buy share at or below which a short is confirmed
    whale_trade_usd: float = 100_000  # a single trade >= this is counted as whale flow (live mode)
    # --- Trend filter ---
    use_trend_filter: bool = True
    trend_ema: int = 200
    # --- Risk management ---
    atr_period: int = 14
    stop_atr_buffer: float = 0.25   # stop placed this many ATRs beyond the sweep wick
    reward_risk: float = 2.0        # take profit at R multiple
    risk_per_trade: float = 0.01    # fraction of equity risked per trade (1%)
    max_bars_in_trade: int = 48     # time stop
    fee_rate: float = 0.001         # per side (0.1%)
    allow_shorts: bool = True
