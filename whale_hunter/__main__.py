import argparse

from .backtest import run_backtest
from .config import Config
from .data import load_csv, synthetic_ohlcv


def main():
    p = argparse.ArgumentParser(description="Whale Hunter backtest / paper trading")
    sub = p.add_subparsers(dest="cmd", required=True)

    bt = sub.add_parser("backtest", help="run a backtest")
    src = bt.add_mutually_exclusive_group(required=True)
    src.add_argument("--csv", help="OHLCV CSV file")
    src.add_argument("--exchange", help="ccxt exchange id, e.g. binance")
    src.add_argument("--synthetic", action="store_true", help="synthetic data (pipeline test only)")
    bt.add_argument("--symbol", default="BTC/USDT")
    bt.add_argument("--timeframe", default="1h")
    bt.add_argument("--equity", type=float, default=1000.0)
    bt.add_argument("--no-trend", action="store_true", help="disable the EMA trend filter")
    bt.add_argument("--long-only", action="store_true")

    pp = sub.add_parser("paper", help="paper-trade live market data (no real orders)")
    pp.add_argument("--exchange", default="binance")
    pp.add_argument("--symbol", default="BTC/USDT")
    pp.add_argument("--timeframe", default="15m")
    pp.add_argument("--equity", type=float, default=1000.0)
    pp.add_argument("--long-only", action="store_true")

    args = p.parse_args()
    cfg = Config(allow_shorts=not args.long_only)

    if args.cmd == "paper":
        from .live import PaperTrader

        PaperTrader(args.exchange, args.symbol, args.timeframe, cfg, args.equity).run()
        return

    cfg.use_trend_filter = not args.no_trend
    if args.csv:
        df = load_csv(args.csv)
    elif args.synthetic:
        df = synthetic_ohlcv()
    else:
        import ccxt

        from .data import fetch_ohlcv

        df = fetch_ohlcv(getattr(ccxt, args.exchange)(), args.symbol, args.timeframe)

    res = run_backtest(df, cfg, args.equity)
    print(f"Bars: {len(df)}  |  {df.index[0]} -> {df.index[-1]}")
    for k, v in res.stats.items():
        print(f"{k:>15}: {v:.4f}" if isinstance(v, float) else f"{k:>15}: {v}")
    for t in res.trades[-10:]:
        side = "LONG " if t.side == 1 else "SHORT"
        print(f"{side} {t.entry_time} @ {t.entry:.2f} -> {t.exit:.2f} ({t.reason}) pnl={t.pnl:+.2f}")


if __name__ == "__main__":
    main()
