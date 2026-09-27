import argparse
import json
import os

from .config import Market, Settings, load_settings
from .engine import load_research, run_forever, save_research, simulate
from .feeds import LiveFeed, synthetic_market
from .notify import Notifier
from .research import format_report, research_all


def main():
    p = argparse.ArgumentParser(prog="octopus", description="Autonomous multi-market trading system")
    p.add_argument("--config", default="octopus.toml")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("research", help="validate every arm on every market and save approvals")
    sub.add_parser("run", help="run autonomously (paper or live, from the config)")
    sim = sub.add_parser("simulate", help="offline out-of-sample simulation of the whole system")
    sim.add_argument("--synthetic", action="store_true", help="use synthetic markets (no network)")
    sim.add_argument("--split", type=float, default=0.6)
    sub.add_parser("status", help="show saved state")
    sub.add_parser("resume", help="clear the kill switch after you reviewed what happened")
    args = p.parse_args()

    if args.cmd == "simulate" and args.synthetic:
        settings = Settings(markets=[
            Market("TREND-A", source="synthetic", asset_class="a"),
            Market("TREND-B", source="synthetic", asset_class="b"),
            Market("NOISE-A", source="synthetic", asset_class="c"),
            Market("NOISE-B", source="synthetic", asset_class="d"),
        ])
        kinds = ["trend", "trend", "random", "random"]
        data = {m.key: synthetic_market(4000, seed=i + 1, kind=k) for i, (m, k) in enumerate(zip(settings.markets, kinds))}
        run_simulation(settings, data, args.split)
        return

    settings = load_settings(args.config)
    notify = Notifier(os.path.join(settings.state_dir, "octopus.log"), settings.telegram)
    os.makedirs(settings.state_dir, exist_ok=True)

    if args.cmd == "research":
        research = research_all(settings, LiveFeed(), notify)
        save_research(settings.state_dir, research)
        print(format_report(research["report"]))
        print(f"\napproved: {len(research['approvals'])} (saved to {settings.state_dir}/research.json)")
    elif args.cmd == "run":
        run_forever(settings, notify)
    elif args.cmd == "simulate":
        feed = LiveFeed()
        data = {m.key: feed.history(m, settings.research.history_bars) for m in settings.markets}
        run_simulation(settings, data, args.split)
    elif args.cmd in ("status", "resume"):
        path = os.path.join(settings.state_dir, "state.json")
        if not os.path.exists(path):
            print("no state yet")
            return
        with open(path) as f:
            state = json.load(f)
        if args.cmd == "resume":
            state["risk"]["halted"], state["risk"]["halt_reason"] = False, ""
            state["risk"]["peak"] = None  # re-anchored to current equity on next start
            with open(path, "w") as f:
                json.dump(state, f, indent=1)
            print("kill switch cleared")
            return
        print(json.dumps(state, indent=1))
        research = load_research(settings.state_dir)
        if research:
            print(f"\nresearch from {research['created']}: {len(research['approvals'])} approved arms")


def run_simulation(settings, data, split):
    engine, report = simulate(settings, data, split)
    print("=== Research on the first {:.0%} of history ===".format(split))
    print(format_report(report))
    print("\n=== Autonomous trading on the unseen remaining {:.0%} ===".format(1 - split))
    for k, v in engine.summary().items():
        print(f"{k:>15}: {v:.4f}" if isinstance(v, float) else f"{k:>15}: {v}")
    for t in engine.closed[-10:]:
        print(f"  {t['market']:<28}{t['arm']:<8}{'L' if t['side'] > 0 else 'S'} {t['reason']:<9} {t['r']:+.2f}R")


if __name__ == "__main__":
    main()
