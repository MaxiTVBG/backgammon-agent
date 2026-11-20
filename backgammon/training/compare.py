"""Compare the TD agent (1-ply) with the TD+Expectimax agent (2-ply).

    python -m backgammon.training.compare --model checkpoints/td_final.pt --games 1000

Prints a Markdown table (win rate and points per game) that can be pasted
into the documentation.
"""

from __future__ import annotations

import argparse
import time

from backgammon.agents.factory import default_model_path
from backgammon.training.evaluate import default_workers, make_pool, play_match


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default=None)
    ap.add_argument("--games", type=int, default=1000)
    ap.add_argument("--expectimax-games", type=int, default=None,
                    help="games for matches involving expectimax (slower), default = --games / 2")
    ap.add_argument("--top-k", type=int, default=6)
    ap.add_argument("--workers", type=int, default=default_workers())
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args(argv)

    model = args.model or default_model_path()
    td = {"kind": "td", "model": model}
    ex = {"kind": "expectimax", "model": model, "top_k": args.top_k}
    ex_games = args.expectimax_games or max(args.games // 2, 2)
    matches = [
        ("TD (1-ply)", td, "Random", {"kind": "random"}, args.games),
        ("TD (1-ply)", td, "Heuristic", {"kind": "heuristic"}, args.games),
        ("TD+Expectimax (2-ply)", ex, "Random", {"kind": "random"}, ex_games),
        ("TD+Expectimax (2-ply)", ex, "Heuristic", {"kind": "heuristic"}, ex_games),
        ("TD+Expectimax (2-ply)", ex, "TD (1-ply)", td, ex_games),
    ]
    print(f"model: {model}\n")
    print("| Agent | Opponent | Games | Win rate | Points/game | Time |")
    print("|---|---|---|---|---|---|")
    pool = make_pool(args.workers)
    try:
        for name_a, a, name_b, b, n in matches:
            t0 = time.perf_counter()
            r = play_match(a, b, n, seed=args.seed, workers=args.workers, pool=pool, chunk=10)
            print(f"| {name_a} | {name_b} | {r.games} | {100 * r.win_rate:.1f}% | "
                  f"{r.ppg:+.3f} | {time.perf_counter() - t0:.0f}s |", flush=True)
    finally:
        if pool:
            pool.shutdown()


if __name__ == "__main__":
    main()
