"""Tournament between two agents.

Examples::

    python -m backgammon.tournament heuristic random --games 1000
    python -m backgammon.tournament td heuristic --model-a checkpoints/td_final.pt --games 2000 --workers 8
"""

from __future__ import annotations

import argparse
import time

from backgammon.agents.factory import AGENT_KINDS
from backgammon.training.evaluate import default_workers, play_match


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Play N games between two agents (colours alternate).")
    ap.add_argument("agent_a", choices=AGENT_KINDS)
    ap.add_argument("agent_b", choices=AGENT_KINDS)
    ap.add_argument("--games", type=int, default=1000)
    ap.add_argument("--model-a", default=None, help="checkpoint for agent A (td/expectimax)")
    ap.add_argument("--model-b", default=None, help="checkpoint for agent B (td/expectimax)")
    ap.add_argument("--workers", type=int, default=default_workers())
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    spec_a = {"kind": args.agent_a, "model": args.model_a}
    spec_b = {"kind": args.agent_b, "model": args.model_b}
    t0 = time.perf_counter()
    res = play_match(spec_a, spec_b, args.games, seed=args.seed, workers=args.workers)
    dt = time.perf_counter() - t0
    print(res.summary(args.agent_a, args.agent_b))
    print(f"Win rate {args.agent_a}: {100 * res.win_rate:.2f}%   "
          f"Points per game: {res.ppg:+.3f}   "
          f"Avg turns: {res.turns / max(res.games, 1):.1f}   "
          f"Time: {dt:.1f}s ({res.games / dt:.1f} games/s)")


if __name__ == "__main__":
    main()
