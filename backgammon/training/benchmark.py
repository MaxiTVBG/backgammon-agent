"""Benchmarks: rules engine speed and CPU vs CUDA for TD training.

    python -m backgammon.training.benchmark
    python -m backgammon.training.benchmark --hidden 160 --games 50

The TD network is tiny (198 -> 80 -> 1) and learning is strictly sequential
(one position at a time, each move depends on the previous update), so the
GPU spends most of its time waiting for kernel launches and CPU<->GPU
synchronisation. This script measures it instead of guessing; ``train.py
--device auto`` runs a short version of the same test.
"""

from __future__ import annotations

import argparse
import random
import time

import torch

from backgammon.engine.board import State, generate_moves
from backgammon.engine.game import opening_roll, roll_dice


def benchmark_engine(seconds: float = 3.0) -> dict:
    rng = random.Random(0)
    t0 = time.perf_counter()
    calls = games = n_after = 0
    while time.perf_counter() - t0 < seconds:
        first, dice = opening_roll(rng)
        s = State.initial(first)
        while not s.is_over():
            res = generate_moves(s, dice)
            calls += 1
            n_after += len(res)
            s = res[rng.randrange(len(res))][1]
            dice = roll_dice(rng)
        games += 1
    el = time.perf_counter() - t0
    return {"movegen_per_sec": calls / el, "afterstates_per_sec": n_after / el,
            "random_games_per_sec": games / el, "avg_afterstates": n_after / max(calls, 1)}


def benchmark_training(device: str, hidden: int = 80, games: int = 30, warmup: int = 3) -> float:
    """Self-play TD training speed in games/s on the given device."""
    from backgammon.agents.network import TDNetwork
    from backgammon.training.td_learner import TDLambdaLearner

    torch.manual_seed(0)
    learner = TDLambdaLearner(TDNetwork(hidden=hidden), lam=0.7, device=device)
    rng = random.Random(0)
    for _ in range(warmup):
        learner.play_game(rng, 0.1)
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(games):
        learner.play_game(rng, 0.1)
    if device == "cuda":
        torch.cuda.synchronize()
    return games / (time.perf_counter() - t0)


def benchmark_inference(device: str, hidden: int = 80, batch: int = 20, reps: int = 2000) -> float:
    """Batched forward passes per second (as used when choosing a move)."""
    from backgammon.agents.network import TDNetwork

    net = TDNetwork(hidden=hidden).to(device)
    x = torch.rand(batch, 198)
    with torch.no_grad():
        for _ in range(20):
            net(x.to(device)).cpu()
        t0 = time.perf_counter()
        for _ in range(reps):
            int(torch.argmax(net(x.to(device))))
    return reps / (time.perf_counter() - t0)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--hidden", type=int, default=80)
    ap.add_argument("--games", type=int, default=40, help="training games per device")
    ap.add_argument("--threads", type=int, default=1, help="torch CPU threads")
    args = ap.parse_args(argv)
    torch.set_num_threads(args.threads)

    print("== Rules engine (pure Python, one core) ==")
    e = benchmark_engine()
    print(f"  move generation : {e['movegen_per_sec']:>10,.0f} calls/s "
          f"({e['avg_afterstates']:.1f} afterstates per call)")
    print(f"  afterstates     : {e['afterstates_per_sec']:>10,.0f} /s")
    print(f"  random games    : {e['random_games_per_sec']:>10,.1f} games/s")

    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    if torch.cuda.is_available():
        print(f"\nCUDA device: {torch.cuda.get_device_name(0)}")
    else:
        print("\nCUDA not available - only CPU is measured")
    print(f"\n== Network (hidden={args.hidden}, cpu threads={args.threads}) ==")
    results = {}
    for d in devices:
        inf = benchmark_inference(d, args.hidden)
        tr = benchmark_training(d, args.hidden, args.games)
        results[d] = tr
        print(f"  {d:>4}: batch inference {inf:>8,.0f} /s   TD self-play training {tr:>6.1f} games/s")
    best = max(results, key=results.get)
    print(f"\nFaster device for training: {best}")
    print(f"Estimated time for 100k games: {100_000 / results[best] / 3600:.1f} h, "
          f"300k games: {300_000 / results[best] / 3600:.1f} h (without evaluation)")


if __name__ == "__main__":
    main()
