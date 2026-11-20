"""Plot training curves from the evaluation CSV written by train.py.

    python -m backgammon.training.plot runs/td_h80/eval.csv
    python -m backgammon.training.plot runs/td_h40/eval.csv runs/td_h80/eval.csv --out docs/img/compare.png
"""

from __future__ import annotations

import argparse
import csv
import os


def read_csv(path: str) -> dict[str, list[float]]:
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    cols: dict[str, list[float]] = {}
    for row in rows:
        for k, v in row.items():
            try:
                cols.setdefault(k, []).append(float(v))
            except (TypeError, ValueError):
                cols.setdefault(k, []).append(float("nan"))
    return cols


def plot(paths: list[str], out: str, title: str | None = None) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    styles = {"random": "-", "heuristic": "--"}
    for path in paths:
        data = read_csv(path)
        label = os.path.basename(os.path.dirname(os.path.abspath(path))) if len(paths) > 1 else ""
        games = data["games"]
        for opp, ls in styles.items():
            key = f"win_rate_{opp}"
            if key in data and any(v == v for v in data[key]):
                axes[0].plot(games, [100 * v for v in data[key]], ls, marker="o", markersize=3,
                             label=f"{label} vs {opp}".strip())
            key = f"ppg_{opp}"
            if key in data and any(v == v for v in data[key]):
                axes[1].plot(games, data[key], ls, marker="o", markersize=3,
                             label=f"{label} vs {opp}".strip())
    axes[0].axhline(50, color="grey", lw=0.8, ls=":")
    axes[0].set_ylim(0, 100)
    axes[0].set_xlabel("training games (self-play)")
    axes[0].set_ylabel("win rate, %")
    axes[0].set_title("Win rate of the TD agent")
    axes[1].axhline(0, color="grey", lw=0.8, ls=":")
    axes[1].set_xlabel("training games (self-play)")
    axes[1].set_ylabel("points per game")
    axes[1].set_title("Average net points per game (gammons count 2, backgammons 3)")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend()
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Plot win rate vs training games")
    ap.add_argument("csv", nargs="+", help="eval.csv file(s) written by train.py")
    ap.add_argument("--out", default="docs/img/training_curve.png")
    ap.add_argument("--title", default=None)
    args = ap.parse_args(argv)
    print("saved", plot(args.csv, args.out, args.title))


if __name__ == "__main__":
    main()
