"""Train the TD(lambda) agent by self-play.

Quick test (a few minutes on a laptop, already beats RandomAgent)::

    python -m backgammon.training.train --games 2000

Full training on a desktop::

    python -m backgammon.training.train --games 300000 --hidden 80 --eval-every 5000 --workers 15

Interrupt with Ctrl+C at any time - a checkpoint is saved, continue with
``--resume`` (training continues up to the ``--games`` total).

Hyper-parameters can also be given in a JSON file (``--config cfg.json``);
values given explicitly on the command line override the file.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
import sys
import time

import torch

from backgammon.agents.network import TDNetwork, load_checkpoint, save_network
from backgammon.training.evaluate import default_workers, make_pool, play_match
from backgammon.training.td_learner import TDLambdaLearner

DEFAULTS = {
    "games": 2000,
    "hidden": 80,
    "lr": 0.1,
    "lr_min": 0.01,
    "lr_decay_games": 200_000,
    "lambda": 0.7,
    "device": "auto",
    "threads": 1,
    "eval_every": 1000,
    "eval_games": 500,
    "save_every": 1000,
    "seed": 1,
    "workers": 0,          # 0 -> number of CPU cores - 1
    "run_name": None,
    "checkpoint_dir": "checkpoints",
    "log_dir": "runs",
    "export": None,        # copy the final model to this path
    "no_heuristic_eval": False,
}

CSV_FIELDS = ["games", "elapsed_sec", "games_per_sec", "lr",
              "win_rate_random", "ppg_random", "gammon_rate_random",
              "win_rate_heuristic", "ppg_heuristic", "gammon_rate_heuristic",
              "avg_turns", "white_win_rate"]


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="TD(lambda) self-play training for backgammon",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--config", help="JSON file with hyper-parameters")
    ap.add_argument("--games", type=int, help=f"total number of training games (default {DEFAULTS['games']})")
    ap.add_argument("--hidden", type=int, help="hidden units (default 80)")
    ap.add_argument("--lr", type=float, help="initial learning rate alpha (default 0.1)")
    ap.add_argument("--lr-min", type=float, help="final learning rate (default 0.01)")
    ap.add_argument("--lr-decay-games", type=int,
                    help="games over which alpha decays geometrically from --lr to --lr-min (default 200000)")
    ap.add_argument("--lambda", dest="lambda_", type=float, help="trace decay lambda (default 0.7)")
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"],
                    help="auto = benchmark and use the faster device (default auto)")
    ap.add_argument("--threads", type=int, help="torch CPU threads for training (default 1)")
    ap.add_argument("--resume", nargs="?", const="latest",
                    help="resume from a checkpoint (no value = latest of this run)")
    ap.add_argument("--eval-every", type=int, help="evaluate every N games (default 1000, 0 = off)")
    ap.add_argument("--eval-games", type=int, help="games per evaluation and opponent (default 500)")
    ap.add_argument("--save-every", type=int, help="save checkpoint every N games (default 1000)")
    ap.add_argument("--seed", type=int, help="random seed (default 1)")
    ap.add_argument("--workers", type=int, help="processes for evaluation (default: cores - 1)")
    ap.add_argument("--run-name", help="name of the run (default td_h<hidden>)")
    ap.add_argument("--checkpoint-dir", help="where checkpoints are written (default checkpoints)")
    ap.add_argument("--log-dir", help="where CSV / TensorBoard logs go (default runs)")
    ap.add_argument("--export", help="copy the final model to this path, e.g. checkpoints/td_final.pt")
    ap.add_argument("--no-heuristic-eval", action="store_true", default=None,
                    help="evaluate only against RandomAgent")
    return ap


def resolve_config(args: argparse.Namespace) -> dict:
    cfg = dict(DEFAULTS)
    if args.config:
        with open(args.config, encoding="utf-8") as f:
            file_cfg = json.load(f)
        unknown = set(file_cfg) - set(cfg)
        if unknown:
            raise SystemExit(f"unknown keys in config file: {sorted(unknown)}")
        cfg.update(file_cfg)
    for key in DEFAULTS:
        val = getattr(args, "lambda_" if key == "lambda" else key, None)
        if val is not None:
            cfg[key] = val
    if not cfg["run_name"]:
        cfg["run_name"] = f"td_h{cfg['hidden']}"
    if not cfg["workers"]:
        cfg["workers"] = default_workers()
    return cfg


def learning_rate(cfg: dict, games_done: int) -> float:
    """Geometric decay from lr to lr_min over lr_decay_games, then constant."""
    lr0, lr1, n = cfg["lr"], cfg["lr_min"], cfg["lr_decay_games"]
    if n <= 0 or lr1 >= lr0:
        return lr0
    frac = min(games_done / n, 1.0)
    return lr0 * (lr1 / lr0) ** frac


def choose_device(requested: str, hidden: int) -> str:
    if requested == "cpu":
        return "cpu"
    if not torch.cuda.is_available():
        if requested == "cuda":
            print("CUDA requested but not available - using CPU", file=sys.stderr)
        return "cpu"
    if requested == "cuda":
        return "cuda"
    from backgammon.training.benchmark import benchmark_training
    res = {d: benchmark_training(d, hidden=hidden, games=15) for d in ("cpu", "cuda")}
    best = max(res, key=res.get)
    print(f"device benchmark: cpu {res['cpu']:.1f} games/s, cuda {res['cuda']:.1f} games/s "
          f"-> using {best}")
    return best


class Logger:
    def __init__(self, log_dir: str, resume: bool):
        os.makedirs(log_dir, exist_ok=True)
        self.csv_path = os.path.join(log_dir, "eval.csv")
        new = not (resume and os.path.exists(self.csv_path))
        self._f = open(self.csv_path, "w" if new else "a", newline="", encoding="utf-8")
        self._w = csv.DictWriter(self._f, fieldnames=CSV_FIELDS)
        if new:
            self._w.writeheader()
        self.tb = None
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.tb = SummaryWriter(log_dir)
        except Exception:  # tensorboard not installed
            pass

    def log(self, row: dict) -> None:
        self._w.writerow({k: row.get(k, "") for k in CSV_FIELDS})
        self._f.flush()
        if self.tb:
            step = row["games"]
            for k, v in row.items():
                if k != "games" and isinstance(v, (int, float)):
                    self.tb.add_scalar(k, v, step)
            self.tb.flush()

    def scalar(self, name: str, value: float, step: int) -> None:
        if self.tb:
            self.tb.add_scalar(name, value, step)

    def close(self) -> None:
        self._f.close()
        if self.tb:
            self.tb.close()


def evaluate(net: TDNetwork, cfg: dict, pool, games_done: int) -> dict:
    spec = {"kind": "td", "hidden": net.hidden,
            "weights": {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}}
    out = {}
    opponents = ["random"] if cfg["no_heuristic_eval"] else ["random", "heuristic"]
    for opp in opponents:
        r = play_match(spec, {"kind": opp}, cfg["eval_games"], seed=games_done + 17,
                       workers=cfg["workers"], pool=pool)
        out[f"win_rate_{opp}"] = round(r.win_rate, 4)
        out[f"ppg_{opp}"] = round(r.ppg, 4)
        out[f"gammon_rate_{opp}"] = round(r.gammons_a / max(r.games, 1), 4)
    return out


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    cfg = resolve_config(args)

    torch.set_num_threads(max(1, cfg["threads"]))
    run_ckpt_dir = os.path.join(cfg["checkpoint_dir"], cfg["run_name"])
    run_log_dir = os.path.join(cfg["log_dir"], cfg["run_name"])
    os.makedirs(run_ckpt_dir, exist_ok=True)

    games_done = 0
    elapsed_before = 0.0
    best_score = -1.0
    rng = random.Random(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    net = TDNetwork(hidden=cfg["hidden"])

    if args.resume:
        path = os.path.join(run_ckpt_dir, "latest.pt") if args.resume == "latest" else args.resume
        if not os.path.exists(path):
            raise SystemExit(f"checkpoint not found: {path}")
        ck = load_checkpoint(path)
        if ck["hidden"] != cfg["hidden"]:
            print(f"note: using hidden={ck['hidden']} from the checkpoint")
            cfg["hidden"] = ck["hidden"]
            net = TDNetwork(hidden=cfg["hidden"])
        net.load_state_dict(ck["state_dict"])
        games_done = ck.get("games", 0)
        elapsed_before = ck.get("elapsed", 0.0)
        best_score = ck.get("best_score", -1.0)
        if ck.get("rng_state") is not None:
            rng.setstate(ck["rng_state"])
        print(f"resumed from {path} at {games_done} games")

    device = choose_device(cfg["device"], cfg["hidden"])
    learner = TDLambdaLearner(net, lam=cfg["lambda"], device=device)
    logger = Logger(run_log_dir, resume=bool(args.resume))
    with open(os.path.join(run_ckpt_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(dict(cfg, device_used=device), f, indent=2)

    print(f"run '{cfg['run_name']}': {cfg['games']} games, hidden={cfg['hidden']}, "
          f"lambda={cfg['lambda']}, lr={cfg['lr']}->{cfg['lr_min']}, device={device}, "
          f"eval workers={cfg['workers']}")
    print(f"checkpoints: {run_ckpt_dir}   logs: {run_log_dir}  (tensorboard --logdir {cfg['log_dir']})")

    pool = make_pool(cfg["workers"]) if cfg["eval_every"] > 0 else None

    def save(name: str) -> str:
        path = os.path.join(run_ckpt_dir, name)
        save_network(net, path, games=games_done, config=cfg, rng_state=rng.getstate(),
                     elapsed=elapsed_before + (time.perf_counter() - t_start),
                     best_score=best_score)
        return path

    def run_eval() -> None:
        nonlocal best_score
        t_eval = time.perf_counter()
        res = evaluate(net, cfg, pool, games_done)
        elapsed = elapsed_before + (time.perf_counter() - t_start)
        row = {"games": games_done, "elapsed_sec": round(elapsed, 1),
               "games_per_sec": round(window_games / max(window_time, 1e-9), 2),
               "lr": round(learning_rate(cfg, games_done), 5),
               "avg_turns": round(window_turns / max(window_games, 1), 2),
               "white_win_rate": round(window_white / max(window_games, 1), 4), **res}
        logger.log(row)
        save(f"td_{games_done}.pt")
        score = res.get("ppg_heuristic", res.get("ppg_random"))
        if score > best_score:
            best_score = score
            save("best.pt")
        msg = "  ".join(f"{k}={v}" for k, v in res.items())
        tqdm.write(f"[eval @ {games_done} games, {time.perf_counter() - t_eval:.1f}s] {msg}")
        bar.set_postfix(rnd=f"{res['win_rate_random']:.0%}",
                        heur=f"{res.get('win_rate_heuristic', 0):.0%}", refresh=False)

    from tqdm import tqdm

    t_start = time.perf_counter()
    window_games = window_turns = window_white = 0
    window_time = 0.0
    bar = tqdm(total=cfg["games"], initial=games_done, unit="game", dynamic_ncols=True,
               smoothing=0.05)
    try:
        if games_done == 0 and cfg["eval_every"] > 0:
            run_eval()
        while games_done < cfg["games"]:
            lr = learning_rate(cfg, games_done)
            t0 = time.perf_counter()
            winner, points, turns = learner.play_game(rng, lr)
            window_time += time.perf_counter() - t0
            games_done += 1
            window_games += 1
            window_turns += turns
            window_white += winner == 0
            bar.update(1)
            if cfg["save_every"] and games_done % cfg["save_every"] == 0:
                save("latest.pt")
            if cfg["eval_every"] and games_done % cfg["eval_every"] == 0:
                run_eval()
                logger.scalar("train/lr", lr, games_done)
                window_games = window_turns = window_white = 0
                window_time = 0.0
    except KeyboardInterrupt:
        bar.close()
        path = save("latest.pt")
        print(f"\ninterrupted - saved {path}; continue with --resume")
        logger.close()
        if pool:
            pool.shutdown(cancel_futures=True)
        return

    bar.close()
    if cfg["eval_every"] and games_done % cfg["eval_every"] != 0:
        run_eval()
    save("latest.pt")
    final = save("final.pt")
    if cfg["export"]:
        os.makedirs(os.path.dirname(cfg["export"]) or ".", exist_ok=True)
        shutil.copyfile(final, cfg["export"])
        print(f"exported final model to {cfg['export']}")
    logger.close()
    if pool:
        pool.shutdown()
    total = elapsed_before + time.perf_counter() - t_start
    print(f"done: {games_done} games in {total / 60:.1f} min. Final model: {final}")


if __name__ == "__main__":
    main()
