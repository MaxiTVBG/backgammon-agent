"""Play backgammon against an agent in the terminal.

    python -m backgammon.play_cli --agent heuristic
    python -m backgammon.play_cli --agent td --model checkpoints/td_final.pt --color black

Enter a move by its number from the list, or in notation such as
``24/18 13/11`` (absolute numbering shown on the board, "bar" and "off").
Commands: ``l`` list moves again, ``h`` help, ``q`` quit.
"""

from __future__ import annotations

import argparse
import random

from backgammon.agents.factory import AGENT_KINDS, make_agent
from backgammon.engine import BLACK, PLAYER_NAMES, WHITE, Game, format_move, render_board
from backgammon.engine.render import SYMBOLS

HELP = __doc__


def normalise(text: str) -> tuple[str, ...]:
    return tuple(sorted(text.lower().replace(",", " ").split()))


def ask_move(game: Game, human: int) -> int:
    moves = game.moves
    labels = [format_move(m, human, absolute=True) for m, _ in moves]
    lookup = {normalise(lbl): i for i, lbl in enumerate(labels)}

    def show():
        for i, lbl in enumerate(labels):
            print(f"  [{i:>2}] {lbl}")

    show()
    while True:
        try:
            text = input("Your move> ").strip()
        except EOFError:
            raise SystemExit(0)
        if not text:
            continue
        if text in ("q", "quit", "exit"):
            raise SystemExit(0)
        if text in ("h", "help", "?"):
            print(HELP)
            continue
        if text in ("l", "list"):
            show()
            continue
        if text.isdigit() and int(text) < len(moves):
            return int(text)
        if normalise(text) in lookup:
            return lookup[normalise(text)]
        print("Not a legal move - type a number from the list, 'l' to list or 'h' for help.")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Human vs agent in the terminal")
    ap.add_argument("--agent", choices=AGENT_KINDS, default="heuristic")
    ap.add_argument("--model", default=None, help="checkpoint for td/expectimax")
    ap.add_argument("--color", choices=["white", "black"], default="white",
                    help="your colour (white = O, moves 24 -> 1)")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args(argv)

    human = WHITE if args.color == "white" else BLACK
    agent = make_agent({"kind": args.agent, "model": args.model})
    game = Game(random.Random(args.seed))
    score = [0, 0]

    while True:
        w, b = game.opening_dice
        print(f"\nOpening roll: O rolls {w}, X rolls {b} -> {SYMBOLS[game.first_player]} starts.")
        while not game.is_over():
            st = game.state
            print()
            print(render_board(st, game.dice))
            if st.turn == human:
                if len(game.moves) == 1 and not game.moves[0][0]:
                    print("You have no legal move.")
                    game.play(0)
                else:
                    game.play(ask_move(game, human))
            else:
                idx = agent.choose(st, game.dice, game.moves) if len(game.moves) > 1 else 0
                move = game.moves[idx][0]
                prob = agent.win_probability(game.moves[idx][1])
                extra = f"  (agent's win estimate {100 * (1 - prob):.0f}%)" if prob is not None else ""
                print(f"{SYMBOLS[st.turn]} ({args.agent}) plays: "
                      f"{format_move(move, st.turn, absolute=True)}{extra}")
                game.play(idx)
            if not game.is_over():
                game.roll()
        final = game.state
        winner = final.winner()
        pts = final.result_points()
        score[winner] += pts
        kind = {1: "single game", 2: "gammon (mars)", 3: "backgammon"}[pts]
        print()
        print(render_board(final))
        who = "You win" if winner == human else "Agent wins"
        print(f"{who}: {kind}, {pts} point(s).  Score - you: {score[human]}, agent: {score[1 - human]}")
        try:
            again = input("Play again? [y/N] ").strip().lower()
        except EOFError:
            break
        if again != "y":
            break
        game.reset()
    print(f"Final score - {PLAYER_NAMES[human]} (you): {score[human]}, agent: {score[1 - human]}")


if __name__ == "__main__":
    main()
