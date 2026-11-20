"""Hand-written rule based agent used as a baseline opponent."""

from __future__ import annotations

from backgammon.engine.board import BAR, OFF, State

from .base import Agent

# Value of owning (2+ checkers) each of our own points 1..24.
POINT_VALUE = [0.0,
               1.0, 1.2, 1.6, 2.2, 3.0, 3.0,   # home board (5 and 6 are key)
               2.4, 1.4, 0.8, 0.5, 0.3, 0.2,   # bar point (7) blocks well
               0.2, 0.2, 0.2, 0.2, 0.3, 0.5,
               0.8, 1.0, 1.1, 1.1, 0.9, 0.6]   # anchors in the opponent's home


class HeuristicAgent(Agent):
    """Scores every afterstate with hand tuned features and picks the best.

    Features (from the mover's point of view):
      * hitting: opponent checkers on the bar
      * making points (more for the home board and the bar point), primes
      * safety: blots weighted by how many opponent shots hit them and how
        many pips we lose when hit
      * advancing: pip count difference and checkers borne off
      * in a pure race only pips, borne off checkers and home-board entry count
    """

    name = "heuristic"

    def __init__(self, hit_weight: float = 4.0, blot_weight: float = 1.0,
                 point_weight: float = 1.0, pip_weight: float = 0.05,
                 prime_weight: float = 1.0, off_weight: float = 1.0):
        self.hit_weight = hit_weight
        self.blot_weight = blot_weight
        self.point_weight = point_weight
        self.pip_weight = pip_weight
        self.prime_weight = prime_weight
        self.off_weight = off_weight

    # ------------------------------------------------------------- scoring
    def score(self, me: tuple[int, ...], op: tuple[int, ...]) -> float:
        """Score position for the player owning ``me`` (who just moved)."""
        my_pips = sum(i * me[i] for i in range(1, 26))
        op_pips = sum(i * op[i] for i in range(1, 26))

        # Opponent checker at its own point k is at our point 25 - k; opponent
        # moves towards higher numbers in our numbering. Bar = our point 0.
        op_positions = [25 - k for k in range(1, 26) if op[k]]  # bar -> 0
        my_positions = [p for p in range(1, 26) if me[p]]
        contact = bool(op_positions and my_positions) and min(op_positions) < max(my_positions)

        score = self.pip_weight * (op_pips - my_pips) + self.off_weight * me[OFF]

        if not contact:
            # Pure race: bring checkers home quickly, bear off.
            home = sum(me[1:7]) + me[OFF]
            return score + 0.3 * home + 0.5 * me[OFF] - 0.02 * my_pips

        score += self.hit_weight * op[BAR]
        if op[BAR]:
            # Closed points in our home board make the hit far stronger.
            score += 0.5 * op[BAR] * sum(1 for p in range(1, 7) if me[p] >= 2)

        # Points and primes.
        run = best_run = 0
        for p in range(1, 25):
            if me[p] >= 2:
                score += self.point_weight * POINT_VALUE[p]
                run += 1
                best_run = max(best_run, run)
            else:
                run = 0
            if me[p] > 4:
                score -= 0.15 * (me[p] - 4)  # avoid big stacks
        if best_run >= 3:
            score += self.prime_weight * (best_run - 2) ** 1.5

        # Blot safety: count opponent shots (direct <= 6 weighted more).
        op_cnt = {}
        for k in range(1, 26):
            if op[k]:
                op_cnt[25 - k if k != BAR else 0] = op[k]
        for p in range(1, 25):
            if me[p] != 1:
                continue
            shots = 0.0
            for q in op_cnt:
                dist = p - q
                if 0 < dist <= 6:
                    shots += 1.0
                elif 6 < dist <= 12:
                    shots += 0.35
            if shots:
                loss = (25 - p) / 24.0  # pips lost when hit, normalised
                score -= self.blot_weight * min(shots, 3.0) * (0.6 + loss)
        return score

    def rank_moves(self, state: State, dice, moves):
        turn = state.turn
        scored = [(i, self.score(after.pos[turn], after.pos[1 - turn]))
                  for i, (_, after) in enumerate(moves)]
        scored.sort(key=lambda t: t[1], reverse=True)
        return scored
