"""Create agents from simple, picklable specifications.

A spec is a dict such as ``{"kind": "td", "model": "checkpoints/td_final.pt"}``.
Specs can be sent to worker processes, which build the agents themselves.
Supported kinds: ``random``, ``heuristic``, ``td``, ``expectimax``.
For ``td``/``expectimax`` either ``model`` (path to a checkpoint) or
``weights`` (a state dict, plus ``hidden``) must be given.
"""

from __future__ import annotations

from .base import Agent

AGENT_KINDS = ("random", "heuristic", "td", "expectimax")


def make_agent(spec: dict | str, device: str = "cpu") -> Agent:
    if isinstance(spec, str):
        spec = {"kind": spec}
    kind = spec["kind"].lower()
    if kind == "random":
        from .random_agent import RandomAgent
        return RandomAgent(seed=spec.get("seed"))
    if kind == "heuristic":
        from .heuristic import HeuristicAgent
        return HeuristicAgent()
    if kind in ("td", "expectimax"):
        from .network import TDNetwork, load_network
        from .td_agent import TDAgent
        if spec.get("weights") is not None:
            net = TDNetwork(hidden=spec["hidden"])
            net.load_state_dict(spec["weights"])
            net.to(device)
        else:
            net = load_network(spec.get("model") or default_model_path(), device=device)
        if kind == "td":
            return TDAgent(net, device=device)
        from .expectimax import ExpectimaxAgent
        return ExpectimaxAgent(net, device=device, top_k=spec.get("top_k", 6))
    raise ValueError(f"unknown agent kind {kind!r}; choose from {AGENT_KINDS}")


def default_model_path() -> str:
    """Path of the bundled final model (or the newest checkpoint)."""
    import glob
    import os

    root = os.path.join(os.path.dirname(__file__), "..", "..", "checkpoints")
    root = os.path.normpath(root)
    final = os.path.join(root, "td_final.pt")
    if os.path.exists(final):
        return final
    cands = sorted(glob.glob(os.path.join(root, "**", "*.pt"), recursive=True),
                   key=os.path.getmtime)
    if not cands:
        raise FileNotFoundError("no trained model found in checkpoints/ - train one first")
    return cands[-1]
