"""Backgammon agents."""

from .base import Agent
from .factory import AGENT_KINDS, make_agent
from .heuristic import HeuristicAgent
from .random_agent import RandomAgent

__all__ = ["Agent", "AGENT_KINDS", "make_agent", "HeuristicAgent", "RandomAgent"]
