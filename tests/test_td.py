"""Tests for the encoding, the network gradients and the TD learner."""

from __future__ import annotations

import random

import numpy as np
import torch

from backgammon.agents.encoding import NUM_INPUTS, encode_state, encode_states
from backgammon.agents.expectimax import ExpectimaxAgent
from backgammon.agents.network import TDNetwork, load_network, save_network
from backgammon.agents.td_agent import TDAgent
from backgammon.engine import BLACK, WHITE, State, generate_moves
from backgammon.training.td_learner import TDLambdaLearner


def test_encoding_size_and_start_values():
    x = encode_state(State.initial(WHITE))
    assert x.shape == (NUM_INPUTS,) == (198,)
    # White's 24 point holds 2 checkers: units 1,1,0,0
    base = (24 - 1) * 4
    assert list(x[base:base + 4]) == [1, 1, 0, 0]
    # White's 6 point holds 5: units 1,1,1,(5-3)/2 = 1
    base = (6 - 1) * 4
    assert list(x[base:base + 4]) == [1, 1, 1, 1.0]
    assert x[96] == 0 and x[97] == 0          # bar, off
    assert x[196] == 1 and x[197] == 0        # white to move
    assert encode_state(State.initial(BLACK))[197] == 1


def test_encoding_bar_and_off():
    s = State.from_points({6: 10}, {6: 13}, WHITE, white_bar=2, white_off=3, black_off=2)
    x = encode_state(s)
    assert x[96] == 1.0 and abs(x[97] - 3 / 15) < 1e-6
    assert abs(x[195] - 2 / 15) < 1e-6


def test_batch_encoding_matches_single():
    s = State.initial()
    states = [a for _, a in generate_moves(s, (6, 5))]
    batch = encode_states(states)
    for i, st in enumerate(states):
        assert np.allclose(batch[i], encode_state(st))


def test_manual_gradient_matches_autograd():
    torch.manual_seed(0)
    net = TDNetwork(hidden=12)
    x = torch.rand(198)
    v, grads = net.value_and_grad(x)
    net.zero_grad()
    out = net(x.unsqueeze(0))[0]
    out.backward()
    auto = [net.fc1.weight.grad, net.fc1.bias.grad, net.fc2.weight.grad, net.fc2.bias.grad]
    assert torch.allclose(v, out.detach())
    for g, a in zip(grads, auto):
        assert g.shape == a.shape
        assert torch.allclose(g, a, atol=1e-6)


def test_td_learning_moves_value_towards_result():
    torch.manual_seed(1)
    net = TDNetwork(hidden=10)
    learner = TDLambdaLearner(net, lam=0.7)
    before = [p.clone() for p in net.parameters()]
    winner, points, turns = learner.play_game(random.Random(0), lr=0.1)
    assert winner in (WHITE, BLACK) and points in (1, 2, 3) and turns > 10
    assert any(not torch.equal(a, b) for a, b in zip(before, net.parameters()))


def test_td_agent_choose_and_probability(tmp_path):
    net = TDNetwork(hidden=8)
    path = tmp_path / "m.pt"
    save_network(net, str(path), games=5)
    net2 = load_network(str(path))
    agent = TDAgent(net2)
    s = State.initial()
    moves = generate_moves(s, (4, 2))
    idx = agent.choose(s, (4, 2), moves)
    ranked = agent.rank_moves(s, (4, 2), moves)
    assert ranked[0][0] == idx
    p = agent.win_probability(s)
    assert 0.0 < p < 1.0


def test_td_agent_takes_winning_move():
    agent = TDAgent(TDNetwork(hidden=8))
    s = State.from_points({1: 1, 2: 1}, {12: 15}, WHITE, white_off=13)
    moves = generate_moves(s, (2, 1))
    idx = agent.choose(s, (2, 1), moves)
    assert moves[idx][1].is_over()


def test_expectimax_ranks_all_moves():
    agent = ExpectimaxAgent(TDNetwork(hidden=8), top_k=3)
    s = State.initial()
    moves = generate_moves(s, (3, 1))
    ranked = agent.rank_moves(s, (3, 1), moves)
    assert sorted(i for i, _ in ranked) == list(range(len(moves)))
    assert all(0.0 <= sc <= 1.0 for _, sc in ranked)
    assert agent.choose(s, (3, 1), moves) == ranked[0][0]
