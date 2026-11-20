"""The value network: 198 inputs -> sigmoid hidden layer -> sigmoid output.

The output V(s) is the estimated probability that **White** wins from
position ``s`` (with the side to move given by the two turn inputs). The
probability that the player to move wins is therefore ``V`` when White is to
move and ``1 - V`` when Black is to move (see :func:`win_prob_for`).

Gradients of V with respect to all weights are computed *by hand*
(:meth:`TDNetwork.value_and_grad`) - the TD(lambda) learner uses them to
build its eligibility traces without autograd.
"""

from __future__ import annotations

import torch
from torch import nn

from .encoding import NUM_INPUTS


class TDNetwork(nn.Module):
    def __init__(self, hidden: int = 80, inputs: int = NUM_INPUTS, init_scale: float = 0.1):
        super().__init__()
        self.hidden = hidden
        self.inputs = inputs
        self.fc1 = nn.Linear(inputs, hidden)
        self.fc2 = nn.Linear(hidden, 1)
        with torch.no_grad():
            for p in self.parameters():
                p.uniform_(-init_scale, init_scale)

    # --------------------------------------------------------- evaluation
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (N, 198) -> (N,) probabilities that White wins."""
        h = torch.sigmoid(self.fc1(x))
        return torch.sigmoid(self.fc2(h)).squeeze(-1)

    @torch.no_grad()
    def value_and_grad(self, x: torch.Tensor):
        """Value of one input vector x (198,) and dV/dtheta, computed manually.

        Returns (v, [dW1, db1, dW2, db2]) where the gradients have the same
        shapes as fc1.weight, fc1.bias, fc2.weight, fc2.bias.

        Derivation (sigma' = sigma (1 - sigma)):
            h  = sigma(W1 x + b1),  v = sigma(w2 . h + b2)
            dv/db2 = v (1 - v)                 =: g
            dv/dw2 = g h
            dv/db1 = g w2 * h (1 - h)          =: gz
            dv/dW1 = gz x^T
        """
        w1, b1 = self.fc1.weight, self.fc1.bias
        w2, b2 = self.fc2.weight[0], self.fc2.bias
        h = torch.sigmoid(torch.mv(w1, x) + b1)
        v = torch.sigmoid(torch.dot(w2, h) + b2[0])
        g = v * (1.0 - v)
        gz = g * w2 * h * (1.0 - h)
        grads = [torch.outer(gz, x), gz, (g * h).unsqueeze(0), g.reshape(1)]
        return v, grads


def win_prob_for(v_white: torch.Tensor | float, player: int):
    """Convert P(White wins) into P(player wins)."""
    return v_white if player == 0 else 1.0 - v_white


def save_network(net: TDNetwork, path: str, **extra) -> None:
    data = {"hidden": net.hidden, "inputs": net.inputs,
            "state_dict": {k: v.detach().cpu() for k, v in net.state_dict().items()}}
    data.update(extra)
    torch.save(data, path)


def load_checkpoint(path: str) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def load_network(path: str, device: str = "cpu") -> TDNetwork:
    data = load_checkpoint(path)
    net = TDNetwork(hidden=data["hidden"], inputs=data.get("inputs", NUM_INPUTS))
    net.load_state_dict(data["state_dict"])
    net.to(device)
    net.eval()
    return net
