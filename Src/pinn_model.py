"""Physics-Informed Neural Network (PINN) for the vehicle lateral path-tracking
and stability problem.

The network maps time ``t`` to the path-relative state ``[vy, r, ey, epsi]``.
Training enforces the closed-loop vehicle dynamics through a physics residual
(see ``train.py``) together with the initial condition, so the learned
trajectory is physically consistent rather than purely data-driven.
"""

import torch
import torch.nn as nn


class PINN(nn.Module):
    def __init__(self, hidden: int = 64, layers: int = 4):
        super().__init__()
        modules = [nn.Linear(1, hidden), nn.Tanh()]
        for _ in range(layers - 1):
            modules += [nn.Linear(hidden, hidden), nn.Tanh()]
        modules += [nn.Linear(hidden, 4)]   # outputs: vy, r, ey, epsi
        self.net = nn.Sequential(*modules)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """t: tensor of shape [N, 1] -> state of shape [N, 4]."""
        return self.net(t)

