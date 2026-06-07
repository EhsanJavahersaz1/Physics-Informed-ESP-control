"""Evaluate the trained PINN and compare it against a numerical (RK4) integration
of the same closed-loop dynamics, which serves as the physical ground truth.

Produces ``results/pinn_vs_rk4.png`` showing that the physics-informed network
reproduces the true vehicle response and that the lateral/heading errors decay
(the vehicle tracks the path and stays stable).

Example:
    python evaluate.py --checkpoint pinn_model.pt
"""

import argparse
import os
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from vehicle_model import VehicleParams, dynamics
from pinn_model import PINN
from train import reference_curvature


def rk4_reference(params, gains, T, n, s0):
    """Reference closed-loop trajectory via fixed-step RK4 integration."""
    dt = T / (n - 1)
    s = s0.clone()
    states, t = [s.clone()], 0.0

    def f(st, tt):
        kappa = reference_curvature(torch.tensor([tt]))
        return dynamics(st.unsqueeze(0), kappa, params, gains).squeeze(0)

    for _ in range(n - 1):
        k1 = f(s, t)
        k2 = f(s + 0.5 * dt * k1, t + 0.5 * dt)
        k3 = f(s + 0.5 * dt * k2, t + 0.5 * dt)
        k4 = f(s + dt * k3, t + dt)
        s = s + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        t += dt
        states.append(s.clone())
    return torch.stack(states)


def main(args):
    params = VehicleParams(vx=args.vx)
    gains = (args.k_ey, args.k_epsi, args.k_r)

    model = PINN(hidden=args.hidden, layers=args.layers)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    model.eval()

    t = torch.linspace(0.0, args.T, args.n).reshape(-1, 1)
    with torch.no_grad():
        pred = model(t).numpy()

    s0 = torch.tensor([0.0, 0.0, args.ey0, args.epsi0])
    ref = rk4_reference(params, gains, args.T, args.n, s0).numpy()
    tt = t.numpy().ravel()

    os.makedirs("results", exist_ok=True)
    labels = ["lateral velocity vy [m/s]", "yaw rate r [rad/s]",
              "lateral error ey [m]", "heading error epsi [rad]"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for i, ax in enumerate(axes.ravel()):
        ax.plot(tt, ref[:, i], "k-", label="RK4 (physics)")
        ax.plot(tt, pred[:, i], "r--", label="PINN")
        ax.set_xlabel("time [s]")
        ax.set_ylabel(labels[i])
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.suptitle("PINN vs. physical ground truth - lateral path-tracking / ESP control")
    fig.tight_layout()
    fig.savefig("results/pinn_vs_rk4.png", dpi=150)
    print("Saved figure to results/pinn_vs_rk4.png")

    mae = abs(pred - ref).mean(axis=0)
    print("Mean absolute error per state (vy, r, ey, epsi):", mae)


def build_parser():
    p = argparse.ArgumentParser(description="Evaluate the trained PINN")
    p.add_argument("--checkpoint", type=str, default="pinn_model.pt")
    p.add_argument("--T", type=float, default=10.0)
    p.add_argument("--n", type=int, default=400)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--vx", type=float, default=20.0)
    p.add_argument("--ey0", type=float, default=1.0)
    p.add_argument("--epsi0", type=float, default=0.0)
    p.add_argument("--k_ey", type=float, default=0.15)
    p.add_argument("--k_epsi", type=float, default=0.8)
    p.add_argument("--k_r", type=float, default=0.2)
    return p


if __name__ == "__main__":
    main(build_parser().parse_args())

