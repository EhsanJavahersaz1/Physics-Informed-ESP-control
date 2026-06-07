"""Train the PINN for vehicle lateral path-tracking and ESP stability control.

The network learns the closed-loop state trajectory that satisfies the vehicle
dynamics (physics loss) and starts from a given initial condition (IC loss).
A non-zero initial lateral error ``ey0`` lets us check that the ESP-style
controller brings the vehicle back onto the path in a physically consistent way.

Example:
    python train.py --epochs 5000 --ey0 1.0
"""

import argparse
import torch

from vehicle_model import VehicleParams, dynamics
from pinn_model import PINN


def reference_curvature(t: torch.Tensor) -> torch.Tensor:
    """Reference path curvature kappa(t) [1/m] (smooth lane-change-like path)."""
    return 0.02 * torch.sin(0.5 * t)


def physics_residual(model, t, params, gains):
    """Residual of d(state)/dt - f(state) at the collocation times ``t``."""
    t = t.clone().requires_grad_(True)
    state = model(t)                              # [N, 4]
    grads = []
    for i in range(4):
        gi = torch.autograd.grad(state[:, i].sum(), t, create_graph=True)[0]
        grads.append(gi[:, 0])
    dstate = torch.stack(grads, dim=-1)           # [N, 4]
    kappa = reference_curvature(t[:, 0])
    f = dynamics(state, kappa, params, gains)
    return dstate - f


def train(args):
    torch.manual_seed(args.seed)
    params = VehicleParams(vx=args.vx)
    gains = (args.k_ey, args.k_epsi, args.k_r)

    model = PINN(hidden=args.hidden, layers=args.layers)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    t_col = torch.linspace(0.0, args.T, args.n_col).reshape(-1, 1)
    t0 = torch.zeros(1, 1)
    s0 = torch.tensor([[0.0, 0.0, args.ey0, args.epsi0]])

    for it in range(args.epochs):
        opt.zero_grad()
        res = physics_residual(model, t_col, params, gains)
        loss_phys = (res ** 2).mean()
        loss_ic = ((model(t0) - s0) ** 2).mean()
        loss = loss_phys + args.w_ic * loss_ic
        loss.backward()
        opt.step()
        if it % args.log_every == 0 or it == args.epochs - 1:
            print(f"iter {it:5d} | loss {loss.item():.3e} "
                  f"(phys {loss_phys.item():.3e}, ic {loss_ic.item():.3e})")

    torch.save(model.state_dict(), args.out)
    print(f"Saved trained model to {args.out}")


def build_parser():
    p = argparse.ArgumentParser(description="Train PINN for lateral path-tracking / ESP control")
    p.add_argument("--T", type=float, default=10.0, help="time horizon [s]")
    p.add_argument("--n_col", type=int, default=2000, help="number of collocation points")
    p.add_argument("--epochs", type=int, default=5000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--vx", type=float, default=20.0, help="longitudinal speed [m/s]")
    p.add_argument("--ey0", type=float, default=1.0, help="initial lateral error [m]")
    p.add_argument("--epsi0", type=float, default=0.0, help="initial heading error [rad]")
    p.add_argument("--k_ey", type=float, default=0.15, help="ESP gain on lateral error")
    p.add_argument("--k_epsi", type=float, default=0.8, help="ESP gain on heading error")
    p.add_argument("--k_r", type=float, default=0.2, help="ESP gain on yaw rate")
    p.add_argument("--w_ic", type=float, default=10.0, help="weight of the IC loss")
    p.add_argument("--log_every", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default="pinn_model.pt")
    return p


if __name__ == "__main__":
    train(build_parser().parse_args())

