# Physics-Informed Neural Network (PINN) for Vehicle Lateral Path-Tracking and ESP Stability Control

A physics-informed deep-learning framework (PyTorch) for **lateral path-tracking
and ESP-style stability control** of a road vehicle. A neural network is trained
to reproduce the closed-loop response of a single-track (bicycle) vehicle model:
steering the vehicle so that it follows a lateral reference path without leaving
it, while keeping the lateral error, heading error and yaw rate bounded (stable).

The physics of the vehicle's lateral dynamics is embedded directly in the
training objective, so the learned model stays physically consistent instead of
fitting data alone.

> Builds on my BSc thesis, *Modelling the Steering of a Four-Wheeled Vehicle and
> Design of an Electronic Lateral-Stability (ESP) Controller* (graded 20/20).

## Idea

The model is a single-track (bicycle) vehicle with path-relative state
`[vy, r, ey, epsi]` (lateral velocity, yaw rate, lateral error, heading error)
and a constant longitudinal speed. An ESP-style feedback steering law,

```
delta = -(k_ey * ey + k_epsi * epsi + k_r * r),
```

stabilizes the vehicle and tracks the path. A neural network `t -> [vy, r, ey, epsi]`
is trained as a **PINN**: its time-derivatives (via automatic differentiation)
are forced to satisfy the closed-loop equations of motion (physics loss), subject
to the initial condition (data loss):

```
L = L_physics + w_ic * L_ic
```

## Repository structure

```
.
├── src/
│   ├── vehicle_model.py   # single-track lateral dynamics + ESP steering law
│   ├── pinn_model.py      # the neural network (PINN)
│   ├── train.py           # physics-informed training loop
│   └── evaluate.py        # compare PINN vs RK4 ground truth, save plots
├── results/               # generated figures (created on first run)
├── requirements.txt
└── README.md
```

## Installation

```bash
git clone https://github.com/EhsanJavahersaz1/Physics-Informed-ESP-control.git
cd Physics-Informed-ESP-control
pip install -r requirements.txt
```

Requirements: Python 3.9+, PyTorch, NumPy, Matplotlib.

## Usage

Train the PINN (starts with a 1 m initial lateral error to test recovery):

```bash
cd src
python train.py --epochs 5000 --ey0 1.0
```

Evaluate and plot the PINN against the physical (RK4) ground truth:

```bash
python evaluate.py --checkpoint pinn_model.pt
```

This writes `results/pinn_vs_rk4.png`, comparing the learned trajectory with a
numerical integration of the same dynamics, and prints the mean absolute error
per state.

## Method summary

- **Vehicle model:** linear-tire single-track (bicycle) lateral dynamics.
- **Control:** ESP-style state-feedback steering for path-tracking and stability.
- **Learning:** PINN whose physics residual enforces the closed-loop ODEs through
  automatic differentiation, combined with an initial-condition loss.
- **Validation:** comparison against a fixed-step RK4 integration of the same
  dynamics.

## Author

**Roohollah Javahersaz**
MSc Control Engineering · BSc Mechanical Engineering
Email: javahersaz.ehsan2021@gmail.com · r.javahersaz@yahoo.com

## License

Released under the MIT License (see `LICENSE`).
