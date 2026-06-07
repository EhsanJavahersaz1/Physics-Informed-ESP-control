"""Single-track (bicycle) lateral vehicle dynamics for path-tracking and
ESP-style stability control.

Path-relative state vector:
    vy    : lateral velocity                [m/s]
    r     : yaw rate                        [rad/s]
    ey    : lateral path-tracking error     [m]
    epsi  : heading (orientation) error     [rad]

The longitudinal speed ``vx`` is treated as constant, which is standard for a
lateral-dynamics / ESP study. The steering angle ``delta`` is the control input;
an ESP-style feedback law steers the vehicle so that it tracks the reference
path while keeping the lateral error, heading error and yaw rate bounded
(i.e. stable).
"""

from dataclasses import dataclass
import torch


@dataclass
class VehicleParams:
    m: float = 1500.0       # mass                        [kg]
    Iz: float = 2250.0      # yaw moment of inertia       [kg m^2]
    a: float = 1.2          # CG-to-front-axle distance   [m]
    b: float = 1.6          # CG-to-rear-axle distance    [m]
    Cf: float = 80000.0     # front cornering stiffness   [N/rad]
    Cr: float = 80000.0     # rear cornering stiffness    [N/rad]
    vx: float = 20.0        # longitudinal speed          [m/s]


def esp_steering(state, gains):
    """ESP-style stabilizing feedback steering law.

        delta = -(k_ey * ey + k_epsi * epsi + k_r * r)

    Drives the lateral error, heading error and yaw rate toward zero, which is
    what keeps the vehicle on the path and stable.
    """
    vy, r, ey, epsi = state[..., 0], state[..., 1], state[..., 2], state[..., 3]
    k_ey, k_epsi, k_r = gains
    return -(k_ey * ey + k_epsi * epsi + k_r * r)


def dynamics(state, kappa_ref, params: VehicleParams, gains):
    """Closed-loop continuous-time dynamics; returns d(state)/dt.

    Args:
        state:     tensor [..., 4] = (vy, r, ey, epsi)
        kappa_ref: reference path curvature at the current point [1/m]
        params:    VehicleParams
        gains:     (k_ey, k_epsi, k_r) feedback gains of the ESP steering law
    """
    m, Iz, a, b = params.m, params.Iz, params.a, params.b
    Cf, Cr, vx = params.Cf, params.Cr, params.vx

    vy, r, ey, epsi = state[..., 0], state[..., 1], state[..., 2], state[..., 3]
    delta = esp_steering(state, gains)

    # --- Lateral + yaw dynamics (linear-tire single-track model) ---
    vy_dot = (-(Cf + Cr) / (m * vx) * vy
              + (-(a * Cf - b * Cr) / (m * vx) - vx) * r
              + Cf / m * delta)
    r_dot = (-(a * Cf - b * Cr) / (Iz * vx) * vy
             - (a ** 2 * Cf + b ** 2 * Cr) / (Iz * vx) * r
             + a * Cf / Iz * delta)

    # --- Path-tracking error dynamics ---
    ey_dot = vy + vx * epsi
    epsi_dot = r - vx * kappa_ref

    return torch.stack([vy_dot, r_dot, ey_dot, epsi_dot], dim=-1)
