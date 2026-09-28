"""The robot's kinematics, shared by the solver (as CasADi expressions) and the runtime (numpy)."""
from typing import Union

import casadi.casadi as cs  # type: ignore
import numpy as np


def _normalize_angle(angle: Union[float, cs.SX]) -> Union[float, cs.SX]:
    """Normalize an angle to [-pi, pi]."""
    if isinstance(angle, cs.SX):
        return cs.fmod(angle + cs.pi, 2 * cs.pi) - cs.pi
    return (angle + np.pi) % (2 * np.pi) - np.pi


def unicycle_model(state: Union[np.ndarray, cs.SX], action: Union[np.ndarray, cs.SX], ts: float, rk4:bool=True) -> Union[np.ndarray, cs.SX]:
    """Unicycle model.
    
    Args:
        ts: Sampling time.
        state: x, y, and theta.
        action: speed and angular speed.
        rk4: If True, use Runge-Kutta 4 to refine the model.
    """
    def d_state_f(state, action):
        if isinstance(state, cs.SX):
            return ts * cs.vertcat(action[0]*cs.cos(state[2]), action[0]*cs.sin(state[2]), action[1])
        return ts * np.array([action[0]*np.cos(state[2]), action[0]*np.sin(state[2]), action[1]])
    if rk4:
        k1 = d_state_f(state, action)
        k2 = d_state_f(state + 0.5*k1, action)
        k3 = d_state_f(state + 0.5*k2, action)
        k4 = d_state_f(state + k3, action)
        d_state = (1/6) * (k1 + 2*k2 + 2*k3 + k4)
    else:
        d_state = d_state_f(state, action)
    next_state = state + d_state
    next_state[2] = _normalize_angle(next_state[2])
    return next_state
