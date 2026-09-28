from __future__ import annotations
from dataclasses import dataclass, field

import yaml # type: ignore


@dataclass
class MPCConfig():
    vehicle_width : float = 0.5   # Vehicle width in meters
    vehicle_margin : float = 0.1  # Vehicle extra safe margin
    social_margin : float = 0.2   # Vehicle extra social margin for soft loss terms
    lin_vel_min : float = -0.5    # Vehicle contraint on the minimal velocity possible
    lin_vel_max : float = 1.5     # Vehicle contraint on the maximal velocity possible
    lin_acc_min : float = -1      # Vehicle contraint on the maximal linear retardation
    lin_acc_max : float = 1       # Vehicle contraint on the maximal linear acceleration
    ang_vel_max : float = 0.5     # Vehicle contraint on the maximal angular velocity
    ang_acc_max : float = 3       # Vehicle contraint on the maximal angular acceleration (considered to be symmetric)

    # Velocity profile (proportional to the maximal speed)
    full_speed : float = 1.0
    high_speed : float = 0.8
    medium_speed : float = 0.5
    low_speed : float = 0.2

    # Parameters specific to the MPC
    ts : float = 0.2        # Size of the time-step (sampling time)
    N_hor : int = 20        # The length of the receding horizon controller
    action_steps : int = 1  # How many steps should be taken from each mpc-solution. Range (1 - N_hor)

    # Penalty weights
    lin_vel_penalty : float = 0      # Cost for linear velocity control action (should be 0)
    lin_acc_penalty : float = 10.0   # Cost for linear acceleration 
    ang_vel_penalty : float = 0      # Cost angular velocity control action
    ang_acc_penalty : float = 20.0   # Cost angular acceleration
    qrpd : float = 100.0             # Cost for reference path deviation
    qpos : float = 0.0               # Cost for position deviation each time step to the reference
    qvel : float = 10.0              # Cost for speed    deviation each time step to the reference
    qtheta : float = 0.0             # Cost for heading  deviation each time step to the reference
    # Terminal weights
    qpN : float = 0.0                # Terminal cost; error relative to final reference position         
    qthetaN : float = 0.0            # Terminal cost; error relative to final reference heading     

    # Helper variables (Generally does not have to be changed)
    nu : int = 2        # Number of control inputs (speed and angular speed)
    ns : int = 3        # Number of states for the robot (x,y,theta,e) [e is the channel width/permitted error, not included yet]
    nq : int = 10       # Number of optimization penalties
    Nother : int = 10   # Maximal number of other robots
    Nstcobs : int = 10  # Maximal number of static obstacles
    nstcobs : int = 12  # Number of variables per obstacles, (4 edges * 3 per edge)
    Ndynobs : int = 15  # Maximal number of dynamic obstacles
    ndynobs : int = 6   # Number of variables per dynamic obstacle

    # Building options in the optimizer
    build_type : str = 'release'          # Can have 'debug' or 'release'
    build_directory : str = 'mpc_solver'   # Name of the directory where the build is created
    bad_exit_codes : list[str] = field(default_factory=lambda: ["NotConvergedIterations", "NotConvergedOutOfTime"]) # Optimizer specific names, otherwise "Converged"
    optimizer_name : str = 'navi_default'   # optimizer name
    max_solver_time_micros : int = 500_000  # Hard wall-clock budget per solve, baked into the solver at
                                            # build time (requires a rebuild to change). On timeout OpEn
                                            # returns the best iterate so far with exit_status
                                            # "NotConvergedOutOfTime" (a bad_exit_code), it does not fail.
                                            # Cutting this to 180ms (i.e. inside the 200ms control period)
                                            # makes the grid scene collide: a half-converged iterate
                                            # violates the obstacle penalties. 500ms clips the ~0.9s tail
                                            # without changing the outcome of any scene tested.
    
    @classmethod
    def from_yaml(cls, file_path: str):
        with open(file_path, "r") as f:
            raw = yaml.safe_load(f) or {}
        if not isinstance(raw, dict):
            raise TypeError("YAML root must be a dict")
        return cls(**raw)
    
    def validate_config(self,  *args, **kwargs) -> None:
        raise NotImplementedError
    
    def to_dict(self) -> dict:
        return self.__dict__.copy()