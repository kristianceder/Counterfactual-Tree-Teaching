from typing import Callable, Tuple, Union, List
import os
import sys
import math
import itertools

import numpy as np

from . import MPCConfig
from .solver_generator import MpcModule
from ._path import PathNode, PathNodeList, TrajectoryNode, TrajectoryNodeList

from .geometry import polygon_halfspace_representation
from .motion_model import unicycle_model


'''
File description:
    Generate the trajectory by generating/using the defined MPC solver. 
File content:
    TrajectoryGenerator <class> - Run the MPC problem. Calculate the trajectory step by step.
Comments:
                                                                                      V [MPC] V
    [GPP] --global path & static obstacles--> [LPP] --refernece path & tube width--> [TG(Config)] <--dynamic obstacles-- [OS]
'''

class Solver(): # this is not found in the .so file (in ternimal: nm -D  navi_test.so)
    def run(self, 
            p : list, 
            initial_guess:np.ndarray | None = None,
            initial_lagrange_multipliers:np.ndarray | None = None,
            initial_penalty:float | None = None
        ) -> object: pass


class TrajectoryGenerator:
    '''
    Description:
        Generate a smooth trajectory based on the reference path and obstacle information.
        Use a configuration specified by 'utils/config'
    Args:
        config  <dotdict> - A dictionary in the dot form contains all information/parameters needed.
        verbose <bool>    - If true, show verbose.
    Functions
        run <run>  - Run.
    Comments:
        Have fun but may need to modify the dynamic obstacle part (search NOTE).
    '''
    def __init__(self, config: MPCConfig, motion_model: Callable | None = None, build_solver:bool=False, verbose=False):
        self.vb = verbose
        self.config = config
        if motion_model is None:
            self.motion_model = unicycle_model
        else:
            self.motion_model = motion_model

        # Common used parameters from config
        self.ts = self.config.ts
        self.ns = self.config.ns
        self.nu = self.config.nu
        self.N_hor = self.config.N_hor

        # Initialization
        self.set_work_mode(mode='safe')
        self.set_obstacle_weights(stc_weights=1e3, dyn_weights=1e3) # default obstacle weights

        self.__import_solver(build_solver=build_solver)

        # Initialize some placeholders
        self._last_action = np.zeros(self.nu)
        self.stc_constraints = [0.0] * self.config.Nstcobs * self.config.nstcobs
        self.dyn_constraints = [0.0] * self.config.Ndynobs * self.config.ndynobs * self.config.N_hor
        self.other_robot_states = [0] * self.config.ns * self.config.N_hor * self.config.Nother

    @property
    def last_action(self):
        return self._last_action

    def __import_solver(self, build_solver:bool=False):
        # The compiled solver lives in `build_directory` at the repository root (see scripts/build_solver.py).
        root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        solver_path = os.path.join(root_dir, self.config.build_directory, self.config.optimizer_name)

        if build_solver:
            MpcModule(self.config).build(self.motion_model)

        sys.path.append(solver_path)
        built_solver = __import__(self.config.optimizer_name) # it loads a .so (shared library object)
        self.solver:Solver = built_solver.solver() # Return a Solver object with run method, cannot find it though

    def load_init_state(self, current_state: np.ndarray, goal_state: np.ndarray):
        if (not isinstance(current_state, np.ndarray)) or (not isinstance(goal_state, np.ndarray)):
            raise TypeError(f'State and action should be numpy.ndarry, got {type(current_state)}/{type(goal_state)}.')
        self.state = current_state
        self.final_goal = goal_state # used to check terminal condition

        self.past_states:list[np.ndarray]  = []
        self.past_actions:list[np.ndarray] = []
        self.cost_timelist:list[float] = []
        self.solver_time_timelist:list[float] = []
        self.exit_status: str | None = None # solver exit status of the most recent run_step() call

        self.idx_ref = 0 # for reference trajectory following
        self.speed_ref_override: list[float] | None = None # see run_step(); e.g. failure_monitor.WaitStrategy
        self.holding = False # see get_local_ref_traj(); e.g. failure_monitor.WaitStrategy
        self.heading_override: float | None = None
        """Heading (rad) the hold should *target* instead of simply freezing the current one --
        only meaningful while `holding`. With position held (zero cross-track error) and the
        speed reference zeroed, the heading cost (`qtheta`) is the only term left with a
        gradient, so the solver turns the robot on the spot toward this angle: a unicycle's
        angular velocity is an independent control input, so it can rotate at v=0. Used by
        failure_monitor.ReorientStrategy to point the robot along its route before resuming,
        which is what breaks the reverse-tracking local minimum (see that strategy, and
        src/mpc_traj_tracker/README.md's "Why the heading cost exists")."""

    def set_obstacle_weights(self, stc_weights:Union[list, float], dyn_weights:Union[list, float]):
        '''
        Attribute
            stc_weights [list]: penalty weights for static obstacles (only useful if soft constraints activated)
            dyn_weights [list]: penalty weights for dynamic obstacles (only useful if soft constraints activated)
        '''
        if isinstance(stc_weights, list):
            self.stc_weights = stc_weights
        elif isinstance(stc_weights, (float,int)):
            self.stc_weights = [stc_weights]*self.N_hor
        else:
            raise TypeError(f'Unsupported datatype for obstacle weights, got {type(stc_weights)}.')
        if isinstance(dyn_weights, list):
            self.dyn_weights = dyn_weights
        elif isinstance(dyn_weights, (float,int)):
            self.dyn_weights = [dyn_weights]*self.N_hor
        else:
            raise TypeError(f'Unsupported datatype for obstacle weights, got {type(dyn_weights)}.')

    def set_work_mode(self, mode:str='safe'):
        '''
        Attribute
            base_speed: The reference speed
            tuning_params: Penalty parameters for MPC
        '''
        nparams = 10

        ### Base/reference speed
        if mode == 'aligning':
            self.base_speed = self.config.lin_vel_max*self.config.medium_speed
            self.tuning_params = [0.0] * nparams
            self.tuning_params[2] = 100
        else:
            self.tuning_params = [self.config.qpos, self.config.qvel, self.config.qtheta, self.config.lin_vel_penalty, self.config.ang_vel_penalty,
                                  self.config.qpN, self.config.qthetaN, self.config.qrpd, self.config.lin_acc_penalty, self.config.ang_acc_penalty]
            if mode == 'safe':
                self.base_speed = self.config.lin_vel_max*self.config.low_speed
            elif mode == 'work':
                self.base_speed = self.config.lin_vel_max*self.config.high_speed
            elif mode == 'super':
                self.base_speed = self.config.lin_vel_max*self.config.full_speed
            else:
                raise ModuleNotFoundError(f'There is no mode called {mode}.')

    def set_current_state(self, current_state: np.ndarray):
        if not isinstance(current_state, np.ndarray):
            raise TypeError(f'State should be numpy.ndarry, got {type(current_state)}.')
        self.state = current_state

    def set_ref_trajectory(self, ref_path: PathNodeList | list):
        '''
        Attributes:
            :idx_ref, start from 0
            :ref_traj, global reference trajectory
        '''
        if not isinstance(ref_path, PathNodeList):
            ref_path = PathNodeList([PathNode(x[0], x[1]) for x in ref_path])
        self.idx_ref = 0
        self.ref_traj = self.get_global_ref_traj(self.ts, ref_path, (self.state[0], self.state[1]), self.base_speed)

    def update_static_constraints(self, obstacle_list:list):
        if len(obstacle_list) > self.config.Nstcobs:
            raise ValueError(f"Got {len(obstacle_list)} static obstacles, but config.Nstcobs only allows "
                              f"{self.config.Nstcobs}. Raise Nstcobs (requires a solver rebuild) or check "
                              f"whether every obstacle here should actually be static (e.g. a moving "
                              f"obstacle wrongly missing `is_static=False`, see simulation.objects.Obstacle).")
        for i, map_obstacle in enumerate(obstacle_list):
            b, a0, a1 = polygon_halfspace_representation(np.array(map_obstacle))
            n_params = len(b) + len(a0) + len(a1)
            if n_params != self.config.nstcobs:
                raise ValueError(f"Static obstacle {i} has {len(map_obstacle)} vertices -> {n_params} halfspace "
                                  f"parameters, but config.nstcobs assumes exactly {self.config.nstcobs} (i.e. "
                                  f"{self.config.nstcobs // 3}-edge polygons only). A silent slice-assignment "
                                  f"here would otherwise corrupt every parameter after this obstacle's slot "
                                  f"rather than fail loudly -- see the MPC README's parameter-vector layout.")
            self.stc_constraints[i*self.config.nstcobs : (i+1)*self.config.nstcobs] = (b+a0+a1)

    def update_dynamic_constraints(self, full_dyn_obstacle_list):
        params_per_dyn_obs =self.config.ndynobs * self.config.N_hor
        if len(full_dyn_obstacle_list) > self.config.Ndynobs:
            raise ValueError(f"Got {len(full_dyn_obstacle_list)} dynamic obstacles, but config.Ndynobs only "
                              f"allows {self.config.Ndynobs} (requires a solver rebuild to raise).")
        for i, dyn_obstacle in enumerate(full_dyn_obstacle_list):
            self.dyn_constraints[i*params_per_dyn_obs:(i+1)*params_per_dyn_obs] = list(itertools.chain(*dyn_obstacle))
    
    def check_termination_condition(self, state: np.ndarray, action: np.ndarray, final_goal: np.ndarray) -> bool:
        if np.allclose(state[:2], final_goal[:2], atol=0.05, rtol=0) and abs(action[0]) < 0.05:
            terminated = True
            print(f"[{self.__class__.__name__}] MPC solution found.")
        else:
            terminated = False
        return terminated


    @staticmethod
    def get_global_ref_traj(ts: float, ref_path: PathNodeList, state: tuple, speed:float) -> TrajectoryNodeList:
        '''
        Description:
            Generate the reference trajectory from the reference path.
        Return:
            ref_traj: x, y coordinates and the heading angles
        '''
        x, y = state[0], state[1]
        x_next, y_next = ref_path[0].x, ref_path[0].y
        
        ref_traj = TrajectoryNodeList([])
        path_idx = 0
        traveling = True
        while(traveling):# for n in range(N):
            while(True):
                dist_to_next = math.hypot(x_next-x, y_next-y)
                if dist_to_next < 1e-9:
                    path_idx += 1
                    x_next, y_next = ref_path[path_idx].x, ref_path[path_idx].y
                    break
                x_dir = (x_next-x) / dist_to_next
                y_dir = (y_next-y) / dist_to_next
                eta = dist_to_next/speed # estimated time of arrival
                if eta > ts: # move to the target node for t
                    x = x+x_dir*speed*ts
                    y = y+y_dir*speed*ts
                    break # to append the position
                else: # move to the target node then set a new target
                    x = x+x_dir*speed*eta
                    y = y+y_dir*speed*eta
                    path_idx += 1
                    if path_idx > len(ref_path)-1 :
                        traveling = False
                        break
                    else:
                        x_next, y_next = ref_path[path_idx][0], ref_path[path_idx][1]
            if not dist_to_next < 1e-9:
                ref_traj.append(TrajectoryNode(x, y, math.atan2(y_dir,x_dir)))
        return ref_traj
    
    @staticmethod
    def _get_local_ref_traj(idx_ref: int, ref_traj_global: TrajectoryNodeList, state: tuple, action_steps=1, horizon=20) -> Tuple[np.ndarray, int]:
        '''
        Returns:
            ref_traj_local: every row is a state, the number of rows should be equal to the horizon
        '''
        x_ref     = ref_traj_global.numpy()[:,0].tolist()
        y_ref     = ref_traj_global.numpy()[:,1].tolist()
        theta_ref = ref_traj_global.numpy()[:,2].tolist()

        lb_idx = max(0, idx_ref-1*action_steps)                # reduce search space for closest reference point
        ub_idx = min(len(ref_traj_global), idx_ref+5*action_steps)    # reduce search space for closest reference point

        distances = [math.hypot(state[0]-x[0], state[1]-x[1]) for x in ref_traj_global[lb_idx:ub_idx]]
        idx_next = distances.index(min(distances))

        idx_next += lb_idx  # idx in orignal reference trajectory list
        if (idx_next+horizon >= len(x_ref)):
            tmpx = x_ref[idx_next:]      + [x_ref[-1]]*(horizon-(len(x_ref)-idx_next))
            tmpy = y_ref[idx_next:]      + [y_ref[-1]]*(horizon-(len(y_ref)-idx_next))
            tmpt = theta_ref[idx_next:]  + [theta_ref[-1]]*(horizon-(len(theta_ref)-idx_next))
        else:
            tmpx = x_ref[idx_next:idx_next+horizon]
            tmpy = y_ref[idx_next:idx_next+horizon]
            tmpt = theta_ref[idx_next:idx_next+horizon]
        ref_traj_local = np.array([tmpx, tmpy, tmpt]).transpose()
        return ref_traj_local, idx_next
    
    def heading_override_error(self) -> float:
        """Signed angle (rad, -pi..pi) from the current heading to `heading_override`; 0 if unset.
        Lets a caller tell when a turn-in-place hold has finished (see failure_monitor's
        `TURN_ALIGNED_RAD`)."""
        if self.heading_override is None:
            return 0.0
        delta = float(self.heading_override) - float(self.state[2])
        return math.atan2(math.sin(delta), math.cos(delta))

    def get_local_ref_traj(self, local_ref_traj: np.ndarray | None = None):
        if self.holding:
            # e.g. failure_monitor.WaitStrategy: re-anchor to the *current* state every call,
            # rather than a fixed point recorded once when the hold started. A one-shot fixed
            # point leaves a stale target as soon as the robot has any residual momentum/overshoot
            # while decelerating into the hold -- the position-hold cost (qrpd) then reads that as
            # real cross-track error and pulls the solver back to it with no directional preference
            # of its own, which can select a reverse correction. Rebuilding the target from `self.state`
            # every step means it's always exactly where the robot already is -- zero cross-track
            # error by construction, nothing to correct back to.
            x, y, theta = float(self.state[0]), float(self.state[1]), float(self.state[2])
            if self.heading_override is not None:
                # Hold the position, but target a *different* heading -- turn on the spot.
                # See `heading_override`'s docstring; used by failure_monitor.ReorientStrategy.
                theta = float(self.heading_override)
            hold_traj = np.array([[x, y, theta]] * self.N_hor)
            if local_ref_traj is not None and local_ref_traj.shape[1] == 2:
                local_ref_traj = np.concatenate((local_ref_traj, hold_traj[:, [2]]), axis=1)
            return hold_traj, local_ref_traj, hold_traj

        idx_ref = self.idx_ref

        extra_ref_traj, *_ = self._get_local_ref_traj(idx_ref, self.ref_traj, (self.state[0], self.state[1]), action_steps=self.config.action_steps, horizon=self.config.N_hor)
        original_ref_traj, idx_ref = self._get_local_ref_traj(idx_ref, self.ref_traj, (self.state[0], self.state[1]), action_steps=self.config.action_steps, horizon=self.config.N_hor)
        self.idx_ref = idx_ref
        if local_ref_traj is not None:
            if local_ref_traj.shape[1] == 2:
                local_ref_traj = np.concatenate((local_ref_traj, original_ref_traj[:,[2]]), axis=1)
        return original_ref_traj, local_ref_traj, extra_ref_traj
    

    def get_action(self, current_ref_traj: np.ndarray, mode='work', initial_guess: np.ndarray | None = None):
        if self.check_termination_condition(self.state, self._last_action, self.final_goal):
            return None
        actions, pred_states, cost = self.run_step(self.stc_constraints, self.dyn_constraints, 
                                                   self.other_robot_states, current_ref_traj, mode, initial_guess)
        self._last_action = actions[0]
        return actions[0], pred_states, cost

    def run_step(self, 
                 stc_constraints : list, 
                 dyn_constraints : list, 
                 other_robot_states : list, 
                 current_ref_traj : np.ndarray, 
                 mode : str = 'safe', 
                 initial_guess : np.ndarray|None = None):
        '''
        Description:
            Run the trajectory planner for one step.
        Arguments:
            other_robot_states: A list with length "ns*N_hor*Nother" (E.x. [0,0,0] * (self.N_hor*self.config.Nother))
            current_ref_traj: from get_local_ref_traj
        Return:
            :actions
            :pred_states
            :cost
        '''
        ### Mode selection
        # if mode == 'aligning':
        #     if abs(theta_ref[idx_next]-self.state[3])<(math.pi/6):
        #         mode =='work'
        self.set_work_mode(mode)

        ### Get reference states ###
        finish_state = current_ref_traj[-1,:]
        current_refs = current_ref_traj.reshape(-1).tolist()

        ### Get reference velocities ###
        if self.speed_ref_override is not None:
            # e.g. WaitStrategy (failure_monitor.recovery): the position reference is frozen
            # at the current pose, but *this* is what actually stops the robot -- left at the
            # normal distance-to-final-goal formula below, the reference speed (qvel's target)
            # would keep pulling for a nonzero forward speed the whole time the position hold
            # (qrpd) pulls the other way, and the solver settles on a small nonzero-velocity
            # compromise between the two instead of a true stop.
            speed_ref_list = self.speed_ref_override
        else:
            dist_to_goal = math.hypot(self.state[0]-self.final_goal[0], self.state[1]-self.final_goal[1]) # change ref speed if final goal close
            if dist_to_goal >= self.base_speed*self.N_hor*self.ts:
                speed_ref_list = [self.base_speed]*self.N_hor
            else:
                speed_ref = dist_to_goal / self.N_hor / self.ts
                speed_ref = max(speed_ref, self.config.low_speed)
                speed_ref_list = [speed_ref]*self.N_hor

        if len(self.past_actions):
            last_u = self.past_actions[-1]
        else:
            last_u = np.zeros(self.nu)
            
        ### Assemble parameters for solver & Run MPC###
        params = list(self.state) + list(finish_state) + list(last_u) + \
                 self.tuning_params + current_refs + speed_ref_list + \
                 other_robot_states + \
                 stc_constraints + dyn_constraints + self.stc_weights + self.dyn_weights

        try:
            taken_states, pred_states, actions, cost, solver_time, exit_status = self.run_solver(params, self.state, self.config.action_steps, initial_guess)
        except RuntimeError as err:
            raise RuntimeError(f"Fatal: Cannot run solver. {err}.")

        self.past_states.append(self.state)
        self.past_states += taken_states[:-1]
        self.past_actions += actions
        self.state = taken_states[-1]
        self.cost_timelist.append(cost)
        self.solver_time_timelist.append(solver_time)
        self.exit_status = exit_status

        if exit_status in self.config.bad_exit_codes and self.vb:
            print(f"[{self.__class__.__name__}] Bad converge status: {exit_status}")

        return actions, pred_states, cost

    def run_solver(self, parameters:list, state: np.ndarray, take_steps:int=1, initial_guess : np.ndarray | None = None):
        '''Run the solver for the pre-defined MPC problem.

        Args:
            parameters: All parameters used by MPC, defined in 'build'.
            state: The overall states.
            take_steps: The number of control step taken by the input (default 1).
            initial_guess: The initial guess for the solver (default None).

        Returns:
            taken_states: List of taken states, length equal to take_steps.
            pred_states: List of predicted states at this step, length equal to horizon N.
            actions: List of taken actions, length equal to take_steps.
            cost: The cost value of this step.
            solver_time: Time cost for solving MPC of the current time step
            exit_status: The exit state of the solver.

        Comments:
            The motion model (dynamics) is defined initially.
        '''
        solution = self.solver.run(parameters, initial_guess=initial_guess)
        # Two generated-binding shapes are in the wild and which one a solver has depends on the
        # opengen version it was built with, not on anything here: older bindings wrap the answer
        # in a Rust-style Result (`is_ok()`/`get()`), newer ones return the solution itself and
        # `None` when the invocation failed (the reason is printed by the Rust side).
        if hasattr(solution, 'is_ok'):
            if not solution.is_ok():
                solver_error = solution.get()
                raise RuntimeError(f"MPC Solver error: [{solver_error.code}]{solver_error.message}")
            solution_data = solution.get()
        elif solution is None:
            raise RuntimeError("MPC Solver error: the solver returned no solution "
                               "(the reason was printed to stdout by the solver itself).")
        else:
            solution_data = solution
        u = solution_data.solution
        cost: float = solution_data.cost
        exit_status: str = solution_data.exit_status
        solver_time: float = solution_data.solve_time_ms
        
        taken_states:List[np.ndarray] = []
        for i in range(take_steps):
            state_next = self.motion_model( state, np.array(u[(i*self.nu):((i+1)*self.nu)]), self.ts )
            taken_states.append(state_next)

        pred_states:List[np.ndarray] = [taken_states[-1]]
        for i in range(len(u)//self.nu):
            pred_state_next = self.motion_model( pred_states[-1], np.array(u[(i*self.nu):(2+i*self.nu)]), self.ts )
            pred_states.append(pred_state_next)
        pred_states = pred_states[1:]

        actions = u[:self.nu*take_steps]
        actions = np.array(actions).reshape(take_steps, self.nu).tolist()
        actions = [np.array(action) for action in actions]
        return taken_states, pred_states, actions, cost, solver_time, exit_status
