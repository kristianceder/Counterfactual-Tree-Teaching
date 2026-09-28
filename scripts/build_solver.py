"""Compile the MPC solver (CasADi + OpEn -> Rust) from config/mpc_default.yaml.

    python scripts/build_solver.py

Needs a Rust toolchain (`cargo`). Writes `mpc_solver/` at the repository root, which
`TrajectoryGenerator` loads at run time. Run it once after installing, and again after changing
anything in the config that shapes the optimisation problem (horizon, obstacle counts,
`max_solver_time_micros`, ...); cost weights alone do not need a rebuild.
"""
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main() -> None:
    os.chdir(REPO)   # the build is written relative to the working directory
    from mpc_traj_tracker import MPCConfig, TrajectoryGenerator
    from mpc_traj_tracker.motion_model import unicycle_model
    config = MPCConfig.from_yaml(os.path.join(REPO, "config", "mpc_default.yaml"))
    TrajectoryGenerator(config, unicycle_model, build_solver=True, verbose=True)
    print(f"built {os.path.join(REPO, config.build_directory, config.optimizer_name)}")


if __name__ == "__main__":
    main()
