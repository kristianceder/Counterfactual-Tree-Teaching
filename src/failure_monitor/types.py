from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class FailureMode(Enum):
    """Failure modes the decision-maker can name. Each is judged on the evidence in
    `report.FAILURE_MODE_EVIDENCE` / `report.deadline_miss_evidence`."""
    STUCK = 'stuck'                                   # Local optimum / deadlock: no progress toward the goal
    SOLVER_DEADLINE_MISS = 'solver_deadline_miss'      # MPC solve time exceeds the control period
    COLLISION = 'collision'                            # Robot footprint overlaps an obstacle
    OSCILLATION = 'oscillation'                        # Heading/position oscillating without net progress
    REVERSE_TRACKING = 'reverse_tracking'              # Driving the route backwards instead of turning to face it
    TIMEOUT = 'timeout'                                # Step budget about to run out before reaching the goal


@dataclass
class RobotSnapshot:
    """One control cycle's worth of state, fed to `report.SituationBuilder.update` every step."""
    step: int
    position: tuple[float, float]
    heading: float
    goal: tuple[float, float]
    action: tuple[float, float] | None = None          # (linear velocity, angular velocity)
    solver_time_ms: float | None = None                 # wall-clock time the last MPC solve took (ms)
    obstacles: list[list[tuple[float, float]]] = field(default_factory=list)
    """Static obstacles as raw (uninflated) polygons."""
    speed: float | None = None                          # commanded robot speed (m/s); falls back to action[0]
    reference_path: list[tuple[float, float]] = field(default_factory=list)
    """The local reference handed to the MPC this step. Collapses to a single held point during a
    recovery hold."""
    route_ahead: list[tuple[float, float]] = field(default_factory=list)
    """The robot's current *intended* route to the goal, which a hold does not change. Obstacle-ahead
    checks use this, so waiting does not blind the report to what it is waiting for."""
    dynamic_obstacles: list[tuple[float, float]] = field(default_factory=list)  # dynamic-obstacle centres
    boundary: list[tuple[float, float]] = field(default_factory=list)
    """The map's outer boundary polygon (raw), a wall for `clearance_by_direction`."""


@dataclass
class FailureEvent:
    """One failure verdict: the mode, the step of the report it was made on, and a message."""
    mode: FailureMode
    step: int
    message: str

    def __str__(self) -> str:
        return f"[step {self.step}] {self.mode.value}: {self.message}"
