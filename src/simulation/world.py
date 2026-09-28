"""The simulated world one episode runs in: a scenario-grid layout, its moving obstacles, and the
ground-truth checks for collision and arrival.

The robot is driven by the MPC (`mpc_traj_tracker.TrajectoryGenerator`), which integrates its own
state. Each control cycle the loop copies that state in (`set_robot_state`) and calls `step`, which
advances the dynamic obstacles by one period, coasts the robot one period ahead, and checks that
position against every obstacle's footprint (grown by the robot radius) and against the goal.
"""
from __future__ import annotations

import random

from numpy.linalg import norm
from shapely.geometry import LineString, Polygon  # type: ignore
from shapely.geometry import JOIN_STYLE  # type: ignore

from path_planning import GeometricMap, VisibilityPathFinder
from .scenario_grid import (DYNAMIC_RADIUS, MPC_BOUNDARY_MARGIN, MPC_INFLATE_MARGIN, REF_BOUNDARY_MARGIN,
                            REF_INFLATE_MARGIN, ScenarioLayout, build, draw_layout)

DYNAMIC_OBSTACLE_RADIUS = DYNAMIC_RADIUS
"""Radius (m) of every dynamic obstacle; the report's `dynamic_obstacle_gap` assumes it."""


def draw(option: int = 1) -> ScenarioLayout:
    """A scenario-grid layout drawn from the global `random` state, so seeding `random` first
    (`episode.seed_everything`) makes the layout a function of the seed."""
    return draw_layout(option, random.getrandbits(32))


class World:
    """One episode's world, built from `layout`."""

    def __init__(self, layout: ScenarioLayout, time_step: float = 0.2):
        self.layout = layout
        self.time_step = time_step
        self.reset()

    def reset(self) -> None:
        self.robot, self.boundary, self.obstacles, self.goal = build(self.layout)
        # The initial reference route is planned around the obstacles the robot knows about
        # (blocks, kerbs, visible walls); hidden walls and moving obstacles are met on the way.
        finder = VisibilityPathFinder(
            boundary_coords=self.boundary.get_mitred_vertices(REF_BOUNDARY_MARGIN),
            obstacle_list=[o.get_mitred_vertices(REF_INFLATE_MARGIN) for o in self.obstacles
                           if o.visible_on_reference_path])
        path, _ = finder.get_ref_path(self.robot.position.tolist(), self.goal.position.tolist())
        if not path:
            raise RuntimeError(f"no reference route for layout seed {self.layout.seed}")
        self.path = LineString(path)
        self.collided = False
        self.reached_goal = False
        self._check()

    @property
    def dynamic_obstacles(self) -> list:
        return [o for o in self.obstacles if not o.is_static]

    def set_robot_state(self, position, angle: float, speed: float, angular_velocity: float) -> None:
        self.robot.position = position
        self.robot.angle = angle
        self.robot.speed = speed
        self.robot.angular_velocity = angular_velocity

    def step(self) -> bool:
        """Advance one control period. Returns True once the robot has collided or arrived."""
        for obstacle in self.obstacles:
            obstacle.step(self.time_step)
        self.robot.coast(self.time_step)
        self._check()
        return self.collided or self.reached_goal

    def _check(self) -> None:
        self.collided |= any(o.collides(self.robot) for o in self.obstacles)
        self.reached_goal |= bool(norm(self.goal.position - self.robot.position) < self.robot.cfg.RADIUS)

    def geometric_map(self, inflate_margin: float = MPC_INFLATE_MARGIN,
                      boundary_margin: float = MPC_BOUNDARY_MARGIN) -> GeometricMap:
        """The static map as the MPC and the replanner see it: every static obstacle (hidden walls
        included) grown by `inflate_margin`, the boundary shrunk by `boundary_margin`."""
        def inflate(polygon, margin):
            return list(Polygon(polygon).buffer(margin, join_style=JOIN_STYLE.mitre).exterior.coords)[:-1]
        boundary = self.boundary.vertices.tolist()
        obstacles = [o.nodes.tolist() for o in self.obstacles if o.is_static]
        return GeometricMap(boundary_coords=boundary, obstacle_list=obstacles,
                            processed_boundary_coords=inflate(boundary, -boundary_margin),
                            processed_obstacle_list=[inflate(o, inflate_margin) for o in obstacles])
