"""The simulated world's objects: the robot, static and moving obstacles, the map boundary and the goal.

The robot's own motion is integrated by the MPC (`mpc_traj_tracker`); `MobileRobot` here is the
simulator's copy of it, used for the ground-truth collision and goal checks in `world.World`.
Dynamic obstacles follow cyclic keyframe animations (`Animation`), so an obstacle's position is a
pure function of time and the layout that placed it.
"""
from __future__ import annotations

from typing import Callable, List, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray
from shapely.geometry import JOIN_STYLE, Point, Polygon  # type: ignore


def _exterior_nodes(polygon: Polygon, orient: int):
    """The exterior coordinates of `polygon`, clockwise for `orient` 1 and counter-clockwise for -1."""
    exterior = polygon.exterior
    if exterior.is_ccw == (orient > 0):
        coords = exterior.coords[-2::-1]
    else:
        coords = exterior.coords[:-1]
    return np.asarray(coords, dtype=np.float32)


def _orient(nodes: ArrayLike, orient: int):
    return _exterior_nodes(Polygon(nodes), orient)


class MobileRobotSpecification:
    RADIUS = 0.5
    SPEED_MIN = -0.5
    SPEED_MAX = 1.5
    ANGULAR_VELOCITY_MIN = -0.5
    ANGULAR_VELOCITY_MAX = 0.5


class MobileRobot:
    """A unicycle robot with state (x, y, theta, v, w)."""

    def __init__(self, state: NDArray):
        self.state = state
        self.cfg = MobileRobotSpecification()

    @property
    def state(self) -> NDArray:
        return self._state

    @state.setter
    def state(self, state: NDArray) -> None:
        self._state = state
        self.point = Point(self.position)

    @property
    def position(self) -> NDArray:
        return self.state[:2]

    @position.setter
    def position(self, position: NDArray) -> None:
        self.state[:2] = position
        self.point = Point(position)

    @property
    def angle(self) -> float:
        return self.state[2]

    @angle.setter
    def angle(self, angle: float) -> None:
        self.state[2] = angle

    @property
    def speed(self) -> float:
        return self.state[3]

    @speed.setter
    def speed(self, speed: float) -> None:
        self.state[3] = speed

    @property
    def angular_velocity(self) -> float:
        return self.state[4]

    @angular_velocity.setter
    def angular_velocity(self, angular_velocity: float) -> None:
        self.state[4] = angular_velocity

    def coast(self, time_step: float) -> None:
        """Advance one time step at the current (clamped) speed and angular velocity."""
        self.speed = min(max(self.speed, self.cfg.SPEED_MIN), self.cfg.SPEED_MAX)
        self.angular_velocity = min(max(self.angular_velocity, self.cfg.ANGULAR_VELOCITY_MIN),
                                    self.cfg.ANGULAR_VELOCITY_MAX)
        self.angle += time_step * self.angular_velocity
        self.position += time_step * self.speed * np.asarray((np.cos(self.angle), np.sin(self.angle)))


class KeyFrame:
    """Position and rotation of an object at one instant of an animation."""

    def __init__(self, position: ArrayLike, rotation: float):
        self.position = np.asarray(position, dtype=np.float32)
        self.rotation = rotation

    def get_rotation_matrix(self) -> NDArray[np.float32]:
        c = np.cos(self.rotation)
        s = np.sin(self.rotation)
        return np.array([[c, -s], [s, c]], dtype=np.float32)


class Animation:
    """The movement of an obstacle as a cyclic keyframe animation."""

    def __init__(self, time_steps: List[float], keyframes: List[KeyFrame],
                 interp: Callable[[float], float] = lambda x: x, offset: float = 0):
        """
        Args:
            time_steps: Keyframe durations; the first must be 0, and there is one more entry than
                keyframes (the last is the time taken to loop back to the first keyframe).
            keyframes: Keyframes of the animation.
            interp: Interpolation between consecutive keyframes.
            offset: Animation time offset.
        """
        assert time_steps[0] == 0, "First keyframe must be valid at t = 0"
        assert len(time_steps) == len(keyframes) + 1, "Time steps must be one more than the number of keyframes"
        self.time_steps = time_steps
        self.keyframes = keyframes
        self.interp = interp
        self.offset = offset
        self.length = sum(time_steps)

    def get_keyframe(self, time: float) -> KeyFrame:
        time = (time + self.offset) % self.length
        t = 0.0
        for i in range(len(self.keyframes)):
            t += self.time_steps[i]
            if t <= time < t + self.time_steps[i + 1]:
                alpha = self.interp((time - t) / self.time_steps[i + 1])
                k0 = self.keyframes[i]
                k1 = self.keyframes[(i + 1) % len(self.keyframes)]
                return KeyFrame(k0.position * (1 - alpha) + k1.position * alpha,
                                k0.rotation * (1 - alpha) + k1.rotation * alpha)
        raise RuntimeError("Time value did not match any keyframe interval")

    @staticmethod
    def static(position: ArrayLike = (0, 0), angle: float = 0):
        return Animation([0, 1], [KeyFrame(position, angle)])

    @staticmethod
    def periodic_with_pause(p1: ArrayLike, p_mid: ArrayLike, p2: ArrayLike, freq: float,
                            pause_time: float, angle: float = 0.0, offset: float = 0.0):
        """Back and forth between `p1` and `p2`, dwelling at `p_mid` for `pause_time` seconds on
        each pass: p1 -> p_mid -> [pause] -> p2 -> p_mid -> [pause] -> p1. Each leg is eased and
        timed in proportion to its length; `freq` sets the one-way pace (pi / freq seconds)."""
        interp = lambda x: (1 - np.cos(x * np.pi)) / 2
        d1 = float(np.linalg.norm(np.asarray(p_mid, dtype=float) - np.asarray(p1, dtype=float)))
        d2 = float(np.linalg.norm(np.asarray(p2, dtype=float) - np.asarray(p_mid, dtype=float)))
        total_d = d1 + d2
        leg_time = np.pi/freq if freq != 0 else 1
        t1 = leg_time * d1/total_d if total_d > 0 else leg_time/2
        t2 = leg_time * d2/total_d if total_d > 0 else leg_time/2
        keyframes = [KeyFrame(p1, angle), KeyFrame(p_mid, angle), KeyFrame(p_mid, angle),
                     KeyFrame(p2, angle), KeyFrame(p_mid, angle), KeyFrame(p_mid, angle)]
        time_steps = [0.0, t1, pause_time, t2, t2, pause_time, t1]
        return Animation(time_steps, keyframes, interp, offset)

    @staticmethod
    def path_with_stops(waypoints: List[ArrayLike], speed: float, pause_time: Union[float, List[float]] = 0.0,
                        angle: float = 0.0, offset: float = 0.0):
        """A loop through `waypoints` (back from the last to the first) at `speed` m/s, stopping
        `pause_time` seconds at each waypoint (one value, or one per waypoint)."""
        n = len(waypoints)
        if isinstance(pause_time, (int, float)):
            pause_time = [float(pause_time)] * n
        if len(pause_time) != n:
            raise ValueError(f"pause_time must be a scalar or have one entry per waypoint ({n}), got {len(pause_time)}.")
        interp = lambda x: (1 - np.cos(x * np.pi)) / 2
        keyframes: List[KeyFrame] = []
        time_steps = [0.0]
        for i in range(n):
            wp = np.asarray(waypoints[i], dtype=float)
            # Two identical keyframes `pause_time[i]` apart hold the obstacle still.
            keyframes.append(KeyFrame(wp, angle))
            keyframes.append(KeyFrame(wp, angle))
            time_steps.append(pause_time[i])
            nxt = np.asarray(waypoints[(i + 1) % n], dtype=float)
            dist = float(np.linalg.norm(nxt - wp))
            time_steps.append(dist/speed if speed > 0 else 1.0)
        return Animation(time_steps, keyframes, interp, offset)


def _circle_nodes(rx: float, ry: float, corners: int) -> NDArray:
    nodes = np.zeros((corners, 2))
    for i in range(corners):
        angle = 2 * np.pi * i / corners
        nodes[i, :] = (rx * np.cos(angle), -ry * np.sin(angle))
    return nodes


class Obstacle:
    """A static or moving polygonal obstacle."""
    _nodes: NDArray[np.float32]
    _padded_nodes: NDArray[np.float32] | None = None
    _keyframe: KeyFrame
    _padded_polygon: Polygon | None = None

    def __init__(self, nodes: ArrayLike, visible_on_reference_path: bool, animation: Animation, is_static: bool = True):
        """
        Args:
            nodes: Vertices of the obstacle's polygon, relative to its animated position.
            visible_on_reference_path: Whether the initial reference route is planned around it.
                Hidden walls and all dynamic obstacles are not: the robot only meets them on the way.
            animation: The obstacle's movement.
            is_static: Whether it is part of the static map the MPC and the replanner see.
        """
        self.nodes = _orient(nodes, 1)
        self.visible_on_reference_path = visible_on_reference_path
        self.animation = animation
        self.time = 0.0
        self.keyframe = animation.get_keyframe(self.time)
        self.is_static = is_static

    def step(self, time_step: float) -> None:
        self.time += time_step
        self.keyframe = self.animation.get_keyframe(self.time)

    @property
    def nodes(self) -> NDArray[np.float32]:
        return self._nodes

    @nodes.setter
    def nodes(self, nodes: ArrayLike) -> None:
        self._nodes = np.asarray(nodes, dtype=np.float32)
        self._padded_polygon = None
        self._padded_nodes = None

    @property
    def padded_nodes(self) -> NDArray[np.float32]:
        if self._padded_nodes is None:
            polygon = Polygon(self.nodes)
            self._padded_nodes = _exterior_nodes(
                polygon.buffer(MobileRobotSpecification.RADIUS, join_style=JOIN_STYLE.round, resolution=4), 1)
        return self._padded_nodes

    @property
    def keyframe(self) -> KeyFrame:
        return self._keyframe

    @keyframe.setter
    def keyframe(self, keyframe: KeyFrame) -> None:
        self._keyframe = keyframe
        self._padded_polygon = None

    @property
    def padded_polygon(self) -> Polygon:
        if self._padded_polygon is None:
            self._padded_polygon = Polygon(self.get_padded_vertices())
        return self._padded_polygon

    def get_mitred_vertices(self, inflation_margin: float) -> NDArray[np.float32]:
        """The obstacle's current corners, grown by `inflation_margin` with mitred joins."""
        polygon = Polygon(self.get_vertices())
        return _exterior_nodes(polygon.buffer(inflation_margin, join_style=JOIN_STYLE.mitre, mitre_limit=2), 1)

    def get_vertices(self) -> NDArray[np.float32]:
        return self.keyframe.position + (self.keyframe.get_rotation_matrix() @ self.nodes.T).T

    def get_padded_vertices(self) -> NDArray[np.float32]:
        """The obstacle's current corners grown by the robot radius (round joins)."""
        return self.keyframe.position + (self.keyframe.get_rotation_matrix() @ self.padded_nodes.T).T

    def collides(self, robot: MobileRobot) -> bool:
        return self.padded_polygon.contains(robot.point)

    @staticmethod
    def create_static(nodes: ArrayLike) -> "Obstacle":
        return Obstacle(nodes, True, Animation.static(), is_static=True)

    @staticmethod
    def create_dynamic_with_pause(p1: ArrayLike, p_mid: ArrayLike, p2: ArrayLike, freq: float, pause_time: float,
                                  rx: float, ry: float, angle: float = 0.0, corners: int = 12) -> "Obstacle":
        """A `corners`-gon crossing back and forth with a dwell (`Animation.periodic_with_pause`)."""
        animation = Animation.periodic_with_pause(p1, p_mid, p2, freq, pause_time, angle, offset=0.0)
        return Obstacle(_circle_nodes(rx, ry, corners), False, animation, is_static=False)

    @staticmethod
    def create_dynamic_path(waypoints: List[ArrayLike], speed: float, pause_time: Union[float, List[float]],
                            rx: float, ry: float, angle: float = 0.0, corners: int = 12) -> "Obstacle":
        """A `corners`-gon following a looping route (`Animation.path_with_stops`)."""
        animation = Animation.path_with_stops(waypoints, speed, pause_time, angle, offset=0.0)
        return Obstacle(_circle_nodes(rx, ry, corners), False, animation, is_static=False)


class Boundary:
    """Outer boundary of a map."""

    def __init__(self, vertices: ArrayLike):
        self.vertices = np.asarray(_orient(vertices, -1), dtype=np.float32)

    def get_mitred_vertices(self, inflation_margin: float) -> NDArray[np.float32]:
        """The boundary shrunk inward by `inflation_margin` (mitred joins)."""
        polygon = Polygon(self.vertices)
        return _exterior_nodes(polygon.buffer(-inflation_margin, join_style=JOIN_STYLE.mitre, mitre_limit=2), -1)


class Goal:
    def __init__(self, position: ArrayLike):
        self.position = np.asarray(position, dtype=np.float32)
