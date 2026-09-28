"""The scenario grid: a 3x3 city-block world assembled from staged situations, each with a known
right answer, so that every recovery action has situations where it is the right one.

Every episode draws a new layout from a `layout_seed`: street width 5-7 m (so the blocks and the
whole map shift), start and goal on two dead-end stubs at the map edge attached to different
intersections, and a mix of:

- **wall** -- a hidden wall across a loop segment the route uses, with the way round the loop
  open. The robot stops at it; **REPLAN_ROUTE** finds the detour.
- **gate** -- a chokepoint slit on the route with a dynamic obstacle that parks *in* the slit for
  14-19 s and then withdraws into the kerb for 20-30 s. Nothing fits past it, a replan cannot see
  it and returns the same route, and it is gone well before it counts as abandoned: **WAIT** (and
  **RESUME_ROUTE** once it has withdrawn mid-hold). The gate's first closure is timed to the
  robot's estimated arrival, so most gates are met closed.
- **sealed** -- hidden walls that leave no route at all: both ways round the loop, or the goal's
  own dead-end street. The first REPLAN_ROUTE reports `goal_reachable: false`, after which the
  only action that ends the episode well is **REQUEST_HUMAN**.
- **breakdown** -- a gate whose obstacle never leaves, on a loop cut by a *visible* wall so that
  the slit is the only way through. WAIT is the right first answer and the wrong second one:
  once the obstacle has stood still for 30 s, **REQUEST_HUMAN**.
- **reversed** -- the robot starts facing away from its route, which the MPC tracks in reverse:
  **REORIENT**.
- **traffic** -- pedestrians crossing mid-block and a kerb-side patrol, never within 1.5 m of an
  intersection corner (`CORNER_CLEAR_M`), in streets wide enough to drive round: **CONTINUE**.
  These are the false-alarm pressure, not a situation to recover from.

`needs_human` is the layout's ground truth: true exactly for *sealed* and *breakdown*. It is what
an episode's REQUEST_HUMAN is scored against, and what makes a false alarm on a *gate* (where the
report also says "stopped, dynamic obstacle on the route") count as the failure it is.

**Nothing moves near the start.** Every dynamic obstacle's whole swept path stays at least
`SPAWN_CLEAR_M` from the start pose -- the distance the robot covers before the supervisor's
first assessment (25 warm-up steps at ~1.1 m/s) plus a braking margin -- and `GOAL_CLEAR_M` from
the goal. A layout that cannot satisfy this is redrawn, never relaxed.

**One layout per seed.** `draw_layout` is a pure function of its seed; `build` turns a layout into
fresh objects and is deterministic, so the world can be rebuilt as often as needed within an
episode. A layout is kept only if its solvability (with every static obstacle inflated the way the
MPC and the replanner see them) matches what was staged.

`option` 1 is the mixed stream used for every experiment. 2-7 stage exactly one situation each
(wall, gate, sealed, breakdown, reversed, traffic only), for checking that situation in isolation.

Capacity: 9 blocks + 2 gates x 2 kerb boxes + 3 walls = 16 static obstacles (the compiled solver's
`Nstcobs` is 30), 2 gate obstacles + 1 patrol + 2 pedestrians = 5 dynamic ones (`Ndynobs` is 15).
Every static obstacle is a 4-vertex rectangle, as the solver requires.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from math import atan2, pi

import numpy as np
from shapely.geometry import LineString, Point, Polygon, box as shapely_box  # type: ignore
from shapely.ops import unary_union  # type: ignore

from path_planning import VisibilityPathFinder
from .objects import Boundary, Goal, MobileRobot, Obstacle

BLOCK = 4.0
N_BLOCKS = 3
MPC_INFLATE_MARGIN = 1.2     # MUST match world.MPC_INFLATE_MARGIN (the MPC's static constraints)
MPC_BOUNDARY_MARGIN = 0.4    # ... and world.MPC_BOUNDARY_MARGIN
REF_INFLATE_MARGIN = 0.8     # world.World's initial reference route
REF_BOUNDARY_MARGIN = 0.5    # ... and the boundary margin it uses
DYNAMIC_RADIUS = 0.8         # the report's obstacle gaps assume it (world.DYNAMIC_OBSTACLE_RADIUS)
CHOKE_RAW_GAP = 3.0          # 0.6 m effective after MPC inflation
CHOKE_HALF_WIDTH = 0.8
WALL_HALF_THICKNESS = 0.15
MAX_ATTEMPTS = 200

Rect = list[tuple[float, float]]


def _rect(x0: float, y0: float, x1: float, y1: float) -> Rect:
    return [(x0, y0), (x0, y1), (x1, y1), (x1, y0)]


class _Grid:
    """The fixed topology: two vertical and two horizontal streets crossing at four
    intersections, a loop of four street segments between them, and eight dead-end stubs
    running out to the map edge."""

    def __init__(self, street: float):
        self.street = street
        self.step = BLOCK + street
        self.span = N_BLOCKS * BLOCK + (N_BLOCKS - 1) * street
        self.centers = [k * self.step + BLOCK + street / 2 for k in range(N_BLOCKS - 1)]

    def blocks(self) -> list[Rect]:
        return [_rect(i * self.step, j * self.step, i * self.step + BLOCK, j * self.step + BLOCK)
                for i in range(N_BLOCKS) for j in range(N_BLOCKS)]

    def intersection(self, kx: int, ky: int) -> tuple[float, float]:
        return self.centers[kx], self.centers[ky]

    def loop_segments(self) -> list[dict]:
        """The four street segments between intersections, each spanning block row/column 1.
        `axis` is the direction of travel along it; `line` the street's centre coordinate."""
        lo, hi = self.step, self.step + BLOCK
        segments = []
        for k in range(2):
            segments.append({"axis": "x", "line": self.centers[k], "lo": lo, "hi": hi,
                             "ends": (self.intersection(0, k), self.intersection(1, k))})
            segments.append({"axis": "y", "line": self.centers[k], "lo": lo, "hi": hi,
                             "ends": (self.intersection(k, 0), self.intersection(k, 1))})
        return segments

    def band(self, seg: dict) -> Polygon:
        half = self.street / 2
        if seg["axis"] == "x":
            return shapely_box(seg["lo"], seg["line"] - half, seg["hi"], seg["line"] + half)
        return shapely_box(seg["line"] - half, seg["lo"], seg["line"] + half, seg["hi"])

    def stubs(self) -> list[dict]:
        """The eight dead ends: (intersection, outward direction). A point on one is placed
        `inset` metres from the map edge, facing back towards its intersection."""
        stubs = []
        for kx in range(2):
            for ky in range(2):
                stubs.append({"node": (kx, ky), "axis": "x", "outward": -1 if kx == 0 else 1})
                stubs.append({"node": (kx, ky), "axis": "y", "outward": -1 if ky == 0 else 1})
        return stubs

    def stub_point(self, stub: dict, inset: float) -> tuple[float, float, float]:
        cx, cy = self.intersection(*stub["node"])
        edge = 0.0 if stub["outward"] < 0 else self.span
        along = edge - stub["outward"] * inset
        if stub["axis"] == "x":
            return along, cy, pi if stub["outward"] > 0 else 0.0
        return cx, along, -pi / 2 if stub["outward"] > 0 else pi / 2


def _chokepoint(seg: dict, street: float) -> list[Rect]:
    """The two kerb boxes that narrow a street to a slit, for either street orientation."""
    mid = (seg["lo"] + seg["hi"]) / 2
    half_gap, half_street = CHOKE_RAW_GAP / 2, street / 2
    a0, a1 = mid - CHOKE_HALF_WIDTH, mid + CHOKE_HALF_WIDTH
    c = seg["line"]
    if seg["axis"] == "x":
        return [_rect(a0, c - half_street, a1, c - half_gap), _rect(a0, c + half_gap, a1, c + half_street)]
    return [_rect(c - half_street, a0, c - half_gap, a1), _rect(c + half_gap, a0, c + half_street, a1)]


def _reference_path(span: float, visible: list[Rect], start, goal) -> list[tuple[float, float]]:
    """The route `world.World` will plan, computed the same way, so walls and gates can be placed
    on the route the robot actually commits to."""
    obstacles = [Obstacle.create_static(r).get_mitred_vertices(REF_INFLATE_MARGIN) for r in visible]
    boundary = Boundary(_rect(0.0, 0.0, span, span)).get_mitred_vertices(REF_BOUNDARY_MARGIN)
    path, _ = VisibilityPathFinder(boundary_coords=boundary, obstacle_list=obstacles).get_ref_path(
        list(start[:2]), list(goal))
    return [(float(x), float(y)) for x, y in path]


def _solvable(span: float, statics: list[Rect], start, goal) -> bool:
    """Whether start and goal share a free region once every static obstacle is inflated the
    way the MPC and `ReplanRouteStrategy` see them."""
    free = shapely_box(0, 0, span, span).buffer(-MPC_BOUNDARY_MARGIN, join_style=2)
    grown = unary_union([Polygon(r).buffer(MPC_INFLATE_MARGIN, join_style=2) for r in statics])
    free = free.difference(grown)
    a, b = Point(start[:2]), Point(goal)
    parts = getattr(free, "geoms", [free])
    return any(p.contains(a) and p.contains(b) for p in parts)


def _street_axis_at(grid: _Grid, x: float, y: float, rng: random.Random) -> str | None:
    """Direction of travel of the street a point is in ("x" or "y"), random at an intersection,
    None if the point is not inside a street."""
    half = grid.street / 2
    in_vertical = any(abs(x - c) <= half for c in grid.centers)
    in_horizontal = any(abs(y - c) <= half for c in grid.centers)
    if in_vertical and in_horizontal:
        return rng.choice(["x", "y"])
    return "y" if in_vertical else "x" if in_horizontal else None


SPAWN_CLEAR_M = 7.0
GOAL_CLEAR_M = 3.0
STAGE_CLEAR_M = 4.0          # traffic keeps this far from a wall or gate, so a staged stall reads as what it is
ROBOT_SPEED_MPS = 1.1        # measured free-driving speed; times the gates
WALL_DELAY_S = 12.0          # what a hidden wall costs before the detour starts: the stall, the verdict, the turn
GATE_CLOSED_S = (14.0, 19.0) # below tree.ABANDONED_OBSTACLE_S (30 s) on purpose: a gate is never a breakdown
GATE_OFF_CENTRE_M = (0.3, 0.5)
"""How far from the slit's centre line the gate's obstacle parks. It still shuts the slit (the free
channel after the MPC's inflation is 0.6 m wide, and the obstacle plus the robot need 1.3 m), but
the robot no longer meets it exactly head-on. Head-on is a symmetric problem the MPC answers by
simply stopping, solver idle, and then nothing distinguishes CONTINUE from WAIT; off-centre it
tries to slide past on the wider side and the solver runs at its time cap until something holds
the robot. Measured over 20 layouts: all 8 dead-centre approaches stopped cleanly, all 12 offset
or skewed ones ground (18-57 consecutive solves over the control period)."""
GATE_MIN_TRAVEL_M = 10.0
"""The least the robot drives before it reaches a gate. `SPAWN_CLEAR_M` is a straight-line distance
and keeps the robot from being hit; this is a driving distance and keeps the supervisor in the
game: the first assessment is at step 25 and a verdict needs evidence that takes a second to
build, so a gate 7 m down the road (reached at step 31, in the one layout where it happened) is
met by the MPC alone, which steered off the parked obstacle into the kerb."""
PATROL_LANE_M = 1.0
"""A patrol drives this far from the block wall, up and down one block's length. In the middle of a
5 m street a 0.8 m obstacle leaves 0.5 m either side of it once the MPC's 1.2 m inflation is taken
off, so a patrol that pauses there shuts the street; and one that ends in an intersection's mouth
sits on every route through it (one layout spent 150 steps behind exactly that). Kerb-side, the
robot always has at least 1.5 m to pass in."""
PATROL_MIN_STREET_M = 6.0
CORNER_CLEAR_M = 1.5
"""Traffic stays this far, along the street, from the end of any block -- a robot's length and a
half from every intersection corner. The spawn guarantee keeps traffic away from the *start*; this
keeps it away from *corners*, where a robot squeezing past a slow pedestrian or the patrol at
1-1.5 m/s clips the block corner before any order could matter. Avoiding a moving obstacle in a straight street is the MPC's job and it does it; avoiding one while
rounding a corner it does not, and that is not what this map is for."""
GATE_APPROACH = (0.3, 3.0)
"""A gate is staged only where the route reaches the slit at least this far off the street's centre
line (m) or at least this far off its direction (deg) -- the other half of the same measurement.
Shifting the slit itself to force an angle was tried and dropped: the wider kerb box then stands
dead ahead of a robot at full speed, and the MPC, steering off the parked obstacle, drove into it
in 4 of 16 layouts before the first assessment."""
GATE_OPEN_S = (20.0, 30.0)
GATE_SPEED_MPS = (0.5, 0.8)
PARKED_FOREVER_S = 1.0e6

OPTIONS = {1: "mixed", 2: "wall", 3: "gate", 4: "sealed", 5: "breakdown", 6: "reversed", 7: "traffic"}

EXPECTED = {
    "wall": "REPLAN_ROUTE",
    "gate": "WAIT",
    "sealed": "REQUEST_HUMAN",
    "breakdown": "REQUEST_HUMAN",
    "reversed": "REORIENT",
}
"""The recovery each staged situation exists to call for. *sealed* passes through a REPLAN_ROUTE
(the one that finds no route) and *breakdown* through a WAIT first; this is where they end."""

@dataclass
class ScenarioLayout:
    """Everything drawn for one episode. `build` turns it into fresh map objects; `describe` is
    what a recording keeps of it."""
    seed: int
    option: int
    street: float
    span: float
    start: tuple[float, float, float]
    goal: tuple[float, float]
    reference: list[tuple[float, float]]
    blocks: list[Rect] = field(default_factory=list)
    kerbs: list[Rect] = field(default_factory=list)          # the gates' slit boxes; visible
    visible_walls: list[Rect] = field(default_factory=list)  # known to the reference planner
    hidden_walls: list[Rect] = field(default_factory=list)   # known to the MPC and the replanner only
    dynamic: list[tuple[str, dict]] = field(default_factory=list)
    situations: list[dict] = field(default_factory=list)
    needs_human: bool = False

    @property
    def kinds(self) -> list[str]:
        return [s["kind"] for s in self.situations]

    def describe(self) -> dict:
        return {"generator": "grid_scenarios", "seed": self.seed, "option": self.option,
                "needs_human": self.needs_human, "situations": self.situations,
                "traffic": [k for k, _ in self.dynamic if k in ("patrol", "pedestrian")]}

    def summary(self) -> str:
        kinds = [k for k, _ in self.dynamic]
        staged = ", ".join(self.kinds) or "nothing staged"
        return (f"grid-scenarios seed={self.seed} option={self.option} street={self.street:.1f}m "
                f"start=({self.start[0]:.1f},{self.start[1]:.1f}) goal=({self.goal[0]:.1f},{self.goal[1]:.1f}) "
                f"[{staged}] needs_human={'yes' if self.needs_human else 'no'} "
                f"patrols={kinds.count('patrol')} pedestrians={kinds.count('pedestrian')}")


def _wall_across(seg: dict, street: float, pos: float) -> Rect:
    half, c, t = street / 2, seg["line"], WALL_HALF_THICKNESS
    return (_rect(pos - t, c - half, pos + t, c + half) if seg["axis"] == "x"
            else _rect(c - half, pos - t, c + half, pos + t))


def _approach(route: LineString, seg: dict) -> tuple[float, float]:
    """How the route meets a slit in the middle of `seg`: its lateral offset from the street's
    centre line there (m), and the angle between its direction over the last 2 m and the street's
    (deg). Both zero is a dead-ahead approach."""
    mid = (seg["lo"] + seg["hi"]) / 2
    centre = Point(_segment_point(seg, mid))
    at = route.project(centre)
    a, b = route.interpolate(max(0.0, at - 2.5)), route.interpolate(max(0.0, at - 0.5))
    heading = atan2(b.y - a.y, b.x - a.x)
    street = 0.0 if seg["axis"] == "x" else pi / 2
    skew = abs((heading - street + pi / 2) % pi - pi / 2)
    return route.distance(centre), skew * 180 / pi


def _segment_point(seg: dict, pos: float) -> tuple[float, float]:
    return (pos, seg["line"]) if seg["axis"] == "x" else (seg["line"], pos)


def _used(grid: _Grid, route: LineString, segments: list[dict]) -> list[dict]:
    """The loop segments `route` really drives along, in the order it reaches them."""
    used = [s for s in segments if route.intersection(grid.band(s)).length > 1.0]
    return sorted(used, key=lambda s: route.project(Point(_segment_point(s, (s["lo"] + s["hi"]) / 2))))


def _swept(kind: str, p: dict) -> LineString:
    return LineString(p["points"])


def _draw_plan(option: int, rng: random.Random) -> dict:
    """Which situations this layout stages. Option 1 draws them; the others fix one."""
    plan = {"human": None, "wall": False, "gates": 0, "reversed": False, "cut": False, "traffic": True}
    name = OPTIONS[option]
    if name == "mixed":
        plan["human"] = rng.choices([None, "sealed", "breakdown"], weights=[0.72, 0.14, 0.14])[0]
        plan["reversed"] = rng.random() < 0.25
        if plan["human"] is None:
            plan["wall"] = rng.random() < 0.55
            plan["gates"] = rng.choices([0, 1, 2], weights=[0.25, 0.55, 0.2])[0]
            # Without a hidden wall a gate's street can be the only way through or not; with one,
            # the wall itself is what makes the detour the only way.
            plan["cut"] = plan["gates"] > 0 and not plan["wall"] and rng.random() < 0.5
        elif plan["human"] == "sealed":
            plan["gates"] = rng.choices([0, 1], weights=[0.7, 0.3])[0]
        else:
            plan["cut"] = True
    elif name == "wall":
        plan["wall"] = True
    elif name == "gate":
        plan["gates"] = 1
        plan["cut"] = rng.random() < 0.5
    elif name == "sealed":
        plan["human"] = "sealed"
    elif name == "breakdown":
        plan["human"], plan["cut"] = "breakdown", True
    elif name == "reversed":
        plan["reversed"] = True
    if name not in ("mixed", "traffic"):
        plan["traffic"] = rng.random() < 0.5
    return plan


def draw_layout(option: int, seed: int) -> ScenarioLayout:
    if option not in OPTIONS:
        raise ValueError(f"Invalid scenario grid option {option}, should be one of {sorted(OPTIONS)}.")
    rng = random.Random(seed)
    # Drawn once, outside the retry loop: redrawing the plan with the geometry would let every
    # rejection re-roll towards whatever is easiest to place, and the mixed stream drifted to 27%
    # layouts with nothing staged that way.
    plan = _draw_plan(option, rng)
    for attempt in range(3 * MAX_ATTEMPTS):
        if attempt == 2 * MAX_ATTEMPTS and plan["gates"] > 1:
            # Two gates need two staged streets at least GATE_MIN_TRAVEL_M down the road, which
            # some draws of start and goal never offer. One gate is the same situation once.
            plan = {**plan, "gates": 1}
        layout = _try_layout(option, seed, rng, plan)
        if layout is not None:
            return layout
    raise RuntimeError(f"No valid scenario grid layout found in {MAX_ATTEMPTS} attempts (seed {seed}).")


def _try_layout(option: int, seed: int, rng: random.Random, plan: dict) -> ScenarioLayout | None:
    grid = _Grid(street=rng.uniform(5.0, 7.0))
    blocks, segments, half = grid.blocks(), grid.loop_segments(), grid.street / 2

    stubs = grid.stubs()
    s_stub = rng.choice(stubs)
    g_stub = rng.choice([s for s in stubs if s["node"] != s_stub["node"]])
    sx, sy, facing = grid.stub_point(s_stub, rng.uniform(0.8, 2.0))
    gx, gy, _ = grid.stub_point(g_stub, rng.uniform(0.6, 1.2))
    goal = (gx, gy)
    heading = facing + rng.uniform(-0.3, 0.3)
    if plan["reversed"]:
        heading = facing + pi + rng.uniform(-0.6, 0.6)
    start = (sx, sy, heading)

    statics_known = list(blocks)
    try:
        bare = LineString(_reference_path(grid.span, statics_known, start, goal))
    except Exception:
        return None
    used = _used(grid, bare, segments)
    if not used:
        return None
    unused = [s for s in segments if s not in used]

    # -- a visible wall that cuts the loop, so that what is staged on the route has no way round --
    visible_walls: list[Rect] = []
    if plan["cut"]:
        if not unused:
            return None
        cut = rng.choice(unused)
        visible_walls.append(_wall_across(cut, grid.street, (cut["lo"] + cut["hi"]) / 2))
        statics_known += visible_walls

    # -- hidden walls ---------------------------------------------------------------------------
    hidden_walls: list[Rect] = []
    situations: list[dict] = []
    wall_seg = None
    sealed_at_goal = plan["human"] == "sealed" and rng.random() < 0.5
    if plan["wall"] or (plan["human"] == "sealed" and not sealed_at_goal):
        wall_seg = rng.choice(used)
        pos = rng.uniform(wall_seg["lo"] + 0.6, wall_seg["hi"] - 0.6)
        hidden_walls.append(_wall_across(wall_seg, grid.street, pos))
        if plan["wall"]:
            situations.append({"kind": "wall", "expected": EXPECTED["wall"], "at": _segment_point(wall_seg, pos)})
    if plan["human"] == "sealed":
        if sealed_at_goal:
            # The goal's own dead-end street: the robot drives the whole route before it finds out.
            cx, cy = grid.intersection(*g_stub["node"])
            edge = 0.0 if g_stub["outward"] < 0 else grid.span
            along = edge - g_stub["outward"] * 3.2
            stub_seg = {"axis": g_stub["axis"], "line": cy if g_stub["axis"] == "x" else cx}
            hidden_walls.append(_wall_across(stub_seg, grid.street, along))
            where = _segment_point(stub_seg, along)
        else:
            # Both ways round the loop: a second hidden wall on the way the first one leaves open.
            if not unused:
                return None
            seg = rng.choice(unused)
            hidden_walls.append(_wall_across(seg, grid.street, rng.uniform(seg["lo"] + 0.6, seg["hi"] - 0.6)))
            where = _segment_point(wall_seg, pos)
        situations.append({"kind": "sealed", "expected": EXPECTED["sealed"], "at": where})

    # -- what the robot ends up driving: the reference, then (after a hidden wall) the detour a
    #    replan finds from just short of that wall ------------------------------------------------
    detour = None
    wall_at_s = None
    if plan["wall"]:
        wall_at_s = bare.project(Polygon(hidden_walls[0]).centroid)
        turn_back = bare.interpolate(max(0.0, wall_at_s - 1.5))
        try:
            detour = LineString(_reference_path(grid.span, statics_known + hidden_walls,
                                                (turn_back.x, turn_back.y, 0.0), goal))
        except Exception:
            return None

    # -- gates ----------------------------------------------------------------------------------
    n_gates = 1 if plan["human"] == "breakdown" else plan["gates"]
    gate_segments = [s for s in used if s is not wall_seg]
    if detour is not None:
        gate_segments += [s for s in _used(grid, detour, segments) if s is not wall_seg and s not in gate_segments]
    rng.shuffle(gate_segments)
    if len(gate_segments) < n_gates:
        return None
    gate_segments = gate_segments[:n_gates]
    kerbs = [r for s in gate_segments for r in _chokepoint(s, grid.street)]

    # The kerb boxes are visible, so they move the reference a little: plan it again, and check it
    # still goes the way the staging assumed.
    try:
        reference = _reference_path(grid.span, statics_known + kerbs, start, goal)
        if detour is not None:
            detour = LineString(_reference_path(grid.span, statics_known + kerbs + hidden_walls,
                                                (turn_back.x, turn_back.y, 0.0), goal))
    except Exception:
        return None
    if len(reference) < 2:
        return None
    route = LineString(reference)
    if _used(grid, route, segments) != used:
        return None
    if wall_seg is not None and not route.intersects(Polygon(hidden_walls[0])):
        return None   # the wall has to be on the route the robot commits to, or it stages nothing

    # A gate is only staged where the robot comes at it off-centre or at an angle (see
    # GATE_APPROACH): met dead ahead, the MPC just stops and the gate stages nothing.
    for seg in gate_segments:
        path = detour if (detour is not None and detour.distance(Point(_segment_point(
            seg, (seg["lo"] + seg["hi"]) / 2))) < route.distance(Point(_segment_point(
                seg, (seg["lo"] + seg["hi"]) / 2)))) else route
        offset, skew = _approach(path, seg)
        if offset < GATE_APPROACH[0] and skew < GATE_APPROACH[1]:
            return None

    statics = statics_known + kerbs + hidden_walls
    if _solvable(grid.span, statics, start, goal) != (plan["human"] != "sealed"):
        return None
    if not _solvable(grid.span, statics_known + kerbs, start, goal):
        return None   # sealed has to mean sealed by the hidden walls, not by something having gone wrong

    layout = ScenarioLayout(seed=seed, option=option, street=grid.street, span=grid.span, start=start,
                            goal=goal, reference=reference, blocks=blocks, kerbs=kerbs,
                            visible_walls=visible_walls, hidden_walls=hidden_walls,
                            situations=situations, needs_human=plan["human"] is not None)
    if plan["reversed"]:
        layout.situations.insert(0, {"kind": "reversed", "expected": EXPECTED["reversed"],
                                     "at": (start[0], start[1])})

    start_pt, goal_pt = Point(start[:2]), Point(goal)
    staged_at = [Polygon(w).centroid for w in hidden_walls + visible_walls]

    def acceptable(path: LineString, clear_of_stage: bool) -> bool:
        if path.distance(start_pt) < SPAWN_CLEAR_M or path.distance(goal_pt) < GOAL_CLEAR_M:
            return False
        return not clear_of_stage or all(path.distance(p) >= STAGE_CLEAR_M for p in staged_at)

    # -- the gates' obstacles --------------------------------------------------------------------
    for seg in gate_segments:
        mid = (seg["lo"] + seg["hi"]) / 2
        side_sign = rng.choice([-1, 1])
        off = rng.choice([-1, 1]) * rng.uniform(*GATE_OFF_CENTRE_M)
        slit = ((mid, seg["line"] + off) if seg["axis"] == "x" else (seg["line"] + off, mid))
        side = ((mid, seg["line"] + side_sign * half) if seg["axis"] == "x"
                else (seg["line"] + side_sign * half, mid))
        if not acceptable(LineString([slit, side]), clear_of_stage=False):
            return None
        if any(Point(slit).distance(p) < STAGE_CLEAR_M for p in staged_at):
            return None
        # When the robot gets there: along the reference if the slit comes before any hidden wall,
        # otherwise out to the wall, a stall, and along the detour.
        along_route = route.project(Point(slit))
        on_route = route.distance(Point(slit)) < 1.0
        if detour is None or (on_route and along_route < wall_at_s):
            if along_route < GATE_MIN_TRAVEL_M:
                return None
            arrival_s = along_route / ROBOT_SPEED_MPS
        else:
            arrival_s = (wall_at_s + detour.project(Point(slit))) / ROBOT_SPEED_MPS + WALL_DELAY_S
        broken = plan["human"] == "breakdown"
        closed_s = PARKED_FOREVER_S if broken else rng.uniform(*GATE_CLOSED_S)
        params = {"points": [slit, side], "speed": rng.uniform(*GATE_SPEED_MPS),
                  "pauses": [closed_s, rng.uniform(*GATE_OPEN_S)],
                  # In the slit from `closes_at_s`: a few seconds before the robot gets there, so
                  # it arrives to a gate that is visibly shut rather than one shutting on it.
                  "closes_at_s": 0.0 if broken else max(0.0, arrival_s - rng.uniform(2.0, 6.0))}
        layout.dynamic.append(("gate", params))
        kind = "breakdown" if broken else "gate"
        layout.situations.append({"kind": kind, "expected": EXPECTED[kind], "at": slit,
                                  "closed_s": None if broken else round(closed_s, 1),
                                  "closes_at_s": round(params["closes_at_s"], 1),
                                  "only_way_through": bool(plan["cut"] or plan["wall"] or broken)})
        staged_at.append(Point(slit))

    # -- traffic: never near the start, the goal, or anything staged ------------------------------
    if plan["traffic"]:
        quiet = [s for s in segments if s not in gate_segments and s is not wall_seg]
        rng.shuffle(quiet)
        # Only in a street wide enough to pass in comfortably: at 5 m the band left beside a
        # kerb-lane patrol is 1.5 m, and two robots in 40 crawled past one a WAIT at a time until
        # their step budget ran out.
        n_patrols = rng.choice([0, 1, 1]) if grid.street >= PATROL_MIN_STREET_M else 0
        for seg in quiet[:n_patrols]:
            # One block's length, kerb-side, never into an intersection: see PATROL_LANE_M.
            # The kerb the route does not run along: a visibility route hugs the inside of every
            # corner, so it is on one kerb lane or the other, and a patrol sharing it stalled one
            # robot for 80 steps and was hit by another.
            mid_pt = Point(_segment_point(seg, (seg["lo"] + seg["hi"]) / 2))
            drive = detour if detour is not None else route
            near = drive.interpolate(drive.project(mid_pt))
            side = (near.y if seg["axis"] == "x" else near.x) - seg["line"]
            sign = -1 if side > 0 else 1 if side < 0 else rng.choice([-1, 1])
            lane = seg["line"] + sign * (half - PATROL_LANE_M)
            lo, hi = seg["lo"] + CORNER_CLEAR_M, seg["hi"] - CORNER_CLEAR_M
            pts = [(lo, lane), (hi, lane)] if seg["axis"] == "x" else [(lane, lo), (lane, hi)]
            if not acceptable(LineString(pts), clear_of_stage=True):
                continue
            layout.dynamic.append(("patrol", {"points": pts, "speed": rng.uniform(0.2, 0.35),
                                              "pause": rng.uniform(2.0, 5.0), "phase": rng.random()}))
        for _ in range(rng.choice([0, 1, 1, 2])):
            for _attempt in range(20):
                driven = detour if detour is not None else route
                if driven.length <= SPAWN_CLEAR_M + GOAL_CLEAR_M:
                    break
                p = driven.interpolate(rng.uniform(0.0, driven.length - GOAL_CLEAR_M))
                axis = _street_axis_at(grid, p.x, p.y, rng)
                if axis is None:
                    continue
                # Mid-block only: see CORNER_CLEAR_M. (This also rules out intersections, where the
                # street's axis was a coin toss.)
                along = p.x if axis == "x" else p.y
                if not any(k * grid.step + CORNER_CLEAR_M <= along <= k * grid.step + BLOCK - CORNER_CLEAR_M
                           for k in range(N_BLOCKS)):
                    continue
                line = min(grid.centers, key=lambda c: abs((p.y if axis == "x" else p.x) - c))
                # It crosses the route but pauses by the kerb on the far side of it, not on it: one
                # that keeps coming back to stand on the robot's line beside a block corner can hold
                # a robot for 150 steps. Traffic is there to be driven past.
                lateral = (p.y if axis == "x" else p.x) - line
                rest = line + (-1 if lateral > 0 else 1) * (half - PATROL_LANE_M)
                across = ([(p.x, line - half), (p.x, rest), (p.x, line + half)] if axis == "x"
                          else [(line - half, p.y), (rest, p.y), (line + half, p.y)])
                if not acceptable(LineString(across), clear_of_stage=True):
                    continue
                # Slow on purpose: at 1-1.4 m/s a pedestrian can walk into the side of the robot
                # faster than the MPC gets out of the way.
                layout.dynamic.append(("pedestrian", {"points": across, "freq": rng.uniform(0.05, 0.1),
                                                      "pause": rng.uniform(2.0, 5.0), "phase": rng.random()}))
                staged_at.append(p)
                break

    # The guarantee, checked on what was actually built rather than trusted to the code above.
    for kind, p in layout.dynamic:
        path = _swept(kind, p)
        if path.distance(start_pt) < SPAWN_CLEAR_M or path.distance(goal_pt) < GOAL_CLEAR_M:
            return None
    if OPTIONS[option] not in ("mixed", "traffic") and OPTIONS[option] not in layout.kinds:
        return None
    return layout


def build(layout: ScenarioLayout):
    """Fresh objects from a layout: (robot, boundary, obstacles, goal)."""
    obstacles = [Obstacle.create_static(r) for r in layout.blocks + layout.kerbs + layout.visible_walls]
    hidden: list[Obstacle] = [Obstacle.create_static(r) for r in layout.hidden_walls]
    for kind, p in layout.dynamic:
        if kind == "gate":
            obs = Obstacle.create_dynamic_path(waypoints=p["points"], speed=p["speed"], pause_time=p["pauses"],
                                               rx=DYNAMIC_RADIUS, ry=DYNAMIC_RADIUS, corners=20)
            # The animation's t=0 is the moment the obstacle arrives in the slit.
            obs.animation.offset = (-p["closes_at_s"]) % obs.animation.length
        elif kind == "patrol":
            obs = Obstacle.create_dynamic_path(waypoints=p["points"], speed=p["speed"], pause_time=p["pause"],
                                               rx=DYNAMIC_RADIUS, ry=DYNAMIC_RADIUS, corners=20)
            obs.animation.offset = p["phase"] * obs.animation.length
        else:
            p1, p_mid, p2 = p["points"]
            obs = Obstacle.create_dynamic_with_pause(p1=p1, p_mid=p_mid, p2=p2, freq=p["freq"],
                                                     pause_time=p["pause"], rx=DYNAMIC_RADIUS,
                                                     ry=DYNAMIC_RADIUS, angle=0.0, corners=20)
            obs.animation.offset = p["phase"] * obs.animation.length
        obs.keyframe = obs.animation.get_keyframe(obs.time)
        hidden.append(obs)
    for o in hidden:
        o.visible_on_reference_path = False

    robot = MobileRobot(np.array([layout.start[0], layout.start[1], layout.start[2], 0, 0]))
    return robot, Boundary(_rect(0.0, 0.0, layout.span, layout.span)), obstacles + hidden, Goal(layout.goal)
