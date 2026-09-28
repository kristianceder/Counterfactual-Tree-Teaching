# MPC Trajectory Tracker

A receding-horizon MPC controller that turns a reference path into feasible
unicycle control commands, subject to static/dynamic obstacle avoidance,
other-robot avoidance, and kinematic limits. Built on
[CasADi](https://web.casadi.org/) + [OpEn](https://alphaville.github.io/optimization-engine/)
(PANOC, a first-order proximal-gradient solver), which compiles the problem
below into a native Rust solver (`solver_generator.py`'s `MpcModule.build()`,
invoked via `python scripts/build_solver.py`).

```
[visibility-graph planner] --reference path--> [TrajectoryGenerator(Config)] <--predicted dynamic obstacles-- [simulated world]
                                                         ^--hold / heading / new route-- [recovery strategies]
```

`TrajectoryGenerator` (`trajectory_generator.py`) is the runtime driver: it
loads the compiled solver, converts a reference *path* into a reference
*trajectory* (`get_global_ref_traj`/`set_ref_trajectory`), and calls the
solver once per control cycle (`run_step`/`get_action`) with the current
state, a sliding window of that reference (`get_local_ref_traj`), and the
current obstacle set. `solver_generator.py`'s `MpcModule.build()` is where the
actual optimization problem -- states, cost, constraints -- is defined; this
file documents that formulation.

## State, inputs, and dynamics

State `s = (x, y, θ)` (position, heading), input `u = (v, ω)` (linear,
angular velocity). Discretized with RK4 under the unicycle model
(`mpc_traj_tracker.motion_model.unicycle_model`):

```
ẋ = v·cos(θ)      x_{k+1} = x_k + ts · ẋ   (RK4-integrated, not forward-Euler)
ẏ = v·sin(θ)      y_{k+1} = y_k + ts · ẏ
θ̇ = ω             θ_{k+1} = θ_k + ts · θ̇
```

The solver optimizes over `u_0, ..., u_{N-1}` (flattened into the decision
variable `u` in `build()`) across a horizon of `N_hor` steps (default 20,
`ts=0.2s` → a 4s lookahead), forward-simulating the state trajectory
`state_next` from the current state at each step of the horizon loop.

## Cost function

Summed over every horizon step `kt`, plus a terminal term. Each term's
config-file weight is in parens; `0.0` means the term is present in the
formula but currently switched off.

| Term | Formula | Weight | What it does |
|---|---|---|---|
| Reference path deviation | `qrpd · min_i dist(state, segment_i)²` | `qrpd` (100.0) | Cross-track cost: distance from the predicted position to the closest segment of the local reference *path* (not a specific timed point on it) -- direction-agnostic. |
| Reference velocity deviation | `qvel · (v − v_ref)²` | `qvel` (10.0) | Tracks the reference *speed* (a positive scalar per horizon step, see `run_step`), not just its magnitude -- driving in reverse is scored as a large error here. |
| **Reference heading deviation** | `qtheta · (1 − cos(θ − θ_ref))` | `qtheta` (20.0) | Tracks the local reference trajectory's heading (the direction of travel between consecutive reference points, computed via `atan2` in `get_global_ref_traj`). Uses `1−cos(Δθ)` rather than a raw squared difference so it wraps smoothly through ±π. **Added to fix a real bug** -- see "Why the heading cost exists" below. |
| Control action | `rv·v² + rw·ω²` | `lin_vel_penalty` (0), `ang_vel_penalty` (0) | Penalizes control effort directly (both currently off; effort is regulated via the acceleration cost/limits instead). |
| Fleet collision | `1000 · Σ_j max(0, d_safe² − ‖p − p_j‖²)` | fixed (1000) | Soft repulsion from other robots' predicted positions (`d_safe = vehicle_width`); not the primary defense -- see "Other robots" below. |
| Dynamic obstacle proximity | `q_dyn[kt] · max(0, inside_ellipse)²` | `q_dyn` (per-step, set via `TrajectoryGenerator.set_obstacle_weights`, default 1e3) | Soft cost for being near/inside a dynamic obstacle's ellipse, inflated by `social_margin` (0.2) beyond its actual footprint -- a "personal space" buffer on top of the hard-ish avoidance below. |
| Position deviation | `qpos · (state − ref_point)²` | `qpos` (0.0) | **Unused** -- unpacked from the parameter vector but never added to `cost` in `build()`. Present for a future point-exact (rather than path-cross-track) tracking mode. |
| Acceleration | `acc_penalty·‖a‖² + w_acc_penalty·‖α‖²` | `lin_acc_penalty` (10.0), `ang_acc_penalty` (20.0) | Penalizes linear/angular acceleration (finite difference of consecutive `v`/`ω`, including the transition from the previous step's applied action). |
| Terminal position | `qN · ‖p_N − p_goal‖²` | `qpN` (0.0) | **Also effectively unused** at the default weight -- reaching the goal is instead handled by `check_termination_condition`'s explicit tolerance check outside the solver. |
| Terminal heading | `qthetaN · (θ_N − θ_goal)²` | `qthetaN` (0.0) | Same -- off by default. Unlike the per-step heading cost above, this uses a raw squared difference (a latent ±π-wrap bug if ever turned on with a `θ_goal` near ±π). |

## Constraints

Three different mechanisms are used, in increasing order of strictness --
see `problem.with_*` in `build()`:

1. **Box constraints** (`with_constraints`, hard, exact): control bounds
   `lin_vel_min/max` (**-0.5 / 1.5** m/s -- note reverse driving is allowed
   by design), `ang_vel_max` (0.5 rad/s), applied to every `u_t` directly.
2. **Augmented-Lagrangian constraints** (`with_aug_lagrangian_constraints`,
   dual-variable-driven, converges to true feasibility): acceleration limits
   (`lin_acc_min/max`, `ang_acc_max`) **and static-obstacle avoidance**,
   combined into one mapping since OpEn's `Problem` only holds one ALM
   constraint set (a second call would silently replace the first, not add
   to it -- see the comment in `build()`).
3. **Penalty-method constraints** (`with_penalty_constraints`, soft, a
   weighted term the solver is only encouraged to satisfy): dynamic-obstacle
   avoidance only.

### Static obstacles: hardened via ALM

Each static obstacle is a convex polygon given as a half-space (H-)
representation `b − a0·x − a1·y > 0` per edge (`util.utils_geo.polygon_halfspace_representation`,
computed once when `update_static_constraints()` is called -- typically
once per episode for a static map). `inside_pollygon()` returns a smooth,
always-`≥0` "how far inside every half-space at once" indicator (a product
of `max(0, edge_value)²` terms -- zero the moment *any* edge is violated,
i.e. the point is outside the polygon).

**It is a hard (ALM) constraint, not a soft penalty.** As a penalty (`with_penalty_constraints`) the
indicator, summed over every obstacle and horizon step into one scalar, is only *discouraged* by a
fixed, slowly escalating weight, and under the real-time iteration budget the solve can converge to
a solution that lets the robot's footprint penetrate an obstacle instead of stopping short of it. As
an ALM constraint there is one aggregate per horizon step (`Σ_obstacles inside_pollygon(state_next)`,
the `Nstcobs` obstacle slots collapsed into a single per-`kt` value) constrained to `== 0`. Since
every term in that sum is individually `≥0`, forcing the sum to zero forces every obstacle to be
fully outside at every step, and ALM's dual variable, one per horizon step, drives that correction
far more reliably under the same iteration budget.

Dynamic obstacles were deliberately left on the softer penalty method:
their predicted trajectories (constant-velocity extrapolation, see
`_predict` in `src/failure_monitor/episode.py`) are inherently uncertain, so a
hard ALM constraint against a prediction that's often slightly wrong risks
making the solver infeasible or jittery in busy dynamic scenes.

### Other robots: cost only, not a constraint

Fleet ("other robot") avoidance is cost-only (see the table above) -- there
is no ALM/penalty constraint for it at all. The scenarios in this repository
have a single robot, so the fleet terms are inactive (`Nother` slots filled
with zeros).

## Full formulation

The prose sections above walk through the code; this is the same problem as a
single formal statement, solved fresh every control cycle over a horizon of
$N = N_{hor}$ steps ($N{=}20$, $t_s{=}0.2\,\mathrm{s}$ by default). $s_0$ is
fixed to the current measured state, so the decision variables are just the
controls $u_0,\dots,u_{N-1}$:

$$
\min_{u_0,\dots,u_{N-1}} \quad \sum_{k=0}^{N-1} \ell_k(s_{k+1}, u_k) \;+\; \ell_N(s_N)
$$

**Stage cost** (every horizon step $k$; $p_k=(x_k,y_k)$):

$$
\begin{aligned}
\ell_k(s_{k+1}, u_k) = \;
& q_{rpd} \cdot \min_{i} \operatorname{dist}\!\big(p_{k+1},\, \mathrm{seg}_i(\text{path})\big)^2
&&\text{(reference path deviation)} \\
+\; & q_{vel} \cdot (v_k - v_k^{ref})^2
&&\text{(reference velocity deviation)} \\
+\; & q_{\theta} \cdot \big(1 - \cos(\theta_{k+1} - \theta_k^{ref})\big)
&&\text{(reference heading deviation)} \\
+\; & r_v v_k^2 + r_\omega \omega_k^2
&&\text{(control effort; } r_v{=}r_\omega{=}0 \text{ by default)} \\
+\; & 1000 \sum_{j \,\in\, \text{other robots}} \max\!\big(0,\; d_{safe}^2 - \lVert p_{k+1} - p_{k,j}^{\text{oth}} \rVert^2\big)
&&\text{(fleet collision, cost-only)} \\
+\; & \sum_{d \,\in\, \text{dyn. obstacles}} q_{dyn,k} \cdot \max\!\big(0,\, E_d(p_{k+1})\big)^2
&&\text{(dynamic obstacle proximity)} \\
+\; & a_{pen} \cdot \Big(\dfrac{v_k - v_{k-1}}{t_s}\Big)^{\!2} + \alpha_{pen} \cdot \Big(\dfrac{\omega_k - \omega_{k-1}}{t_s}\Big)^{\!2}
&&\text{(acceleration cost, } v_{-1}, \omega_{-1} \text{ = previous applied action)}
\end{aligned}
$$

**Terminal cost** ($s_N = (x_N,y_N,\theta_N)$, the state at the end of the horizon):

$$
\ell_N(s_N) = q_N \cdot \big\lVert p_N - p_{goal} \big\rVert^2 \;+\; q_{\theta N} \cdot (\theta_N - \theta_{goal})^2
$$

$q_{pos}$ (a planned but unused per-step position-tracking weight) does not
appear above -- see the cost table for why. Both terminal weights default to
$0$.

**Dynamics** (unicycle, RK4-integrated over $t_s$; $\mathrm{RK4}$ denotes one
classical 4th-order Runge-Kutta step of the ODE below, not forward-Euler):

$$
s_{k+1} = s_k + \mathrm{RK4}_{t_s}\big(f,\, s_k,\, u_k\big),
\qquad
f(s, u) = \begin{bmatrix} v\cos\theta \\ v\sin\theta \\ \omega \end{bmatrix}
$$

**Box constraints** (hard, exact -- $\forall k = 0,\dots,N-1$):

$$
v_k \in [v_{min},\, v_{max}] = [-0.5,\, 1.5]\ \mathrm{m/s},
\qquad
\omega_k \in [-\omega_{max},\, \omega_{max}] = [-0.5,\, 0.5]\ \mathrm{rad/s}
$$

**Augmented-Lagrangian constraints** (converge to true feasibility; combined
into one mapping since the solver only supports a single ALM constraint set --
see "Static obstacles: hardened via ALM" above):

$$
\frac{v_k - v_{k-1}}{t_s} \in [a_{min}, a_{max}],
\qquad
\frac{\omega_k - \omega_{k-1}}{t_s} \in [-\alpha_{max}, \alpha_{max}],
\qquad k = 0,\dots,N-1
$$

$$
\sum_{i=1}^{N_{stcobs}} \mathrm{inside}\big(p_{k+1};\, O_i\big) \;=\; 0, \qquad k = 0,\dots,N-1
$$

where each static obstacle $O_i$ is a convex polygon in half-space form
(edges $j=1,\dots,m_i$, each $b_{ij} - a_{ij}^{(0)} x - a_{ij}^{(1)} y > 0$
when $(x,y)$ is on the inner side of that edge), and

$$
\mathrm{inside}(p; O) = \prod_{j=1}^{m} \Big(\max\big(0,\; b_j - a_j^{(0)} x - a_j^{(1)} y\big)\Big)^{2}
$$

is $0$ the instant $p=(x,y)$ is outside *any* edge's half-space (i.e. outside
the polygon) and positive only when strictly inside every edge at once --
so constraining the sum to exactly $0$ forces the state fully outside every
static obstacle at every horizon step (each term is individually $\geq 0$, so
a zero sum means every term is zero).

**Penalty-method constraint** (soft; dynamic obstacles only):

$$
\max\big(0,\; E_d(p_{k+1})\big) \;\longrightarrow\; 0 \quad \text{(encouraged, not enforced)}, \qquad \forall d,\ k = 0,\dots,N-1
$$

with the ellipse indicator (center $(c_x,c_y)$, semi-axes $(r_x,r_y)$,
rotation $\phi$; positive inside, negative outside) used both here and in the
dynamic-obstacle stage cost above:

$$
E(p;\, c_x,c_y,r_x,r_y,\phi) = 1 - \frac{\big((x{-}c_x)\cos\phi + (y{-}c_y)\sin\phi\big)^2}{r_x^2} - \frac{\big((x{-}c_x)\sin\phi - (y{-}c_y)\cos\phi\big)^2}{r_y^2}
$$

## Runtime data flow

Everything above is compiled once into a fixed cost/constraint *structure*;
what actually varies call-to-call is the parameter vector `z` that
`TrajectoryGenerator.run_step()` assembles and passes to the solver, without
needing a rebuild:

```
z = [ current_state, goal_state, previous_action,       # s
      qpos, qvel, qtheta, rv, rw, qN, qthetaN, qrpd,     # q  (tuning_params, from TrajectoryGenerator.set_work_mode)
      acc_penalty, w_acc_penalty,
      reference_trajectory (x,y,θ per horizon step),     # r  (from get_local_ref_traj)
      reference_speed per horizon step,
      other_robots' predicted states,                    # c
      static_obstacles (half-space form),                # o_s (from update_static_constraints)
      dynamic_obstacles (ellipse form, per horizon step), # o_d (from update_dynamic_constraints)
      static/dynamic obstacle cost weights ]              # q_stc, q_dyn
```

`set_work_mode(mode)` (`'safe'`/`'work'`/`'super'`/`'aligning'`) selects the
reference *speed* (`base_speed`, a fraction of `lin_vel_max`) and swaps in a
config-defined `tuning_params` vector -- it's how the same compiled solver
serves both a cautious, slow-approach mode and a faster cruising mode
without a rebuild. It's called fresh at the top of every `run_step()`, so
whatever `mode` the caller passes to `get_action()` each cycle is what's
actually used *this* step, regardless of what mode a reference trajectory
happened to be built under (`set_ref_trajectory()`, called once per replan,
also reads `base_speed` at that moment to space the new trajectory's nodes --
see `ReplanRouteStrategy` in `src/failure_monitor/recovery.py` for a case
where that ordering mattered).

## Why the heading cost exists

Without a heading term, the only thing tying the robot's heading to
*anything* along the way is `cost_refvalue_deviation` on the *velocity
magnitude* (not sign) and `qvel`'s reference speed (always positive) -- and
the position-tracking cost (`qrpd`) is a pure distance-to-line-segment
check, entirely direction-agnostic. For a unicycle, "drive backward with a
flipped heading" and "drive forward facing the right way" are then
cost-equivalent ways to track the same `(x, y)` path.

The solver's non-convex, warm-started optimizer does land in the reverse
branch, reliably, right after a `REPLAN_ROUTE` recovery (`src/failure_monitor/recovery.py`): the robot's
heading at the moment it gets unstuck already points roughly away from the
escape route, and once the reverse-tracking solution is established, the
receding horizon just keeps warm-starting from it every step, so nothing
ever forces a re-evaluation: the MPC drives the entire remaining route to
the goal at `lin_vel_min` (-0.5 m/s, the hard reverse-speed bound) instead
of turning to face forward.

**This cost is necessary but not sufficient, and is not a tuning knob.** On
the grid scenario the robot can also end up reverse-tracking a replanned route *while already facing the right
way* — heading aligned to within 0-13 deg, so `qtheta` had no error left to
act on. Raising the weight does not help and is not even monotonic: 20 -> 80
fixed one seed, 20 -> 150 made two others worse, which is what you would
expect when the weight is only shifting which basin a warm-started non-convex
solve happens to land in. That case is handled where it belongs — detected as
its own failure mode (REVERSE_TRACKING) and corrected by turning the robot on
the spot -- see `ReorientStrategy` in `src/failure_monitor/recovery.py`. Retune `qtheta` for
tracking quality if you like, but not as a fix for reverse-tracking.

`cost_heading_deviation` gives the solver a real, per-step reason
to prefer a heading that matches the direction it's actually meant to be
traveling, breaking that symmetry. `qtheta: 20.0` (`config/mpc_default.yaml`)
is a starting point, not a tuned final answer -- picked so its maximum
per-step contribution at a fully-reversed `Δθ=π` (`2·qtheta=40`) is roughly
on par with `qvel`'s maximum contribution for a full-reverse velocity
mismatch (`10·(1.5−(−0.5))²=40`). Since `qtheta` is a runtime parameter (not
baked into the compiled solver), it can be retuned in the config file alone
-- no rebuild needed unless the cost *structure* itself changes again.

## Rebuilding

The compiled solver (`mpc_solver/`, git-ignored) is platform-specific and
not shipped in the repo. Build it once after install, and again any time
`solver_generator.py`'s cost/constraint *structure* changes (not needed for
a pure weight/tuning-parameter change in `config/mpc_default.yaml`, which is
read at runtime):

```bash
python scripts/build_solver.py
```

It takes several minutes -- most of it spent compiling the CasADi-generated C code
(`auto_casadi_grad.c`, `auto_preconditioning_functions.c`) at `-O3`, twice
(once for the direct solver, once for its Python bindings).
