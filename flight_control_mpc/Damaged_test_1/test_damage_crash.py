import numpy as np
import matplotlib.pyplot as plt

from aircraft_model_damaged import AircraftModel
from path_planner_crash import GlidePathPlanner, CrashSitePlanner, polyline_length, max_reachable_ground_range
from guidance_mpc_damage_crash import GuidanceMPC
from plot import plot_report_figures
from animation import animate_results

# --------------------------------------------------------------
# Controller Settings
# --------------------------------------------------------------
MPC_DT     = 1        # MPC time step (s)
MPC_N      = 10        # MPC horizon length (steps)
PLANNER_N  = 100      # Number of waypoints in the global planner polyline

# --------------------------------------------------------------
# Damage / Crash-landing Settings
# --------------------------------------------------------------
DAMAGE_ACTIVE = True
DAMAGE_GAMMA_MAX_DEG = -3.0   # gamma upper bound when damaged (deg); must be < 0 to force descent

# If the planned runway path length exceeds this reachable range under DAMAGE_GAMMA_MAX_DEG,
# switch to crash MPC and crash-site planner.
REACHABILITY_SAFETY_FACTOR = 0.9

# No-land zones (2D ellipses in N/E meters)
NO_LAND_ELLIPSES = [
    {"cx": -500.0, "cy": -800.0, "a": 400.0, "b": 250.0, "angle_deg": 20.0},
    {"cx": -1200.0, "cy": -1500.0, "a": 500.0, "b": 300.0, "angle_deg": -35.0},
]

# --------------------------------------------------------------
# Environment and Airplane Initial Conditions
# --------------------------------------------------------------
RUNWAY_HEADING_DEG          = 90                            # runway heading (deg)
AIRPLANE_START_POS          = (-2000.0, -4000.0, 500.0)   # (N, E, h) in meters
AIRPLANE_START_VEL_KT       = 80.0                          # initial speed (kt)
AIRPLANE_START_HEADING_DEG  = 170                           # initial heading (deg)

# --------------------------------------------------------------
# End of settings
# --------------------------------------------------------------

# --------------------------------------------------------------
# Instantiate model + planner + MPC
# --------------------------------------------------------------
aircraft = AircraftModel(
    pos_north=AIRPLANE_START_POS[0],
    pos_east=AIRPLANE_START_POS[1],
    altitude=AIRPLANE_START_POS[2],
    vel_kt=AIRPLANE_START_VEL_KT,
    heading_deg=AIRPLANE_START_HEADING_DEG,
    climb_angle_deg=0.0,
    dt=MPC_DT,
)

# Activate damage (gamma upper bound) if requested
if DAMAGE_ACTIVE:
    aircraft.set_damage(True, gamma_max_damage_deg=DAMAGE_GAMMA_MAX_DEG)

gamma_max_damage_rad = np.deg2rad(DAMAGE_GAMMA_MAX_DEG)
aircraft.gamma = min(aircraft.gamma, gamma_max_damage_rad) # enforce initial climb angle within damage limits
planner = GlidePathPlanner(RUNWAY_HEADING_DEG, PLANNER_N)
guidance_mpc = GuidanceMPC(
    planner,
    aircraft,
    MPC_N,
    MPC_DT,
    gamma_min_rad=np.deg2rad(-30.0),
    gamma_max_rad=(gamma_max_damage_rad if DAMAGE_ACTIVE else np.deg2rad(30.0)),
    terminal_vz_weight=0.0,
)

# Crash planner + crash MPC (constructed lazily when needed)
crash_planner = CrashSitePlanner(N=PLANNER_N, no_land_ellipses=NO_LAND_ELLIPSES)
crash_mpc = None
crash_mode = False
crash_trigger_step = None


# --------------------------------------------------------------
# Logging buffers (for plotting / analysis)
# --------------------------------------------------------------
# Planned path (polyline) at each sim step (store full arrays per step)
x_planned, y_planned, h_planned = [], [], []

# MPC predicted trajectory at each sim step (store horizon arrays per step)
x_mpc, y_mpc, h_mpc = [], [], []

# Applied control inputs (first input of MPC solution)
u_thrust_mpc_m_s2       = []  # accel command (m/s^2)
u_heading_rate_mpc_deg_s = [] # chi_dot (deg/s)
u_climb_rate_mpc_deg_s   = [] # gamma_dot (deg/s)

# Closed-loop simulated state history (absolute)
x_sim_m, y_sim_m, h_sim_m = [], [], []
vel_sim_ms = []
heading_sim_deg = []
climb_angle_sim_deg = []

# Seed initial state
x_sim_m.append(aircraft.pos_north)
y_sim_m.append(aircraft.pos_east)
h_sim_m.append(aircraft.altitude)
vel_sim_ms.append(aircraft.vel_ms)
heading_sim_deg.append(np.rad2deg(aircraft.chi))
climb_angle_sim_deg.append(np.rad2deg(aircraft.gamma))

# --------------------------------------------------------------
# Simulation loop
# --------------------------------------------------------------
sim_step = 0

# Event-triggered replanning flag
replan = True
cached_waypoints = None

replan_flags = []          # one bool per MPC step
replan_times = []          # timestamps (seconds)
replan_counts = 0
t_step = 0                # 1 step = 1 second if mpc_dt=1

# Initialize logs
x_sim_m = [float(aircraft.pos_north)]
y_sim_m = [float(aircraft.pos_east)]
h_sim_m = [float(aircraft.altitude)]
vel_sim_ms = [float(aircraft.vel_ms)]
heading_sim_deg = [float(np.rad2deg(aircraft.chi))]
climb_angle_sim_deg = [float(np.rad2deg(aircraft.gamma))]

u_thrust_mpc_m_s2 = []
u_heading_rate_mpc_deg_s = []
u_climb_rate_mpc_deg_s = []

x_planned, y_planned, h_planned = [], [], []
x_mpc, y_mpc, h_mpc = [], [], []

# Run until we reach ground level
while h_sim_m[-1] > 0.1:


    # 1) Build / refresh the global waypoint polyline (replan only when requested)
    if replan or cached_waypoints is None:
        if crash_mode:
            cached_waypoints = crash_planner.solve_for_waypoints(
                aircraft,
                gamma_max_rad=gamma_max_damage_rad,
                verbose=False,
            )
        else:
            cached_waypoints = planner.solve_for_waypoints(aircraft, verbose=False)

        # Reset replan memory for whichever controller is active
        (crash_mpc if crash_mode else guidance_mpc).reset_replan_memory()
        replan_counts += 1
        replan = False

    # 1b) Build local MPC reference along the (possibly cached) polyline
    Xref_abs, _ = (crash_mpc if crash_mode else guidance_mpc).build_local_reference(
        aircraft, waypoints=cached_waypoints
    )
    # 2) Reachability check (only when damaged and not already in crash mode)
    #    - Estimate planned ground-track distance to runway from the current cached waypoints
    #    - Estimate conservative maximum reachable ground range under gamma_max_damage_rad
    if DAMAGE_ACTIVE and (not crash_mode):
        planned_len_m = polyline_length(cached_waypoints, use_altitude=False)
        reachable_m = max_reachable_ground_range(aircraft.altitude, gamma_max_damage_rad, safety_factor=0.90)

        if planned_len_m > reachable_m:
            crash_mode = True
            crash_trigger_step = sim_step
            print(
                f"[CRASH MPC] ⚠️  Triggered at step {sim_step}: planned_len={planned_len_m:.1f} m > reachable={reachable_m:.1f} m "
                f"(h={aircraft.altitude:.1f} m, gamma_max={DAMAGE_GAMMA_MAX_DEG:.2f} deg)"
            )

            # Re-plan to a crash touchdown point (outside forbidden ellipses)
            cached_waypoints = crash_planner.solve_for_waypoints(
                aircraft,
                gamma_max_rad=gamma_max_damage_rad,
                verbose=True,
            )

            # Build a dedicated crash MPC instance with a terminal vertical-speed objective
            crash_mpc = GuidanceMPC(
                crash_planner,  # any planner-like object with solve_for_waypoints()
                aircraft,
                MPC_N,
                MPC_DT,
                gamma_min_rad=np.deg2rad(-30.0),
                gamma_max_rad=gamma_max_damage_rad,
                terminal_vz_weight=200.0,  # tune
            )
            crash_mpc.reset_replan_memory()
            replan = False

        try:
            u0, X_pred, U_pred, _, replan = (crash_mpc if crash_mode else guidance_mpc).solve_for_control_input(
                aircraft, cached_waypoints, Xref_abs
            )
        except RuntimeError:
            print("MPC failed")
            break

    # 3) Log replan flag (this step)
    replan_flags.append(bool(replan))
    replan_times.append(t_step)
    t_step += 1

    # 4) Log applied control inputs
    u_thrust_mpc_m_s2.append(float(u0[0]))
    u_heading_rate_mpc_deg_s.append(float(np.rad2deg(u0[1])))
    u_climb_rate_mpc_deg_s.append(float(np.rad2deg(u0[2])))

    # 5) Log the current global plan (for visualization)
    x_planned.append(np.asarray(cached_waypoints[:, 0], dtype=float).copy())
    y_planned.append(np.asarray(cached_waypoints[:, 1], dtype=float).copy())
    h_planned.append(np.asarray(cached_waypoints[:, 2], dtype=float).copy())

    # 6) Log MPC predicted horizon trajectory (for visualization)
    x_mpc.append(np.asarray(X_pred[0, :], dtype=float).copy())  # north
    y_mpc.append(np.asarray(X_pred[1, :], dtype=float).copy())  # east
    h_mpc.append(np.asarray(X_pred[2, :], dtype=float).copy())  # altitude

    # 7) Advance TRUE simulation by one step
    aircraft.step(u0)
    sim_step += 1

    # 8) Log TRUE simulated state from aircraft (NOT from X_pred)
    x_sim_m.append(float(aircraft.pos_north))
    y_sim_m.append(float(aircraft.pos_east))
    h_sim_m.append(float(aircraft.altitude))
    vel_sim_ms.append(float(aircraft.vel_ms))
    heading_sim_deg.append(float(np.rad2deg(aircraft.chi)))
    climb_angle_sim_deg.append(float(np.rad2deg(aircraft.gamma)))


# --------------------------------------------------------------
# Plot results
# --------------------------------------------------------------
plot_report_figures(
    x_sim_m, y_sim_m, h_sim_m,
    vel_sim_ms, heading_sim_deg, climb_angle_sim_deg,
    u_thrust_mpc_m_s2, u_heading_rate_mpc_deg_s, u_climb_rate_mpc_deg_s,
    x_planned, y_planned, h_planned,
    runway_heading_deg=RUNWAY_HEADING_DEG,
    out_dir="report_figs",
    glide_angle_deg=3.0,
    runway_length_m=2000.0,
    input_limits=None,
    mpc_dt = MPC_DT,
    replan_flags=replan_flags,
    replan_times=replan_times,   # optional; you can omit this
)

# --------------------------------------------------------------
# Play Simulation Results
# --------------------------------------------------------------
plt.ion()
replay_step = 0

while True:

    animate_results(
        replay_step,
        x_planned[replay_step],
        y_planned[replay_step],
        h_planned[replay_step],
        RUNWAY_HEADING_DEG,
        x_mpc=x_mpc[replay_step],
        y_mpc=y_mpc[replay_step],
        h_mpc=h_mpc[replay_step],
        u_thrust=u_thrust_mpc_m_s2[replay_step],
        u_heading=u_heading_rate_mpc_deg_s[replay_step],
        u_climb_rate=u_climb_rate_mpc_deg_s[replay_step],
        vel_ms=vel_sim_ms[replay_step],
        heading_deg=heading_sim_deg[replay_step],
        climb_angle_deg=climb_angle_sim_deg[replay_step],
    )

    if replay_step == 0:
        plt.pause(3.0)

    replay_step = replay_step + 1
    if replay_step >= len(x_planned):
        plt.pause(3.0)
        replay_step = 0

    plt.pause(0.1)