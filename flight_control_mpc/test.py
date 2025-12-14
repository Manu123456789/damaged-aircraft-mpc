import numpy as np
import matplotlib.pyplot as plt

from aircraft_model import AircraftModel
from path_planner import GlidePathPlanner
from guidance_mpc import GuidanceMPC
from plot import plot_report_figures
from animation import animate_results

# --------------------------------------------------------------
# Controller Settings
# --------------------------------------------------------------
MPC_DT     = 1        # MPC time step (s)
MPC_N      = 5        # MPC horizon length (steps)
PLANNER_N  = 100      # Number of waypoints in the global planner polyline

# --------------------------------------------------------------
# Environment and Airplane Initial Conditions
# --------------------------------------------------------------
RUNWAY_HEADING_DEG          = 90                            # runway heading (deg)
AIRPLANE_START_POS          = (200.0, -3000.0, 150.0)   # (N, E, h) in meters
AIRPLANE_START_VEL_KT       = 80.0                          # initial speed (kt)
AIRPLANE_START_HEADING_DEG  = 110.0                           # initial heading (deg)

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

planner = GlidePathPlanner(RUNWAY_HEADING_DEG, PLANNER_N)
guidance_mpc = GuidanceMPC(planner, aircraft, MPC_N, MPC_DT)

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

# Run until we reach (approximately) ground level
while h_sim_m[-1] > 0.1:

    # 1) Build flight path reference for MPC (if needed)
    if replan or cached_waypoints is None:
        Xref_abs, cached_waypoints = guidance_mpc.build_local_reference(
            aircraft, waypoints=None
        )
        replan = False
        guidance_mpc.reset_replan_memory()
    else:
        Xref_abs, _ = guidance_mpc.build_local_reference(
            aircraft, waypoints=cached_waypoints
        )

    # 2) Solve MPC for control input
    try:
        state_vector = aircraft.get_state_vector()
        u0, X_pred, U_pred, _, replan = guidance_mpc.solve_for_control_input(aircraft, cached_waypoints, Xref_abs)

    except RuntimeError as e:
        print("MPC failed")
        break

    # 3) Log applied control inputs
    u_thrust_mpc_m_s2.append(u0[0])
    u_heading_rate_mpc_deg_s.append(np.rad2deg(u0[1]))
    u_climb_rate_mpc_deg_s.append(np.rad2deg(u0[2]))

    # 4) Log the current global plan (for visualization)
    x_planned.append(cached_waypoints[:, 0])
    y_planned.append(cached_waypoints[:, 1])
    h_planned.append(cached_waypoints[:, 2])

    # 5) Log MPC predicted horizon trajectory (for visualization)
    x_mpc.append(X_pred[0, :])  # north
    y_mpc.append(X_pred[1, :])  # east
    h_mpc.append(X_pred[2, :])  # altitude

    # 6) Advance simulation by one step
    x_next = X_pred[:, 1]  # next state in the predicted trajectory

    x_sim_m.append(x_next[0])
    y_sim_m.append(x_next[1])
    h_sim_m.append(x_next[2])
    vel_sim_ms.append(x_next[3])
    heading_sim_deg.append(np.rad2deg(x_next[4]))
    climb_angle_sim_deg.append(np.rad2deg(x_next[5]))


    # Apply the control input and update aircraft state
    aircraft.step(u0)
    sim_step += 1


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
    mpc_dt = MPC_DT
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