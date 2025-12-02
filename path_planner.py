import numpy as np
import casadi as ca
import matplotlib.pyplot as plt

WEIGHT_SMOOTH = 1.0
WEIGHT_HEIGHT_SMOOTH = 10.0
WEIGHT_LENGTH = 1.0

W_GLIDE = 10.0  # tune this


class GlidePathPlanner:
    def __init__(self, N, runway_pos, runway_heading_deg):
        self.N = N
        self.runway_pos = runway_pos
        self.runway_heading_rad = np.deg2rad(runway_heading_deg)

    def _build_glide_path_qp(self, start_pos):
        
        # extract current position
        x0, y0, h0 = start_pos
        x_end, y_end, h_end = self.runway_pos
        
        # decision variable dimension
        N = self.N
        n_waypoint = N + 1
        n_var = 3 * n_waypoint

        
        # index helpers rom decision vector
        def idx_x(i): return 3 * i
        def idx_y(i): return 3 * i + 1
        def idx_h(i): return 3 * i + 2

        opti = ca.Opti()

        # decision vector
        z = opti.variable(n_var)

        # --------------------------------------------------------------
        # Objective: smoothness + length
        # --------------------------------------------------------------
        J = 0

        # Smoothness term: sum_i ||p_{i+1} - 2 p_i + p_{i-1}||^2
        for i in range(1, N):
            # second difference for x,y,h
            for d in range(3):
                if d == 2:
                    w = WEIGHT_HEIGHT_SMOOTH
                else:
                    w = WEIGHT_SMOOTH
                if w == 0.0:
                    continue

                id0 = 3 * (i - 1) + d
                id1 = 3 * i + d
                id2 = 3 * (i + 1) + d

                second_diff = z[id2] - 2 * z[id1] + z[id0]
                J += w * second_diff**2

        # Length-like term: sum_i ||p_{i+1} - p_i||^2
        if WEIGHT_LENGTH > 0.0:
            for i in range(N):
                for d in range(3):
                    id0 = 3 * i + d
                    id1 = 3 * (i + 1) + d
                    diff = z[id1] - z[id0]
                    J += WEIGHT_LENGTH * diff**2

        # ----------------- Glideslope + centerline tracking -----------------
        W_GS = 5.0    # weight for glideslope altitude error
        W_LAT = 1.0   # weight for lateral (cross-track) error

        dx = np.cos(self.runway_heading_rad)
        dy = np.sin(self.runway_heading_rad)
        x_end, y_end, h_end = self.runway_pos

        gamma = np.deg2rad(2.0)  # 2 degree glideslope
        tan_gamma = np.tan(gamma)

        for i in range(self.N + 1):
            xi = z[idx_x(i)]
            yi = z[idx_y(i)]
            hi = z[idx_h(i)]

            # Along-track distance from runway threshold, positive outwards
            s_i = (xi - x_end) * dx + (yi - y_end) * dy

            # Cross-track error (lateral distance to runway centerline)
            c_i = (xi - x_end) * (-dy) + (yi - y_end) * dx

            # Reference altitude on a 2° glideslope from runway
            h_ref = h_end + s_i * tan_gamma

            w_scale = (i / self.N)**2  # small far out, big near runway
            J += w_scale * W_GS * (hi - h_ref)**2
            J += w_scale * W_LAT * c_i**2


        opti.minimize(J)

        # --------------------------------------------------------------
        # Constraints
        # --------------------------------------------------------------

        # Start fixed
        opti.subject_to(z[idx_x(0)] == x0)
        opti.subject_to(z[idx_y(0)] == y0)
        opti.subject_to(z[idx_h(0)] == h0)

        # End fixed (runway position)
        opti.subject_to(z[idx_x(N)] == x_end)
        opti.subject_to(z[idx_y(N)] == y_end)
        opti.subject_to(z[idx_h(N)] == h_end)

        # --------------------------------------------------------------
        # Altitude monotonicity: altitude must not increase
        # --------------------------------------------------------------
        for i in range(N):
            h_i   = z[idx_h(i)]
            h_ip1 = z[idx_h(i + 1)]
            opti.subject_to(h_ip1 <= h_i)

        # Runway direction in ground plane
        dx = np.cos(self.runway_heading_rad)
        dy = np.sin(self.runway_heading_rad)

        # --------------------------------------------------------------
        # Final-alignment region: last D_align meters nearly straight-in
        # --------------------------------------------------------------
        D_align = 2000.0  # [m] length of desired straight-in segment

        # Horizontal distance from start to runway
        D0 = np.hypot(x0 - x_end, y0 - y_end)

        if D0 > 1e-3:
            # fraction of the path where we START alignment
            # e.g. if D0 = 3.2 km and D_align = 2.0 km,
            # we start aligning when ~1.2 km out.
            s_align_start = max(0.0, (D0 - D_align) / D0)

            # convert to waypoint index
            i_align_start = int(np.floor(s_align_start * N))

            for i in range(i_align_start, N):
                x_i   = z[idx_x(i)]
                x_ip1 = z[idx_x(i + 1)]
                y_i   = z[idx_y(i)]
                y_ip1 = z[idx_y(i + 1)]

                seg_dx = x_ip1 - x_i
                seg_dy = y_ip1 - y_i

                # parallel to runway: cross = 0
                opti.subject_to(seg_dx * dy - seg_dy * dx == 0)

                # forward along runway: dot >= 0
                opti.subject_to(seg_dx * dx + seg_dy * dy >= 0)


        # --------------------------------------------------------------
        # Solver setup
        # --------------------------------------------------------------
        opti.solver('ipopt', {
            "print_time": True,
            "ipopt.print_level": 0,
        })

        self.opti = opti
        self.z = z
        self.n_waypoint = n_waypoint

        return opti, z, n_waypoint

    def solve_QP(self, start_pos):
        """
        Convenience method: build + solve, returns waypoints as (N+1, 3) ndarray.
        """
        opti, z, n_wp = self._build_glide_path_qp(start_pos)
        sol = opti.solve()
        z_opt = np.array(sol.value(z)).flatten()
        waypoints = z_opt.reshape(n_wp, 3)

        return waypoints
    


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from glide_plots import (
        plot_3d_path,
        plot_ground_track_with_runway,
        plot_altitude_vs_runway_distance,
    )

    N = 50

    # Runway at origin, heading along +x (change if needed)
    runway_pos = (0.0, 0.0, 0.0)
    runway_heading_deg = 0.0   # 0 deg = +x

    planner = GlidePathPlanner(N, runway_pos, runway_heading_deg)

    # Example start position
    start_pos = (3000.0, -5000.0, 500.0)

    waypoints = planner.solve_QP(start_pos)

    print("Waypoints (x, y, h):")
    print(waypoints)

    x = waypoints[:, 0]
    y = waypoints[:, 1]
    h = waypoints[:, 2]

    # 3D path
    plot_3d_path(x, y, h, runway_pos)

    # 2D ground track with runway rectangle + centerline
    plot_ground_track_with_runway(
        x,
        y,
        runway_pos,
        runway_heading_deg,
    )

    # Altitude vs runway distance
    plot_altitude_vs_runway_distance(
        x,
        y,
        h,
        runway_pos,
        runway_heading_deg,
    )

    plt.show()
