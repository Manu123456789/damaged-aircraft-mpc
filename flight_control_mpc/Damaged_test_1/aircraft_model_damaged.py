import numpy as np
from numpy import sin, cos

# --------------------------------------------------------------
# Aircraft Performance Constants
# --------------------------------------------------------------
BEST_GLIDE_SPEED_KT = 65          # best glide speed (kt)
APPROACH_SPEED_KT = 80            # approach speed (kt)
STALL_SPEED_KT = 50.0             # stall speed (kt)
NEVER_EXCEED_SPEED_KT = 150.0     # never exceed speed (kt)


class AircraftModel:
    def __init__(self, pos_north, pos_east, altitude, vel_kt, heading_deg, climb_angle_deg, dt = 0.05):
        # Initial Configuration
        self.pos_north = pos_north                  # North Position (m)
        self.pos_east = pos_east                    # East Position (m)
        self.altitude = altitude                    # Altitude (m)
        self.vel_ms = vel_kt * 0.514444             # Airspeed (m/s) converted from knots
        self.chi = np.deg2rad(heading_deg)          # Heading angle(rad)
        self.gamma = np.deg2rad(climb_angle_deg)    # Flight path (climb) angle (rad)

        self.glide_speed_ms = BEST_GLIDE_SPEED_KT * 0.514444               # Best glide speed (m/s) converted from knots
        self.stall_speed_ms = STALL_SPEED_KT * 0.514444                    # Stall speed (m/s) converted from knots
        self.never_exceed_speed_ms = NEVER_EXCEED_SPEED_KT * 0.514444      # Never exceed speed (m/s) converted from knots
        self.approach_speed_ms = APPROACH_SPEED_KT * 0.514444              # Approach speed (m/s) converted from knots
        self.dt = dt

        # --------------------------------------------------------------
        # Damage model (simple pitch / flight-path angle authority loss)
        # --------------------------------------------------------------
        # If damage is active, enforce an upper bound on gamma (flight-path angle).
        # Example: gamma_max_damage_deg = -3.0 means the aircraft cannot fly level/climb;
        # it must maintain gamma <= -3 deg (i.e., continuous descent).
        self.damage_active = False
        self.gamma_max_damage_rad = None  # when active, a scalar (rad)
        self._update_linearized_kinematics()

    def get_state_vector(self):
        return np.array([
            self.pos_north,
            self.pos_east,
            self.altitude,
            self.vel_ms,
            self.chi,
            self.gamma,
        ])
    
    def update_from_vector(self, x):
        self.pos_north, self.pos_east, self.altitude, \
        self.vel_ms, self.chi, self.gamma = x

        self._update_linearized_kinematics()

    def set_damage(self, active: bool, gamma_max_damage_deg: float = -3.0):
        """Enable/disable damage.

        Parameters
        ----------
        active:
            If True, apply a hard upper bound on gamma each integration step.
        gamma_max_damage_deg:
            Upper bound for gamma (deg). Should be <= 0 for 'cannot sustain flight'.
        """
        self.damage_active = bool(active)
        if self.damage_active:
            self.gamma_max_damage_rad = np.deg2rad(float(gamma_max_damage_deg))
        else:
            self.gamma_max_damage_rad = None

    def gamma_upper_bound(self):
        """Return the current gamma upper bound (rad), accounting for damage."""
        if self.damage_active and (self.gamma_max_damage_rad is not None):
            return float(self.gamma_max_damage_rad)
        return None

    def _update_linearized_kinematics(self):

        """ 
        Update the dynamics matricies using linearized kinematics model 
        """
        V = self.vel_ms
        gamma = self.gamma
        chi = self.chi
        dt = self.dt

        cgamma = cos(gamma)
        sgamma = sin(gamma)
        cchi = cos(chi)
        schi = sin(chi)

        # Continuous-time state-space matrices linearized about current state
        Ac = np.array([
            [0.0, 0.0, 0.0,  cgamma * cchi,  -V * cgamma * schi,  -V * sgamma * cchi],
            [0.0, 0.0, 0.0,  cgamma * schi,   V * cgamma * cchi,  -V * sgamma * schi],
            [0.0, 0.0, 0.0,  sgamma,          0.0,                 V * cgamma],
            [0.0, 0.0, 0.0,  0.0,             0.0,                 0.0],
            [0.0, 0.0, 0.0,  0.0,             0.0,                 0.0],
            [0.0, 0.0, 0.0,  0.0,             0.0,                 0.0],
        ])

        Bc = np.array([
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ])

        # Discretize using Euler forward method
        self.Ad = np.eye(6) + Ac * dt
        self.Bd = Bc * dt

    def step(self, u):
        u_accel, u_chidot, u_gammadot = u
        dt = self.dt

        # nonlinear kinematics step
        self.pos_north += self.vel_ms * np.cos(self.gamma) * np.cos(self.chi) * dt
        self.pos_east  += self.vel_ms * np.cos(self.gamma) * np.sin(self.chi) * dt
        self.altitude  += self.vel_ms * np.sin(self.gamma) * dt

        self.vel_ms += u_accel * dt
        self.chi    += u_chidot * dt
        self.gamma  += u_gammadot * dt

        # Damage enforcement: cap gamma from above
        if self.damage_active and (self.gamma_max_damage_rad is not None):
            self.gamma = min(self.gamma, self.gamma_max_damage_rad)
        self._update_linearized_kinematics()