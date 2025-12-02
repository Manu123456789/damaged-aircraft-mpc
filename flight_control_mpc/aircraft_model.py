import numpy as np
from numpy import sin, cos

class AircraftModel:
    def __init__(self):

        # Initial Configuration
        self.pos_north = 0.0        # North Position (m)
        self.pos_east = 0.0         # East Position (m)
        self.altitude = 5000.0      # Altitude (m)
        self.vel = 300              # Airspeed (m/s)
        self.chi = 0.0              # Heading angle(rad)
        self.gamma = 0.0            # Flight path (climb) angle (rad)

    def update_linearized_kinematics(self, dt):
        """ 
        Update the dynamics matricies using linearized kinematics model 
        """
        V = self.vel
        gamma = self.gamma
        chi = self.chi

        cgamma = cos(gamma)
        sgamma = sin(gamma)
        cchi = cos(chi)
        schi = sin(chi)

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

        self.Ad = np.eye(6) + Ac * dt
        self.Bd = Bc * dt