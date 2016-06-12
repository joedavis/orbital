import numpy as np
from scipy.integrate import odeint

from orbital.ode import ode_system


class NBodySystem:
    """
    A system of N particles of different masses in N dimensional space.
    """

    def __init__(self, masses=None, positions=None, velocities=None):
        assert len(masses) == len(positions) == len(velocities)

        self.body_count = len(masses)
        self.masses = masses
        # Store the coordinates in a packed form, ready to be integrated
        # Needs dtype=np.double
        self._coords = np.reshape(
            np.array([positions, velocities * masses[:, np.newaxis]]),
            (6*self.body_count,)
        )

    @property
    def velocities(self):
        return self.momenta / self.masses

    @property
    def momenta(self):
        return np.reshape(self._coords[self.body_count*3:],
                          (self.body_count, 3))

    @property
    def positions(self):
        return np.reshape(self._coords[0:self.body_count*3],
                          (self.body_count, 3))

    def step(self, t):
        """Step the system t seconds into the future"""
        N = 2
        self._coords = self._integrate(np.linspace(0, t, N))[N-1]

    def _integrate(self, t):
        """Find the state of the t seconds in the future."""
        return odeint(ode_system, self._coords, t,
                      (self.body_count, self.masses,))


def random_system(n, xmax, ymax, zmax):
    """Generate a random system of particles."""

    xs = 2 * xmax * (0.5 - np.random.rand(n))
    ys = 2 * ymax * (0.5 - np.random.rand(n))
    zs = 2 * zmax * (0.5 - np.random.rand(n))
    positions = np.transpose([xs, ys, zs])

    xs = 4 * (0.5 - np.random.rand(n))
    ys = 4 * (0.5 - np.random.rand(n))
    zs = 4 * (0.5 - np.random.rand(n))
    velocities = np.transpose([xs, ys, zs])

    masses = np.random.rand(n) * 4
    return NBodySystem(masses, positions, velocities)
