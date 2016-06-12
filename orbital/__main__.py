from orbital.system import random_system

import matplotlib.pyplot as plt
import matplotlib.animation as anim
from mpl_toolkits.mplot3d import Axes3D
del Axes3D  # hide warnings

T = 0.1  # timestep
n = 50
w = 10
system = random_system(n, w, w, w)

fig = plt.figure()
ax = fig.add_subplot('111', projection='3d')

ax.set_xlim3d([-w, w])
ax.set_ylim3d([-w, w])
ax.set_zlim3d([-w, w])

particles = ax.plot([], [], 'b.', zs=[])


def update(t, particles):
    pos = system.positions
    system.step(T)
    particles.set_data(pos[:, 0], pos[:, 1])
    particles.set_3d_properties(pos[:, 2])
    return particles

system_animation = anim.FuncAnimation(fig, update, 60000, fargs=particles,
                                      interval=30)

plt.show()
