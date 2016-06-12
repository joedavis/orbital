import numpy as np
cimport numpy as np

import cython

cdef extern from "math.h":
    double sqrt(double)

#from scipy.constants import G

# Planck Units

DTYPE = np.double
ctypedef double DTYPE_t

cdef inline double norm3(double[] v):
    return sqrt(v[0]**2 + v[1]**2 + v[2]**2)

@cython.cdivision(True)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.boundscheck(False)
def ode_system(np.ndarray[DTYPE_t, ndim=1] coords,
               double t,
               unsigned int n,
               np.ndarray[DTYPE_t, ndim=1] masses):
    """The system of ordinary differential equations to solve."""
    cdef unsigned int i = 0
    cdef unsigned int j = 0
    cdef unsigned int k = 0
    cdef double m = 0.0
    cdef double[3] dr = [0,0,0]
    cdef double dr_norm = 0
    cdef np.ndarray[DTYPE_t,ndim=1] dydx = np.zeros((n * 6,), dtype=DTYPE)

    for i in range(n):
        for j in range(3):
            dydx[i*3+j] = coords[(n+i)*3+j] / masses[i]

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            m = -1.0 * masses[i] * masses[j]
            for k in range(3):
                dr[k] = coords[i*3+k] - coords[j*3+k]
            dr_norm = m / (norm3(dr) ** 3)
            
            for k in range(3):
                dydx[(n+i)*3+k] += dr[k] * dr_norm

    return dydx

