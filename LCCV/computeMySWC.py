import numpy as np
from computeSWC import *


def computeMySWC(A, cl, ncl, cores, pdist, local_core):
    ncores = len(cores) - 1
    n, dim = A.shape
    D = np.zeros((ncores + 1, dim))
    cl_cores = np.zeros(ncores + 1)

    for i in range(1, ncores + 1):
        D[i, :] = A[cores[i], :]
        cl_cores[i] = cl[cores[i], 1]

    nl = np.zeros(ncores + 1)

    # Count the number of points belonging to each core
    for i in range(1, ncores + 1):
        for j in range(1, n):
            if any(local_core[j] == cores[i]) and cl[j, 1] > 0:
                nl[i] += 1

    _, s = computeSWC(D, cl_cores, ncl, pdist)
    mcv = 0
    for i in range(1, ncores + 1):
        mcv += s[i] * nl[i]

    mcv = mcv / (n - 1)
    return mcv
