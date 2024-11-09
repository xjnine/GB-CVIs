import numpy as np


def computeSWC(D, cl, ncl, dist):
    n, d = D.shape
    n -= 1
    d -= 1
    cdata = [None] * (ncl + 1)  # The number of points in each cluster
    cindex = np.zeros((ncl + 1, n + 1), dtype=int)
    numc = ncl

    for i in range(1, ncl + 1):
        nump = 0
        for j in range(1, n + 1):
            if cl[j] == i:
                nump += 1
                if cdata[i] is None:
                    cdata[i] = [[0, 0]]
                cdata[i].append(D[j, :])
                cindex[i, nump] = j
    numo = 0
    # Don't compute the swc of outliers
    if min(cl[1:]) <= 0:
        for i in range(1, n + 1):
            if cl[i] <= 0:
                numo += 1
    swc = 0
    s1 = np.zeros(n + 1)
    for i in range(1, numc + 1):
        a = []
        b = []
        s = []
        np_ = len(cdata[i]) - 1
        if np_ > 1:
            for j in range(1, np_ + 1):
                # compute a[j]
                suma = 0
                for k in range(1, np_ + 1):
                    if j != k:
                        suma += dist[cindex[i, j], cindex[i, k]]
                if a.__len__() == 0:
                    a.append(0)
                a.append(suma / (np_ - 1))

                # compute b[j]
                d = [float('inf')] * (numc + 1)
                for k in range(1, numc + 1):
                    if k != i:
                        np2 = len(cdata[k]) - 1
                        sumd = 0
                        for l in range(1, np2 + 1):
                            sumd += dist[cindex[i, j], cindex[k, l]]
                        d[k] = sumd / np2
                if b.__len__() == 0:
                    b.append(0)
                b.append(min(d))

                # compute s[j]
                if s.__len__() == 0:
                    s.append(0)
                s.append((b[j] - a[j]) / max(a[j], b[j]))
                s1[cindex[i, j]] = s[j]
                swc += s[j]
    swc = swc / (n - numo)
    return swc, s1
