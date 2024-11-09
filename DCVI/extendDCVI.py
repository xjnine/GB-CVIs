from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.neighbors import KDTree
from scipy.spatial.distance import pdist, squareform
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from sklearn.metrics import pairwise_distances


def showCPV(D, CPV, LPS, CL):
    plt.scatter(D[:, 0], D[:, 1], c=CL)
    plt.scatter(D[LPS, 0], D[LPS, 1], c='r')
    for i in range(D.shape[0]):
        if CL[i] != -1:
            plt.plot([D[i, 0], D[CPV[i], 0]], [D[i, 1], D[CPV[i], 1]], 'k-')
    plt.show()


def extendDVCI(D):
    CL, LPS, CP, CPV = findCore(D)  # 寻找密度核
    showCPV(D, CPV, LPS, CL)
    D1 = D[LPS, :]

    k = D1.shape[0]
    kk = int(np.ceil(np.sqrt(D1.shape[0])))
    avgdcvi = np.zeros(kk - 1)
    CL = np.zeros((kk - 1, k))
    Distance = pdist(D1)
    Z = linkage(Distance, 'single')

    for ii in range(2, kk + 1):
        cl = fcluster(Z, ii, criterion='maxclust')
        cd = np.zeros(ii)
        sd = np.zeros(ii)
        dcvi = np.zeros(ii)

        for mm in range(1, ii + 1):
            DD = D1[cl == mm, :]
            DDD = D1[cl != mm, :]

            tree = KDTree(DD)
            dist, _ = tree.query(DDD, k=1)
            cd[mm - 1] = np.max(dist)

            temp = pdist(np.vstack((DD, DDD)))
            sd[mm - 1] = np.min(temp)

        for nn in range(ii):
            if cd[nn] == 0:
                cd[nn] = np.min(cd[np.nonzero(cd)])

        for nn in range(ii):
            dcvi[nn] = cd[nn] / sd[nn]

        avgdcvi[ii - 2] = np.mean(dcvi)
        CL[ii - 2, :] = cl

    d = np.argmin(avgdcvi)
    on = d + 2
    oncvi = avgdcvi[d]
    bins = CL[d, :]

    return D1, on, oncvi, avgdcvi, bins


def NaNSearching(D):
    r = 1
    nb = np.zeros(D.shape[0])
    C = []
    NN = []
    RNN = []
    NNN = []
    A = squareform(pdist(D))
    Numb1 = 0
    Numb2 = 0

    for ii in range(D.shape[0]):
        sorted_indices = np.argsort(A[:, ii])
        C.append([A[sorted_indices, ii], sorted_indices])
        NN.append([])
        RNN.append([])
        NNN.append([])

    while r < D.shape[0]:
        for kk in range(D.shape[0]):
            x = kk
            y = C[x][1][r]
            nb[y] += 1
            NN[x].append(y)
            RNN[y].append(x)

        Numb1 = np.sum(nb == 0)
        if Numb2 != Numb1:
            Numb2 = Numb1
        else:
            break
        r += 1

    for jj in range(D.shape[0]):
        NNN[jj] = np.intersect1d(NN[jj], RNN[jj])

    Sup = r
    return Sup, NN, RNN, NNN, nb, A


def findCenter2(D, NN, NNN, A, nb):
    r1 = np.zeros(D.shape[0])
    r2 = np.zeros(D.shape[0])
    rf = np.zeros(D.shape[0])
    Pr1 = np.zeros(D.shape[0])
    Pr2 = np.zeros(D.shape[0])
    CPV = np.zeros(D.shape[0])
    CP = np.zeros(D.shape[0])
    Nei1 = []
    Nei2 = []
    CL = np.zeros(D.shape[0])

    for kk in range(D.shape[0]):
        CL[kk] = 0

    for ii in range(D.shape[0]):
        if len(NN[ii]) > 0:
            r1[ii] = 1 * np.max(pdist([D[ii, :], D[NN[ii], :]]))
            r2[ii] = np.max(pdist([D[ii, :], D[NN[ii], :]]))
            rf = r1 * 0.95
            Nei1.append(np.where(A[:, ii] < r1[ii])[0])
            Nei2.append(np.where(A[:, ii] < rf[ii])[0])
            Pr1[ii] = len(Nei1[ii])
            Pr2[ii] = len(Nei2[ii])
        else:
            r1[ii] = 0
            r2[ii] = 0
            rf[ii] = 0
            Nei1.append([])
            Nei2.append([])

    B = np.mean(r2) + 2 * np.std(r2)

    for ii in range(D.shape[0]):
        if r2[ii] > B:
            CL[ii] = -1
        if r2[ii] == 0:
            CL[ii] = -1
        if nb[ii] < 2:
            CL[ii] = -1

    for jj in range(D.shape[0]):
        Nei1[jj] = Nei1[jj][CL[Nei1[jj]] != -1]

    for ii in range(D.shape[0]):
        if len(Nei1[ii]) > 0:
            if CL[ii] != -1:
                y = np.argmin(pdist([D[Nei1[ii], :], np.mean(D[Nei1[ii], :], axis=0)]))
                if CPV[Nei1[ii][y]] == ii:
                    CPV[ii] = ii
                else:
                    CPV[ii] = Nei1[ii][y]
            else:
                CPV[ii] = ii
        else:
            CPV[ii] = ii

    for ii in range(D.shape[0]):
        if CL[ii] != -1:
            CP[ii] = ii
            while CP[ii] != CPV[CP[ii]]:
                CP[ii] = CPV[CP[ii]]
        else:
            CP[ii] = ii

    return CPV, CP, Pr1, Pr2, r1, rf, r2, Nei1, CL


def findDensityPeak(CP, D, r1, rf, Pr1, Pr2, nb, Sup, CL):
    LPS = []
    FLP = []
    T2 = []

    for ii in range(D.shape[0]):
        if CL[ii] != -1:
            if CP[ii] == ii:
                LPS.append(ii)
                # T2 = D.shape[1] * np.log(rf[ii] / r1[ii]) + np.log(Pr1[ii])
                # if nb[ii] < Sup/2:
                #     FLP.append(ii)

    return LPS, FLP, T2


def showCPV(D, CPV, LPS, CL):
    fig, ax = plt.subplots()

    ax.plot(D[CL != -1, 0], D[CL != -1, 1], '.', color=[1, 0.5, 0], markersize=10)

    for ii in range(D.shape[0]):
        if CPV[ii] != ii and CL[ii] != -1:
            x = [D[ii, 0], D[CPV[ii], 0]]
            y = [D[ii, 1], D[CPV[ii], 1]]
            ax.plot(x, y, '-', color=[0.4, 0.8, 0.9], markersize=5)

    ax.plot(D[LPS, 0], D[LPS, 1], 'r.', markersize=20)

    plt.show()


def Kruskal(D):
    # Construct adjacency matrix
    a = np.zeros((D.shape[0], D.shape[0]))
    for i in range(D.shape[0]):
        for j in range(D.shape[0]):
            if i < j:
                a[i, j] = np.sqrt((D[i, 0] - D[j, 0]) ** 2 + (D[i, 1] - D[j, 1]) ** 2)

    # Create graph and find minimum spanning tree
    G = nx.Graph(a)
    T = nx.minimum_spanning_tree(G)

    # Get result matrix
    result = np.array([(u, v, G[u][v]['weight']) for u, v in T.edges()])

    return T, result


def DBSCAN(X, epsilon, MinPts):
    C = 0
    n = X.shape[0]
    IDX = np.zeros(n)

    D = pairwise_distances(X)

    visited = np.zeros(n, dtype=bool)
    isnoise = np.zeros(n, dtype=bool)

    def ExpandCluster(i, Neighbors, C):
        IDX[i] = C

        k = 0
        while k < len(Neighbors):
            j = Neighbors[k]

            if not visited[j]:
                visited[j] = True
                Neighbors2 = RegionQuery(j)
                if len(Neighbors2) >= MinPts:
                    Neighbors.extend(Neighbors2)

            if IDX[j] == 0:
                IDX[j] = C

            k += 1

    def RegionQuery(i):
        return np.where(D[i] <= epsilon)[0]

    for i in range(n):
        if not visited[i]:
            visited[i] = True

            Neighbors = RegionQuery(i)
            if len(Neighbors) < MinPts:
                isnoise[i] = True
            else:
                C += 1
                ExpandCluster(i, Neighbors, C)

    return IDX, isnoise
