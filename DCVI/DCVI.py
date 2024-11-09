import numpy as np
import networkx as nx
import pandas as pd
from scipy.spatial import distance_matrix
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt


def DCVI(D):
    CL, LPS, CP, CPV = findCore(D)
    # showCPV(D, CPV, LPS, CL)
    D1 = D[LPS, :]  # 核心点

    G, result = Kruskal(D1)
    # MST(D1, result)

    return result, D1, LPS, CP, CL, CPV


# 7
def MST(D, result):
    edge_indices = np.argsort(result[2])
    max_edge_indices = edge_indices[-4:]
    # plot the points

    plt.scatter(D[:, 0], D[:, 1], color=[1, 0.5, 0], s=10)

    # plot the edges
    for i in range(result.shape[1]):
        x = [D[int(result[0, i]), 0], D[int(result[1, i]), 0]]
        y = [D[int(result[0, i]), 1], D[int(result[1, i]), 1]]
        if i in max_edge_indices:
            plt.plot(x, y, '-', color='red')
        else:
            plt.plot(x, y, '-', color=[0.4, 0.8, 0.9])
    plt.title("MST")
    # show the plot
    plt.show()


# 5
def showCPV(D, CPV, LPS, CL):
    plt.figure(1)

    plt.plot(D[CL != -1, 0], D[CL != -1, 1], '.', color=[1, 0.5, 0], markersize=10)

    for ii in range(D.shape[0]):
        if CPV[ii] != ii and CL[ii] != -1:
            x = [D[ii, 0], D[int(CPV[ii]), 0]]
            y = [D[ii, 1], D[int(CPV[ii]), 1]]
            plt.plot(x, y, '-', color=[0.3, 0.8, 0.9], markersize=5)

    plt.plot(D[LPS, 0], D[LPS, 1], 'r.', markersize=10)
    plt.title("showCPV")
    plt.show()


# 6
def Kruskal(D):
    # Construct the adjacency matrix
    a = np.zeros((D.shape[0], D.shape[0]))
    for i in range(D.shape[0]):
        for j in range(D.shape[0]):
            if i < j:
                a[i, j] = np.sqrt((D[i, 0] - D[j, 0]) ** 2 + (D[i, 1] - D[j, 1]) ** 2)

    # Construct graph
    G = nx.from_numpy_matrix(a)

    # Compute the minimum spanning tree of the graph
    T = nx.minimum_spanning_tree(G)

    # Prepare result
    edges = T.edges(data=True)
    result = [[u, v, data['weight']] for u, v, data in edges]
    result = np.transpose(result)

    return T, result


# 1
def findCore(D):
    Sup, NN, RNN, NNN, nb, A = NaNSearching(D)
    CPV, CP, Pr1, Pr2, r1, rf, r2, Nei1, CL = findCenter2(D, NN, NNN, A, nb)
    LPS, FLP, T2 = findDensityPeak(CP, D, r1, rf, Pr1, Pr2, nb, Sup, CL)

    return CL, LPS, CP, CPV


# 2
# def NaNSearching(D):
#     r = 1
#     nb = np.zeros((D.shape[0], 1))
#     C = [None] * D.shape[0]
#     NN = [[] for _ in range(D.shape[0])]
#     RNN = [[] for _ in range(D.shape[0])]
#     NNN = [[] for _ in range(D.shape[0])]
#     NaN = [[] for _ in range(D.shape[0])]
#     A = distance_matrix(D, D)
#     Numb1 = 0
#     Numb2 = 0
#     for ii in range(D.shape[0]):
#         sa_index = np.argsort(A[:, ii])
#         C[ii] = np.column_stack([A[:, ii][sa_index], sa_index])
#     while (r < D.shape[0]):
#         for kk in range(D.shape[0]):
#             x = kk
#             y = C[x][r, 1]
#             nb[int(y)] = nb[int(y)] + 1
#             NN[x].append(int(y))
#             RNN[int(y)].append(x)
#         Numb1 = np.sum(nb == 0)
#         if Numb2 != Numb1:
#             Numb2 = Numb1
#         else:
#             break
#         r = r + 1
#     for jj in range(D.shape[0]):
#         NNN[jj] = np.intersect1d(NN[jj], RNN[jj])
#         id = len(RNN[jj]) + 1
#         temp = C[jj][1:id, 1]
#         NaN.append(temp)
#     Sup = r
#     return Sup, NN, RNN, NNN, nb, A,NaN
def NaNSearching(D):
    r = 1
    nb = np.zeros((D.shape[0], 1))
    C = [None] * D.shape[0]
    NN = [[] for _ in range(D.shape[0])]
    RNN = [[] for _ in range(D.shape[0])]
    NNN = [[] for _ in range(D.shape[0])]
    A = distance_matrix(D, D)
    Numb1 = 0
    Numb2 = 0
    for ii in range(D.shape[0]):
        sa_index = np.argsort(A[:, ii])
        C[ii] = np.column_stack([A[:, ii][sa_index], sa_index])
    while (r < D.shape[0]):
        for kk in range(D.shape[0]):
            x = kk
            y = C[x][r + 1, 1]
            nb[int(y)] = nb[int(y)] + 1
            NN[x].append(int(y))
            RNN[int(y)].append(x)
        Numb1 = np.sum(nb == 0)
        if Numb2 != Numb1:
            Numb2 = Numb1
        else:
            break
        r = r + 1
    for jj in range(D.shape[0]):
        NNN[jj] = np.intersect1d(NN[jj], RNN[jj])
    Sup = r
    return Sup, NN, RNN, NNN, nb, A


# 3
# def findCenter2(D, NN, NNN, A, nb, NaN):
#     r1 = np.zeros(D.shape[0])
#     r2 = np.zeros(D.shape[0])
#     rf = np.zeros(D.shape[0])
#     Pr1 = np.zeros(D.shape[0])
#     Pr2 = np.zeros(D.shape[0])
#     CPV = np.zeros(D.shape[0])  # 用于噪声点的检查
#     CP = np.zeros(D.shape[0])
#
#     Nei1 = [None] * D.shape[0]
#     Nei2 = [None] * D.shape[0]
#     CL = np.zeros(D.shape[0])
#
#     for kk in range(D.shape[0]):
#         CL[kk] = 0
#
#     for ii in range(D.shape[0]):
#         if NN[ii]:
#             # r1[ii] = 1 * np.max(np.linalg.norm(D[ii, :] - D[NN[ii], :], axis=1))
#             # r2[ii] = np.max(np.linalg.norm(D[ii, :] - D[NN[ii], :], axis=1))
#             # rf = r1 * 0.95
#             # Nei1[ii] = np.where(A[:, ii] < r1[ii])[0]
#             # Nei2[ii] = np.where(A[:, ii] < rf[ii])[0]
#             # Pr1[ii] = Nei1[ii].shape[0]
#             # Pr2[ii] = Nei2[ii].shape[0]
#             # 计算点p与周围自然邻居的距离极大值作为寻求density core的半径
#             # r1[ii] = 1.5 * np.max(pdist(D[ii].reshape(1, -1), D[NN[ii]]))
#             r1[ii] = 1 * np.max(np.linalg.norm(D[ii, :] - D[NN[ii], :], axis=1))
#             Nei1[ii] = np.where(A[:, ii] < r1[ii])[0]
#             Pr1[ii] = Nei1[ii].shape[0]
#         else:
#             r1[ii] = 0
#         if NN[ii]:
#             # 计算点p与周围自然邻居的距离极大值作为寻求convergent point的半径
#             # r2[ii] = np.max(pdist(D[ii].reshape(1, -1), D[NN[ii]]))
#             r2[ii] = np.max(np.linalg.norm(D[ii, :] - D[NN[ii], :], axis=1))
#         else:
#             r2[ii] = 0
#
#         # else:
#         #     r1[ii] = 0
#         #     r2[ii] = 0
#         #     rf[ii] = 0
#
#     B = np.mean(r2) + 2 * np.std(r2)
#     for ii in range(D.shape[0]):
#         if r2[ii] > B:
#             CL[ii] = -1
#         if r2[ii] == 0:
#             CL[ii] = -1
#         if nb[ii] < 2:
#             CL[ii] = -1
#
#     for jj in range(D.shape[0]):
#         Nei1[ii] = np.setdiff1d(Nei1[ii], np.where(CL[Nei1[ii]] == -1))
#
#     for ii in range(D.shape[0]):
#         if len(Nei1[ii]) != 0:
#             if CL[ii] != -1:
#                 y = np.argmin(np.linalg.norm(D[Nei1[ii], :] - np.mean(D[Nei1[ii], :], axis=0), axis=1))
#                 if CPV[Nei1[ii][y]] == ii:
#                     CPV[ii] = ii
#                 else:
#                     CPV[ii] = Nei1[ii][y]
#             else:
#                 CPV[ii] = ii
#         else:
#             CPV[ii] = ii
#
#     for ii in range(D.shape[0]):
#         if CL[ii] != -1:
#             CP[ii] = ii
#             while CP[ii] != CPV[int(CP[ii])]:
#                 CP[ii] = CPV[int(CP[ii])]
#         else:
#             CP[ii] = ii
#
#     return CPV, CP, Pr1, Pr2, r1, rf, r2, Nei1, CL
def findCenter2(D, NN, NNN, A, nb):
    r1 = np.zeros(D.shape[0])
    r2 = np.zeros(D.shape[0])
    rf = np.zeros(D.shape[0])
    Pr1 = np.zeros(D.shape[0])
    Pr2 = np.zeros(D.shape[0])
    CPV = np.zeros(D.shape[0])  # 用于噪声点的检查
    CP = np.zeros(D.shape[0])

    Nei1 = [None] * D.shape[0]
    Nei2 = [None] * D.shape[0]
    CL = np.zeros(D.shape[0])

    for kk in range(D.shape[0]):
        CL[kk] = 0

    for ii in range(D.shape[0]):
        if len(NN[ii]) != 0:
            r1[ii] = 1 * np.max(np.linalg.norm(D[ii, :] - D[NN[ii], :], axis=1))
            r2[ii] = np.max(np.linalg.norm(D[ii, :] - D[NN[ii], :], axis=1))
            rf = r1 * 0.95
            Nei1[ii] = np.where(A[:, ii] < r1[ii])[0]
            Nei2[ii] = np.where(A[:, ii] < rf[ii])[0]
            Pr1[ii] = Nei1[ii].shape[0]
            Pr2[ii] = Nei2[ii].shape[0]
        else:
            r1[ii] = 0
            r2[ii] = 0
            rf[ii] = 0

    B = np.mean(r2) + 2 * np.std(r2)
    for ii in range(D.shape[0]):
        if r2[ii] > B:
            CL[ii] = -1
        if r2[ii] == 0:
            CL[ii] = -1
        if nb[ii] < 2:
            CL[ii] = -1

    for jj in range(D.shape[0]):
        Nei1[ii] = np.setdiff1d(Nei1[ii], np.where(CL[Nei1[ii]] == -1))

    for ii in range(D.shape[0]):
        if len(Nei1[ii]) != 0:
            if CL[ii] != -1:
                y = np.argmin(np.linalg.norm(D[Nei1[ii], :] - np.mean(D[Nei1[ii], :], axis=0), axis=1))
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
            while CP[ii] != CPV[int(CP[ii])]:
                CP[ii] = CPV[int(CP[ii])]
        else:
            CP[ii] = ii

    return CPV, CP, Pr1, Pr2, r1, rf, r2, Nei1, CL


# 4
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
