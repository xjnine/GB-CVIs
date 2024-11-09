import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


def findCore(D):
    Sup, NN, RNN, NNN, nb, A = NaNSearching(D)
    CPV, CP, Pr1, Pr2, r1, rf, r2, Nei1, CL = findCenter2(D, NN, NNN, A, nb)
    LPS, FLP, T2 = findDensityPeak(CP, D, r1, rf, Pr1, Pr2, nb, Sup, CL)
    return CL, LPS, CP, CPV


def NaNSearching(D):
    r = 1
    nb = np.zeros(D.shape[0])
    C = [None] * D.shape[0]
    NN = [None] * D.shape[0]  # 初始化每个点的KNN邻居
    RNN = [None] * D.shape[0]  # 初始化每个点的RKNN邻居
    NNN = [None] * D.shape[0]  # 是NN和RNN的交集，也就是每个点的
    A = np.zeros((D.shape[0], D.shape[0]))
    Numb1 = 0
    Numb2 = 0

    for ii in range(D.shape[0]):
        sa, index = np.sort(A[:, ii]), np.argsort(A[:, ii])
        C[ii] = np.column_stack((sa, index))

    while r < D.shape[0]:
        for kk in range(D.shape[0]):
            x = kk
            y = C[x][r, 1]
            nb[y] += 1
            NN[x] = NN[x] + [y]
            RNN[y] = RNN[y] + [x]

        Numb1 = np.sum(nb == 0)
        if Numb2 != Numb1:
            Numb2 = Numb1
        else:
            break
        r += 1

    for jj in range(D.shape[0]):
        NNN[jj] = list(set(NN[jj]).intersection(RNN[jj]))

    Sup = r

    return Sup, NN, RNN, NNN, nb, A


def findCenter2(D, NN, NNN, A, nb):
    r1 = []
    r2 = []
    rf = []
    Pr1 = []
    Pr2 = []
    CPV = np.zeros(D.shape[0])  # 用于噪声点的检查***
    CP = []
    Nei1 = [None] * D.shape[0]
    Nei2 = [None] * D.shape[0]

    CL = np.zeros(D.shape[0])

    for kk in range(D.shape[0]):
        CL[kk] = 0

    for ii in range(D.shape[0]):
        if len(NN[ii]) > 0:
            r1.append(3.2 * np.max(np.sqrt(np.sum((D[ii, :] - D[NN[ii], :]) ** 2, axis=1))))
            r2.append(np.max(np.sqrt(np.sum((D[ii, :] - D[NN[ii], :]) ** 2, axis=1))))
            rf.append(r1[ii] * 0.95)
            Nei1[ii] = np.where(A[:, ii] < r1[ii])[0]
            Nei2[ii] = np.where(A[:, ii] < rf[ii])[0]
            Pr1.append(len(Nei1[ii]))
            Pr2.append(len(Nei2[ii]))
        else:
            r1.append(0)
            r2.append(0)
            rf.append(0)

    B = np.mean(r2) + 2 * np.std(r2)
    for ii in range(D.shape[0]):
        if r2[ii] > B:
            CL[ii] = -1
        if r2[ii] == 0:
            CL[ii] = -1
        if nb[ii] < 2:
            CL[ii] = -1

    for ii in range(D.shape[0]):
        Nei1[ii] = Nei1[ii][CL[Nei1[ii]] != -1]

    for ii in range(D.shape[0]):
        if CL[ii] != -1:
            y = np.argmin(np.sqrt(np.sum((D[Nei1[ii], :] - np.mean(D[Nei1[ii], :], axis=0)) ** 2, axis=1)))
            if CPV[Nei1[ii][y]] == ii:
                CPV[ii] = ii
            else:
                CPV[ii] = Nei1[ii][y]
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
    #                 T2 = D.shape[1] * np.log(rf[ii] / r1[ii]) + np.log(Pr1[ii])
    #                 if nb[ii] < Sup / 2:
    #                     FLP.append(ii)
    return LPS, FLP, T2


def Kruskal(D):
    a = np.zeros((D.shape[0], D.shape[0]))
    for i in range(D.shape[0]):
        for j in range(D.shape[0]):
            if i < j:
                a[i, j] = np.sqrt((D[i, 0] - D[j, 0]) ** 2 + (D[i, 1] - D[j, 1]) ** 2)

    G = nx.Graph(a)
    T = nx.minimum_spanning_tree(G)
    result = np.array(list(T.edges)).T

    return T, result


def MST(D, result):
    plt.figure()
    plt.plot(D[:, 0], D[:, 1], 'r.', markersize=20)

    for i in range(result.shape[1]):
        x = [D[result[0, i], 0], D[result[1, i], 0]]
        y = [D[result[0, i], 1], D[result[1, i], 1]]
        plt.plot(x, y, '-', color=[0.4, 0.8, 0.9], markersize=5)

    plt.show()
