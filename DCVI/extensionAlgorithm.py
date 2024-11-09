import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.sparse.csgraph import minimum_spanning_tree


def extensionalgorithm(D):
    CL, LPS, CP, CPV = findCore(D)  # 寻找密度核
    D1 = D[LPS, :]  # 核心点
    k = D1.shape[0]
    kk = np.ceil(np.sqrt(D1.shape[0]))
    Distance = np.sqrt(np.sum((D1[:, np.newaxis, :] - D1[np.newaxis, :, :]) ** 2, axis=2))
    Z = linkage(Distance, 'single')
    avgCVI = np.zeros(int(kk) - 1)
    cl = np.zeros((int(kk) - 1, k), dtype=int)

    for ii in range(2, int(kk) + 1):
        CL = fcluster(Z, ii, criterion='maxclust')
        cd = np.zeros(ii)
        sd = np.zeros(ii)
        cvi = np.zeros(ii)

        for mm in range(1, ii + 1):
            a = np.where(CL == mm)[0]
            b = np.where(CL != mm)[0]
            D2 = Distance[a[:, np.newaxis], a]
            D3 = Distance[a[:, np.newaxis], b]
            T = minimum_spanning_tree(D2).toarray()
            weight = T[T > 0]
            if len(weight) > 0:
                cd[mm - 1] = np.max(weight)
            else:
                cd[mm - 1] = 0
            sd[mm - 1] = np.min(D3)

        for mm in range(1, ii + 1):
            if cd[mm - 1] == 0:
                temp = np.min(cd[cd != 0])
                cd[mm - 1] = temp

        for tt in range(1, ii + 1):
            cvi[tt - 1] = cd[tt - 1] / sd[tt - 1]

        avgCVI[ii - 2] = (1 / ii) * np.sum(cvi)
        cl[ii - 2] = CL

    d = np.argmin(avgCVI)
    on = d + 2
    oncvi = avgCVI[d]
    bins = cl[d]

    return on, oncvi, avgCVI, bins, D1


def findCore(D):
    Sup, NN, RNN, NNN, nb, A = NaNSearching(D)
    CPV, CP, Pr1, Pr2, r1, rf, r2, Nei1, CL = findCenter2(D, NN, NNN, A, nb)
    LPS, FLP, T2 = findDensityPeak(CP, D, r1, rf, Pr1, Pr2, nb, Sup, CL)

    return CL, LPS, CP, CPV


def NaNSearching(D):
    r = 1
    nb = np.zeros(D.shape[0])
    C = [None] * D.shape[0]
    NN = [None] * D.shape[0]
    RNN = [None] * D.shape[0]
    NNN = [None] * D.shape[0]
    A = np.zeros((D.shape[0], D.shape[0]))
    Numb1 = 0
    Numb2 = 0

    for ii in range(D.shape[0]):
        sa, index = np.sort(A[:, ii]), np.argsort(A[:, ii])
        C[ii] = [sa, index]

    while r < D.shape[0]:
        for kk in range(D.shape[0]):
            x = kk
            y = C[x][0][r + 1, 2]
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
        NNN[jj] = list(set(NN[jj]) & set(RNN[jj]))

    Sup = r
    return Sup, NN, RNN, NNN, nb, A


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
                _, y = np.min(np.linalg.norm(D[Nei1[ii], :] - np.mean(D[Nei1[ii], :], axis=0), axis=1))
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
