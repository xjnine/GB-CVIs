import numpy as np


def findCore(D):
    NN, RNN, NNN, A, nb, Sup = NaNSearching(D)
    CPV, CP, Pr1, Pr2, r1, rf, r2, Nei1, CL = findCenter2(D, NNN, A, nb)
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

    return NN, RNN, NNN, A, nb, Sup


def findCenter2(D, NNN, A, nb):
    r1 = []
    r2 = []
    rf = []
    Pr1 = []
    Pr2 = []
    CPV = np.zeros(D.shape[0])
    CP = []
    Nei1 = [None] * D.shape[0]
    Nei2 = [None] * D.shape[0]
    CL = np.zeros(D.shape[0])

    for kk in range(D.shape[0]):
        CL[kk] = 0

    for ii in range(D.shape[0]):
        if len(NNN[ii]) > 0:
            r1.append(4 * np.mean(np.sqrt(np.sum((D[ii, :] - D[NNN[ii], :]) ** 2, axis=1))))
            r2.append(np.max(np.sqrt(np.sum((D[ii, :] - D[NNN[ii], :]) ** 2, axis=1))))
            rf.append(r1[ii] * 0.95)
            Nei1[ii] = np.where(A[:, ii] < r1[ii])[0]
            Nei2[ii] = np.where(A[:, ii] < rf[ii])[0]
            Pr1.append(len(Nei1[ii]))
            Pr2.append(len(Nei2[ii]))
        else:
            r1.append(0)
            r2.append(0)
            rf.append(0)

    B = np.mean(r1) + 2.5 * np.std(r1)
    for ii in range(D.shape[0]):
        if r1[ii] > B:
            CL[ii] = -1
        if r1[ii] == 0:
            CL[ii] = -1
        if nb[ii] < 2:
            CL[ii] = -1

    for jj in range(D.shape[0]):
        Nei1[jj] = Nei1[jj][np.where(CL[Nei1[jj]] != -1)[0]]

    for ii in range(D.shape[0]):
        if len(Nei1[ii]) > 0:
            if CL[ii] != -1:
                _, y = np.min(np.sqrt(np.sum((D[Nei1[ii], :] - np.mean(D[Nei1[ii], :], axis=0)) ** 2, axis=1)))
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

    return LPS, FLP, T2
