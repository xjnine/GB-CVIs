import numpy as np


def NaNSearching2(D):
    r = 1
    nb = np.zeros(D.shape[0])
    C = [None] * D.shape[0]
    NN = [None] * D.shape[0]  # Initialize KNN neighbors for each point
    RNN = [None] * D.shape[0]  # Initialize RKNN neighbors for each point
    NNN = [None] * D.shape[0]  # Intersection of NN and RNN, i.e., neighbors for each point
    NaN = [None] * D.shape[0]
    A = np.linalg.norm(D[:, np.newaxis] - D, axis=2)  # Euclidean distance between points

    Numb1 = 0
    Numb2 = 0

    for ii in range(D.shape[0]):
        sa, index = zip(*sorted(enumerate(A[:, ii]), key=lambda x: x[1]))
        C[ii] = np.column_stack((sa, index))

    while r < D.shape[0]:
        for kk in range(D.shape[0]):
            x = kk
            y = C[x][r + 1, 1]
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
        id = RNN[jj].shape[0] + 1
        temp = C[jj][1:id, 1]
        NaN[jj] = temp

    Sup = r

    return Sup, NN, RNN, NNN, nb, A, NaN
