import numpy as np


def Kruskal(D):
    # 构建连接图（邻接矩阵）
    a = np.zeros((D.shape[0], D.shape[0]))
    for i in range(D.shape[0]):
        print('**************************************' + str(i))
        for j in range(D.shape[0]):
            if i < j:
                a[i, j] = np.sqrt((D[i, 0] - D[j, 0]) ** 2 + (D[i, 1] - D[j, 1]) ** 2)
                # a[i, j] = np.linalg.norm(D[i, :] - D[j, :])

    # Prim's Algorithm
    a = a + np.transpose(a)
    a[a == 0] = np.inf
    result = []
    p = [0]
    tb = list(range(1, len(a)))

    while len(result) != len(a) - 1:
        temp = a[p, :][:, tb].flatten()
        d = np.min(temp)
        jb, kb = np.where(a[p, :][:, tb] == d)
        j = p[jb[0]]
        k = tb[kb[0]]
        result.append([j, k, d])
        p.append(k)
        tb.remove(k)

    return result
