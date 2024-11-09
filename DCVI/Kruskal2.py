import numpy as np


def Kruskal2(D):
    a = np.zeros((D.shape[0], D.shape[0]))
    for i in range(D.shape[0]):
        print('**************************************', i)
        for j in range(D.shape[0]):
            if i < j:
                a[i, j] = np.sqrt((D[i, 0] - D[j, 0]) ** 2 + (D[i, 1] - D[j, 1]) ** 2)
    graph_adjacent = a
    len = graph_adjacent.shape[0]
    temp = graph_adjacent.copy()
    superedge = np.zeros((len - 1, 2))
    i = 0

    tag = np.arange(1, len + 1)
    while superedge[len - 2, 0] == 0:
        Y, I = np.sort(temp, axis=0), np.argsort(temp, axis=0)
        cost_min = np.min(Y[0, :])
        index = np.where(Y[0, :] == cost_min)[0][0]
        anotherpoint = I[0, index]
        temp[index, anotherpoint] = 100
        temp[anotherpoint, index] = 100
        if tag[anotherpoint] != tag[index]:
            superedge[i, :] = [index, anotherpoint]
            i += 1
            for j in range(len):
                if (tag[j] == tag[anotherpoint]) and (j != anotherpoint):
                    tag[j] = tag[index]
            tag[anotherpoint] = tag[index]

    s = 0
    for ii in range(len - 1):
        k = '最小生成树第{}条边：（{}，{}），权值为{}'.format(ii, superedge[ii, 0], superedge[ii, 1],
                                                         graph_adjacent[superedge[ii, 0], superedge[ii, 1]])
        s += graph_adjacent[superedge[ii, 0], superedge[ii, 1]]
    print('最小生成树的总代价为：')
    print(s)

    return superedge
