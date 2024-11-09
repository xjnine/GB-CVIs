import numpy as np
import networkx as nx
from scipy.sparse.csgraph import connected_components


def testWeight(k, mst):
    avgDCVI = 0
    DCVI = np.zeros(k)
    # MST的变的权重
    edge = mst[2]
    # 对mst 的权重进行从大到小的排序
    sortmst = mst[:, np.argsort(edge)[::-1]]
    # 将要切的边的个数
    cutnumber = k - 1

    weight = np.zeros(k)
    weight2 = np.zeros(k)
    weight3 = np.zeros(k)
    cutmst = sortmst.copy()
    wmst = sortmst.copy()

    # 创建边
    G = nx.Graph()
    # cutmst.shape[1] mst 的列数 即节点个数
    for ii in range(cutmst.shape[1]):
        a = cutmst[0, ii].astype(int)
        b = cutmst[1, ii].astype(int)
        G.add_edge(a, b)

    #  删除前k-1个最长的边
    for ii in range(cutnumber):
        a = cutmst[0, ii].astype(int)
        b = cutmst[1, ii].astype(int)
        if G.has_edge(a, b):
            G.remove_edge(a, b)

    # 将图的结构信息转换成稀疏矩阵
    sparse_matrix = nx.to_scipy_sparse_matrix(G)
    # 找出图中的连通分量。它返回一个元组，包含两个元素：
    # 第一个元素是连通分量的个数
    # bins ====> 第二个元素是一个列表，其中每个节点的标签表示所属的连通分量。
    bins = connected_components(sparse_matrix)[1]

    # 如果某个连通分量只有一个节点，它将被视为孤立的，而不是形成了完整的连通网络。
    # 在这种情况下，函数会返回一个无穷大的值和表示节点所属连通分量的 bins 数组。

    maxedge = sortmst[:, :cutnumber].copy()
    cutpoint = np.zeros((3, cutnumber))
    # 返回 cutnumber 开始到末尾的部分
    wmst = wmst[:, cutnumber:]

    for ii in range(2):
        for jj in range(cutnumber):
            cutpoint[ii, jj] = bins[int(maxedge[ii, jj])]

    cutpoint[2, :] = maxedge[2, :]

    sep = np.full(k, cutpoint[2, 0])
    maxsep = cutpoint[2, 0]

    # 簇间分离度 找最小值
    for ii in range(k):
        for jj in range(cutnumber):
            if cutpoint[0, jj] == ii or cutpoint[1, jj] == ii:
                sep[ii] = min(cutpoint[2, jj], sep[ii])

    for ii in range(k):
        temp2 = 0
        temp3 = np.inf
        temp = 0
        count = 0
        for jj in range(wmst.shape[1]):
            if bins[int(wmst[0, jj])] == ii:
                temp2 = max(temp2, wmst[2, jj])
                temp3 = min(temp3, wmst[2, jj])
                temp += wmst[2, jj]
                count += 1
        if count - 1 == 0:
            weight[ii] = 0
        else:
            weight[ii] = temp / (count - 1)
        # weight2[ii]  某一个簇内边的总和
        weight2[ii] = temp2

    temp = 0
    for ii in range(k):
        # 簇内只有一个点的处理
        if weight2[ii] == 0:
            temp = min(weight2[weight2 != 0])
            weight2[ii] = temp
        weight3[ii] = weight[ii] / weight2[ii]

    for ii in range(k):
        # 原始版本 com 为 簇内最长边
        # DCVI[ii] = weight2[ii] / sep[ii]
    #     版本1 com 为 簇内所有边的总和/边的个数
        DCVI[ii] = weight[ii] / sep[ii]
    avgDCVI = np.mean(DCVI)

    return avgDCVI, bins
