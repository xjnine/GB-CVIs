import pandas as pd
from scipy.spatial import distance_matrix
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from scipy.sparse.csgraph import connected_components


def testWeight(k, mst):
    # 初始化一些变量和数组
    avgDCVI = 0
    DCVI = np.zeros(k)
    edge = mst[2]
    sortmst = mst[:, np.argsort(edge)[::-1]]
    cutnumber = k - 1
    weight = np.zeros(k)
    weight2 = np.zeros(k)
    weight3 = np.zeros(k)
    cutmst = sortmst.copy()
    wmst = sortmst.copy()
    # 使用Graph对象构建一个网络，该网络包含所有的边
    G = nx.Graph()
    for ii in range(cutmst.shape[1]):
        a = cutmst[0, ii].astype(int)
        b = cutmst[1, ii].astype(int)
        G.add_edge(a, b)
    # 删除足够数量的最大权重边以形成k个连通组件
    for ii in range(cutnumber):
        a = cutmst[0, ii].astype(int)
        b = cutmst[1, ii].astype(int)
        if G.has_edge(a, b):
            G.remove_edge(a, b)
    # 将网络转换为稀疏矩阵，并找出每个连通组件
    sparse_matrix = nx.to_scipy_sparse_array(G)
    bins = connected_components(sparse_matrix)[1]
    # print(k)
    # print("bins:", bins)

    for ii in range(k):
        if np.sum(bins == ii) == 1:
            return float('inf'), bins

    # 确定每个连通组件的最大边（它们都是被删除的边）
    maxedge = sortmst[:, :cutnumber].copy()
    cutpoint = np.zeros((3, cutnumber))
    wmst = wmst[:, cutnumber:]

    # 获取每个连通组件的最大边的两个端点和权重
    for ii in range(2):
        for jj in range(cutnumber):
            cutpoint[ii, jj] = bins[int(maxedge[ii, jj])]

    cutpoint[2, :] = maxedge[2, :]

    # 初始化sep数组以存储每个连通组件的最大边权重，并找出最大的边权重
    sep = np.full(k, cutpoint[2, 0])
    maxsep = cutpoint[2, 0]

    # 更新每个连通组件的sep值为它的最大边权重和它与其他连通组件的边的权重中的最小值
    for ii in range(k):
        for jj in range(cutnumber):
            if cutpoint[0, jj] == ii or cutpoint[1, jj] == ii:
                sep[ii] = min(cutpoint[2, jj], sep[ii])

    # 计算每个连通组件内部的平均边权重和最大边权重
    for ii in range(k):
        temp2 = 0
        temp3 = np.inf
        temp = 0
        count = 0
        for jj in range(wmst.shape[1]):
            if bins[int(wmst[0, jj])] == ii:
                temp2 = max(temp2, wmst[2, jj])  # 寻找连通组件内的最大权重
                temp3 = min(temp3, wmst[2, jj])  # 寻找连通组件内的最小权重
                temp += wmst[2, jj]  # 计算连通组件内所有权重的总和
                count += 1  # 计算连通组件内边的数量
            if count - 1 == 0:  # 如果连通组件只包含一个点
                weight[ii] = 0  # 连通组件的平均权重为0
            else:  # 否则
                weight[ii] = temp / (count - 1)  # 连通组件的平均权重为权重总和除以边的数量减1
            weight2[ii] = temp2  # 连通组件的最大权重

    temp = 0
    for ii in range(k):
        if weight2[ii] == 0:  # 如果连通组件没有边（只有一个点）
            temp = min(weight2[weight2 != 0])  # 则其最大权重为其他连通组件最大权重的最小值
            weight2[ii] = temp
        weight3[ii] = weight[ii] / weight2[ii]  # 连通组件的权重比值（平均权重/最大权重）

    for ii in range(k):  # 计算每个连通组件的DCVI值（最大权重/分离度）
       DCVI[ii] = weight2[ii] / sep[ii]
        # DCVI[ii] = weight2[ii] / (sep[ii] * sep[ii])
    print('JJJ',DCVI)
    avgDCVI = np.mean(DCVI)  # 计算所有连通组件的DCVI值的平均值
    print('运行了',{k})
    return avgDCVI, bins  # 返回平均DCVI值，连通组件数组，每个连通组件的最大权重，分离度，排序后的MST


def TestCVI(result):
    kk = result.shape[1] + 1
    on = 1
    onCVI = 1
    cl = None
    # int(np.ceil(np.sqrt(kk)))
    for ii in range(2, 16):
        # if ii==31:
        #     return
        avgDCVI, bins = testWeight(ii, result)
        print(ii,avgDCVI)
        temp = min(onCVI, avgDCVI)
        if onCVI != temp:
            onCVI = temp
            on = ii
            cl = bins

    return on, onCVI,cl


# 原始
# def DCVI(D):
#     CL, LPS, CP, CPV = findCore(D)
#     showCPV(D, CPV, LPS, CL)
#     D1 = D[LPS, :]  # 核心点
#     G, result = Kruskal(D1)
#     MST(D1, result)
#
#     return result, D1, LPS, CP, CL, CPV
# 粒球
def DCVI(D):
    CL, LPS, CP, CPV = findCore(D)
    # showCPV(D, CPV, LPS, CL)
    D1 = D[LPS, :]  # 核心点
    G, result = Kruskal(D1)
    # MST(D1, result)

    return result, D1, LPS, CP, CL, CPV


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


# 1
def findCore(D):
    Sup, NN, RNN, NNN, nb, A = NaNSearching(D)
    CPV, CP, Pr1, Pr2, r1, rf, r2, Nei1, CL = findCenter2(D, NN, NNN, A, nb)
    LPS, FLP, T2 = findDensityPeak(CP, D, r1, rf, Pr1, Pr2, nb, Sup, CL)

    return CL, LPS, CP, CPV


# 2
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
