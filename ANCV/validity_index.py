import numpy as np
from scipy.spatial.distance import squareform, pdist
import numpy as np
from collections import OrderedDict
from scipy.spatial.distance import pdist


def isharedist(data, p, q, p_knn, q_knn, output, Eu_dist, pp):
    p_point = data[p]
    q_point = data[q]

    snn = np.intersect1d(p_knn, q_knn)

    p_q_sharedneibor = np.setdiff1d(np.union1d(p_knn, [p]),
                                    np.union1d(snn, [q]))  # p relative to q non-shared neighbors
    q_p_sharedneibor = np.setdiff1d(np.union1d(q_knn, [q]),
                                    np.union1d(snn, [p]))  # q relative to p non-shared neighbors

    pairwise = []
    for i in range(len(p_q_sharedneibor)):
        for j in range(len(q_p_sharedneibor)):
            if p_q_sharedneibor[i] < q_p_sharedneibor[j]:
                pairwise.append([p_q_sharedneibor[i], q_p_sharedneibor[j]])
            else:
                pairwise.append([q_p_sharedneibor[j], p_q_sharedneibor[i]])

    # distance = Eu_dist[p_q_sharedneibor, q_p_sharedneibor]
    distance = Eu_dist[p_q_sharedneibor][:, q_p_sharedneibor]

    dist_result = distance.T.flatten()

    number = len(dist_result)

    return dist_result, number, pairwise


def Compactness_r9(data_new, output, Dataset_MST, SNNthr, knn, Eu_dist):
    k_value = len(np.unique(output))
    com = np.zeros(k_value)

    for i in range(1, k_value + 1):
        pairwise = 0  # 满足有相连边且共享近邻个数小于阈值的点对数
        tiaojian = []
        idx = np.where(output == i)[0]
        distance = []
        point_wise_in = []
        n = len(idx)
        clu_mst = Dataset_MST
        clu_mst_tril = np.tril(clu_mst)  # 将 clu_mst 矩阵转换为下三角形矩阵
        # clu_mst_tril = Dataset_MST

        if n == 1:
            com[i - 1] = 0
            continue
        # n 当前簇的簇内个数
        for ii in range(n):
            mst_neibor_clu = np.where(clu_mst_tril[idx[ii], :] != 0)[0]
            no_home1 = np.where(output[mst_neibor_clu] != i)[0]
            mst_neibor_clu = np.delete(mst_neibor_clu, no_home1)  # 去掉该点在mst上相连的不属于同一个簇的点
            i_knn = knn[idx[ii], :]

            len_mst_neibor_clu = len(mst_neibor_clu)
            for jj in range(len_mst_neibor_clu):
                near = mst_neibor_clu[jj]
                j_knn = knn[near, :]

                i_point = i_knn
                j_point = j_knn

                sn = len(np.intersect1d(i_point, j_point))

                if sn < SNNthr:
                    pairwise += 1
                    tiaojian.append([idx[ii], near])
                    sdist, num, Pair_point = isharedist(data_new, idx[ii], near, i_knn, j_knn, output, Eu_dist, 1)
                    # 将 Pair_point 添加到 point_wise_in 的下方
                    # print("Pair_point", Pair_point)
                    # print("point_wise_in", point_wise_in)
                    for pair in Pair_point:
                        point_wise_in.append(pair)

        # P_Wmatrix_in = np.unique(point_wise_in, axis=0)  # 去重后的点对
        # P_Wmatrix_in = np.array([np.array(pair) for pair in point_wise_in], dtype=object)
        P_Wmatrix_in = list(OrderedDict.fromkeys(map(tuple, point_wise_in)))
        P_Wmatrix_in = np.array(P_Wmatrix_in)
        for row_num in range(P_Wmatrix_in.shape[0]):
            # a, b = P_Wmatrix_in[row_num, 0], P_Wmatrix_in[row_num, 1]
            # a, b = P_Wmatrix_in[row_num][0], P_Wmatrix_in[row_num][1]
            a, b = P_Wmatrix_in[row_num, 0], P_Wmatrix_in[row_num, 1]
            distance.append(Eu_dist[a, b])
            # distance.append(Eu_dist[a, b])  # 点对距离矩阵

        CLU_MST = clu_mst_tril[idx[:, np.newaxis], idx]
        if np.count_nonzero(CLU_MST > 0) != 0:
            special = np.mean(CLU_MST[np.where(CLU_MST != 0)])
        else:
            # special = np.mean(Eu_dist[idx[:, np.newaxis], idx])
            special = np.mean(np.mean(Eu_dist[idx][:, idx]))

        if pairwise >= 1:
            com[i - 1] = np.mean(distance)
        else:
            if np.count_nonzero(clu_mst_tril != 0) != 1:
                com[i - 1] = special  # 非孤立点，但是较为紧凑的簇
            else:
                com[i - 1] = 0  # 孤立点

    com[com == 0] = np.max(com)
    return com


def Separation_r1(data_new, output, Dataset_MST, knn, Eu_dist):
    k_value = len(np.unique(output))
    sep = np.zeros((k_value, k_value))
    # 首先判断两个簇的MST之间是否有边相连
    for i in range(1, k_value + 1):
        idx = np.where(output == i)[0]
        idx = idx.T
        num = 0
        clu_mst = Dataset_MST[idx, :]
        n = len(idx)
        for j in range(i + 1, k_value + 1):
            juli = []
            point_wise_out = []
            for L in range(n):
                v1 = np.where(clu_mst[L, :] != 0)[0]
                if len(v1) == 0:
                    continue
                elif len(np.where(output[v1] == j)[0]) > 0:
                    v2 = np.where(output[v1] == j)[0]
                    # print("v2",v2)
                    if len(v2) == 0:
                        continue
                    for J in v2:
                        p = idx[L]
                        q = v1[J]
                        # q = v1[J]
                        [pq_dist, _, Pair_point] = isharedist(data_new, p, q, knn[p], knn[q], output, Eu_dist, 2)
                        # point_wise_out = np.array(Pair_point)
                        for pair in Pair_point:
                            point_wise_out.append(pair)
                        # point_wise_out.append(Pair_point)  # 去重前的点对
                        juli.extend(pq_dist)
                        num += 1
            distance = []
            P_Wmatrix_out = list(OrderedDict.fromkeys(map(tuple, point_wise_out)))
            P_Wmatrix_out = np.array(P_Wmatrix_out)
            # P_Wmatrix_out = np.unique(np.array(point_wise_out), axis=0)  # 去重后的点对
            for row_num in range(P_Wmatrix_out.shape[0]):
                a = P_Wmatrix_out[row_num, 0]
                b = P_Wmatrix_out[row_num, 1]
                distance.append(Eu_dist[a, b])
                # distance.append(Eu_dist[a, b])  # 点对距离矩阵
            if len(juli) != 0:
                sep[i - 1, j - 1] = np.mean(distance)
                sep[j - 1, i - 1] = sep[i - 1, j - 1]
    return sep
