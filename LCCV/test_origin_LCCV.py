import math
import time

import pandas as pd
import matplotlib
from sklearn.preprocessing import StandardScaler, MinMaxScaler

matplotlib.use('TkAgg')

from computeClusterSim1 import *
from computeMySWC import *
from utils import *
from sklearn.decomposition import PCA


def HC_LCCV(A):
    N, dim = A.shape
    dist = np.zeros((N, N))
    for i in range(1, N):
        for j in range(1, N):
            for k in range(1, dim):
                dist[i, j] = dist[i, j] + (A[i, k] - A[j, k]) ** 2
            dist[i, j] = math.sqrt(dist[i, j])
    sdist = np.sort(dist, axis=1)
    sdist[0, :] = 0
    index = np.argsort(dist, axis=1)
    index[0, :] = 0
    print('Start running NaN-Searching algorithm...')
    r = 1
    nb = np.zeros(N, dtype=int)  # The number of each point's reverse neighbor
    count = 0
    count1 = 0
    flag = 0
    RNN = np.zeros((N, N))
    while flag == 0:
        for i in range(1, N):
            k = index[i, r + 1]
            nb[k] = nb[k] + 1
            RNN[k, nb[k]] = i
        r += 1
        count2 = 0
        for i in range(1, N):
            if nb[i] == 0:
                count2 = count2 + 1
        if count1 == count2:
            count = count + 1
        else:
            count = 1
        if count2 == 0 or (r > 2 and count >= 2):
            flag = 1
        count1 = count2
    supk = r - 1
    max_nb = np.max(nb)
    # max_nb = 12
    print('The characteristic value is {}'.format(supk))
    print('The maximum value of nb is {}'.format(max_nb))

    rho = np.zeros((N, 2))
    Non = max_nb
    for i in range(1, N):
        d = 0
        for j in range(1, Non + 2):
            d = d + sdist[i, j]
        rho[i, 1] = Non / d

    second_column = rho[:, 1]
    sorted_indices_in_row = np.argsort(second_column[1:])[::-1] + [1]
    ordrho = np.zeros((N, 2), dtype=int)
    for i in range(1, N):
        ordrho[i, 1] = sorted_indices_in_row[i - 1]
    sorted_values_in_row = np.sort(second_column[1::])[::-1]
    rho_sorted = np.zeros((N, 2))
    for i in range(1, N):
        rho_sorted[i, 1] = sorted_values_in_row[i - 1]
    local_core = np.zeros((N, 2), dtype=int)
    print('Starting running LORE algorithm...')
    for i in range(1, N):
        p = ordrho[i, 1]
        maxrho = rho[p, 1]
        maxindex = p
        # Find the point with maximum density in the local neighbors
        for j in range(1, nb[p] + 2):
            x = index[p, j]
            if maxrho < rho[x, 1]:
                maxrho = rho[x, 1]
                maxindex = x
        # Assign representative of the point with maximum density
        if local_core[maxindex, 1] == 0:
            local_core[maxindex, 1] = maxindex
        # Assign representative of the local neighbors
        for j in range(1, nb[p] + 2):
            if local_core[index[p, j], 1] == 0:
                local_core[index[p, j], 1] = local_core[maxindex, 1]
            else:
                # Determine the representative according to RCR
                q = local_core[index[p, j], 1]
                if dist[index[p, j], q] > dist[index[p, j], local_core[maxindex, 1]]:
                    local_core[index[p, j], 1] = local_core[maxindex, 1]
            # Determine the representative according to RTR
            for m in range(1, N):
                if local_core[m, 1] == index[p, j]:
                    local_core[m, 1] = local_core[index[p, j], 1]

    cluster_number = 0
    cl = np.zeros((N, 2))
    cores = [0]
    for i in range(1, N):
        if local_core[i, 1] == i:
            cluster_number += 1
            cores.append(i)
            cl[i, 1] = cluster_number
    for i in range(1, N):
        cl[i, 1] = cl[local_core[i, 1], 1]
    print('The number of initial clusters is', cluster_number)

    conn = np.zeros((N, N))
    weight = np.zeros((N, N))

    for i in range(1, N):
        for j in range(2, supk + 2):
            x = index[i, j]
            conn[i, x] = 1 / (1 + dist[i, x])
            conn[x, i] = conn[i, x]
            weight[i, x] = dist[i, x]

    print('Start computing the graph-based distance between local cores...')
    short_path = np.zeros((cluster_number + 1, cluster_number + 1))  # The shortest path between local cores
    weight2 = sp.csr_matrix(weight)
    for i in range(1, cluster_number + 1):
        short_path[i, i] = 0
        D, Z = dijkstra(weight2, cores[i])
        for j in range(i + 1, cluster_number + 1):
            short_path[i, j] = D[cores[j]]
            if short_path[i, j] == np.inf:
                short_path[i, j] = 0
            short_path[j, i] = short_path[i, j]
    maxd = np.max(short_path[1:, 1:])
    for i in range(1, cluster_number + 1):
        for j in range(1, cluster_number + 1):
            if short_path[i, j] == 0:
                short_path[i, j] = maxd

    results = np.empty((cluster_number, 6), dtype=object)

    print('Compute the similarity between clusters')
    cdata = [None] * (cluster_number + 1)
    nc = np.zeros(cluster_number + 1, dtype=int)

    for i in range(1, cluster_number + 1):
        nc[i] = 0
        for j in range(1, N):
            if cl[j, 1] == i:
                nc[i] += 1
                if cdata[i] is None:
                    cdata[i] = [0]
                cdata[i].append(j)
    sim = computeClusterSim1(cdata, conn, cluster_number, nc)
    results[1][1] = cdata.copy()
    results[1][2] = cl.copy()

    print('Merge the small clusters')
    small_threshold = (N - 1) / cluster_number
    clunum2 = cluster_number
    canClu = np.zeros(N + 1, dtype=int)
    for i in range(1, cluster_number + 1):
        if nc[i] <= small_threshold:
            v = 0
            ic = 0
            for j in range(1, cluster_number + 1):
                if sim[i][j] > v:
                    ic = j
                    v = sim[i][j]

            # If there are no clusters connecting the small clusters, the small clusters are considered as outliers
            if ic == 0 and nc[i] < small_threshold / 2:
                for j in range(1, nc[i] + 1):
                    x = cdata[i][j]
                    # isnoise[x] = 1
                    cl[x, 1] = 0
                cdata[i] = [0]
                nc[i] = 0
                clunum2 -= 1

            if ic > 0:
                clunum2 -= 1
                # Merge the clusters
                for j in range(1, nc[i] + 1):
                    cdata[ic].append(cdata[i][j])
                nc[ic] += nc[i]
                ncand = 0
                for k in range(1, cluster_number + 1):
                    if k != ic and k != i:
                        if sim[ic][k] != 0 or sim[i][k] != 0:
                            ncand += 1
                            canClu[ncand] = k

                # Update the similarity matrix
                for j in range(1, ncand + 1):
                    ncutedge = 0
                    sumcutedge = 0
                    if canClu[j] != ic and canClu[j] != i:
                        for k in range(1, nc[ic] + 1):
                            for m in range(1, nc[canClu[j]] + 1):
                                if conn[cdata[ic][k]][cdata[canClu[j]][m]] != 0:
                                    ncutedge += 1
                                    sumcutedge += conn[cdata[ic][k]][cdata[canClu[j]][m]]
                    if ncutedge != 0:
                        ave = sumcutedge / ncutedge
                        sim[ic][canClu[j]] = ave ** 3 * ncutedge
                        sim[canClu[j]][ic] = sim[ic][canClu[j]]

                # Set all elements in row i to 0
                for j in range(len(sim[i])):
                    sim[i][j] = 0

                # Set all elements in column i to 0
                for row in sim:
                    row[i] = 0

                nc[i] = 0
                cdata[i] = [0]

    # Obtain the clustering result after merging small clusters
    for i in range(1, cluster_number + 1):
        if cdata[i] is not None:
            for item in cdata[i][1:]:
                cl[item, 1] = i
    for i in range(1, cluster_number + 1):
        if nc[i] == 0:
            for j in range(i + 1, cluster_number + 1):
                if cdata[j] is not None:
                    for item in cdata[j][1:]:
                        cl[item, 1] -= 1

    mcv = computeMySWC(A, cl, clunum2, cores, short_path, local_core)
    results[cluster_number - clunum2 + 1, 1] = cdata.copy()
    results[cluster_number - clunum2 + 1, 3] = sim.copy()
    results[cluster_number - clunum2 + 1, 2] = cl.copy()
    if results[cluster_number - clunum2 + 1, 4] is None:
        results[cluster_number - clunum2 + 1, 4] = np.zeros((2, 3), dtype=object)
    results[cluster_number - clunum2 + 1, 4][1, 1] = mcv
    results[cluster_number - clunum2 + 1, 4][1, 2] = clunum2

    print('Start merging clusters...')
    canditateClu = np.zeros(N, dtype=int)
    clunum = clunum2
    while clunum > 2:
        print('The number of clusters is {}'.format(clunum))
        row_idx = cluster_number - clunum + 1
        results[row_idx, 1] = cdata.copy()
        results[row_idx, 3] = sim.copy()
        results[row_idx, 2] = cl.copy()
        if results[row_idx, 4] is None:
            results[row_idx, 4] = np.zeros((2, 3), dtype=object)
        results[row_idx, 4][1, 1] = mcv
        results[row_idx, 4][1, 2] = clunum
        maxsim = 0
        idx, idy = -1, -1
        for i in range(1, cluster_number + 1):
            for j in range(1, cluster_number + 1):
                if sim[i, j] > maxsim:
                    maxsim = sim[i, j]
                    idx, idy = i, j
        results[row_idx, 5] = np.array([idx, idy])
        # print('In {} times merging, to be merged clusters are: {}, {}'.format(cluster_number - clunum + 1, idx, idy))
        if idx != idy:
            if idx > idy:
                idx, idy = idy, idx
            # cdata[idx] += cdata[idy]  #?
            # nc[idx] += nc[idy]
            # cdata[idy] = []
            for i in range(1, nc[idy] + 1):
                if cdata[idx].__len__() == 0:
                    cdata[idx] = [0]
                cdata[idx].append(cdata[idy][i])  # 将 cdata{1,idy} 中的每个元素添加到 cdata{1,idx} 的末尾
            nc[idx] += nc[idy]  # 更新 nc(idx)
            cdata[idy] = [0]
            nc[idy] = 0
            clunum -= 1
            ncand = 0
            for i in range(1, cluster_number + 1):
                if i != idx and i != idy:
                    if sim[idx, i] != 0 or sim[idy, i] != 0:
                        ncand += 1
                        canditateClu[ncand] = i
            for j in range(1, ncand + 1):
                ncutedge = 0
                sumcutedge = 0
                if canditateClu[j] != idx and canditateClu[j] != idy:
                    for i in range(1, nc[idx] + 1):
                        for m in range(1, nc[canditateClu[j]] + 1):
                            if conn[cdata[idx][i], cdata[canditateClu[j]][m]] != 0:
                                ncutedge += 1
                                sumcutedge += conn[cdata[idx][i], cdata[canditateClu[j]][m]]
                if ncutedge != 0:
                    ave = sumcutedge / ncutedge
                    sim[idx, canditateClu[j]] = ave ** 3 * ncutedge
                    sim[canditateClu[j], idx] = sim[idx, canditateClu[j]]
            sim[idy, :] = 0
            sim[:, idy] = 0
            for i in range(1, cluster_number + 1):
                if cdata[i] is not None:
                    cl[cdata[i][1:], 1] = i
            for i in range(1, cluster_number + 1):
                if nc[i] == 0:
                    for j in range(i + 1, cluster_number + 1):
                        if cdata[j] is not None:
                            cl[cdata[j][1:], 1] -= 1
            mcv = computeMySWC(A, cl, clunum, cores, short_path, local_core)

            if clunum == 2:
                print('The last merge')
                results[cluster_number - clunum + 1, 1] = cdata.copy()
                results[cluster_number - clunum + 1, 3] = sim.copy()
                results[cluster_number - clunum + 1, 2] = cl.copy()
                if results[cluster_number - clunum + 1, 4] is None:
                    results[cluster_number - clunum + 1, 4] = np.zeros((2, 3), dtype=object)
                results[cluster_number - clunum + 1, 4][1, 1] = mcv
                results[cluster_number - clunum + 1, 4][1, 2] = clunum
        else:
            print('There are no clusters to be merged')
            results[cluster_number - 1, :] = np.empty(6, dtype=object)
            break

    cv = np.full(cluster_number, -np.inf)
    for i in range(1, cluster_number):
        if results[i][4] is not None:  # Assuming the 4th element in MATLAB corresponds to the 3rd index in Python
            cv[i] = results[i][4][1][1]

    id = np.argmax(cv)

    bestcl = results[id][2]
    unique_nums, counts = np.unique(bestcl, return_counts=True)

    print('Process end')

    return bestcl


if __name__ == '__main__':
    datasets = ['n2', 'n3', 'face', 'D13', 'spirals', 'network', 'basic5', 'basic3', 'un2', 'boxes',
                'basic1', 'basic4']
    for i in range(datasets.__len__()):
        keys = [datasets[i]]
        st = time.time()
        data_path = "D:\\02_code\\01_Python\\Internal-index\\HB-Acceleration\\datasets\\data\\"
        data = pd.read_csv(data_path + keys[0] + ".csv", header=None).values
        if data.shape[1] >= 3:
            print("降维中....")
            data = StandardScaler().fit_transform(data)
            data = MinMaxScaler(feature_range=(0, 1)).fit_transform(data)
            pca = PCA(n_components=2)
            data = pca.fit(data).transform(data)
            data = MinMaxScaler(feature_range=(0, 1)).fit_transform(data)
            print("降维结束")
        else:
            data = MinMaxScaler(feature_range=(0, 1)).fit_transform(data)
        rows, cols = data.shape
        new_array = np.zeros((rows + 1, cols + 1), dtype=data.dtype)
        new_array[1:, 1:] = data
        data = new_array

        bestcl = HC_LCCV(data)
        bestcl = bestcl[1:, 1:]
        et = time.time()
        t = et - st
        unique_elements = np.unique(bestcl)
        on = len(unique_elements)
        print(f"dataset is: {datasets[i]}, 原始LCCV经历了：{t:.2f}s ,on为：{on}")
        print("-----------------------------------------------------")
